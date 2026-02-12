# エラーハンドリング ── 例外・Result型・堅牢なエラー処理

> エラーハンドリングはプログラムの信頼性を決定づける。正常系だけを書くのは簡単だが、異常系を適切に処理するコードこそがプロフェッショナルの仕事である。例外・Result型・エラーコードの使い分けを理解し、堅牢なシステムを構築する。
>
> -- Robert C. Martin, *Clean Code* Chapter 7

---

## この章で学ぶこと

1. **エラーハンドリングの基本戦略** ── 例外、Result型、エラーコードの使い分けと、それぞれの言語での推奨パターンを理解する
2. **例外設計の原則** ── 検査例外 vs 非検査例外、カスタム例外階層の設計、レイヤー間の例外変換を身につける
3. **Result型による型安全なエラー処理** ── TypeScript・Python・Rustでの実装パターンとモナド的合成を習得する
4. **堅牢なエラー処理パターン** ── Fail Fast、ガードクローズ、Circuit Breaker、グレースフルデグラデーションを実践する
5. **構造化ロギングとエラー監視** ── 本番環境でのエラー追跡・分析・アラートの設計を理解する

---

## 前提知識

| トピック | 必要レベル | 参照ガイド |
|---------|-----------|-----------|
| 関数設計 | 基本 | [関数設計](./01-functions.md) |
| 型システム | 基本 | 各言語の型システムドキュメント |
| クリーンコード概要 | 理解済み | [クリーンコード概要](../00-principles/00-clean-code-overview.md) |
| 結合度・凝集度 | 理解済み | [結合度と凝集度](../00-principles/03-coupling-cohesion.md) |
| デザインパターン | あると望ましい | GoFパターン参照 |

---

## なぜエラーハンドリングが重要なのか

### エラー処理の品質がシステム全体の信頼性を決める

```
  エラー処理品質とシステム障害の関係

  障害率
  (月あたり)
    │
  20│ ●  エラー握りつぶし
    │
  15│    ●  広範囲catch + ログのみ
    │
  10│       ●  基本的な例外処理
    │
   5│          ●  カスタム例外階層
    │              ●  Result型 + 監視
   1│                 ●  Fail Fast + Circuit Breaker + 構造化ログ
    └──────────────────────────────────────────
     低い ──────── エラー処理の成熟度 ────────── 高い
```

### エラー処理が不十分な場合のコスト

| 問題 | 影響 | 実例 |
|------|------|------|
| 例外の握りつぶし | データ消失・整合性破壊 | 決済失敗が通知されず、商品だけ発送される |
| 不適切なリトライ | 障害の増幅 | 決済APIに無限リトライして二重請求発生 |
| エラー情報の不足 | 調査時間の増大 | 「エラーが発生しました」だけのログで原因調査に数日 |
| エラーの過剰キャッチ | バグの隠蔽 | NullPointerExceptionが握りつぶされ、根本原因が発見されない |
| セキュリティ情報のリーク | 攻撃者への情報提供 | スタックトレースがユーザーに表示され、内部構造が露出 |

### Ariane 5の教訓

1996年のAriane 5ロケット打ち上げ失敗は、エラーハンドリングの失敗が引き起こした。64ビット浮動小数点を16ビット整数に変換するコードでオーバーフロー例外が発生したが、適切にハンドリングされずシステムが停止、ロケットは自爆した。損失額は約5億ドル。

**教訓: エラーは必ず発生する。問題はエラーが起きるかどうかではなく、起きたときにどう対処するかである。**

---

## 1. エラーハンドリング戦略の全体像

### 1.1 3つのアプローチ

```
+---------------------------------------------------------------------+
|  エラー処理の3つのアプローチ                                          |
+---------------------------------------------------------------------+
|  1. 例外 (Exception)                                                |
|     → 予期しないエラーをスタックを遡って処理                         |
|     → Java, Python, C#, TypeScript, Ruby, Kotlin                    |
|     → 利点: 正常フローとエラーフローの分離                            |
|     → 欠点: 処理し忘れのリスク、パフォーマンスコスト                  |
+---------------------------------------------------------------------+
|  2. Result/Either型                                                 |
|     → 成功/失敗を型で表現、コンパイル時にチェック                    |
|     → Rust (Result), Go (error), Haskell (Either), Scala (Try)      |
|     → 利点: 型安全、処理強制、ゼロコスト                              |
|     → 欠点: コードが冗長になりがち                                    |
+---------------------------------------------------------------------+
|  3. エラーコード（レガシー）                                         |
|     → 関数の戻り値でエラーを示す                                     |
|     → C言語の伝統。現代では非推奨                                    |
|     → 利点: シンプル、パフォーマンス                                  |
|     → 欠点: 無視されやすい、型安全でない                              |
+---------------------------------------------------------------------+
```

### 1.2 エラーの分類

エラーを正しく分類することが、適切な処理戦略を選ぶ第一歩である。

```
  エラーの分類と対処方針

  ┌─────────────────────────────────────────────────────────────┐
  │              プログラミングエラー（Bug）                      │
  │  ・null参照、配列範囲外、型エラー、assertion failure         │
  │  → 修正すべきバグ。Fail Fastで即座に停止                     │
  │  → 例外（非検査）で報告。リトライ不可                        │
  ├─────────────────────────────────────────────────────────────┤
  │              ビジネスエラー（Expected Failure）               │
  │  ・在庫不足、残高不足、権限なし、バリデーション失敗          │
  │  → 想定内のエラー。Result型や専用例外で表現                  │
  │  → ユーザーに適切なメッセージを返す                          │
  ├─────────────────────────────────────────────────────────────┤
  │              インフラエラー（Transient Failure）              │
  │  ・DB接続失敗、ネットワークタイムアウト、ディスクフル        │
  │  → 一時的な障害。リトライ or Circuit Breaker                 │
  │  → グレースフルデグラデーション                               │
  ├─────────────────────────────────────────────────────────────┤
  │              致命的エラー（Fatal Error）                      │
  │  ・メモリ不足、設定ファイル破損、起動時の依存関係不足        │
  │  → リカバリ不可。ログを残して安全に終了                      │
  │  → プロセス監視で自動再起動                                  │
  └─────────────────────────────────────────────────────────────┘
```

### 1.3 エラー処理の判断フローチャート

```
  エラーが発生
       │
       ▼
  ┌───────────────────┐
  │ プログラミングエラー? │
  └────┬──────┬────────┘
       │Yes   │No
       ▼      ▼
  Fail Fast   ┌───────────────────┐
  (即座停止)  │ ビジネスエラー?    │
              └────┬──────┬────────┘
                   │Yes   │No
                   ▼      ▼
           Result型 or   ┌───────────────────┐
           専用例外      │ 一時的なエラー?    │
                         └────┬──────┬────────┘
                              │Yes   │No
                              ▼      ▼
                         リトライ   致命的エラー
                         + Circuit  → ログ + 安全終了
                         Breaker
```

---

## 2. 例外設計

### 2.1 例外の基本原則

| 原則 | 説明 |
|------|------|
| 具体的な例外をスローする | `Exception` ではなく `UserNotFoundError` |
| 具体的な例外をキャッチする | `Exception` ではなく `PaymentDeclinedError` |
| try-catchの範囲を最小にする | 失敗しうる操作のみを囲む |
| 例外を変換してレイヤーを越える | インフラ例外をドメイン例外に変換 |
| 例外に十分なコンテキストを含める | 何が、なぜ、どのIDで |
| 機密情報を例外に含めない | パスワード、トークンは絶対にNG |

### 2.2 カスタム例外の階層設計

**コード例1: アプリケーション例外階層（Python）**

```python
"""
アプリケーション固有の例外階層

設計方針:
- すべてのアプリケーション例外は AppError を継承
- エラーコード（文字列）でプログラム的に識別可能
- HTTPステータスコードとの対応を持つ
- 機械可読なエラー情報と人間可読なメッセージの両方を提供
"""
from __future__ import annotations
from typing import Any
from datetime import datetime


class AppError(Exception):
    """アプリケーションの基底例外"""

    def __init__(
        self,
        message: str,
        code: str = "UNKNOWN",
        status_code: int = 500,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.details = details or {}
        self.timestamp = datetime.utcnow()

    def to_dict(self) -> dict[str, Any]:
        """API レスポンス用の辞書表現"""
        return {
            "error": {
                "code": self.code,
                "message": str(self),
                "details": self.details,
                "timestamp": self.timestamp.isoformat(),
            }
        }


# ── ビジネスエラー群 ──────────────────────────────

class ValidationError(AppError):
    """入力バリデーションエラー（400 Bad Request）"""

    def __init__(self, field: str, message: str, value: Any = None):
        super().__init__(
            message,
            code="VALIDATION_ERROR",
            status_code=400,
            details={"field": field, "rejected_value": repr(value)},
        )
        self.field = field


class NotFoundError(AppError):
    """リソースが見つからないエラー（404 Not Found）"""

    def __init__(self, resource: str, identifier: str):
        super().__init__(
            f"{resource} (ID: {identifier}) が見つかりません",
            code="NOT_FOUND",
            status_code=404,
            details={"resource": resource, "identifier": identifier},
        )
        self.resource = resource
        self.identifier = identifier


class ConflictError(AppError):
    """リソース競合エラー（409 Conflict）"""

    def __init__(self, resource: str, message: str):
        super().__init__(
            message,
            code="CONFLICT",
            status_code=409,
            details={"resource": resource},
        )


class AuthorizationError(AppError):
    """権限不足エラー（403 Forbidden）"""

    def __init__(self, action: str, resource: str | None = None):
        msg = f"操作 '{action}' の権限がありません"
        if resource:
            msg += f" (対象: {resource})"
        super().__init__(
            msg,
            code="FORBIDDEN",
            status_code=403,
            details={"action": action, "resource": resource},
        )


class BusinessRuleViolationError(AppError):
    """ビジネスルール違反エラー（422 Unprocessable Entity）"""

    def __init__(self, rule: str, message: str):
        super().__init__(
            message,
            code="BUSINESS_RULE_VIOLATION",
            status_code=422,
            details={"rule": rule},
        )


# ── インフラエラー群 ──────────────────────────────

class InfrastructureError(AppError):
    """インフラストラクチャ基底例外"""

    def __init__(self, message: str, code: str = "INFRASTRUCTURE_ERROR"):
        super().__init__(message, code=code, status_code=503)


class DataAccessError(InfrastructureError):
    """データベースアクセスエラー"""

    def __init__(self, message: str, cause: Exception | None = None):
        super().__init__(message, code="DATA_ACCESS_ERROR")
        self.__cause__ = cause


class ExternalServiceError(InfrastructureError):
    """外部サービス通信エラー"""

    def __init__(self, service: str, message: str, cause: Exception | None = None):
        super().__init__(
            f"外部サービス '{service}' エラー: {message}",
            code="EXTERNAL_SERVICE_ERROR",
        )
        self.service = service
        self.__cause__ = cause


# ── 使用例 ────────────────────────────────────────

class UserService:
    def __init__(self, repository, auth_service):
        self.repository = repository
        self.auth_service = auth_service

    def get_user(self, user_id: str) -> User:
        """ユーザーを取得する。見つからない場合は NotFoundError。"""
        user = self.repository.find_by_id(user_id)
        if user is None:
            raise NotFoundError("User", user_id)
        return user

    def update_email(self, user_id: str, new_email: str, actor: User) -> User:
        """ユーザーのメールアドレスを更新する。"""
        # 権限チェック
        if not self.auth_service.can(actor, "update_user", user_id):
            raise AuthorizationError("update_user", f"User:{user_id}")

        # バリデーション
        if not self._is_valid_email(new_email):
            raise ValidationError("email", "不正なメールアドレス形式です", new_email)

        # 重複チェック
        existing = self.repository.find_by_email(new_email)
        if existing and existing.id != user_id:
            raise ConflictError("User", f"メールアドレス '{new_email}' は既に使用されています")

        user = self.get_user(user_id)
        user.email = new_email
        return self.repository.save(user)
```

### 2.3 try-catchの適切な範囲

**コード例2: try-catchの範囲設計（Java）**

```java
// ──────────────────────────────────────────────────
// 悪い例: try の範囲が広すぎる
// ──────────────────────────────────────────────────
// 問題点:
// 1. どの操作で例外が発生したかわからない
// 2. catch(Exception) で全例外を潰している
// 3. メール送信失敗で注文全体がロールバックされてしまう
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

// ──────────────────────────────────────────────────
// 良い例: 操作の重要度に応じてtry-catchを分離
// ──────────────────────────────────────────────────
// 原則:
// 1. 失敗が許されない操作（決済）は細かくキャッチ
// 2. 失敗しても業務に影響しない操作（通知）は別のtryブロック
// 3. 例外は具体的にキャッチ

// Phase 1: 前準備（例外なら処理不要 → tryなし）
User user = userRepository.findById(userId);
if (user == null) {
    return OrderResult.userNotFound(userId);
}

Order order = new Order(user);
order.addItem(item);
order.calculateTotal();

// Phase 2: 決済（失敗は業務に直結 → 細かくキャッチ）
try {
    paymentService.charge(order);
} catch (PaymentDeclinedException e) {
    logger.info("決済拒否: userId={}, reason={}", userId, e.getReason());
    return OrderResult.paymentFailed(e.getReason());
} catch (PaymentGatewayException e) {
    logger.error("決済ゲートウェイ接続エラー: orderId={}", order.getId(), e);
    return OrderResult.systemError("決済システムに接続できません。しばらくしてからお試しください。");
}

// Phase 3: 注文確定（ここで失敗してはいけない）
orderRepository.save(order);

// Phase 4: 通知（失敗しても注文はロールバックしない）
try {
    emailService.sendConfirmation(user, order);
} catch (EmailServiceException e) {
    logger.warn("確認メール送信失敗（注文自体は成功）: orderId={}", order.getId(), e);
    // 失敗した通知を再送キューに入れる
    notificationRetryQueue.enqueue(new EmailNotification(user, order));
}

// Phase 5: 分析（完全にオプション）
try {
    analyticsService.track(order);
} catch (Exception e) {
    logger.debug("分析イベント送信失敗（無視）: orderId={}", order.getId(), e);
    // 分析データの欠損は許容する
}
```

### 2.4 例外の変換（レイヤー間）

例外はレイヤー境界で変換する。これにより上位レイヤーが下位レイヤーの実装詳細に依存しなくなる。

```
  例外の変換フロー

  ┌──────────────────────────────────┐
  │  コントローラ層                   │  HttpException → HTTP Response
  │  ドメイン例外 → HTTPステータス   │  (404, 403, 422, 500)
  ├──────────────────────────────────┤
  │  サービス層                       │  ドメイン例外をそのまま伝播
  │  ビジネスルールの検証             │  (新たな例外を発生させることも)
  ├──────────────────────────────────┤
  │  リポジトリ層                     │  インフラ例外 → ドメイン例外
  │  SQLError → DataAccessError      │  に変換
  ├──────────────────────────────────┤
  │  インフラ層                       │  生のインフラ例外
  │  SQLError, ConnectionError, etc  │  (上位に直接公開しない)
  └──────────────────────────────────┘
```

**コード例3: レイヤー間の例外変換（TypeScript）**

```typescript
// ====================================================================
// インフラ層: 生の例外
// ====================================================================
// pg (PostgreSQL client) は DatabaseError をスローする

// ====================================================================
// リポジトリ層: インフラ例外 → ドメイン例外 に変換
// ====================================================================
class UserRepository {
  constructor(private db: DatabasePool) {}

  async findById(id: string): Promise<User> {
    try {
      const row = await this.db.query(
        'SELECT * FROM users WHERE id = $1',
        [id]
      );

      if (!row) {
        throw new UserNotFoundError(id);
      }

      return this.mapToUser(row);
    } catch (error) {
      // ドメイン例外はそのまま再スロー
      if (error instanceof AppError) throw error;

      // PostgreSQL の一意制約違反 → ドメイン例外
      if (error instanceof DatabaseError && error.code === '23505') {
        throw new ConflictError('User', `ID ${id} は既に存在します`);
      }

      // その他のインフラ例外をラップ
      throw new DataAccessError(
        `ユーザー取得失敗 (ID: ${id})`,
        { cause: error }
      );
    }
  }

  async save(user: User): Promise<User> {
    try {
      await this.db.query(
        'INSERT INTO users (id, name, email) VALUES ($1, $2, $3) ' +
        'ON CONFLICT (id) DO UPDATE SET name = $2, email = $3',
        [user.id, user.name, user.email]
      );
      return user;
    } catch (error) {
      if (error instanceof DatabaseError && error.code === '23505') {
        throw new ConflictError('User', `メールアドレス '${user.email}' は既に使用されています`);
      }
      throw new DataAccessError(
        `ユーザー保存失敗 (ID: ${user.id})`,
        { cause: error }
      );
    }
  }
}

// ====================================================================
// サービス層: ドメイン例外をそのまま伝播（または新規発生）
// ====================================================================
class UserService {
  constructor(
    private userRepository: UserRepository,
    private emailService: EmailService,
  ) {}

  async getUser(id: string): Promise<User> {
    // リポジトリの例外（NotFoundError, DataAccessError）はそのまま伝播
    return this.userRepository.findById(id);
  }

  async updateEmail(userId: string, newEmail: string): Promise<User> {
    const user = await this.userRepository.findById(userId);

    // ビジネスルールの検証（新しいドメイン例外を発生）
    if (!this.isValidEmail(newEmail)) {
      throw new ValidationError('email', '不正なメールアドレス形式です');
    }

    user.email = newEmail;
    return this.userRepository.save(user);
  }
}

// ====================================================================
// コントローラ層: ドメイン例外 → HTTPレスポンスに変換
// ====================================================================
class UserController {
  constructor(private userService: UserService) {}

  async getUser(req: Request, res: Response): Promise<void> {
    try {
      const user = await this.userService.getUser(req.params.id);
      res.json(user);
    } catch (error) {
      this.handleError(res, error);
    }
  }

  async updateEmail(req: Request, res: Response): Promise<void> {
    try {
      const user = await this.userService.updateEmail(
        req.params.id,
        req.body.email
      );
      res.json(user);
    } catch (error) {
      this.handleError(res, error);
    }
  }

  private handleError(res: Response, error: unknown): void {
    if (error instanceof AppError) {
      // ドメイン例外 → 適切なHTTPステータスコード
      res.status(error.statusCode).json(error.toDict());
    } else {
      // 予期しない例外 → 500
      console.error('予期しないエラー:', error);
      res.status(500).json({
        error: {
          code: 'INTERNAL_ERROR',
          message: 'サーバーエラーが発生しました',
        },
      });
    }
  }
}

// ====================================================================
// グローバルエラーハンドラ（Express ミドルウェア）
// ====================================================================
function globalErrorHandler(
  error: Error,
  req: Request,
  res: Response,
  next: NextFunction
): void {
  // 構造化ログ
  const logEntry = {
    timestamp: new Date().toISOString(),
    method: req.method,
    path: req.path,
    errorType: error.constructor.name,
    message: error.message,
    stack: error.stack,
    requestId: req.headers['x-request-id'],
  };

  if (error instanceof AppError) {
    if (error.statusCode >= 500) {
      logger.error(logEntry);
    } else {
      logger.warn(logEntry);
    }
    res.status(error.statusCode).json(error.toDict());
  } else {
    logger.error(logEntry);
    res.status(500).json({
      error: {
        code: 'INTERNAL_ERROR',
        message: 'サーバーエラーが発生しました',
      },
    });
  }
}
```

### 2.5 検査例外 vs 非検査例外（Java固有の議論）

Javaの検査例外（Checked Exception）は長い間議論の的になってきた。

```
  検査例外のメリットとデメリット

  ┌──────────────────────────────────────────────────┐
  │  検査例外 (Checked Exception)                     │
  │  ・FileNotFoundException, SQLException            │
  │  ・メリット: 呼び出し側に処理を強制できる         │
  │  ・デメリット: OCP違反（下位の変更がシグネチャに波及）│
  │  ・デメリット: catch の氾濫で可読性低下            │
  │  → Robert C. Martin: 「検査例外の代償は高すぎる」  │
  ├──────────────────────────────────────────────────┤
  │  非検査例外 (Unchecked Exception)                 │
  │  ・NullPointerException, IllegalArgumentException │
  │  ・メリット: シグネチャが清潔に保たれる           │
  │  ・デメリット: 処理し忘れのリスク                  │
  │  → Kotlin, Scala, C# は非検査例外のみ採用        │
  └──────────────────────────────────────────────────┘
```

**コード例4: 検査例外のOCP違反問題**

```java
// 検査例外がOpen-Closed Principle を違反する例

// Step 1: 最初の実装
class UserRepository {
    User findById(String id) throws SQLException {  // SQLExceptionを宣言
        return db.query("SELECT ...");
    }
}

class UserService {
    User getUser(String id) throws SQLException {   // 伝播が必要
        return repository.findById(id);
    }
}

class UserController {
    void getUser(Request req) throws SQLException {  // さらに伝播
        userService.getUser(req.getParam("id"));
    }
}

// Step 2: MongoDB に変更した場合
// → findById の throws を MongoException に変更
// → UserService, UserController の throws も全部変更が必要！
// → Open-Closed Principle 違反

// 解決策: 非検査例外にラップする
class UserRepository {
    User findById(String id) {
        try {
            return db.query("SELECT ...");
        } catch (SQLException e) {
            throw new DataAccessException("ユーザー取得失敗", e);  // 非検査例外
        }
    }
}
// → UserService, UserController は変更不要
```

---

## 3. Result型パターン

### なぜResult型が必要なのか

例外の問題点を整理する。

```
  例外の問題点

  1. 関数のシグネチャに現れない（TypeScript, Python）
     function getUser(id: string): User
     // この関数が UserNotFoundError をスローすることは
     // シグネチャからは分からない

  2. catch し忘れてもコンパイルエラーにならない
     const user = getUser(id);  // 例外発生 → クラッシュ

  3. パフォーマンスコスト
     例外生成時にスタックトレースを構築 → 高コスト
     ビジネスエラー（残高不足等）は頻繁に発生する可能性あり
```

Result型はこれらの問題を解決する。

### 3.1 TypeScript での Result型

**コード例5: Result型の完全実装（TypeScript）**

```typescript
// ====================================================================
// Result型の定義
// ====================================================================
type Result<T, E = Error> =
  | { readonly ok: true; readonly value: T }
  | { readonly ok: false; readonly error: E };

// ファクトリ関数
function ok<T>(value: T): Result<T, never> {
  return { ok: true, value };
}

function err<E>(error: E): Result<never, E> {
  return { ok: false, error };
}

// ====================================================================
// Result型のユーティリティ関数
// ====================================================================

/** Result を変換する（成功時のみ） */
function map<T, U, E>(
  result: Result<T, E>,
  fn: (value: T) => U
): Result<U, E> {
  if (result.ok) {
    return ok(fn(result.value));
  }
  return result;
}

/** Result をフラットマップする（チェイニング） */
function flatMap<T, U, E>(
  result: Result<T, E>,
  fn: (value: T) => Result<U, E>
): Result<U, E> {
  if (result.ok) {
    return fn(result.value);
  }
  return result;
}

/** 例外をスローする関数を Result に変換 */
function tryCatch<T>(fn: () => T): Result<T, Error> {
  try {
    return ok(fn());
  } catch (error) {
    return err(error instanceof Error ? error : new Error(String(error)));
  }
}

/** async 版 */
async function tryCatchAsync<T>(
  fn: () => Promise<T>
): Promise<Result<T, Error>> {
  try {
    return ok(await fn());
  } catch (error) {
    return err(error instanceof Error ? error : new Error(String(error)));
  }
}

/** 複数の Result をまとめる */
function all<T, E>(results: Result<T, E>[]): Result<T[], E> {
  const values: T[] = [];
  for (const result of results) {
    if (!result.ok) return result;
    values.push(result.value);
  }
  return ok(values);
}

// ====================================================================
// 実践例: 送金処理
// ====================================================================

// ビジネスエラーの型定義（Discriminated Union）
type TransferError =
  | { type: 'INSUFFICIENT_BALANCE'; available: number; requested: number }
  | { type: 'ACCOUNT_LOCKED'; reason: string }
  | { type: 'DAILY_LIMIT_EXCEEDED'; limit: number; current: number }
  | { type: 'SAME_ACCOUNT'; accountId: string }
  | { type: 'INVALID_AMOUNT'; amount: number };

function validateTransferAmount(amount: number): Result<number, TransferError> {
  if (amount <= 0 || !Number.isFinite(amount)) {
    return err({ type: 'INVALID_AMOUNT', amount });
  }
  return ok(amount);
}

function checkAccountStatus(account: Account): Result<Account, TransferError> {
  if (account.isLocked) {
    return err({ type: 'ACCOUNT_LOCKED', reason: account.lockReason });
  }
  return ok(account);
}

function checkBalance(
  account: Account,
  amount: number
): Result<Account, TransferError> {
  if (account.balance < amount) {
    return err({
      type: 'INSUFFICIENT_BALANCE',
      available: account.balance,
      requested: amount,
    });
  }
  return ok(account);
}

function checkDailyLimit(
  account: Account,
  amount: number
): Result<Account, TransferError> {
  const DAILY_LIMIT = 1_000_000;
  if (account.dailyTransferred + amount > DAILY_LIMIT) {
    return err({
      type: 'DAILY_LIMIT_EXCEEDED',
      limit: DAILY_LIMIT,
      current: account.dailyTransferred,
    });
  }
  return ok(account);
}

function transfer(
  from: Account,
  to: Account,
  amount: number
): Result<TransferReceipt, TransferError> {
  // 同一口座チェック
  if (from.id === to.id) {
    return err({ type: 'SAME_ACCOUNT', accountId: from.id });
  }

  // バリデーションチェーン
  const amountResult = validateTransferAmount(amount);
  if (!amountResult.ok) return amountResult as Result<never, TransferError>;

  const statusResult = checkAccountStatus(from);
  if (!statusResult.ok) return statusResult as Result<never, TransferError>;

  const balanceResult = checkBalance(from, amount);
  if (!balanceResult.ok) return balanceResult as Result<never, TransferError>;

  const limitResult = checkDailyLimit(from, amount);
  if (!limitResult.ok) return limitResult as Result<never, TransferError>;

  // すべてのチェックを通過 → 送金実行
  from.balance -= amount;
  to.balance += amount;
  from.dailyTransferred += amount;

  return ok(new TransferReceipt(from, to, amount, new Date()));
}

// 呼び出し側: 全エラーケースの処理が強制される
const result = transfer(accountA, accountB, 50000);

if (result.ok) {
  console.log(`送金成功: ${result.value.receiptId}`);
} else {
  // TypeScript の型絞り込みにより、各ケースで適切な型が推論される
  switch (result.error.type) {
    case 'INSUFFICIENT_BALANCE':
      console.log(
        `残高不足: 残高 ${result.error.available}円 < 送金額 ${result.error.requested}円`
      );
      break;
    case 'ACCOUNT_LOCKED':
      console.log(`口座ロック中: ${result.error.reason}`);
      break;
    case 'DAILY_LIMIT_EXCEEDED':
      console.log(
        `日次上限超過: 上限 ${result.error.limit}円, 本日送金済み ${result.error.current}円`
      );
      break;
    case 'SAME_ACCOUNT':
      console.log(`同一口座への送金はできません: ${result.error.accountId}`);
      break;
    case 'INVALID_AMOUNT':
      console.log(`不正な金額: ${result.error.amount}`);
      break;
  }
}
```

### 3.2 Python での Result型

**コード例6: Python Result型（パターンマッチング対応）**

```python
"""
Python 3.10+ のパターンマッチングを活用した Result 型実装。
Rust の Result<T, E> に近い API を提供する。
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Generic, TypeVar, Callable, Union

T = TypeVar('T')
U = TypeVar('U')
E = TypeVar('E')


@dataclass(frozen=True)
class Ok(Generic[T]):
    """成功を表す型"""
    value: T

    def is_ok(self) -> bool:
        return True

    def is_err(self) -> bool:
        return False

    def map(self, fn: Callable[[T], U]) -> Result[U, E]:
        """成功値を変換する"""
        return Ok(fn(self.value))

    def flat_map(self, fn: Callable[[T], Result[U, E]]) -> Result[U, E]:
        """成功値を変換し、Result を返す関数とチェインする"""
        return fn(self.value)

    def unwrap(self) -> T:
        """成功値を取得する（失敗時は例外）"""
        return self.value

    def unwrap_or(self, default: T) -> T:
        """成功値を取得する（失敗時はデフォルト値）"""
        return self.value

    def map_err(self, fn: Callable) -> Result[T, E]:
        """エラー値を変換する（成功時は何もしない）"""
        return self


@dataclass(frozen=True)
class Err(Generic[E]):
    """失敗を表す型"""
    error: E

    def is_ok(self) -> bool:
        return False

    def is_err(self) -> bool:
        return True

    def map(self, fn: Callable) -> Result:
        """成功値を変換する（失敗時は何もしない）"""
        return self

    def flat_map(self, fn: Callable) -> Result:
        """成功値を変換する（失敗時は何もしない）"""
        return self

    def unwrap(self):
        """成功値を取得する（失敗時は例外）"""
        raise ValueError(f"Err を unwrap しようとしました: {self.error}")

    def unwrap_or(self, default):
        """成功値を取得する（失敗時はデフォルト値）"""
        return default

    def map_err(self, fn: Callable[[E], U]) -> Result[T, U]:
        """エラー値を変換する"""
        return Err(fn(self.error))


Result = Union[Ok[T], Err[E]]


# ── ユーティリティ関数 ────────────────────────────

def try_catch(fn: Callable[[], T]) -> Result[T, Exception]:
    """例外をスローする関数を Result に変換する"""
    try:
        return Ok(fn())
    except Exception as e:
        return Err(e)


# ── 実践例: ユーザー登録バリデーション ────────────

@dataclass(frozen=True)
class RegistrationError:
    field: str
    message: str


def validate_username(name: str) -> Result[str, RegistrationError]:
    if len(name) < 3:
        return Err(RegistrationError("username", "ユーザー名は3文字以上必要です"))
    if len(name) > 20:
        return Err(RegistrationError("username", "ユーザー名は20文字以下にしてください"))
    if not name.isalnum():
        return Err(RegistrationError("username", "ユーザー名は英数字のみ使用できます"))
    return Ok(name.lower())


def validate_email(email: str) -> Result[str, RegistrationError]:
    if "@" not in email:
        return Err(RegistrationError("email", "不正なメールアドレス形式です"))
    if email.count("@") > 1:
        return Err(RegistrationError("email", "@ は1つのみ使用できます"))
    return Ok(email.lower())


def validate_password(password: str) -> Result[str, RegistrationError]:
    if len(password) < 8:
        return Err(RegistrationError("password", "パスワードは8文字以上必要です"))
    if not any(c.isupper() for c in password):
        return Err(RegistrationError("password", "大文字を1つ以上含めてください"))
    if not any(c.isdigit() for c in password):
        return Err(RegistrationError("password", "数字を1つ以上含めてください"))
    return Ok(password)


@dataclass
class RegistrationData:
    username: str
    email: str
    password: str


def validate_registration(
    username: str,
    email: str,
    password: str,
) -> Result[RegistrationData, list[RegistrationError]]:
    """すべてのバリデーションを実行し、エラーをまとめて返す"""
    errors: list[RegistrationError] = []

    username_result = validate_username(username)
    email_result = validate_email(email)
    password_result = validate_password(password)

    # エラーを収集
    for result in [username_result, email_result, password_result]:
        match result:
            case Err(error):
                errors.append(error)

    if errors:
        return Err(errors)

    # すべて成功（unwrap は安全）
    return Ok(RegistrationData(
        username=username_result.unwrap(),
        email=email_result.unwrap(),
        password=password_result.unwrap(),
    ))


# パターンマッチングによる使用（Python 3.10+）
match validate_registration("ab", "invalid", "short"):
    case Ok(data):
        print(f"登録成功: {data.username}")
    case Err(errors):
        for error in errors:
            print(f"  [{error.field}] {error.message}")
        # 出力:
        #   [username] ユーザー名は3文字以上必要です
        #   [email] 不正なメールアドレス形式です
        #   [password] パスワードは8文字以上必要です
```

### 3.3 Rust の Result型

**コード例7: Rust のネイティブ Result型**

```rust
use std::fs;
use std::io;
use std::num::ParseIntError;

// ── カスタムエラー型 ────────────────────────────
#[derive(Debug)]
enum ConfigError {
    IoError(io::Error),
    ParseError(ParseIntError),
    MissingField(String),
    InvalidValue { field: String, value: String, reason: String },
}

// From トレイトで自動変換
impl From<io::Error> for ConfigError {
    fn from(error: io::Error) -> Self {
        ConfigError::IoError(error)
    }
}

impl From<ParseIntError> for ConfigError {
    fn from(error: ParseIntError) -> Self {
        ConfigError::ParseError(error)
    }
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::IoError(e) => write!(f, "IO エラー: {}", e),
            ConfigError::ParseError(e) => write!(f, "パースエラー: {}", e),
            ConfigError::MissingField(field) => write!(f, "フィールド '{}' が見つかりません", field),
            ConfigError::InvalidValue { field, value, reason } =>
                write!(f, "フィールド '{}' の値 '{}' が不正: {}", field, value, reason),
        }
    }
}

impl std::error::Error for ConfigError {}

// ── ? 演算子による自動変換とエラー伝播 ──────────
struct Config {
    host: String,
    port: u16,
    max_connections: u32,
}

fn load_config(path: &str) -> Result<Config, ConfigError> {
    // ? 演算子: io::Error → ConfigError に自動変換して早期リターン
    let content = fs::read_to_string(path)?;

    let mut host = None;
    let mut port = None;
    let mut max_connections = None;

    for line in content.lines() {
        let parts: Vec<&str> = line.splitn(2, '=').collect();
        if parts.len() != 2 { continue; }

        let key = parts[0].trim();
        let value = parts[1].trim();

        match key {
            "host" => host = Some(value.to_string()),
            "port" => {
                // ? 演算子: ParseIntError → ConfigError に自動変換
                let p: u16 = value.parse()?;
                if p == 0 {
                    return Err(ConfigError::InvalidValue {
                        field: "port".to_string(),
                        value: value.to_string(),
                        reason: "ポート番号は1以上必要です".to_string(),
                    });
                }
                port = Some(p);
            },
            "max_connections" => {
                max_connections = Some(value.parse()?);
            },
            _ => {} // 未知のキーは無視
        }
    }

    Ok(Config {
        host: host.ok_or(ConfigError::MissingField("host".to_string()))?,
        port: port.ok_or(ConfigError::MissingField("port".to_string()))?,
        max_connections: max_connections.unwrap_or(100), // デフォルト値
    })
}

fn main() {
    match load_config("server.conf") {
        Ok(config) => {
            println!("サーバー起動: {}:{}", config.host, config.port);
            println!("最大接続数: {}", config.max_connections);
        }
        Err(e) => {
            eprintln!("設定ファイル読み込み失敗: {}", e);
            std::process::exit(1);
        }
    }
}
```

### 3.4 Go のエラー処理

**コード例8: Go の明示的エラー処理**

```go
package user

import (
    "errors"
    "fmt"
    "regexp"
)

// ── センチネルエラー（比較用の定義済みエラー）────
var (
    ErrNotFound       = errors.New("user not found")
    ErrAlreadyExists  = errors.New("user already exists")
    ErrInvalidEmail   = errors.New("invalid email format")
    ErrAccountLocked  = errors.New("account is locked")
)

// ── 構造化エラー型 ────────────────────────────
type ValidationError struct {
    Field   string
    Message string
    Value   interface{}
}

func (e *ValidationError) Error() string {
    return fmt.Sprintf("validation error: field=%s, message=%s, value=%v",
        e.Field, e.Message, e.Value)
}

// ── エラーラッピング（Go 1.13+）────────────────
type RepositoryError struct {
    Operation string
    UserID    string
    Err       error // 元のエラーを保持
}

func (e *RepositoryError) Error() string {
    return fmt.Sprintf("repository error: op=%s, userId=%s: %v",
        e.Operation, e.UserID, e.Err)
}

func (e *RepositoryError) Unwrap() error {
    return e.Err
}

// ── サービス実装 ────────────────────────────────
type UserService struct {
    repo UserRepository
}

func (s *UserService) GetUser(id string) (*User, error) {
    user, err := s.repo.FindByID(id)
    if err != nil {
        // エラーの種類に応じた処理
        if errors.Is(err, ErrNotFound) {
            return nil, fmt.Errorf("user %s: %w", id, ErrNotFound)
        }
        // インフラエラーをラップ
        return nil, &RepositoryError{
            Operation: "FindByID",
            UserID:    id,
            Err:       err,
        }
    }
    return user, nil
}

var emailRegex = regexp.MustCompile(`^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$`)

func (s *UserService) UpdateEmail(userID, newEmail string) error {
    // バリデーション
    if !emailRegex.MatchString(newEmail) {
        return &ValidationError{
            Field:   "email",
            Message: "不正なメールアドレス形式です",
            Value:   newEmail,
        }
    }

    user, err := s.GetUser(userID)
    if err != nil {
        return err  // エラーをそのまま伝播
    }

    user.Email = newEmail
    if err := s.repo.Save(user); err != nil {
        return &RepositoryError{
            Operation: "Save",
            UserID:    userID,
            Err:       err,
        }
    }

    return nil
}

// ── 呼び出し側: errors.Is / errors.As で判別 ────
func handleUpdateEmail(userID, email string) {
    svc := &UserService{repo: NewPostgresRepo()}

    err := svc.UpdateEmail(userID, email)
    if err == nil {
        fmt.Println("更新成功")
        return
    }

    // errors.Is: エラーチェーン内のセンチネルエラーを検索
    if errors.Is(err, ErrNotFound) {
        fmt.Printf("ユーザー %s が見つかりません\n", userID)
        return
    }

    // errors.As: エラーチェーン内の型付きエラーを検索
    var validationErr *ValidationError
    if errors.As(err, &validationErr) {
        fmt.Printf("入力エラー: %s - %s\n",
            validationErr.Field, validationErr.Message)
        return
    }

    var repoErr *RepositoryError
    if errors.As(err, &repoErr) {
        fmt.Printf("システムエラー: %s (operation=%s)\n",
            repoErr.Error(), repoErr.Operation)
        return
    }

    fmt.Printf("予期しないエラー: %v\n", err)
}
```

---

## 4. 例外 vs Result型 の使い分け

### 4.1 判断基準

| 基準 | 例外 | Result型 |
|------|------|---------|
| プログラミングエラー | 最適（即座にクラッシュ） | 不適切（バグは型で表現すべきでない） |
| ビジネスエラー | 可能だが冗長 | 最適（型で表現、処理強制） |
| 予期しないエラー | 最適 | 不適切（予期しないものは型にできない） |
| エラー処理の強制 | 強制できない（catchし忘れ） | コンパイル時に強制（switch exhaustiveness） |
| パフォーマンス | スタックトレース生成コスト大 | ゼロコスト（通常の戻り値） |
| コードの可読性 | 正常フローが清潔 | エラーチェックが散在しがち |
| デバッグ容易性 | スタックトレースあり | 発生箇所の特定が難しい場合あり |
| 合成可能性 | try-catch のネストが必要 | map/flatMap で合成可能 |

### 4.2 ハイブリッドアプローチ（推奨）

```
  ハイブリッドアプローチ

  ┌──────────────────────────────────────────────────┐
  │  インフラ層                                       │
  │  → 例外を使用（DB, ネットワーク等のエラー）       │
  │  → try-catch でキャッチし、Result型に変換         │
  ├──────────────────────────────────────────────────┤
  │  ドメイン層                                       │
  │  → Result型を使用（ビジネスルール違反）           │
  │  → 型安全で処理漏れなし                            │
  ├──────────────────────────────────────────────────┤
  │  アプリケーション層                               │
  │  → Result型を集約、エラーレスポンス生成           │
  │  → 予期しない例外はグローバルハンドラで補足       │
  └──────────────────────────────────────────────────┘
```

**コード例9: ハイブリッドアプローチ（TypeScript）**

```typescript
// インフラ層: 例外 → Result に変換
class UserRepositoryImpl implements UserRepository {
  async findById(id: string): Promise<Result<User, AppError>> {
    try {
      const row = await this.db.query('SELECT * FROM users WHERE id = $1', [id]);
      if (!row) {
        return err(new NotFoundError('User', id));
      }
      return ok(this.mapToUser(row));
    } catch (error) {
      // インフラ例外を Result に変換
      return err(new DataAccessError('ユーザー取得失敗', { cause: error }));
    }
  }
}

// ドメイン層: 純粋な Result 型
function applyDiscount(
  order: Order,
  coupon: Coupon,
): Result<Order, DiscountError> {
  if (coupon.isExpired()) {
    return err({ type: 'COUPON_EXPIRED', expiredAt: coupon.expiresAt });
  }
  if (order.total < coupon.minimumAmount) {
    return err({
      type: 'MINIMUM_NOT_MET',
      minimum: coupon.minimumAmount,
      current: order.total,
    });
  }
  return ok(order.withDiscount(coupon.discountAmount));
}

// アプリケーション層: Result の集約
class OrderService {
  async createOrder(
    userId: string,
    items: OrderItem[],
    couponCode?: string,
  ): Promise<Result<Order, OrderError>> {
    // インフラ操作（Result返却）
    const userResult = await this.userRepo.findById(userId);
    if (!userResult.ok) return userResult;

    const order = Order.create(userResult.value, items);

    // クーポン適用（Result チェイン）
    if (couponCode) {
      const couponResult = await this.couponRepo.findByCode(couponCode);
      if (!couponResult.ok) return couponResult;

      const discountResult = applyDiscount(order, couponResult.value);
      if (!discountResult.ok) return discountResult;

      return this.orderRepo.save(discountResult.value);
    }

    return this.orderRepo.save(order);
  }
}
```

### 4.3 言語別推奨アプローチ

| 言語 | 推奨アプローチ | 理由 |
|------|---------------|------|
| Java | 非検査例外（ビジネス） + ドメイン例外階層 | 検査例外は実用上の問題が多い |
| Python | 例外が標準。カスタム例外階層を設計 | EAFP文化（許可より許しを求めよ） |
| TypeScript | Result型 + 例外の組み合わせ | 型システムが強力、Union型が使いやすい |
| Rust | Result型が標準。panic!は致命的エラーのみ | 型システムが処理を強制、? 演算子が便利 |
| Go | error インターフェース + errors.Is/As | シンプルで明示的、if err != nil パターン |
| Kotlin | sealed class + when 式 | null安全 + exhaustive when |
| Scala | Either[L, R] / Try[T] | 関数型プログラミングとの親和性 |
| Haskell | Either / Maybe + モナド | 純粋関数型、do記法で合成 |

---

## 5. 堅牢なエラー処理パターン

### 5.1 Fail Fast パターン

関数の先頭で前提条件を検証し、不正な入力を早期に拒否する。

```
  Fail Fast の流れ

  入力
    │
    ▼
  ┌─────────────┐
  │ 前提条件     │ ─ 違反 → 即座にエラー（ガード節）
  │ チェック      │
  └──────┬──────┘
         │OK
         ▼
  ┌─────────────┐
  │ ビジネス     │
  │ ロジック     │ ← ここではエラーチェック不要
  └──────┬──────┘   （前提条件が保証されている）
         │
         ▼
       結果
```

**コード例10: Fail Fast パターン（Python）**

```python
"""
Fail Fast: ガード節で前提条件を保証する。

原則:
1. 関数の先頭でバリデーション
2. 失敗したら即座に例外 or エラー返却
3. バリデーション通過後の本体は正常系のみ
"""
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Optional


@dataclass
class Money:
    amount: Decimal
    currency: str


class InsufficientFundsError(Exception):
    def __init__(self, available: Decimal, requested: Decimal):
        self.available = available
        self.requested = requested
        super().__init__(
            f"残高不足: 残高 {available}円 < 要求 {requested}円"
        )


class InvalidAmountError(Exception):
    def __init__(self, amount, reason: str):
        super().__init__(f"不正な金額 ({amount}): {reason}")


class AccountLockedError(Exception):
    def __init__(self, account_id: str, reason: str):
        super().__init__(f"口座 {account_id} はロック中: {reason}")


# ── Fail Fast を適用した関数 ──────────────────────

def withdraw(account: "Account", amount: str, memo: Optional[str] = None) -> Money:
    """
    口座から出金する。

    Fail Fast パターン:
    1. 引数の型・値を検証（プログラミングエラー防止）
    2. ビジネスルールを検証（ビジネスエラー）
    3. すべてのチェックを通過したら処理実行
    """

    # ── ガード節 1: 引数の基本検証 ──
    if account is None:
        raise ValueError("account は None であってはなりません")

    try:
        decimal_amount = Decimal(amount)
    except (InvalidOperation, TypeError):
        raise InvalidAmountError(amount, "数値に変換できません")

    if decimal_amount <= 0:
        raise InvalidAmountError(amount, "金額は正の数である必要があります")

    if decimal_amount != decimal_amount.quantize(Decimal("0.01")):
        raise InvalidAmountError(amount, "小数点以下2桁までです")

    # ── ガード節 2: ビジネスルール検証 ──
    if account.is_locked:
        raise AccountLockedError(account.id, account.lock_reason)

    if account.balance < decimal_amount:
        raise InsufficientFundsError(account.balance, decimal_amount)

    daily_limit = Decimal("1000000")
    if account.daily_withdrawn + decimal_amount > daily_limit:
        raise InvalidAmountError(
            amount,
            f"日次出金上限（{daily_limit}円）を超過します"
        )

    # ── 本体: ここでは正常系のみ ──
    # （すべてのガード節を通過しているので安全）
    account.balance -= decimal_amount
    account.daily_withdrawn += decimal_amount

    transaction = Transaction(
        account_id=account.id,
        type="WITHDRAWAL",
        amount=decimal_amount,
        memo=memo or "出金",
    )
    account.transactions.append(transaction)

    return Money(amount=decimal_amount, currency=account.currency)
```

### 5.2 Circuit Breaker パターン

外部サービスへの呼び出しを監視し、連続失敗が閾値を超えたら一時的に呼び出しを停止する。

```
  Circuit Breaker 状態遷移

  ┌─────────┐  失敗回数 >= 閾値  ┌──────────┐
  │  Closed  │ ──────────────── → │   Open   │
  │ (正常)   │                    │ (遮断中) │
  └────┬─────┘                    └────┬─────┘
       │ ▲                             │
       │ │ 成功                  タイムアウト
       │ │                             │
       │ └────────────────── ┌─────────┘
       │     リセット         │
       │                ┌────▼──────┐
       └────────────── │ Half-Open  │
            成功率 OK    │ (試行中)   │
                        └───────────┘
                          │
                          │ 失敗 → Open に戻る
```

**コード例11: Circuit Breaker の実装（Python）**

```python
"""
Circuit Breaker パターン

Michael T. Nygard 『Release It!』で提唱された安定性パターン。
外部サービスの障害が自システムに波及するのを防ぐ。
"""
import time
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable, TypeVar, Generic
from threading import Lock

T = TypeVar('T')
logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"        # 正常（リクエストを通す）
    OPEN = "open"            # 遮断（リクエストを即座に拒否）
    HALF_OPEN = "half_open"  # 試行（限定的にリクエストを通す）


class CircuitOpenError(Exception):
    """サーキットブレーカーが開いている（サービス利用不可）"""
    def __init__(self, service: str, retry_after: float):
        self.service = service
        self.retry_after = retry_after
        super().__init__(
            f"サービス '{service}' は一時的に利用できません。"
            f"{retry_after:.0f}秒後にリトライしてください。"
        )


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5          # OPEN に遷移する失敗回数
    success_threshold: int = 3          # HALF_OPEN → CLOSED に必要な成功回数
    timeout: float = 30.0               # OPEN → HALF_OPEN のタイムアウト（秒）
    excluded_exceptions: tuple = ()      # カウントしない例外（ビジネスエラー等）


class CircuitBreaker:
    """
    Circuit Breaker の実装

    使い方:
        breaker = CircuitBreaker("payment-api")

        try:
            result = breaker.call(lambda: payment_api.charge(order))
        except CircuitOpenError:
            # フォールバック処理
            return fallback_result()
    """

    def __init__(self, name: str, config: CircuitBreakerConfig | None = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._lock = Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                # タイムアウトチェック
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.config.timeout:
                    logger.info(
                        f"[CircuitBreaker:{self.name}] OPEN → HALF_OPEN "
                        f"(timeout={self.config.timeout}s 経過)"
                    )
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
            return self._state

    def call(self, fn: Callable[[], T]) -> T:
        """
        保護された関数呼び出し。

        - CLOSED: 通常通り呼び出す
        - OPEN: CircuitOpenError をスロー
        - HALF_OPEN: 呼び出して結果を観察
        """
        current_state = self.state

        if current_state == CircuitState.OPEN:
            retry_after = (
                self.config.timeout - (time.time() - self._last_failure_time)
            )
            raise CircuitOpenError(self.name, max(0, retry_after))

        try:
            result = fn()
            self._on_success()
            return result
        except Exception as e:
            # 除外例外（ビジネスエラー等）はカウントしない
            if isinstance(e, self.config.excluded_exceptions):
                raise
            self._on_failure(e)
            raise

    def _on_success(self) -> None:
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    logger.info(
                        f"[CircuitBreaker:{self.name}] HALF_OPEN → CLOSED "
                        f"(success_count={self._success_count})"
                    )
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
            else:
                self._failure_count = 0  # 連続成功でカウントリセット

    def _on_failure(self, error: Exception) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                logger.warning(
                    f"[CircuitBreaker:{self.name}] HALF_OPEN → OPEN "
                    f"(試行中に失敗: {error})"
                )
                self._state = CircuitState.OPEN
            elif self._failure_count >= self.config.failure_threshold:
                logger.error(
                    f"[CircuitBreaker:{self.name}] CLOSED → OPEN "
                    f"(failure_count={self._failure_count}, "
                    f"threshold={self.config.failure_threshold})"
                )
                self._state = CircuitState.OPEN


# ── 使用例: 決済サービスの保護 ────────────────────

class PaymentService:
    def __init__(self, api_client):
        self.api_client = api_client
        self.breaker = CircuitBreaker(
            "payment-api",
            CircuitBreakerConfig(
                failure_threshold=3,
                timeout=60.0,
                excluded_exceptions=(PaymentDeclinedError,),  # 決済拒否はカウントしない
            ),
        )

    def charge(self, order: "Order") -> "PaymentResult":
        try:
            return self.breaker.call(
                lambda: self.api_client.charge(order.total, order.payment_method)
            )
        except CircuitOpenError:
            logger.warning("決済API停止中。キューに保存して後でリトライ")
            return PaymentResult.queued(order.id)
```

### 5.3 リトライパターン（Exponential Backoff）

**コード例12: リトライ with Exponential Backoff（Python）**

```python
"""
リトライパターン: Exponential Backoff + Jitter

注意: すべてのエラーをリトライしてはいけない。
リトライすべき: ネットワークタイムアウト、503、429
リトライすべきでない: 400 (Bad Request)、401 (Unauthorized)、404 (Not Found)
"""
import time
import random
import logging
from functools import wraps
from typing import Callable, TypeVar, Type

T = TypeVar('T')
logger = logging.getLogger(__name__)


class MaxRetriesExceededError(Exception):
    """リトライ回数上限超過"""
    def __init__(self, attempts: int, last_error: Exception):
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"{attempts}回のリトライ後も失敗: {last_error}"
        )


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    retryable_exceptions: tuple[Type[Exception], ...] = (Exception,),
    non_retryable_exceptions: tuple[Type[Exception], ...] = (),
) -> Callable:
    """
    リトライデコレータ（Exponential Backoff + Full Jitter）

    パラメータ:
        max_attempts: 最大試行回数（初回を含む）
        base_delay: 基本待機時間（秒）
        max_delay: 最大待機時間（秒）
        exponential_base: 指数の底（通常は2）
        retryable_exceptions: リトライする例外
        non_retryable_exceptions: リトライしない例外（優先）
    """
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        def wrapper(*args, **kwargs) -> T:
            last_error = None

            for attempt in range(1, max_attempts + 1):
                try:
                    result = fn(*args, **kwargs)
                    if attempt > 1:
                        logger.info(
                            f"[Retry] {fn.__name__} 成功 "
                            f"(attempt={attempt}/{max_attempts})"
                        )
                    return result

                except non_retryable_exceptions as e:
                    # リトライしない例外は即座に再スロー
                    logger.debug(
                        f"[Retry] {fn.__name__} 非リトライ例外: {e}"
                    )
                    raise

                except retryable_exceptions as e:
                    last_error = e

                    if attempt == max_attempts:
                        logger.error(
                            f"[Retry] {fn.__name__} 最大リトライ回数到達 "
                            f"(attempts={max_attempts}): {e}"
                        )
                        raise MaxRetriesExceededError(max_attempts, e) from e

                    # Exponential Backoff + Full Jitter
                    delay = min(
                        base_delay * (exponential_base ** (attempt - 1)),
                        max_delay
                    )
                    jittered_delay = random.uniform(0, delay)

                    logger.warning(
                        f"[Retry] {fn.__name__} 失敗 "
                        f"(attempt={attempt}/{max_attempts}): {e}. "
                        f"{jittered_delay:.2f}秒後にリトライ"
                    )
                    time.sleep(jittered_delay)

            # ここには到達しないはずだが安全のため
            raise MaxRetriesExceededError(max_attempts, last_error)

        return wrapper
    return decorator


# ── 使用例 ────────────────────────────────────────

class ExternalApiClient:

    @with_retry(
        max_attempts=3,
        base_delay=1.0,
        retryable_exceptions=(ConnectionError, TimeoutError),
        non_retryable_exceptions=(ValueError, AuthenticationError),
    )
    def fetch_exchange_rate(self, currency_pair: str) -> float:
        """為替レートを外部APIから取得する"""
        response = self.http_client.get(
            f"https://api.example.com/rates/{currency_pair}",
            timeout=5.0,
        )
        if response.status_code == 429:
            raise ConnectionError("Rate limited")
        if response.status_code == 401:
            raise AuthenticationError("Invalid API key")
        return response.json()["rate"]
```

### 5.4 グレースフルデグラデーション

**コード例13: グレースフルデグラデーション（TypeScript）**

```typescript
/**
 * グレースフルデグラデーション（優雅な劣化）
 *
 * 一部のサービスが障害を起こしても、
 * システム全体は動作し続けるように設計する。
 *
 * 原則:
 * 1. 必須機能と非必須機能を明確に分ける
 * 2. 非必須機能の障害は全体に波及させない
 * 3. フォールバック（代替処理）を用意する
 * 4. 劣化状態であることをユーザーに通知する
 */

interface ProductDetail {
  product: Product;
  recommendations: Product[];
  reviews: Review[];
  inventory: InventoryStatus;
  deliveryEstimate: string | null;
}

class ProductPageService {
  constructor(
    private productService: ProductService,       // 必須
    private recommendationService: RecommendationService,  // 非必須
    private reviewService: ReviewService,         // 非必須
    private inventoryService: InventoryService,   // 準必須
    private deliveryService: DeliveryService,     // 非必須
  ) {}

  async getProductDetail(productId: string): Promise<ProductDetail> {
    // ── 必須: 商品情報の取得（失敗したら全体エラー）──
    const product = await this.productService.getById(productId);

    // ── 非必須: 並行して取得、失敗したらフォールバック ──
    const [recommendations, reviews, inventory, deliveryEstimate] =
      await Promise.all([
        this.getRecommendationsSafe(product),
        this.getReviewsSafe(productId),
        this.getInventorySafe(productId),
        this.getDeliveryEstimateSafe(productId),
      ]);

    return {
      product,
      recommendations,
      reviews,
      inventory,
      deliveryEstimate,
    };
  }

  /** レコメンド: 失敗 → 人気商品を返す */
  private async getRecommendationsSafe(product: Product): Promise<Product[]> {
    try {
      return await withTimeout(
        this.recommendationService.getFor(product),
        3000  // 3秒タイムアウト
      );
    } catch (error) {
      logger.warn('レコメンド取得失敗、人気商品にフォールバック', { error });
      // フォールバック: キャッシュされた人気商品を返す
      return this.recommendationService.getPopularCached();
    }
  }

  /** レビュー: 失敗 → 空配列を返す */
  private async getReviewsSafe(productId: string): Promise<Review[]> {
    try {
      return await withTimeout(
        this.reviewService.getFor(productId),
        3000
      );
    } catch (error) {
      logger.warn('レビュー取得失敗、空配列にフォールバック', {
        productId,
        error,
      });
      return []; // レビューなしでもページは表示可能
    }
  }

  /** 在庫: 失敗 → 「確認中」状態を返す */
  private async getInventorySafe(
    productId: string
  ): Promise<InventoryStatus> {
    try {
      return await withTimeout(
        this.inventoryService.check(productId),
        2000  // 在庫確認は重要なので短いタイムアウト
      );
    } catch (error) {
      logger.warn('在庫確認失敗、不明状態にフォールバック', {
        productId,
        error,
      });
      // 「在庫確認中」を返し、購入ボタンは有効にする
      return { status: 'CHECKING', message: '在庫を確認中です' };
    }
  }

  /** 配送予定: 失敗 → null（非表示）*/
  private async getDeliveryEstimateSafe(
    productId: string
  ): Promise<string | null> {
    try {
      return await withTimeout(
        this.deliveryService.estimate(productId),
        2000
      );
    } catch (error) {
      logger.debug('配送予定取得失敗（非表示にする）', { productId, error });
      return null; // 配送予定は非表示にする
    }
  }
}

/** タイムアウト付きPromise */
function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
  return Promise.race([
    promise,
    new Promise<never>((_, reject) =>
      setTimeout(() => reject(new Error(`${ms}ms timeout`)), ms)
    ),
  ]);
}
```

---

## 6. 構造化ロギングとエラー監視

### 6.1 構造化ログの原則

```
  ログレベルとエラーの対応

  ┌──────────┬────────────────────────────────────────────────┐
  │ レベル    │ 用途                                           │
  ├──────────┼────────────────────────────────────────────────┤
  │ ERROR    │ 即座に対応が必要なエラー                        │
  │          │ データ整合性の破壊、決済失敗、認証システム障害   │
  ├──────────┼────────────────────────────────────────────────┤
  │ WARN     │ 異常だが業務継続可能                            │
  │          │ リトライ成功、フォールバック発動、性能劣化       │
  ├──────────┼────────────────────────────────────────────────┤
  │ INFO     │ 正常な業務イベント                              │
  │          │ ユーザー登録、注文確定、バッチ完了               │
  ├──────────┼────────────────────────────────────────────────┤
  │ DEBUG    │ 開発・調査用の詳細情報                          │
  │          │ SQLクエリ、API リクエスト/レスポンス             │
  └──────────┴────────────────────────────────────────────────┘
```

**コード例14: 構造化ロギング（Python）**

```python
"""
構造化ロギング: JSON形式のログで検索・分析を容易にする

原則:
1. 人間可読なメッセージ + 機械可読なフィールド
2. リクエストIDでトレーシング
3. センシティブ情報のマスキング
4. ログレベルの適切な使い分け
"""
import json
import logging
import uuid
from contextvars import ContextVar
from functools import wraps
from typing import Any

# リクエストスコープのコンテキスト
request_id_var: ContextVar[str] = ContextVar('request_id', default='')
user_id_var: ContextVar[str] = ContextVar('user_id', default='')


class StructuredLogger:
    """構造化ログを出力するロガー"""

    # マスキング対象のフィールド名
    SENSITIVE_FIELDS = {
        'password', 'token', 'api_key', 'secret',
        'credit_card', 'ssn', 'authorization',
    }

    def __init__(self, name: str):
        self._logger = logging.getLogger(name)

    def _mask_sensitive(self, data: dict[str, Any]) -> dict[str, Any]:
        """センシティブ情報をマスキングする"""
        masked = {}
        for key, value in data.items():
            if key.lower() in self.SENSITIVE_FIELDS:
                masked[key] = "***MASKED***"
            elif isinstance(value, dict):
                masked[key] = self._mask_sensitive(value)
            else:
                masked[key] = value
        return masked

    def _build_entry(
        self,
        level: str,
        message: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """ログエントリを構築する"""
        entry = {
            "level": level,
            "message": message,
            "request_id": request_id_var.get(''),
            "user_id": user_id_var.get(''),
        }
        # 追加フィールドをマスキングして追加
        extra = self._mask_sensitive(kwargs)
        entry.update(extra)
        return entry

    def info(self, message: str, **kwargs: Any) -> None:
        entry = self._build_entry("INFO", message, **kwargs)
        self._logger.info(json.dumps(entry, ensure_ascii=False))

    def warning(self, message: str, **kwargs: Any) -> None:
        entry = self._build_entry("WARNING", message, **kwargs)
        self._logger.warning(json.dumps(entry, ensure_ascii=False))

    def error(self, message: str, error: Exception | None = None, **kwargs: Any) -> None:
        entry = self._build_entry("ERROR", message, **kwargs)
        if error:
            entry["error_type"] = type(error).__name__
            entry["error_message"] = str(error)
            # スタックトレースは ERROR のみ
            import traceback
            entry["stack_trace"] = traceback.format_exc()
        self._logger.error(json.dumps(entry, ensure_ascii=False))


logger = StructuredLogger(__name__)


# ── 使用例 ────────────────────────────────────────

class OrderService:
    def create_order(self, user_id: str, items: list) -> "Order":
        logger.info(
            "注文作成開始",
            user_id=user_id,
            item_count=len(items),
            total_amount=sum(item.price for item in items),
        )

        try:
            order = self._process_order(user_id, items)
            logger.info(
                "注文作成成功",
                order_id=order.id,
                user_id=user_id,
                total=order.total,
            )
            return order

        except InsufficientInventoryError as e:
            logger.warning(
                "在庫不足で注文失敗",
                user_id=user_id,
                product_id=e.product_id,
                requested=e.requested,
                available=e.available,
            )
            raise

        except PaymentFailedError as e:
            logger.error(
                "決済処理失敗",
                error=e,
                user_id=user_id,
                payment_method=e.payment_method,
                # password や token は自動マスキング
                token=e.transaction_token,
            )
            raise

# ログ出力例（JSON）:
# {
#   "level": "ERROR",
#   "message": "決済処理失敗",
#   "request_id": "req-abc123",
#   "user_id": "user-456",
#   "error_type": "PaymentFailedError",
#   "error_message": "決済ゲートウェイタイムアウト",
#   "payment_method": "credit_card",
#   "token": "***MASKED***",
#   "stack_trace": "..."
# }
```

### 6.2 エラーログ vs エラーレスポンスの分離

```
  ログとレスポンスの分離

  ┌────────────────────────────────────────────────────────────┐
  │  エラーログ（内部用）                                       │
  │  ・スタックトレース                                        │
  │  ・内部ID（リクエストID、トレースID）                       │
  │  ・DB クエリ、API レスポンス                                │
  │  ・ユーザーID（個人情報ではない識別子）                     │
  │  → 開発者・運用チーム向け。Datadog / CloudWatch 等で管理   │
  ├────────────────────────────────────────────────────────────┤
  │  エラーレスポンス（外部用）                                 │
  │  ・ユーザーが理解できるメッセージ                           │
  │  ・エラーコード（VALIDATION_ERROR, NOT_FOUND）             │
  │  ・リクエストID（サポート問い合わせ用）                     │
  │  → ユーザー・クライアント開発者向け                         │
  │  → 内部実装の詳細は絶対に公開しない                         │
  └────────────────────────────────────────────────────────────┘
```

**悪い例 vs 良い例**:

```json
// 悪い: 内部情報がレスポンスに漏れている
{
  "error": "org.postgresql.util.PSQLException: ERROR: relation \"users\" does not exist",
  "stack": "at com.example.UserRepository.findById(UserRepository.java:42)..."
}

// 良い: ユーザー向けの情報のみ
{
  "error": {
    "code": "INTERNAL_ERROR",
    "message": "サーバーエラーが発生しました。しばらくしてからお試しください。",
    "request_id": "req-abc123"
  }
}
```

---

## 7. アンチパターン

### アンチパターン1: 例外の握りつぶし（Swallowing Exceptions）

最も危険なアンチパターン。エラーが闇に消え、バグの原因究明が不可能になる。

```python
# ──────────────────────────────────────────────────
# NG: エラーを握りつぶす
# ──────────────────────────────────────────────────
try:
    process_payment(order)
except Exception:
    pass  # 何もしない → 決済失敗が闇に消える
          # ユーザーには「注文完了」と表示される
          # 実際は課金されていないが商品が発送される

# ──────────────────────────────────────────────────
# NG: ログだけ書いて放置
# ──────────────────────────────────────────────────
try:
    process_payment(order)
except Exception as e:
    print(f"error: {e}")  # ログも残らない可能性あり
    # そのまま続行 → 同じ問題

# ──────────────────────────────────────────────────
# OK: 例外の種類に応じて適切に処理
# ──────────────────────────────────────────────────
try:
    process_payment(order)
except PaymentDeclinedError as e:
    # ビジネスエラー: ユーザーに通知、注文はキャンセル
    logger.warning(f"決済拒否: order={order.id}, reason={e.reason}")
    order.cancel(reason=f"決済拒否: {e.reason}")
    return PaymentResult.declined(e.reason)
except PaymentGatewayError as e:
    # インフラエラー: リトライキューに投入
    logger.error(f"決済ゲートウェイ障害: order={order.id}", exc_info=True)
    retry_queue.enqueue(order, max_retries=3)
    return PaymentResult.queued("決済処理中です。完了次第メールでお知らせします。")
except Exception as e:
    # 予期しないエラー: ログ + 上位に伝播
    logger.error(f"予期しない決済エラー: order={order.id}", exc_info=True)
    raise PaymentSystemError("決済処理に失敗しました") from e
```

### アンチパターン2: 制御フローとしての例外利用

例外を正常な制御フローに使うと、パフォーマンスが低下し、コードの意図が不明瞭になる。

```python
# ──────────────────────────────────────────────────
# NG: 例外を制御フローに使う
# ──────────────────────────────────────────────────
def find_user(users: list, name: str):
    try:
        for user in users:
            if user.name == name:
                raise StopIteration(user)  # 見つかったら例外で脱出
    except StopIteration as e:
        return e.args[0]
    return None

# NG: 例外で存在チェック（LBYL vs EAFP の誤用）
def get_config_value(config: dict, key: str):
    try:
        return config[key]
    except KeyError:
        try:
            return config[key.upper()]
        except KeyError:
            try:
                return config[key.lower()]
            except KeyError:
                return None  # 3段ネストの try-catch

# ──────────────────────────────────────────────────
# OK: 通常の制御フローを使う
# ──────────────────────────────────────────────────
def find_user(users: list, name: str):
    for user in users:
        if user.name == name:
            return user
    return None

# OK: dict.get() や 条件分岐を使う
def get_config_value(config: dict, key: str):
    for candidate in [key, key.upper(), key.lower()]:
        if candidate in config:
            return config[candidate]
    return None
```

### アンチパターン3: 過剰な防御（Defensive Programming の行き過ぎ）

```python
# ──────────────────────────────────────────────────
# NG: 全メソッドで引数チェック → コードの半分がチェック
# ──────────────────────────────────────────────────
class Calculator:
    def add(self, a, b):
        if a is None:
            raise ValueError("a is None")
        if b is None:
            raise ValueError("b is None")
        if not isinstance(a, (int, float)):
            raise TypeError(f"a must be numeric, got {type(a)}")
        if not isinstance(b, (int, float)):
            raise TypeError(f"b must be numeric, got {type(b)}")
        return a + b  # 実際のロジックは1行

    def multiply(self, a, b):
        if a is None:
            raise ValueError("a is None")
        if b is None:
            raise ValueError("b is None")
        # ... 同じチェックの繰り返し
        return a * b

# ──────────────────────────────────────────────────
# OK: 境界でバリデーション、内部は信頼する
# ──────────────────────────────────────────────────
class Calculator:
    """内部メソッド。型チェックは公開APIの境界で行う。"""
    def add(self, a: float, b: float) -> float:
        return a + b

    def multiply(self, a: float, b: float) -> float:
        return a * b

class CalculatorAPI:
    """公開API。ここで入力を検証する。"""
    def __init__(self):
        self.calc = Calculator()

    def calculate(self, operation: str, a: str, b: str) -> float:
        # 境界でのみ厳密にバリデーション
        try:
            num_a = float(a)
            num_b = float(b)
        except (ValueError, TypeError) as e:
            raise ValidationError(f"数値に変換できません: {e}")

        match operation:
            case "add":
                return self.calc.add(num_a, num_b)
            case "multiply":
                return self.calc.multiply(num_a, num_b)
            case _:
                raise ValidationError(f"不明な演算: {operation}")
```

### アンチパターン4: エラー情報の欠落

```python
# ──────────────────────────────────────────────────
# NG: コンテキストのないエラーメッセージ
# ──────────────────────────────────────────────────
raise Exception("エラーが発生しました")  # 何のエラー？ どこで？ なぜ？
raise ValueError("不正な値です")  # 何の値が不正？
raise RuntimeError("処理に失敗しました")  # 何の処理？ 原因は？

# ──────────────────────────────────────────────────
# OK: 十分なコンテキストを含むエラーメッセージ
# ──────────────────────────────────────────────────
raise UserNotFoundError(
    user_id="user-123",
    searched_in="active_users_table"
)
# → "ユーザー user-123 が active_users_table に見つかりません"

raise ValidationError(
    field="email",
    message="不正なメールアドレス形式です",
    value="not-an-email"  # ※ センシティブでない値のみ
)
# → "バリデーションエラー: email フィールド - 不正なメールアドレス形式です (value: 'not-an-email')"

raise PaymentProcessingError(
    order_id="order-456",
    amount=Decimal("5000"),
    gateway="stripe",
    reason="card_declined",
    # token は含めない（センシティブ）
)
# → "決済処理エラー: order-456, 金額=5000円, gateway=stripe, 理由=card_declined"
```

### アンチパターン5: 例外の型を見ない包括的キャッチ

```typescript
// ──────────────────────────────────────────────────
// NG: catch(error) で何でもキャッチして同じ処理
// ──────────────────────────────────────────────────
try {
  await orderService.createOrder(userId, items);
} catch (error) {
  // すべてのエラーを「500 サーバーエラー」で返す
  res.status(500).json({ error: 'Something went wrong' });
}

// ──────────────────────────────────────────────────
// OK: エラーの型に応じて適切なレスポンス
// ──────────────────────────────────────────────────
try {
  await orderService.createOrder(userId, items);
} catch (error) {
  if (error instanceof ValidationError) {
    res.status(400).json({
      error: { code: 'VALIDATION_ERROR', message: error.message, field: error.field }
    });
  } else if (error instanceof NotFoundError) {
    res.status(404).json({
      error: { code: 'NOT_FOUND', message: error.message }
    });
  } else if (error instanceof AuthorizationError) {
    res.status(403).json({
      error: { code: 'FORBIDDEN', message: error.message }
    });
  } else if (error instanceof ConflictError) {
    res.status(409).json({
      error: { code: 'CONFLICT', message: error.message }
    });
  } else {
    // 予期しないエラーのみ 500
    logger.error('予期しないエラー', { error, userId });
    res.status(500).json({
      error: { code: 'INTERNAL_ERROR', message: 'サーバーエラーが発生しました' }
    });
  }
}
```

---

## 8. 演習問題

### 演習1（基礎）: 例外階層の設計

以下の要件を満たすカスタム例外階層を設計せよ。

**要件:**
- ECサイトのバックエンドAPI
- 発生しうるエラー: 商品が見つからない、在庫不足、クレジットカード決済拒否、配送先住所不正、ユーザー認証失敗、外部API接続失敗

**課題:**
1. 例外階層を設計し、各例外クラスを実装する
2. 各例外に適切なHTTPステータスコード、エラーコード、コンテキスト情報を持たせる
3. `to_response()` メソッドで安全なAPIレスポンスに変換できるようにする

**期待される出力（例外クラス図）:**

```
AppError (500)
├── BusinessError
│   ├── ProductNotFoundError (404)
│   ├── InsufficientStockError (422)
│   ├── PaymentDeclinedError (422)
│   └── InvalidAddressError (400)
├── AuthenticationError (401)
└── InfrastructureError (503)
    └── ExternalApiError (503)
```

**模範解答のヒント:**

```python
class AppError(Exception):
    """基底例外"""
    def __init__(self, message, code, status_code=500, details=None):
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.details = details or {}

    def to_response(self):
        """安全なAPIレスポンス（内部情報を含まない）"""
        return {
            "error": {
                "code": self.code,
                "message": str(self),
            }
        }

class InsufficientStockError(BusinessError):
    def __init__(self, product_id, requested, available):
        super().__init__(
            f"商品 {product_id} の在庫不足: 要求={requested}, 在庫={available}",
            code="INSUFFICIENT_STOCK",
            status_code=422,
            details={
                "product_id": product_id,
                "requested": requested,
                "available": available,
            },
        )
```

---

### 演習2（応用）: Result型によるバリデーションチェーン

以下の注文バリデーションを Result型で実装せよ。

**要件:**
- 注文には商品リスト、配送先住所、決済方法が必要
- バリデーション: 商品リスト非空、各商品の数量 > 0、住所の必須フィールド確認、決済方法の有効性確認
- すべてのバリデーションエラーを収集して一括で返す（最初のエラーで止まらない）

**期待される動作:**

```typescript
const result = validateOrder({
  items: [],
  address: { street: '', city: 'Tokyo', zip: '' },
  payment: { method: 'bitcoin' },  // サポート外
});

// result.ok === false
// result.error === [
//   { field: 'items', message: '商品を1つ以上選択してください' },
//   { field: 'address.street', message: '番地は必須です' },
//   { field: 'address.zip', message: '郵便番号は必須です' },
//   { field: 'payment.method', message: 'サポートされていない決済方法です: bitcoin' },
// ]
```

**模範解答のヒント:**

```typescript
type ValidationErrors = { field: string; message: string }[];

function validateOrder(input: OrderInput): Result<ValidatedOrder, ValidationErrors> {
  const errors: ValidationErrors = [];

  // すべてのバリデーションを実行（途中で止まらない）
  const itemsResult = validateItems(input.items);
  const addressResult = validateAddress(input.address);
  const paymentResult = validatePayment(input.payment);

  // エラーを収集
  if (!itemsResult.ok) errors.push(...itemsResult.error);
  if (!addressResult.ok) errors.push(...addressResult.error);
  if (!paymentResult.ok) errors.push(...paymentResult.error);

  if (errors.length > 0) return err(errors);

  return ok({
    items: itemsResult.value,
    address: addressResult.value,
    payment: paymentResult.value,
  });
}
```

---

### 演習3（発展）: Circuit Breaker + グレースフルデグラデーション

以下のシステムに Circuit Breaker とグレースフルデグラデーションを実装せよ。

**要件:**
- 商品検索API：メインの検索エンジン（Elasticsearch）が障害時、RDBMSにフォールバック
- 検索結果にレコメンド情報を付与：レコメンドAPI障害時は、カテゴリベースの代替結果を返す
- 各外部サービスに Circuit Breaker を適用（失敗3回で遮断、30秒後に試行）

**期待される動作:**

```
正常時:
  SearchEngine(ES) → 商品一覧 + Recommendation API → レコメンド付き結果

ES 障害時:
  SearchEngine(ES) → [Circuit OPEN] → RDBMS フォールバック → 商品一覧
  + Recommendation API → レコメンド付き結果

ES + Recommendation 両方障害時:
  RDBMS フォールバック → 商品一覧
  + カテゴリベース代替レコメンド → 最低限の結果
```

**模範解答のヒント:**

```python
class ProductSearchService:
    def __init__(self):
        self.es_breaker = CircuitBreaker("elasticsearch", threshold=3, timeout=30)
        self.recommend_breaker = CircuitBreaker("recommendation", threshold=3, timeout=30)

    def search(self, query: str) -> SearchResult:
        # 検索: ES → RDBMS フォールバック
        products = self._search_with_fallback(query)

        # レコメンド: API → カテゴリベース フォールバック
        recommendations = self._recommend_with_fallback(products)

        return SearchResult(products=products, recommendations=recommendations)

    def _search_with_fallback(self, query):
        try:
            return self.es_breaker.call(lambda: self.es_client.search(query))
        except CircuitOpenError:
            logger.warning("Elasticsearch unavailable, falling back to RDBMS")
            return self.rdbms_client.search(query)  # 性能は劣るが動作する
```

---

## 9. FAQ

### Q1: すべての関数にtry-catchを書くべきか？

いいえ。**例外は適切なレイヤーで1回だけキャッチする**のが原則。すべての関数でtry-catchすると、エラーの伝播が妨げられ、コードが冗長になる。

**キャッチすべき場所（推奨）:**
- APIのエントリポイント（コントローラ層）
- バッチ処理のループ（1件の失敗で全体を止めない）
- 外部サービスとの境界（インフラ例外をドメイン例外に変換）
- UIイベントハンドラ（ユーザーに適切なメッセージを表示）

**キャッチすべきでない場所:**
- ドメインロジック内部（例外をそのまま伝播させる）
- ユーティリティ関数（呼び出し側に判断を委ねる）

### Q2: 例外メッセージには何を含めるべきか？

**含めるべき情報:**
1. **何が起きたか**: 「ユーザーが見つかりません」
2. **コンテキスト**: 「ID: user-123, テーブル: active_users」
3. **原因（可能なら）**: 「データベース接続タイムアウト（5秒）」
4. **回復方法のヒント**: 「管理者に連絡してください」「しばらくしてからお試しください」

**含めてはいけないもの:**
- パスワード、APIトークン、シークレットキー
- クレジットカード番号、社会保障番号
- 個人情報（メールアドレス、電話番号）はログには可だがレスポンスには不可
- スタックトレース（ログには含めるが、ユーザーレスポンスには含めない）
- SQLクエリ（SQLインジェクションの情報を与えてしまう）

### Q3: エラーログとエラーレスポンスは同じ内容でよいか？

**異なるべき**。これは重要なセキュリティ原則である。

| 項目 | エラーログ（内部） | エラーレスポンス（外部） |
|------|-------------------|----------------------|
| スタックトレース | 含める | 含めない |
| 内部ID | 含める | リクエストIDのみ |
| SQLクエリ | 含める | 含めない |
| ユーザーID | 含める | 含めない |
| 技術的詳細 | 含める | 含めない |
| 対処方法 | 技術的な対処法 | ユーザーが取れるアクション |

### Q4: Result型と例外を同じプロジェクトで混ぜてよいか？

**混ぜてよい（むしろ推奨）**。ハイブリッドアプローチとして以下の使い分けが効果的:

- **ビジネスエラー → Result型**: 「残高不足」「在庫なし」等の予測可能なエラーは型で表現
- **インフラエラー → 例外**: DB接続失敗等は例外でキャッチし、Result型に変換
- **プログラミングエラー → 例外**: null参照等のバグは例外で即座にクラッシュ
- **予期しないエラー → 例外**: グローバルエラーハンドラで補足

重要なのは一貫性。チーム内でどのレイヤーでどちらを使うかを明文化しておく。

### Q5: リトライはすべてのエラーに対して行うべきか？

**いいえ。リトライが有効なのは一時的なエラーのみ**。

| エラー | リトライ | 理由 |
|--------|---------|------|
| 503 Service Unavailable | する | サービスが一時的にダウン |
| 429 Too Many Requests | する（Backoff必須） | レート制限、時間が解決 |
| ネットワークタイムアウト | する | 一時的な通信障害の可能性 |
| 400 Bad Request | しない | リクエストが不正、リトライしても同じ |
| 401 Unauthorized | しない | 認証情報が間違い、リトライしても同じ |
| 404 Not Found | しない | リソースが存在しない |
| 409 Conflict | 状況次第 | 楽観的ロックは再取得してリトライ可能 |

**リトライ時の注意:**
- 必ず Exponential Backoff + Jitter を使う
- 最大リトライ回数を設定する（無限リトライは厳禁）
- 冪等でない操作のリトライは二重処理のリスクがある（冪等キーを使う）

### Q6: Go の `if err != nil` パターンは冗長ではないか？

Goの `if err != nil` は確かに冗長に見えるが、以下のメリットがある:

1. **エラー処理が明示的**: 例外のように暗黙的にスキップされない
2. **制御フローが追跡しやすい**: エラーがどこで発生し、どこで処理されるかが一目瞭然
3. **パフォーマンス**: スタックトレースの生成コストがない

Go 2 のジェネリクスにより、Result型的なパターンも使えるようになった。`errors.Is` / `errors.As`（Go 1.13+）でエラーの型判別も型安全に行える。冗長さが気になる場合は、ヘルパー関数やコード生成で軽減できる。

---

## まとめ

| 戦略 | 用途 | 長所 | 短所 | 採用言語 |
|------|------|------|------|---------|
| 例外 | 予期しないエラー、インフラエラー | 自然な伝播、スタックトレース | 処理漏れのリスク | Java, Python, C#, TS |
| Result型 | ビジネスエラー | 型安全、処理強制、ゼロコスト | 冗長になりがち | Rust, Haskell, TS |
| エラーコード | レガシーシステム | シンプル | 無視されやすい | C |
| Fail Fast | 入力検証、前提条件 | バグの早期発見 | ── | 全言語 |
| Circuit Breaker | 外部サービス呼び出し | 障害の波及防止 | 実装の複雑さ | 全言語 |
| リトライ + Backoff | 一時的なインフラエラー | 自動回復 | 冪等性の保証が必要 | 全言語 |
| グレースフルデグラデーション | 非必須機能の障害 | ユーザー体験の維持 | フォールバックの設計・テスト | 全言語 |
| 構造化ログ | エラー監視・分析 | 検索・分析が容易 | ログ設計のコスト | 全言語 |

### エラーハンドリングのチェックリスト

- [ ] エラーの分類は適切か（プログラミング / ビジネス / インフラ / 致命的）
- [ ] カスタム例外階層を設計しているか
- [ ] try-catchの範囲は最小限か
- [ ] レイヤー境界で例外を変換しているか
- [ ] ビジネスエラーにResult型を検討したか
- [ ] Fail Fastで前提条件を検証しているか
- [ ] 外部サービスにCircuit Breakerを適用しているか
- [ ] リトライはExponential Backoff + Jitterか
- [ ] グレースフルデグラデーションを設計しているか
- [ ] ログとレスポンスを分離しているか
- [ ] センシティブ情報がログやレスポンスに漏れていないか
- [ ] エラーメッセージに十分なコンテキストがあるか

---

## 次に読むべきガイド

- [関数設計](./01-functions.md) ── エラーハンドリングを組み込んだ関数設計
- [テスト原則](./04-testing-principles.md) ── エラーケースのテスト方法（境界値、異常系テスト）
- [クリーンコード概要](../00-principles/00-clean-code-overview.md) ── エラーハンドリングの位置づけ
- [結合度と凝集度](../00-principles/03-coupling-cohesion.md) ── 例外の変換がレイヤー間の結合度を下げる理由
- [関数型原則](../03-practices-advanced/02-functional-principles.md) ── Maybe/Either モナドによるエラー処理

---

## 参考文献

1. **Robert C. Martin** 『Clean Code: A Handbook of Agile Software Craftsmanship』 Prentice Hall, 2008 (Chapter 7: Error Handling) ── 例外設計の基本原則、try-catchの適切な使い方
2. **Michael T. Nygard** 『Release It!: Design and Deploy Production-Ready Software』 Pragmatic Bookshelf, 2018 (2nd Edition) ── Circuit Breaker、Bulkhead等の安定性パターン
3. **Joe Duffy** "The Error Model" (blog post, 2016) ── Result型と例外の比較分析、言語設計の観点からの考察
4. **Martin Fowler** "Fail Fast" (martinfowler.com) ── Fail Fastパターンの解説と適用場面
5. **Eric Evans** 『Domain-Driven Design: Tackling Complexity in the Heart of Software』 Addison-Wesley, 2003 ── ドメイン例外の設計、レイヤー間の例外変換
6. **Rust The Book** "Error Handling" (doc.rust-lang.org) ── Result<T, E>型、? 演算子、From トレイトによるエラー変換
7. **Go Blog** "Error handling and Go" (go.dev/blog) ── Goのエラー処理イディオム、errors.Is/As、エラーラッピング
8. **Sam Newman** 『Building Microservices: Designing Fine-Grained Systems』 O'Reilly, 2021 (2nd Edition) ── 分散システムにおけるエラー処理、リトライ、Circuit Breaker
9. **Joshua Bloch** 『Effective Java』 Addison-Wesley, 2018 (3rd Edition, Items 69-77) ── Javaにおける例外の正しい使い方、検査例外と非検査例外の議論
10. **AWS** "Exponential Backoff And Jitter" (aws.amazon.com/blogs) ── Full Jitter、Equal Jitter、Decorrelated Jitter の比較と推奨

