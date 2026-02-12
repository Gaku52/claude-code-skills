# 関数設計 ── 単一責任・引数・副作用

> 関数はプログラムの基本構成要素である。小さく、1つのことだけを行い、名前が意図を伝える関数は、システム全体の可読性と保守性を決定づける。

---

## この章で学ぶこと

1. **関数の単一責任** ── 1つの関数が1つのことだけを行う設計を理解する
2. **引数設計の原則** ── 引数の数・順序・型の最適化方法を身につける
3. **副作用の管理** ── 予測可能で安全な関数を書くための技法を習得する
4. **抽象レベルの統一** ── 関数内の処理を同じ抽象レベルに揃える技法を理解する
5. **コマンド/クエリ分離** ── CQSの原則と実践的な適用方法を習得する

---

## 前提知識

このガイドを最大限に活用するには、以下の知識が必要です。

| 前提知識 | 必要レベル | 参照先 |
|---------|----------|--------|
| 命名規則 | 読了推奨 | [命名規則](./00-naming.md) |
| SOLID原則（特にSRP） | 基本を理解 | [SOLID原則](../00-principles/01-solid.md) |
| クリーンコードの概要 | 読了推奨 | [クリーンコード概論](../00-principles/00-clean-code-overview.md) |

---

## 1. 関数の基本原則

### 1.1 良い関数の5条件

```
+-----------------------------------------------------------+
|  良い関数の5条件                                          |
|  ─────────────────────────────────────                    |
|  1. 小さい (Small)        → 20行以下が目安               |
|  2. 1つのことだけ (Do One Thing)  → 抽象レベルを揃える    |
|  3. 副作用がない/明示的 (No Hidden Side Effects)          |
|  4. 引数は少なく (Few Arguments) → 理想は0〜2個          |
|  5. コマンド/クエリ分離 (CQS) → 変更か取得かどちらか     |
+-----------------------------------------------------------+
```

### 1.2 なぜ関数を小さくすべきか

```
  関数サイズと認知負荷の関係

  認知負荷
    ^
    |                              /
    |                          /
    |                      /
    |                  /
    |             __/
    |         __/
    |     __/
    |  __/
    +──────────────────────────→ 関数の行数
    0    10    20    50   100   200+

  ・〜10行: 一目で理解可能。テスト容易
  ・〜20行: 許容範囲。スクロール不要
  ・〜50行: 黄色信号。分割を検討
  ・100行+: 赤信号。必ず分割
  ・200行+: 緊急。God Function
```

| メトリクス | 目安 | 超過時のアクション |
|-----------|------|-----------------|
| 行数 | 20行以下 | Extract Method |
| 引数の数 | 2個以下 | 引数オブジェクト化 |
| ネストの深さ | 2段以下 | ガード節、早期リターン |
| 循環的複雑度 | 5以下 | 条件分岐の分離 |
| 認知的複雑度 | 15以下 | 複雑なロジックの関数化 |

### 1.3 抽象レベルの統一

```
  抽象レベルの一貫性

  高い抽象 ┌──────────────────────────────────┐
  レベル   │ processOrder()                    │
           │   ├── validateOrder()             │
           │   ├── calculateTotal()            │
           │   ├── chargePayment()             │
           │   └── sendConfirmation()          │
           └──────────────────────────────────┘
  低い抽象 ┌──────────────────────────────────┐
  レベル   │ calculateTotal()                  │
           │   ├── subtotal = sum(prices)      │
           │   ├── tax = subtotal * rate       │
           │   └── return subtotal + tax       │
           └──────────────────────────────────┘
  ※ 1つの関数内で抽象レベルを混在させない
```

**コード例1: 抽象レベルの統一**

```python
# === 悪い: 抽象レベルが混在 ===
def process_order(order):
    # 高レベル: バリデーション
    if not order.is_valid():
        raise InvalidOrderError()

    # 低レベル: 手動で合計計算（詳細が混在）
    total = 0
    for item in order.items:
        price = item.unit_price * item.quantity
        if item.discount_code:
            discount = db.query(
                "SELECT rate FROM discounts WHERE code = %s",
                item.discount_code
            )
            if discount:
                price *= (1 - discount[0].rate)
        total += price
    tax = total * 0.10
    total_with_tax = total + tax

    # 高レベル: 決済
    charge_payment(order.customer, total_with_tax)

# 問題点:
# ・バリデーション（高レベル）と金額計算の詳細（低レベル）が混在
# ・DB呼び出し（低レベル）が中に紛れ込んでいる
# ・テストが困難（DB依存がインラインに存在）


# === 良い: 各関数が同じ抽象レベル ===
def process_order(order: Order) -> OrderResult:
    """注文を処理する（高レベル）"""
    validate_order(order)
    total = calculate_total(order.items)
    payment_result = charge_payment(order.customer, total)
    return create_order_result(order, payment_result)

def calculate_total(items: list[OrderItem]) -> Money:
    """注文合計を計算する（中レベル）"""
    subtotal = sum(calculate_item_price(item) for item in items)
    tax = calculate_tax(subtotal)
    return subtotal + tax

def calculate_item_price(item: OrderItem) -> Money:
    """個別商品の価格を計算する（低レベル）"""
    base_price = item.unit_price * item.quantity
    discount = get_discount_rate(item.discount_code)
    return base_price * (1 - discount)

# 各関数:
# ・1つの抽象レベルのみ
# ・単独でテスト可能
# ・名前から意図が読み取れる
```

**コード例2: ステップダウンルール（新聞記事スタイル）**

```python
# 新聞記事のように: 概要 → 詳細 → さらに詳細

# === 1. 最上位（記事の見出し）===
def deploy_application(config: DeployConfig) -> DeployResult:
    """アプリケーションをデプロイする"""
    validate_config(config)
    build_artifacts = build_application(config)
    test_results = run_tests(build_artifacts)
    ensure_tests_passed(test_results)
    return deploy_to_target(build_artifacts, config.target)

# === 2. 中位（記事の段落）===
def build_application(config: DeployConfig) -> BuildArtifacts:
    """アプリケーションをビルドする"""
    source = checkout_source(config.repository, config.branch)
    dependencies = install_dependencies(source)
    return compile_and_package(source, dependencies)

def run_tests(artifacts: BuildArtifacts) -> TestResults:
    """テストを実行する"""
    unit_results = run_unit_tests(artifacts)
    integration_results = run_integration_tests(artifacts)
    return TestResults(unit=unit_results, integration=integration_results)

# === 3. 最下位（記事の詳細）===
def checkout_source(repo: str, branch: str) -> SourceCode:
    """ソースコードをチェックアウトする"""
    return git.clone(repo, branch=branch)

def install_dependencies(source: SourceCode) -> Dependencies:
    """依存パッケージをインストールする"""
    return package_manager.install(source.dependency_file)
```

---

## 2. 引数設計

### 2.1 引数の数

```
  引数の数と理解しやすさ

  理解しやすさ
    ^
    |  *****
    |       ****
    |           ***
    |              **
    |                *
    |                 *
    +--+--+--+--+--+--> 引数の数
       0  1  2  3  4  5+

  0個 (niladic)  : 最高。Circle.area()
  1個 (monadic)  : 良い。isValid(email)
  2個 (dyadic)   : 許容。Point(x, y)
  3個 (triadic)  : 要検討。引数オブジェクト化
  4個+ (polyadic): リファクタリング必須
```

**コード例3: 引数オブジェクトパターン**

```typescript
// === 悪い: 引数が多すぎる ===
function createUser(
  name: string,
  email: string,
  age: number,
  address: string,
  phone: string,
  role: string,
  department: string
): User {
  // ...
}

// 引数の順序を間違えやすい
createUser("田中", "tanaka@example.com", 30, "東京都...", "090-...", "admin", "開発部");

// === 良い: 引数オブジェクトで構造化 ===
interface CreateUserRequest {
  name: string;
  email: string;
  age: number;
  address: string;
  phone: string;
  role: UserRole;
  department: string;
}

function createUser(request: CreateUserRequest): User {
  // ...
}

// 名前付きフィールドで意図が明確
createUser({
  name: "田中",
  email: "tanaka@example.com",
  age: 30,
  address: "東京都...",
  phone: "090-...",
  role: UserRole.ADMIN,
  department: "開発部"
});
```

**コード例4: フラグ引数の排除**

```python
# === 悪い: ブール引数で動作を切り替え（SRP違反の兆候）===
def create_report(data, is_pdf: bool):
    if is_pdf:
        return generate_pdf_report(data)
    else:
        return generate_html_report(data)

# 呼び出し側: create_report(data, True) ← True が何を意味するか不明

# === 良い: 関数を分離 ===
def create_pdf_report(data: ReportData) -> bytes:
    return generate_pdf(data)

def create_html_report(data: ReportData) -> str:
    return generate_html(data)

# === さらに良い: 戦略パターンで拡張性を確保 ===
class ReportGenerator(Protocol):
    def generate(self, data: ReportData) -> Report: ...

class PdfReportGenerator:
    def generate(self, data: ReportData) -> PdfReport:
        # PDF生成ロジック
        pass

class HtmlReportGenerator:
    def generate(self, data: ReportData) -> HtmlReport:
        # HTML生成ロジック
        pass

# 使用
generator = PdfReportGenerator()  # or HtmlReportGenerator()
report = generator.generate(data)
```

**コード例5: ビルダーパターンによる複雑な構築**

```python
# === 引数が多い場合: ビルダーパターン ===

class QueryBuilder:
    """複雑なクエリの段階的構築"""
    def __init__(self, table: str):
        self._table = table
        self._conditions: list[str] = []
        self._order_by: str | None = None
        self._limit: int | None = None
        self._offset: int | None = None
        self._columns: list[str] = ['*']

    def select(self, *columns: str) -> 'QueryBuilder':
        self._columns = list(columns)
        return self

    def where(self, condition: str) -> 'QueryBuilder':
        self._conditions.append(condition)
        return self

    def order_by(self, column: str, desc: bool = False) -> 'QueryBuilder':
        direction = 'DESC' if desc else 'ASC'
        self._order_by = f"{column} {direction}"
        return self

    def limit(self, count: int) -> 'QueryBuilder':
        self._limit = count
        return self

    def offset(self, count: int) -> 'QueryBuilder':
        self._offset = count
        return self

    def build(self) -> str:
        columns = ', '.join(self._columns)
        query = f"SELECT {columns} FROM {self._table}"
        if self._conditions:
            query += " WHERE " + " AND ".join(self._conditions)
        if self._order_by:
            query += f" ORDER BY {self._order_by}"
        if self._limit:
            query += f" LIMIT {self._limit}"
        if self._offset:
            query += f" OFFSET {self._offset}"
        return query

# 使用: 段階的に構築、最後に build()
query = (
    QueryBuilder("users")
    .select("name", "email", "age")
    .where("age >= 18")
    .where("is_active = true")
    .order_by("name")
    .limit(50)
    .offset(100)
    .build()
)
```

---

## 3. 副作用の管理

### 3.1 隠れた副作用の排除

**コード例6: 隠れた副作用の排除**

```java
// === 悪い: 名前からは読み取れない副作用がある ===
public boolean checkPassword(String userName, String password) {
    User user = userRepository.findByName(userName);
    if (user == null) return false;

    if (user.getPassword().equals(encrypt(password))) {
        Session.initialize();  // 隠れた副作用！パスワードチェックなのにセッションを初期化
        return true;
    }
    return false;
}
// → 名前は "check"（確認）なのに、セッション初期化という副作用がある
// → テストでも予期しないセッション状態が発生する


// === 良い: 副作用を分離し、名前で意図を明示 ===
public boolean isPasswordValid(String userName, String password) {
    // 純粋な検証のみ。副作用なし
    User user = userRepository.findByName(userName);
    if (user == null) return false;
    return user.getPassword().equals(encrypt(password));
}

public AuthResult authenticateAndCreateSession(String userName, String password) {
    // 名前が副作用を明示している
    if (!isPasswordValid(userName, password)) {
        return AuthResult.failure("認証失敗");
    }
    Session session = sessionManager.createSession(userName);
    return AuthResult.success(session);
}
```

### 3.2 純粋関数と副作用のある関数の分離

```python
# === 純粋関数: 同じ入力に対して常に同じ出力。副作用なし ===

def calculate_discount(price: float, discount_rate: float) -> float:
    """価格に割引を適用する（純粋関数）"""
    return price * (1 - discount_rate)

def validate_email(email: str) -> bool:
    """メールアドレスの形式を検証する（純粋関数）"""
    import re
    return bool(re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', email))

def format_currency(amount: float, currency: str = 'JPY') -> str:
    """金額をフォーマットする（純粋関数）"""
    if currency == 'JPY':
        return f"¥{amount:,.0f}"
    return f"${amount:,.2f}"

# 純粋関数の特徴:
# ・テストが容易（モック不要）
# ・並行実行が安全
# ・キャッシュ可能（メモ化）
# ・推論しやすい


# === 副作用のある関数: 明示的に分離 ===

def save_order(order: Order, repository: OrderRepository) -> None:
    """注文を保存する（副作用あり: DB書き込み）"""
    repository.save(order)

def send_notification(user_id: str, message: str, client: NotificationClient) -> None:
    """通知を送信する（副作用あり: 外部API呼び出し）"""
    client.send(user_id, message)

def log_event(event: str, logger: Logger) -> None:
    """イベントをログに記録する（副作用あり: ファイル/ストリーム書き込み）"""
    logger.info(event)
```

**コード例7: 関数型コアと命令型シェル**

```python
# === Functional Core / Imperative Shell パターン ===
# ビジネスロジック（純粋関数）と外部とのやり取り（副作用）を分離

# --- Functional Core: 純粋関数でビジネスロジック ---
class PricingRules:
    """価格計算のルール（純粋関数の集合）"""

    @staticmethod
    def calculate_subtotal(items: list[OrderItem]) -> Decimal:
        return sum(item.price * item.quantity for item in items)

    @staticmethod
    def apply_discount(subtotal: Decimal, discount: Discount) -> Decimal:
        if discount.is_percentage:
            return subtotal * (1 - discount.rate)
        return subtotal - discount.amount

    @staticmethod
    def calculate_tax(amount: Decimal, tax_rate: Decimal) -> Decimal:
        return amount * tax_rate

    @staticmethod
    def calculate_shipping(
        total_weight: Decimal,
        is_premium_member: bool
    ) -> Decimal:
        if is_premium_member:
            return Decimal('0')
        base = total_weight * Decimal('10')
        return base if base > Decimal('500') else Decimal('500')


# --- Imperative Shell: 副作用を含む外部とのやり取り ---
class OrderService:
    """注文サービス（命令型シェル）"""

    def __init__(
        self,
        order_repo: OrderRepository,
        discount_repo: DiscountRepository,
        notification: NotificationService,
    ):
        self.order_repo = order_repo
        self.discount_repo = discount_repo
        self.notification = notification

    def place_order(self, request: PlaceOrderRequest) -> OrderResult:
        # 副作用: DBからデータ取得
        discount = self.discount_repo.find_by_code(request.discount_code)

        # 純粋関数: ビジネスロジック
        subtotal = PricingRules.calculate_subtotal(request.items)
        discounted = PricingRules.apply_discount(subtotal, discount)
        tax = PricingRules.calculate_tax(discounted, Decimal('0.10'))
        shipping = PricingRules.calculate_shipping(
            request.total_weight, request.is_premium
        )
        total = discounted + tax + shipping

        # 副作用: DB保存
        order = Order(items=request.items, total=total)
        self.order_repo.save(order)

        # 副作用: 通知送信
        self.notification.send_order_confirmation(request.user_id, order.id)

        return OrderResult.success(order)
```

### 3.3 コマンド/クエリ分離 (CQS)

```
  ┌─────────────────────────────────────────────┐
  │  Command-Query Separation (CQS)             │
  ├──────────────┬──────────────────────────────┤
  │ コマンド      │ 状態を変更する。戻り値なし   │
  │ (Command)    │ void setName(String name)    │
  ├──────────────┼──────────────────────────────┤
  │ クエリ        │ 値を返す。状態を変更しない   │
  │ (Query)      │ String getName()             │
  ├──────────────┼──────────────────────────────┤
  │ 混在（避ける）│ 状態変更 + 値返却            │
  │              │ int addAndGetCount(Item i)    │
  └──────────────┴──────────────────────────────┘
```

**コード例8: CQSの適用**

```python
# === CQS違反: 1つのメソッドが変更と取得を同時に行う ===
class UserService:
    def update_and_get_user(self, user_id: str, name: str) -> User:
        """ユーザーを更新して返す（コマンド+クエリ混在）"""
        user = self.repo.find_by_id(user_id)
        user.name = name
        self.repo.save(user)
        return user

# 問題点:
# ・呼び出し側が副作用の存在に気づきにくい
# ・テスト時に副作用の検証が複雑
# ・キャッシュ不可能（副作用があるため）


# === CQS準拠: コマンドとクエリを分離 ===
class UserService:
    def update_user_name(self, user_id: str, name: str) -> None:
        """ユーザー名を更新する（コマンド: 状態変更のみ）"""
        user = self.repo.find_by_id(user_id)
        user.name = name
        self.repo.save(user)

    def get_user(self, user_id: str) -> User:
        """ユーザーを取得する（クエリ: 副作用なし）"""
        return self.repo.find_by_id(user_id)

# 使用
user_service.update_user_name("user-123", "新しい名前")
updated_user = user_service.get_user("user-123")
```

```python
# CQS の例外: Stack.pop() のような古典的操作

# 厳密なCQS準拠
class Stack:
    def peek(self) -> T:
        """先頭要素を返す（クエリ: 状態変更なし）"""
        if self.is_empty():
            raise EmptyStackError()
        return self.items[-1]

    def remove_top(self) -> None:
        """先頭要素を削除する（コマンド: 戻り値なし）"""
        if self.is_empty():
            raise EmptyStackError()
        self.items.pop()

# 注: pop() はCQS違反だが、実用上は広く受け入れられている
# 厳密なCQS適用は場面に応じて判断する

# CQS例外が許容される場面:
# ・pop(): 要素を削除して返す（アトミック操作として有用）
# ・next(): イテレータの進行と値取得
# ・dequeue(): キューからの取り出し
# → 原子性が重要な場合は CQS 違反も許容される
```

---

## 4. ガード節と早期リターン

### 4.1 ネストを減らすテクニック

**コード例9: ガード節による可読性向上**

```python
# === 悪い: 深いネスト ===
def process_payment(order):
    if order is not None:
        if order.is_valid():
            if order.payment_method is not None:
                if order.total > 0:
                    if order.customer.has_sufficient_balance(order.total):
                        result = charge(order)
                        if result.success:
                            send_confirmation(order)
                            return result
                        else:
                            return PaymentResult.failed(result.error)
                    else:
                        return PaymentResult.insufficient_balance()
                else:
                    return PaymentResult.invalid_amount()
            else:
                return PaymentResult.no_payment_method()
        else:
            return PaymentResult.invalid_order()
    else:
        return PaymentResult.null_order()


# === 良い: ガード節で早期リターン ===
def process_payment(order: Order) -> PaymentResult:
    # ガード節: 異常系を先に処理
    if order is None:
        return PaymentResult.null_order()

    if not order.is_valid():
        return PaymentResult.invalid_order()

    if order.payment_method is None:
        return PaymentResult.no_payment_method()

    if order.total <= 0:
        return PaymentResult.invalid_amount()

    if not order.customer.has_sufficient_balance(order.total):
        return PaymentResult.insufficient_balance()

    # Happy Path: 正常系は最下部
    result = charge(order)
    if not result.success:
        return PaymentResult.failed(result.error)

    send_confirmation(order)
    return result

# メリット:
# ・ネスト0段（最大1段）
# ・異常系が先に排除されるので正常系に集中できる
# ・新しい条件の追加が容易（ガード節を追加するだけ）
```

| 手法 | 説明 | 効果 |
|------|------|------|
| ガード節 | 異常系を先に処理してリターン | ネスト削減 |
| 早期リターン | 条件不成立時にすぐ返す | Happy Path の明確化 |
| Fail Fast | 不正入力を即座に拒否 | バグの早期発見 |

| ネスト深度 | 推奨度 | 対策 |
|-----------|--------|------|
| 1〜2 | 良い | そのまま |
| 3 | 注意 | ガード節やメソッド抽出を検討 |
| 4+ | 要リファクタリング | 必ず分解する |

---

## 5. 関数設計の高度なパターン

### 5.1 高階関数による抽象化

**コード例10: 高階関数でパターンを抽象化**

```python
from typing import TypeVar, Callable, Optional
import time
import logging

T = TypeVar('T')

# === リトライパターンの抽象化 ===
def with_retry(
    operation: Callable[[], T],
    max_retries: int = 3,
    delay_seconds: float = 1.0,
    backoff_factor: float = 2.0,
    on_retry: Optional[Callable[[int, Exception], None]] = None,
) -> T:
    """リトライロジックを抽象化した高階関数"""
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return operation()
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                if on_retry:
                    on_retry(attempt + 1, e)
                time.sleep(delay_seconds * (backoff_factor ** attempt))
    raise last_exception

# 使用: リトライの詳細を知らずに使える
user = with_retry(
    lambda: api_client.fetch_user("user-123"),
    max_retries=3,
    delay_seconds=0.5,
    on_retry=lambda attempt, e: logging.warning(f"Retry {attempt}: {e}"),
)


# === タイミング計測の抽象化（デコレータ） ===
def timed(func: Callable) -> Callable:
    """関数の実行時間を計測するデコレータ"""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            logging.info(f"{func.__name__} took {elapsed:.3f}s")
    return wrapper

@timed
def heavy_computation(data: list) -> float:
    return sum(x ** 2 for x in data)


# === トランザクション境界の抽象化 ===
def with_transaction(
    db: Database,
    operation: Callable[[Connection], T]
) -> T:
    """トランザクション管理を抽象化"""
    conn = db.get_connection()
    try:
        result = operation(conn)
        conn.commit()
        return result
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

# 使用: トランザクション管理を意識せずにビジネスロジックに集中
def transfer_money(from_id: str, to_id: str, amount: Decimal) -> None:
    def _transfer(conn: Connection) -> None:
        conn.execute("UPDATE accounts SET balance = balance - ? WHERE id = ?", [amount, from_id])
        conn.execute("UPDATE accounts SET balance = balance + ? WHERE id = ?", [amount, to_id])

    with_transaction(db, _transfer)
```

### 5.2 パイプライン処理

```python
# === データ変換パイプライン ===

from functools import reduce
from typing import TypeVar, Callable

T = TypeVar('T')

def pipe(*functions: Callable) -> Callable:
    """関数を左から右に合成するパイプライン"""
    def pipeline(data):
        return reduce(lambda acc, fn: fn(acc), functions, data)
    return pipeline

# 使用: 段階的なデータ変換
process_users = pipe(
    lambda users: [u for u in users if u.is_active],      # アクティブユーザーのみ
    lambda users: [u for u in users if u.age >= 18],       # 成人のみ
    lambda users: sorted(users, key=lambda u: u.name),     # 名前順ソート
    lambda users: [u.to_summary() for u in users],         # サマリーに変換
    lambda summaries: summaries[:100],                      # 上位100件
)

result = process_users(all_users)
```

---

## 6. アンチパターン

### アンチパターン1: God Function（巨大関数）

```python
# NG: 1つの関数で全てを処理（200行超）
def process_everything(request):
    # バリデーション (30行)
    # データ変換 (40行)
    # ビジネスロジック (60行)
    # DB操作 (30行)
    # 外部API呼び出し (20行)
    # レスポンス生成 (20行)
    pass

# OK: 責任ごとに関数を分割
def handle_request(request: Request) -> Response:
    validated = validate_request(request)
    domain_data = transform_to_domain(validated)
    result = apply_business_rules(domain_data)
    persisted = save_to_database(result)
    notify_external_services(persisted)
    return create_response(persisted)
```

### アンチパターン2: Output引数（出力パラメータ）

```java
// NG: 引数を変更して結果を返す
void calculateTotal(Order order, Money result) {
    result.setAmount(order.getSubtotal() + order.getTax());
}

// OK: 戻り値で返す
Money calculateTotal(Order order) {
    return order.getSubtotal().add(order.getTax());
}
```

### アンチパターン3: 隠れた時間的結合

```python
# NG: メソッドの呼び出し順序が暗黙に重要
class DataProcessor:
    def initialize(self): ...   # 1. 先に呼ぶ必要がある
    def load_data(self): ...    # 2. initialize の後に呼ぶ
    def process(self): ...      # 3. load_data の後に呼ぶ
    def save(self): ...         # 4. process の後に呼ぶ

# OK: 順序を強制する設計
class DataProcessor:
    def run(self, input_path: str, output_path: str) -> None:
        """全処理を1つのメソッドで順序保証"""
        config = self._initialize()
        data = self._load_data(input_path, config)
        result = self._process(data)
        self._save(result, output_path)
```

### アンチパターン4: フラグ引数による分岐

```python
# NG: ブール引数で関数内の振る舞いを切り替える
def render_page(content: str, is_admin: bool) -> str:
    if is_admin:
        header = render_admin_header()
        sidebar = render_admin_sidebar()
    else:
        header = render_user_header()
        sidebar = render_user_sidebar()
    return f"{header}{sidebar}{content}"

# OK: 役割ごとに関数を分割する
def render_admin_page(content: str) -> str:
    header = render_admin_header()
    sidebar = render_admin_sidebar()
    return f"{header}{sidebar}{content}"

def render_user_page(content: str) -> str:
    header = render_user_header()
    sidebar = render_user_sidebar()
    return f"{header}{sidebar}{content}"
```

**問題点**: フラグ引数は関数が2つの責任を持っている兆候。呼び出し側で `render_page(content, True)` と書くと、`True` が何を意味するのか即座には分からない。関数を分割するか、Strategy パターンを適用して、名前で意図を明確にする。

---

## 7. 演習問題

### 演習1（基礎）: 関数の分割

以下の巨大関数を、単一責任の関数群にリファクタリングしてください。

```python
def register_user(data):
    # バリデーション
    if not data.get('email') or '@' not in data['email']:
        return {'error': 'Invalid email'}
    if not data.get('password') or len(data['password']) < 8:
        return {'error': 'Password too short'}
    if not data.get('name') or len(data['name']) < 2:
        return {'error': 'Name too short'}

    # ユーザー作成
    import hashlib
    salt = os.urandom(32)
    hashed = hashlib.pbkdf2_hmac('sha256', data['password'].encode(), salt, 100000)
    user = {
        'name': data['name'],
        'email': data['email'],
        'password_hash': hashed.hex(),
        'salt': salt.hex(),
        'created_at': datetime.now()
    }

    # DB保存
    db.execute("INSERT INTO users ...", user)

    # メール送信
    smtp = smtplib.SMTP('smtp.example.com')
    smtp.sendmail('noreply@example.com', data['email'], f'Welcome {data["name"]}!')

    return {'success': True, 'user_id': user['id']}
```

**期待される回答:**

```python
def register_user(request: RegisterUserRequest) -> RegisterResult:
    """ユーザーを登録する（高レベル）"""
    validation_error = validate_registration(request)
    if validation_error:
        return RegisterResult.invalid(validation_error)

    user = create_user_entity(request)
    saved_user = save_user(user)
    send_welcome_email(saved_user)
    return RegisterResult.success(saved_user.id)

def validate_registration(request: RegisterUserRequest) -> str | None:
    if not is_valid_email(request.email):
        return "Invalid email"
    if len(request.password) < MINIMUM_PASSWORD_LENGTH:
        return "Password too short"
    if len(request.name) < MINIMUM_NAME_LENGTH:
        return "Name too short"
    return None

def create_user_entity(request: RegisterUserRequest) -> User:
    password_hash = hash_password(request.password)
    return User(name=request.name, email=request.email, password_hash=password_hash)
```

### 演習2（応用）: CQS の適用

以下のCQS違反を修正してください。

```python
class ShoppingCart:
    def add_item_and_get_total(self, item):
        self.items.append(item)
        return sum(i.price for i in self.items)

    def remove_item_and_check_empty(self, item_id):
        self.items = [i for i in self.items if i.id != item_id]
        return len(self.items) == 0
```

**期待される回答:**

```python
class ShoppingCart:
    # コマンド（状態変更）
    def add_item(self, item: CartItem) -> None:
        self.items.append(item)

    def remove_item(self, item_id: str) -> None:
        self.items = [i for i in self.items if i.id != item_id]

    # クエリ（値取得）
    def get_total(self) -> Decimal:
        return sum(i.price for i in self.items)

    def is_empty(self) -> bool:
        return len(self.items) == 0
```

### 演習3（発展）: Functional Core / Imperative Shell

以下のコードを Functional Core と Imperative Shell に分離してください。

```python
def process_order(order_id, discount_code):
    order = db.get_order(order_id)
    discount = db.get_discount(discount_code)

    total = 0
    for item in order.items:
        total += item.price * item.quantity

    if discount:
        total *= (1 - discount.rate)

    tax = total * 0.1
    final = total + tax

    db.update_order_total(order_id, final)
    email_service.send_receipt(order.customer_email, final)

    return final
```

**期待される回答:**

```python
# Functional Core（純粋関数）
def calculate_order_total(items: list[Item], discount_rate: float, tax_rate: float) -> Decimal:
    subtotal = sum(item.price * item.quantity for item in items)
    discounted = subtotal * (1 - discount_rate)
    tax = discounted * tax_rate
    return discounted + tax

# Imperative Shell（副作用あり）
def process_order(order_id: str, discount_code: str) -> Decimal:
    order = db.get_order(order_id)
    discount = db.get_discount(discount_code)
    discount_rate = discount.rate if discount else 0.0

    # 純粋関数でビジネスロジック
    final_total = calculate_order_total(order.items, discount_rate, TAX_RATE)

    # 副作用
    db.update_order_total(order_id, final_total)
    email_service.send_receipt(order.customer_email, final_total)

    return final_total
```

---

## 8. FAQ

### Q1: 関数は何行以下にすべきか？

Robert C. Martin は「4〜5行が理想」と述べているが、実践的には**20行以下**が目安。重要なのは行数ではなく「1つのことだけをしているか」。20行でも1つの責任なら問題なく、5行でも複数の責任が混在していればリファクタリング対象。

### Q2: 引数にnullを渡すのは許容されるか？

原則として**null引数は避けるべき**。Optional型、デフォルト値、メソッドオーバーロードで代替する。null引数はNullPointerExceptionの温床であり、呼び出し側にnullチェックの責任を押し付ける。

### Q3: privateメソッドが増えすぎた場合はどうすべきか？

privateメソッドが多いクラスは、**隠れたクラスが存在する兆候**。関連するprivateメソッド群を新しいクラスに抽出（Extract Class）することで、各クラスの凝集度が上がる。

### Q4: CQSを常に厳密に適用すべきか？

CQSは原則であり、絶対的なルールではない。`pop()`, `next()`, `dequeue()` のように、原子性が重要な操作ではCQS違反も許容される。重要なのは**意図しない副作用を排除すること**であり、明示的に設計されたCQS違反は問題ない。

### Q5: 早期リターンはパフォーマンスに影響するか？

現代のコンパイラ/インタープリタでは、早期リターンによるパフォーマンスへの影響は無視できる。可読性の向上のほうが遥かに重要。唯一注意すべきはリソース解放で、early returnの前に`try-finally`や`with`文でリソースを確実に解放する。

### Q6: ラムダ式と通常の関数のどちらを使うべきか？

**1〜2行の単純な変換・フィルタにはラムダ式**、**3行以上のロジックや再利用する処理には名前付き関数**が適切である。ラムダ式は「何をするか」が即座に分かる場合に有効だが、複雑になるとデバッグしにくくなる。

```python
# OK: ラムダ式が適切（単純な変換）
names = sorted(users, key=lambda u: u.last_name)
active = filter(lambda u: u.is_active, users)

# NG: ラムダ式が複雑すぎる
result = map(lambda x: x.price * x.quantity * (1 - x.discount) if x.is_taxable else x.price * x.quantity, items)

# OK: 名前付き関数に抽出
def calculate_item_total(item: Item) -> float:
    subtotal = item.price * item.quantity
    if item.is_taxable:
        return subtotal * (1 - item.discount)
    return subtotal

result = map(calculate_item_total, items)
```

### Q7: 再帰関数はいつ使うべきか？

再帰はツリー構造の走査やフラクタル的な問題（分割統治法）に自然にフィットする。ただし以下の点に注意する。

1. **スタックオーバーフロー**: 再帰の深さに上限がある（Python: デフォルト1000）
2. **末尾再帰最適化**: 多くの言語では最適化されないため、ループに変換するほうが安全
3. **可読性**: 再帰で書くと自然に読める問題（ディレクトリ走査、JSON解析等）には積極的に使う

```python
# 再帰が自然なケース: ディレクトリツリーの走査
def find_files(directory: Path, extension: str) -> list[Path]:
    found = []
    for entry in directory.iterdir():
        if entry.is_dir():
            found.extend(find_files(entry, extension))  # 再帰
        elif entry.suffix == extension:
            found.append(entry)
    return found

# 再帰が不適切なケース: 単純な集計（ループで十分）
# NG
def sum_recursive(numbers: list[int]) -> int:
    if not numbers:
        return 0
    return numbers[0] + sum_recursive(numbers[1:])

# OK
def sum_iterative(numbers: list[int]) -> int:
    return sum(numbers)
```

---

## まとめ

| 原則 | ガイドライン | 違反の兆候 |
|------|------------|-----------|
| サイズ | 20行以下 | スクロールが必要 |
| 責任 | 1つだけ | 「〜して〜して〜する」という説明 |
| 引数 | 2個以下推奨 | 引数オブジェクト未使用で3個以上 |
| 副作用 | なし/明示的 | 名前に表れない状態変更 |
| 抽象レベル | 統一 | 高レベル操作と低レベル操作の混在 |
| CQS | コマンドとクエリを分離 | 状態変更しつつ値を返す |
| ネスト | 2段以下 | 深いif/forの入れ子 |

| 設計パターン | 効果 | 適用場面 |
|------------|------|---------|
| ガード節 | ネスト削減、異常系の明確化 | 複数の前提条件チェック |
| 引数オブジェクト | 引数の削減、意味の明確化 | 引数が3個以上 |
| 戦略パターン | フラグ引数の排除 | 条件による動作切替 |
| 高階関数 | 共通パターンの抽象化 | リトライ、計測、トランザクション |
| Functional Core | テスト容易性、推論容易性 | ビジネスロジック |

---

## 次に読むべきガイド

- [エラーハンドリング](./02-error-handling.md) ── 関数のエラー処理設計
- [テスト原則](./04-testing-principles.md) ── テストしやすい関数の条件
- [関数型原則](../03-practices-advanced/02-functional-principles.md) ── 純粋関数と参照透過性
- [SOLID原則](../00-principles/01-solid.md) ── SRPと関数の単一責任

---

## 参考文献

1. **Robert C. Martin** 『Clean Code: A Handbook of Agile Software Craftsmanship』 Prentice Hall, 2008 (Chapter 3: Functions) ── 関数設計の基本原則
2. **Bertrand Meyer** 『Object-Oriented Software Construction』 Prentice Hall, 1997 ── Command-Query Separation の提唱
3. **Martin Fowler** 『Refactoring: Improving the Design of Existing Code』 Addison-Wesley, 2018 ── Extract Function, Introduce Parameter Object, Replace Flag Argument
4. **Gary Bernhardt** "Boundaries" (talk, 2012) ── Functional Core / Imperative Shell パターン
5. **Michael Feathers** 『Working Effectively with Legacy Code』 Prentice Hall, 2004 ── レガシーコードにおける関数の抽出技法
6. **Thomas J. McCabe** "A Complexity Measure" IEEE Transactions on Software Engineering, 1976 ── 循環的複雑度の定義
7. **G. Ann Campbell** "Cognitive Complexity" SonarSource, 2018 ── 認知的複雑度メトリクスの定義
8. **Eric Normand** 『Grokking Simplicity: Taming complex software with functional thinking』 Manning, 2021 ── 純粋関数とアクションの分離
