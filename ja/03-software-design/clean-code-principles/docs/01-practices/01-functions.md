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

## トラブルシューティング

### よくあるエラーと解決策

| エラー | 原因 | 解決策 |
|--------|------|--------|
| 初期化エラー | 設定ファイルの不備 | 設定ファイルのパスと形式を確認 |
| タイムアウト | ネットワーク遅延/リソース不足 | タイムアウト値の調整、リトライ処理の追加 |
| メモリ不足 | データ量の増大 | バッチ処理の導入、ページネーションの実装 |
| 権限エラー | アクセス権限の不足 | 実行ユーザーの権限確認、設定の見直し |
| データ不整合 | 並行処理の競合 | ロック機構の導入、トランザクション管理 |

### デバッグの手順

1. **エラーメッセージの確認**: スタックトレースを読み、発生箇所を特定する
2. **再現手順の確立**: 最小限のコードでエラーを再現する
3. **仮説の立案**: 考えられる原因をリストアップする
4. **段階的な検証**: ログ出力やデバッガを使って仮説を検証する
5. **修正と回帰テスト**: 修正後、関連する箇所のテストも実行する

```python
# デバッグ用ユーティリティ
import logging
import traceback
from functools import wraps

# ロガーの設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def debug_decorator(func):
    """関数の入出力をログ出力するデコレータ"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"呼び出し: {func.__name__}(args={args}, kwargs={kwargs})")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"戻り値: {func.__name__} -> {result}")
            return result
        except Exception as e:
            logger.error(f"例外発生: {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

@debug_decorator
def process_data(items):
    """データ処理（デバッグ対象）"""
    if not items:
        raise ValueError("空のデータ")
    return [item * 2 for item in items]
```

### パフォーマンス問題の診断

パフォーマンス問題が発生した場合の診断手順:

1. **ボトルネックの特定**: プロファイリングツールで計測
2. **メモリ使用量の確認**: メモリリークの有無をチェック
3. **I/O待ちの確認**: ディスクやネットワークI/Oの状況を確認
4. **同時接続数の確認**: コネクションプールの状態を確認

| 問題の種類 | 診断ツール | 対策 |
|-----------|-----------|------|
| CPU負荷 | cProfile, py-spy | アルゴリズム改善、並列化 |
| メモリリーク | tracemalloc, objgraph | 参照の適切な解放 |
| I/Oボトルネック | strace, iostat | 非同期I/O、キャッシュ |
| DB遅延 | EXPLAIN, slow query log | インデックス、クエリ最適化 |

---

## 設計判断ガイド

### 選択基準マトリクス

技術選択を行う際の判断基準を以下にまとめます。

| 判断基準 | 重視する場合 | 妥協できる場合 |
|---------|------------|-------------|
| パフォーマンス | リアルタイム処理、大規模データ | 管理画面、バッチ処理 |
| 保守性 | 長期運用、チーム開発 | プロトタイプ、短期プロジェクト |
| スケーラビリティ | 成長が見込まれるサービス | 社内ツール、固定ユーザー |
| セキュリティ | 個人情報、金融データ | 公開データ、社内利用 |
| 開発速度 | MVP、市場投入スピード | 品質重視、ミッションクリティカル |

### アーキテクチャパターンの選択

```
┌─────────────────────────────────────────────────┐
│              アーキテクチャ選択フロー              │
├─────────────────────────────────────────────────┤
│                                                 │
│  ① チーム規模は？                                │
│    ├─ 小規模（1-5人）→ モノリス                   │
│    └─ 大規模（10人+）→ ②へ                       │
│                                                 │
│  ② デプロイ頻度は？                               │
│    ├─ 週1回以下 → モノリス + モジュール分割         │
│    └─ 毎日/複数回 → ③へ                          │
│                                                 │
│  ③ チーム間の独立性は？                            │
│    ├─ 高い → マイクロサービス                      │
│    └─ 中程度 → モジュラーモノリス                   │
│                                                 │
└─────────────────────────────────────────────────┘
```

### トレードオフの分析

技術的な判断には必ずトレードオフが伴います。以下の観点で分析を行いましょう:

**1. 短期 vs 長期のコスト**
- 短期的に速い方法が長期的には技術的負債になることがある
- 逆に、過剰な設計は短期的なコストが高く、プロジェクトの遅延を招く

**2. 一貫性 vs 柔軟性**
- 統一された技術スタックは学習コストが低い
- 多様な技術の採用は適材適所が可能だが、運用コストが増加

**3. 抽象化のレベル**
- 高い抽象化は再利用性が高いが、デバッグが困難になる場合がある
- 低い抽象化は直感的だが、コードの重複が発生しやすい

```python
# 設計判断の記録テンプレート
class ArchitectureDecisionRecord:
    """ADR (Architecture Decision Record) の作成"""

    def __init__(self, title: str):
        self.title = title
        self.context = ""
        self.decision = ""
        self.consequences = []
        self.alternatives = []

    def set_context(self, context: str):
        """背景と課題の記述"""
        self.context = context
        return self

    def set_decision(self, decision: str):
        """決定内容の記述"""
        self.decision = decision
        return self

    def add_consequence(self, consequence: str, positive: bool = True):
        """結果の追加"""
        self.consequences.append({
            'description': consequence,
            'type': 'positive' if positive else 'negative'
        })
        return self

    def add_alternative(self, name: str, reason_rejected: str):
        """却下した代替案の追加"""
        self.alternatives.append({
            'name': name,
            'reason_rejected': reason_rejected
        })
        return self

    def to_markdown(self) -> str:
        """Markdown形式で出力"""
        md = f"# ADR: {self.title}\n\n"
        md += f"## 背景\n{self.context}\n\n"
        md += f"## 決定\n{self.decision}\n\n"
        md += "## 結果\n"
        for c in self.consequences:
            icon = "✅" if c['type'] == 'positive' else "⚠️"
            md += f"- {icon} {c['description']}\n"
        md += "\n## 却下した代替案\n"
        for a in self.alternatives:
            md += f"- **{a['name']}**: {a['reason_rejected']}\n"
        return md
```

---

## 実務での適用シナリオ

### シナリオ1: スタートアップでのMVP開発

**状況:** 限られたリソースで素早くプロダクトをリリースする必要がある

**アプローチ:**
- シンプルなアーキテクチャを選択
- 必要最小限の機能に集中
- 自動テストはクリティカルパスのみ
- モニタリングは早期から導入

**学んだ教訓:**
- 完璧を求めすぎない（YAGNI原則）
- ユーザーフィードバックを早期に取得
- 技術的負債は意識的に管理する

### シナリオ2: レガシーシステムのモダナイゼーション

**状況:** 10年以上運用されているシステムを段階的に刷新する

**アプローチ:**
- Strangler Fig パターンで段階的に移行
- 既存のテストがない場合はCharacterization Testを先に作成
- APIゲートウェイで新旧システムを共存
- データ移行は段階的に実施

| フェーズ | 作業内容 | 期間目安 | リスク |
|---------|---------|---------|--------|
| 1. 調査 | 現状分析、依存関係の把握 | 2-4週間 | 低 |
| 2. 基盤 | CI/CD構築、テスト環境 | 4-6週間 | 低 |
| 3. 移行開始 | 周辺機能から順次移行 | 3-6ヶ月 | 中 |
| 4. コア移行 | 中核機能の移行 | 6-12ヶ月 | 高 |
| 5. 完了 | 旧システム廃止 | 2-4週間 | 中 |

### シナリオ3: 大規模チームでの開発

**状況:** 50人以上のエンジニアが同一プロダクトを開発する

**アプローチ:**
- ドメイン駆動設計で境界を明確化
- チームごとにオーナーシップを設定
- 共通ライブラリはInner Source方式で管理
- APIファーストで設計し、チーム間の依存を最小化

```python
# チーム間のAPI契約定義
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class APIContract:
    """チーム間のAPI契約"""
    endpoint: str
    method: str
    owner_team: str
    consumers: List[str]
    sla_ms: int  # レスポンスタイムSLA
    priority: Priority

    def validate_sla(self, actual_ms: int) -> bool:
        """SLA準拠の確認"""
        return actual_ms <= self.sla_ms

    def to_openapi(self) -> dict:
        """OpenAPI形式で出力"""
        return {
            'path': self.endpoint,
            'method': self.method,
            'x-owner': self.owner_team,
            'x-consumers': self.consumers,
            'x-sla-ms': self.sla_ms
        }

# 使用例
contracts = [
    APIContract(
        endpoint="/api/v1/users",
        method="GET",
        owner_team="user-team",
        consumers=["order-team", "notification-team"],
        sla_ms=200,
        priority=Priority.HIGH
    ),
    APIContract(
        endpoint="/api/v1/orders",
        method="POST",
        owner_team="order-team",
        consumers=["payment-team", "inventory-team"],
        sla_ms=500,
        priority=Priority.CRITICAL
    )
]
```

### シナリオ4: パフォーマンスクリティカルなシステム

**状況:** ミリ秒単位のレスポンスが求められるシステム

**最適化ポイント:**
1. キャッシュ戦略（L1: インメモリ、L2: Redis、L3: CDN）
2. 非同期処理の活用
3. コネクションプーリング
4. クエリ最適化とインデックス設計

| 最適化手法 | 効果 | 実装コスト | 適用場面 |
|-----------|------|-----------|---------|
| インメモリキャッシュ | 高 | 低 | 頻繁にアクセスされるデータ |
| CDN | 高 | 低 | 静的コンテンツ |
| 非同期処理 | 中 | 中 | I/O待ちが多い処理 |
| DB最適化 | 高 | 高 | クエリが遅い場合 |
| コード最適化 | 低-中 | 高 | CPU律速の場合 |

---

## チーム開発での活用

### コードレビューのチェックリスト

このトピックに関連するコードレビューで確認すべきポイント:

- [ ] 命名規則が一貫しているか
- [ ] エラーハンドリングが適切か
- [ ] テストカバレッジは十分か
- [ ] パフォーマンスへの影響はないか
- [ ] セキュリティ上の問題はないか
- [ ] ドキュメントは更新されているか

### ナレッジ共有のベストプラクティス

| 方法 | 頻度 | 対象 | 効果 |
|------|------|------|------|
| ペアプログラミング | 随時 | 複雑なタスク | 即時のフィードバック |
| テックトーク | 週1回 | チーム全体 | 知識の水平展開 |
| ADR (設計記録) | 都度 | 将来のメンバー | 意思決定の透明性 |
| 振り返り | 2週間ごと | チーム全体 | 継続的改善 |
| モブプログラミング | 月1回 | 重要な設計 | 合意形成 |

### 技術的負債の管理

```
優先度マトリクス:

        影響度 高
          │
    ┌─────┼─────┐
    │ 計画 │ 即座 │
    │ 的に │ に   │
    │ 対応 │ 対応 │
    ├─────┼─────┤
    │ 記録 │ 次の │
    │ のみ │ Sprint│
    │     │ で   │
    └─────┼─────┘
          │
        影響度 低
    発生頻度 低  発生頻度 高
```

---

## セキュリティの考慮事項

### 一般的な脆弱性と対策

| 脆弱性 | リスクレベル | 対策 | 検出方法 |
|--------|------------|------|---------|
| インジェクション攻撃 | 高 | 入力値のバリデーション・パラメータ化クエリ | SAST/DAST |
| 認証の不備 | 高 | 多要素認証・セッション管理の強化 | ペネトレーションテスト |
| 機密データの露出 | 高 | 暗号化・アクセス制御 | セキュリティ監査 |
| 設定の不備 | 中 | セキュリティヘッダー・最小権限の原則 | 構成スキャン |
| ログの不足 | 中 | 構造化ログ・監査証跡 | ログ分析 |

### セキュアコーディングのベストプラクティス

```python
# セキュアコーディング例
import hashlib
import secrets
import hmac
from typing import Optional

class SecurityUtils:
    """セキュリティユーティリティ"""

    @staticmethod
    def generate_token(length: int = 32) -> str:
        """暗号学的に安全なトークン生成"""
        return secrets.token_urlsafe(length)

    @staticmethod
    def hash_password(password: str, salt: Optional[str] = None) -> tuple:
        """パスワードのハッシュ化"""
        if salt is None:
            salt = secrets.token_hex(16)
        hashed = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            iterations=100000
        )
        return hashed.hex(), salt

    @staticmethod
    def verify_password(password: str, hashed: str, salt: str) -> bool:
        """パスワードの検証"""
        new_hash, _ = SecurityUtils.hash_password(password, salt)
        return hmac.compare_digest(new_hash, hashed)

    @staticmethod
    def sanitize_input(value: str) -> str:
        """入力値のサニタイズ"""
        dangerous_chars = ['<', '>', '"', "'", '&', '\\']
        result = value
        for char in dangerous_chars:
            result = result.replace(char, '')
        return result.strip()

# 使用例
token = SecurityUtils.generate_token()
hashed, salt = SecurityUtils.hash_password("my_password")
is_valid = SecurityUtils.verify_password("my_password", hashed, salt)
```

### セキュリティチェックリスト

- [ ] 全ての入力値がバリデーションされている
- [ ] 機密情報がログに出力されていない
- [ ] HTTPS が強制されている
- [ ] CORS ポリシーが適切に設定されている
- [ ] 依存パッケージの脆弱性スキャンが実施されている
- [ ] エラーメッセージに内部情報が含まれていない
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


## FAQ

### Q1: このトピックを学ぶ上で最も重要なポイントは何ですか？

実践的な経験を積むことが最も重要です。理論だけでなく、実際にコードを書いて動作を確認することで理解が深まります。

### Q2: 初心者がよく陥る間違いは何ですか？

基礎を飛ばして応用に進むことです。このガイドで説明している基本概念をしっかり理解してから、次のステップに進むことをお勧めします。

### Q3: 実務ではどのように活用されていますか？

このトピックの知識は、日常的な開発業務で頻繁に活用されます。特にコードレビューやアーキテクチャ設計の際に重要になります。

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
