# テスト原則 ── 信頼性の高いテストスイートを構築する技法

> テストはコードの品質を保証する安全網であり、設計を改善するフィードバックメカニズムである。AAA パターン・FIRST 原則・テストダブルの使い分けを理解し、信頼性の高いテストスイートを構築する手法を解説する。テストは「バグを見つけるもの」ではなく「安心して変更できる環境を作るもの」である。

---

## この章で学ぶこと

1. **テストの基本原則と設計哲学** ── テストピラミッド、AAA パターン、FIRST 原則による構造化と、テストが果たすべき本質的な役割を理解する
2. **テストダブルの使い分け** ── Stub、Mock、Spy、Fake の役割と適切な選択基準を身につけ、テスト容易性の高い設計を実現する
3. **実践的なテスト設計技法** ── 境界値テスト、パラメタライズテスト、プロパティベーステスト、TDD サイクルを習得する
4. **テスト品質の維持と改善** ── Flaky テスト対策、テストカバレッジの適切な運用、ミューテーションテストによるテスト品質の検証方法を学ぶ
5. **CI/CD パイプラインとの統合** ── テストの自動実行、並列化、選択実行によるフィードバックループの高速化を実現する

---

## 前提知識

この章を理解するために、以下の知識があると望ましい。

| 前提知識 | 参照先 |
|---------|--------|
| 関数設計の原則 | [関数設計](./01-functions.md) |
| クラス設計の基本 | [クラス設計](./02-classes.md) |
| Python の基本構文 | 基礎プログラミング知識 |
| pytest の基本的な使い方 | pytest 公式ドキュメント |

---

## 1. テストの本質 ── なぜテストを書くのか

### 1.1 テストの3つの役割

テストは単に「バグを見つけるもの」ではない。Vladimir Khorikov は『Unit Testing Principles, Practices, and Patterns』で、テストの役割を以下の3つに整理している。

```
テストの3つの役割
────────────────────────────────────
1. 回帰防止 (Regression Protection)
   → コード変更が既存機能を壊していないことを保証する
   → リファクタリングの安全網として機能する

2. 設計フィードバック (Design Feedback)
   → テストが書きにくい = 設計に問題がある
   → テスタビリティが高い設計 = 良い設計（相関が高い）

3. ドキュメンテーション (Living Documentation)
   → テストは「コードがどう使われるか」の生きた仕様書
   → テストが通る限り、その振る舞いは保証されている
────────────────────────────────────
```

### 1.2 テストと設計の関係

テストが書きにくい場合、それはコードの設計に問題がある兆候である。

```
テストの書きにくさと設計問題の対応
────────────────────────────────────
テストの困難さ             → 設計上の問題
──────────────────────── → ────────────────────
準備(Arrange)が長い       → 依存が多すぎる（SRP違反）
テストダブルが多い        → 結合度が高い
テスト名が長い            → 関数の責務が多い
テストケースが多すぎる    → 条件分岐が複雑すぎる
外部システムに依存        → インターフェースが未分離
テストの順序に依存        → グローバル状態がある
────────────────────────────────────
```

### 1.3 テストのコストとリターン

```
  テストの費用対効果

  リターン（バグ防止、安心感）
  高 |         *  (ユニットテスト:
    |        *    コアロジック)
    |       *
    |      *        * (統合テスト)
    |     *       *
    |    *      *
    |   *     *       * (E2Eテスト)
    |  *    *       *
    | *   *       *
  低 |___*______*________*___
     低                  高
        コスト（作成+保守）

  最も費用対効果が高いのは:
  「ビジネスロジックのユニットテスト」
```

---

## 2. テストピラミッド

### 2.1 全体構造

```
                  /\
                 /  \          E2E テスト
                / E2E \        (少数・遅い・高コスト)
               /      \
              /--------\
             /          \      統合テスト
            / Integration\     (中程度)
           /              \
          /----------------\
         /                  \   ユニットテスト
        /      Unit Tests    \  (多数・速い・低コスト)
       /                      \
      +------------------------+

  ユニット : 統合 : E2E = 70% : 20% : 10% (目安)
```

この比率は Mike Cohn が『Succeeding with Agile』で提唱したものであり、絶対的なルールではなく目安である。プロジェクトの性質（API中心、UI中心、データパイプラインなど）によって最適な比率は変わる。

### 2.2 各レベルの特性と比較

```
レベル        速度      信頼性    保守コスト    フィードバック   検出するバグ
---------------------------------------------------------------------------
Unit         < 1ms     高        低           即座          ロジックエラー
Integration  < 1sec    中        中           数秒          接続・設定ミス
E2E          < 30sec   低(Flaky) 高           数分          フロー全体の不整合
```

| テストレベル | テスト対象 | 具体例 | 使用するツール |
|-------------|-----------|--------|-------------|
| Unit | 単一関数/メソッド | 価格計算、バリデーション | pytest, JUnit, Jest |
| Integration | モジュール間の連携 | DB操作、API呼び出し | pytest + testcontainers |
| E2E | ユーザーフロー全体 | 「ログイン→商品選択→決済」 | Playwright, Cypress |

### 2.3 テストピラミッドの変形

```
テスティングトロフィー（Kent C. Dodds 提唱）
── フロントエンドでは統合テストを重視する考え方

            /\
           /  \          E2E (少数)
          /----\
         /      \
        / 統合    \      統合テスト (最多)
       /  テスト   \
      /------------\
     /              \
    /  ユニット      \   ユニットテスト (中程度)
   /                  \
  +--------------------+
  |     Static Types    |   型チェック (基盤)
  +--------------------+

※ フロントエンドでは、個々の関数テストよりも
   「コンポーネント間の連携」が正しいことの方が重要
```

---

## 3. AAA パターン

### 3.1 Arrange-Act-Assert の基本

テストの構造化パターンとして最も広く使われている。Given-When-Then（BDD スタイル）とも対応する。

**コード例1: AAA パターンの基本形**

```python
def test_order_total_calculation():
    # Arrange: テスト対象と前提条件を準備
    order = Order(id="order-1", user_id="user-1")
    order.add_item(OrderItem(product_id="p1", name="商品A", price=1000, quantity=2))
    order.add_item(OrderItem(product_id="p2", name="商品B", price=500, quantity=3))

    # Act: テスト対象のメソッドを実行
    total = order.total_amount

    # Assert: 期待結果を検証
    assert total == 3500  # 1000*2 + 500*3

def test_order_cancel_shipped_raises_error():
    # Arrange
    order = Order(id="order-1", user_id="user-1", status="shipped")

    # Act & Assert (例外の場合は一体化してよい)
    with pytest.raises(ValueError, match="出荷済みの注文は取消不可"):
        order.cancel()
```

### 3.2 AAA パターンのガイドライン

```
AAA パターンのガイドライン
────────────────────────────────────
1. Arrange が長い場合 → ファクトリ関数やフィクスチャに抽出
2. Act は原則1行 → 複数行になるなら関数が大きすぎる兆候
3. Assert は1テスト1概念 → 複数の概念を検証しない
4. 各セクションは空行で区切る → 視覚的に構造を明確にする
5. コメントは省略可 → AAA の構造自体がドキュメント
────────────────────────────────────
```

**コード例2: Arrange が長い場合の対処**

```python
# BAD: Arrange が長すぎる
def test_monthly_report_generation():
    user = User(id="u1", name="Alice", role="admin")
    department = Department(id="d1", name="Engineering")
    project1 = Project(id="p1", name="Alpha", budget=1000000)
    project2 = Project(id="p2", name="Beta", budget=2000000)
    team = Team(members=[user], department=department)
    time_entries = [
        TimeEntry(user=user, project=project1, hours=80),
        TimeEntry(user=user, project=project2, hours=40),
    ]
    report_config = ReportConfig(
        period="monthly", include_overtime=True, format="pdf"
    )

    report = generate_report(team, time_entries, report_config)

    assert report.total_hours == 120
    assert report.overtime_hours == 20


# GOOD: ファクトリ関数でテストの意図を明確にする
def test_monthly_report_generation():
    team = create_team_with_single_member()
    time_entries = create_time_entries(regular_hours=120, overtime_hours=20)
    config = create_monthly_report_config()

    report = generate_report(team, time_entries, config)

    assert report.total_hours == 120
    assert report.overtime_hours == 20
```

### 3.3 テスト名の命名規則

テスト名は「何がテストされているか」を読むだけで理解できるべきである。

**コード例3: テスト名の命名パターン**

```python
# 命名パターン1: [対象]_[状況]_[期待結果]
def test_order_when_empty_items_raises_validation_error():
    ...

def test_discount_when_total_exceeds_10000_applies_10_percent():
    ...

def test_user_registration_with_duplicate_email_returns_conflict():
    ...

# 命名パターン2: BDD スタイル (ネストクラス)
class TestOrder:
    class TestPlace:
        def test_changes_status_to_placed(self): ...
        def test_raises_error_when_items_empty(self): ...
        def test_emits_order_placed_event(self): ...

    class TestCancel:
        def test_changes_status_to_cancelled(self): ...
        def test_raises_error_when_already_shipped(self): ...
        def test_restores_inventory(self): ...

# 命名パターン3: should スタイル
def test_order_should_calculate_total_including_tax():
    ...

def test_user_should_be_locked_after_five_failed_attempts():
    ...
```

| 命名パターン | 例 | 適用場面 |
|-------------|-----|---------|
| `test_[対象]_[状況]_[期待結果]` | `test_order_when_empty_raises_error` | 最も一般的 |
| BDD ネストクラス | `TestOrder.TestCancel.test_restores_inventory` | テストが多い場合 |
| should スタイル | `test_order_should_calculate_total` | 仕様の読み下し |
| it スタイル (JS) | `it('calculates total including tax')` | Jest, Mocha |

---

## 4. FIRST 原則

FIRST 原則は、良いユニットテストの5つの特性を定義する。Robert C. Martin が『Clean Code』で提唱した。

| 原則 | 意味 | 具体的なガイドライン |
|------|------|------------------|
| **F**ast | 高速 | ユニットテスト1件 < 10ms。全スイート < 10秒 |
| **I**ndependent | 独立 | テスト間に依存なし。順序を変えても結果が同じ |
| **R**epeatable | 再現可能 | 環境・時刻に依存しない。CI でも同じ結果 |
| **S**elf-Validating | 自己検証 | Pass/Fail が自動判定。手動確認不要 |
| **T**imely | 適時 | プロダクションコードの直前・直後に書く |

### 4.1 Fast（高速）

**コード例4: テスト速度の改善**

```python
# BAD (Fast 違反): 本物の外部APIを呼ぶ
def test_payment_processing():
    result = stripe.Charge.create(amount=1000, currency="jpy")  # 実際のAPI呼び出し
    assert result.status == "succeeded"

# GOOD: テストダブルで即座に応答
def test_payment_processing():
    gateway = StubPaymentGateway(always_succeeds=True)
    processor = PaymentProcessor(gateway=gateway)

    result = processor.process(Payment(amount=1000, currency="jpy"))

    assert result.status == "succeeded"
```

### 4.2 Independent（独立）

**コード例5: テスト間の依存を排除**

```python
# BAD (Independent 違反): テスト間で状態を共有
class TestUserService:
    user_id = None  # クラス変数で状態共有

    def test_create_user(self):
        TestUserService.user_id = service.create_user("Alice")

    def test_get_user(self):
        user = service.get_user(TestUserService.user_id)  # 前のテストに依存
        assert user.name == "Alice"

# GOOD: 各テストが独立
class TestUserService:
    def test_create_user(self):
        user_id = service.create_user("Alice")
        assert user_id is not None

    def test_get_user(self):
        user_id = service.create_user("Bob")  # 自分で準備
        user = service.get_user(user_id)
        assert user.name == "Bob"
```

### 4.3 Repeatable（再現可能）

**コード例6: 時刻依存の排除**

```python
# BAD (Repeatable 違反): 現在時刻に依存
def test_is_expired():
    token = Token(expires_at=datetime(2026, 3, 1))
    assert token.is_expired()  # 2026年3月以降にしか通らない

# GOOD: 時刻を注入可能にする（依存性注入）
def test_is_expired():
    token = Token(expires_at=datetime(2026, 3, 1))
    now = datetime(2026, 3, 2)  # テスト用の固定時刻
    assert token.is_expired(now=now)

# GOOD (代替): freezegun で時刻を固定
from freezegun import freeze_time

@freeze_time("2026-03-02")
def test_is_expired():
    token = Token(expires_at=datetime(2026, 3, 1))
    assert token.is_expired()
```

```python
# BAD (Repeatable 違反): ランダム値に依存
import random

def test_shuffle_changes_order():
    items = [1, 2, 3, 4, 5]
    shuffled = shuffle(items)
    assert items != shuffled  # まれに同じ順序になる

# GOOD: シードを固定
def test_shuffle_changes_order():
    items = [1, 2, 3, 4, 5]
    shuffled = shuffle(items, seed=42)  # シード固定
    assert shuffled == [3, 1, 4, 5, 2]  # 決定的な結果
```

### 4.4 Self-Validating（自己検証）

```python
# BAD (Self-Validating 違反): 手動確認が必要
def test_report_generation():
    report = generate_report(data)
    print(report)  # 目視で確認 → Pass/Fail が自動判定できない

# GOOD: 自動判定可能なアサーション
def test_report_generation():
    report = generate_report(data)
    assert report.total_rows == 100
    assert report.summary == "月次レポート: 売上 ¥1,000,000"
    assert report.generated_at is not None
```

### 4.5 Timely（適時）

TDD（テスト駆動開発）では、テストをプロダクションコードの前に書く。TDD でなくても、機能実装と同じタイミングでテストを書くべきである。「後でテストを書く」はほぼ「テストを書かない」と同義になりがちである。

```
TDD サイクル (Red-Green-Refactor)
────────────────────────────────────
  1. Red   : 失敗するテストを書く（まだ実装がない）
  2. Green : テストが通る最小限の実装を書く
  3. Refactor : テストが通ったまま、コードを整理する

  ┌─────────┐     ┌──────────┐     ┌────────────┐
  │  Red     │ ──→ │  Green   │ ──→ │  Refactor  │
  │ (テスト) │     │ (実装)   │     │ (整理)     │
  └─────────┘     └──────────┘     └────────────┘
       ↑                                   │
       └───────────────────────────────────┘
────────────────────────────────────
```

---

## 5. テストダブル

### 5.1 種類と使い分け

```
テストダブルの分類

  テストダブル
  ├── Dummy ─── 引数を埋めるだけ（使われない）
  ├── Stub ──── 固定値を返す
  ├── Spy ───── 実際の処理 + 呼び出し記録
  ├── Mock ──── 呼び出しの検証（期待を設定）
  └── Fake ──── 簡易だが動作する実装
```

| 種類 | 目的 | 検証対象 | 例 |
|------|------|---------|-----|
| **Dummy** | 引数を満たす | なし | テスト対象が使わない引数 |
| **Stub** | 固定値を返す | 戻り値 | `find_by_id()` が固定のユーザーを返す |
| **Mock** | 呼び出しを検証 | メソッド呼び出し | `send_email()` が正しい引数で呼ばれたか |
| **Spy** | 実際の処理 + 記録 | 呼び出し回数・引数 | 実際にメール送信し、何回呼ばれたか記録 |
| **Fake** | 簡易実装 | ロジック全体 | In-memory DB で Repository を代替 |

### 5.2 テストダブルの選択フロー

```
テストダブル選択のフローチャート

Q1: テスト対象が外部システムの「出力」に依存する？
    （例: DB読み取り、API応答、設定値取得）
    → Yes → Stub を使用

Q2: テスト対象が外部システムに「入力」する？
    （例: メール送信、DB書き込み、イベント発行）
    → Yes → Mock を使用

Q3: 完全な代替実装が必要？
    （例: インメモリDB、ローカルファイルシステム）
    → Yes → Fake を使用

Q4: 引数を満たすだけで良い？
    → Yes → Dummy を使用
```

### 5.3 実装例

**コード例7: Stub（固定値を返す）**

```python
class StubProductRepository:
    """テスト用: 固定の商品データを返す。"""
    def find_by_id(self, product_id: str) -> Product:
        return Product(id=product_id, name="テスト商品", price=1000)

    def find_all(self) -> list[Product]:
        return [
            Product(id="p1", name="商品A", price=1000),
            Product(id="p2", name="商品B", price=2000),
        ]

def test_create_order_calculates_total():
    # Arrange
    product_repo = StubProductRepository()
    use_case = CreateOrderUseCase(product_repo=product_repo)

    # Act
    result = use_case.execute(CreateOrderInput(
        items=[{"product_id": "p1", "quantity": 3}]
    ))

    # Assert
    assert result.total_amount == 3000
```

**コード例8: Mock（呼び出しを検証）**

```python
from unittest.mock import Mock, call

def test_order_placement_sends_notification():
    # Arrange
    notifier = Mock()
    service = OrderService(notifier=notifier)

    # Act
    service.place_order(order_id="order-1")

    # Assert: 正しい引数で呼び出されたか検証
    notifier.send.assert_called_once_with(
        recipient="customer@example.com",
        subject="注文確定のお知らせ",
    )

def test_bulk_notification_sends_to_all_users():
    # Arrange
    notifier = Mock()
    service = NotificationService(notifier=notifier)

    # Act
    service.notify_all(user_ids=["u1", "u2", "u3"], message="セール開始")

    # Assert: 3回呼び出されたか
    assert notifier.send.call_count == 3
    notifier.send.assert_any_call(user_id="u1", message="セール開始")
```

**コード例9: Fake（簡易実装）**

```python
class FakeOrderRepository:
    """テスト用: インメモリで動作するリポジトリ。"""
    def __init__(self):
        self._store: dict[str, Order] = {}

    def save(self, order: Order) -> None:
        self._store[order.id] = order

    def find_by_id(self, order_id: str) -> Order | None:
        return self._store.get(order_id)

    def find_by_user(self, user_id: str) -> list[Order]:
        return [o for o in self._store.values() if o.user_id == user_id]

    def count(self) -> int:
        return len(self._store)

def test_order_persistence_and_retrieval():
    # Arrange
    repo = FakeOrderRepository()
    order = Order(id="o1", user_id="u1", items=[
        OrderItem(product_id="p1", quantity=2, price=1000)
    ])

    # Act
    repo.save(order)
    found = repo.find_by_id("o1")

    # Assert
    assert found is not None
    assert found.id == "o1"
    assert found.user_id == "u1"
```

### 5.4 Mock の過剰使用の危険性

Mock を過剰に使うと、テストが実装詳細に結合し、リファクタリング時にテストが壊れる。Vladimir Khorikov は「Mock は出力（コマンド）の検証にのみ使い、入力（クエリ）には Stub を使え」と主張している。

```python
# BAD: Mock の過剰使用（実装詳細に結合）
def test_order_creation_uses_correct_sql():
    db = Mock()
    service = OrderService(db=db)
    service.create_order(user_id="u1", items=[...])

    # SQL 文の詳細をテスト → リファクタリングで壊れる
    db.execute.assert_called_with(
        "INSERT INTO orders (user_id, total) VALUES (%s, %s)",
        ("u1", 3000)
    )

# GOOD: 振る舞いをテスト（Fake を使用）
def test_order_creation_persists_order():
    repo = FakeOrderRepository()
    service = OrderService(repo=repo)
    service.create_order(user_id="u1", items=[...])

    # 結果の検証（実装詳細に依存しない）
    orders = repo.find_by_user("u1")
    assert len(orders) == 1
    assert orders[0].total == 3000
```

---

## 6. 高度なテスト技法

### 6.1 パラメタライズテスト

同じロジックを異なる入力でテストする場合、パラメタライズテストが効率的である。

**コード例10: パラメタライズテスト**

```python
import pytest

@pytest.mark.parametrize("total, expected_discount", [
    (5000,  0),         # 5000円: 割引なし
    (9999,  0),         # 9999円: 割引なし（境界値-1）
    (10000, 0),         # 10000円: 割引なし（境界値）
    (10001, 1000),      # 10001円: 10%割引（境界値+1）
    (20000, 2000),      # 20000円: 10%割引
    (49999, 4999),      # 49999円: 10%割引（次の境界値-1）
    (50000, 7500),      # 50000円: 15%割引
    (100000, 20000),    # 100000円: 20%割引
])
def test_discount_calculation(total, expected_discount):
    calculator = DiscountCalculator()
    assert calculator.calculate(total) == expected_discount
```

```python
# 複数パラメータの組み合わせ
@pytest.mark.parametrize("user_type, order_total, expected", [
    ("regular",  5000,  0),
    ("regular",  10001, 1000),
    ("premium",  5000,  250),    # プレミアムは5%
    ("premium",  10001, 1500),   # プレミアムは15%
    ("vip",      5000,  500),    # VIPは10%
    ("vip",      10001, 2000),   # VIPは20%
])
def test_discount_by_user_type(user_type, order_total, expected):
    calculator = DiscountCalculator()
    assert calculator.calculate(order_total, user_type) == expected
```

### 6.2 境界値テスト

境界値分析は、バグが最も発生しやすい「境界」に集中してテストする技法である。

**コード例11: 境界値テスト**

```python
class TestPasswordValidation:
    """パスワードバリデーションの境界値テスト。"""

    @pytest.mark.parametrize("password, is_valid, description", [
        ("1234567", False, "7文字: 最小-1 → 無効"),
        ("12345678", True, "8文字: 最小境界 → 有効"),
        ("123456789", True, "9文字: 最小+1 → 有効"),
        ("A" * 19, True, "19文字: 最大-1 → 有効"),
        ("A" * 20, True, "20文字: 最大境界 → 有効"),
        ("A" * 21, False, "21文字: 最大+1 → 無効"),
    ])
    def test_length_boundary(self, password, is_valid, description):
        result = validate_password(password)
        assert result.is_valid == is_valid, description

    @pytest.mark.parametrize("password, is_valid, description", [
        ("", False, "空文字列"),
        ("a", False, "1文字"),
        ("A" * 1000, False, "極端に長い文字列"),
    ])
    def test_edge_cases(self, password, is_valid, description):
        result = validate_password(password)
        assert result.is_valid == is_valid, description
```

```
境界値分析のテンプレート
────────────────────────────────────
  任意の範囲 [min, max] に対して以下をテスト:

  1. min - 1  (範囲外: 無効)
  2. min      (境界値: 有効)
  3. min + 1  (範囲内: 有効)
  4. 代表的な中間値
  5. max - 1  (範囲内: 有効)
  6. max      (境界値: 有効)
  7. max + 1  (範囲外: 無効)

  加えて:
  8. 空入力 (None, "", [], 0)
  9. 極端な値 (最大整数, 非常に長い文字列)
────────────────────────────────────
```

### 6.3 プロパティベーステスト

特定の入力値ではなく、「任意の入力に対して成り立つ性質」をテストする。

**コード例12: プロパティベーステスト**

```python
from hypothesis import given, strategies as st

# 性質1: ソートされたリストは元のリストと同じ要素を持つ
@given(st.lists(st.integers()))
def test_sort_preserves_elements(lst):
    sorted_lst = sorted(lst)
    assert sorted(sorted_lst) == sorted(lst)
    assert len(sorted_lst) == len(lst)

# 性質2: ソートされたリストは昇順
@given(st.lists(st.integers(), min_size=2))
def test_sort_is_ordered(lst):
    sorted_lst = sorted(lst)
    for i in range(len(sorted_lst) - 1):
        assert sorted_lst[i] <= sorted_lst[i + 1]

# 性質3: JSONエンコード→デコードで元のデータが復元される
@given(st.dictionaries(
    keys=st.text(min_size=1, max_size=50),
    values=st.one_of(st.integers(), st.text(), st.booleans(), st.none()),
))
def test_json_roundtrip(data):
    encoded = json.dumps(data)
    decoded = json.loads(encoded)
    assert decoded == data

# 性質4: 金額計算で丸め誤差が発生しない
@given(
    price=st.decimals(min_value=1, max_value=1000000, places=0),
    quantity=st.integers(min_value=1, max_value=100),
)
def test_total_is_positive(price, quantity):
    total = price * quantity
    assert total >= price  # 合計は単価以上
    assert total >= quantity  # 合計は数量以上
```

### 6.4 スナップショットテスト

出力が複雑な場合、初回実行時の出力を「スナップショット」として保存し、以降の実行結果と比較する。

**コード例13: スナップショットテスト（pytest-snapshot）**

```python
def test_user_serialization(snapshot):
    user = User(
        id="u1", name="Alice", email="alice@example.com",
        created_at=datetime(2024, 1, 1)
    )
    result = user.to_dict()

    # 初回: スナップショットを保存
    # 2回目以降: 保存されたスナップショットと比較
    snapshot.assert_match(json.dumps(result, indent=2), "user_serialization.json")
```

### 6.5 テーブル駆動テスト（Go スタイル）

Go で広く使われるテストパターン。テストケースをテーブル（リスト）で定義し、ループで実行する。

```python
# テーブル駆動テスト
class TestEmailValidation:
    test_cases = [
        {"input": "user@example.com", "valid": True, "desc": "正常なメール"},
        {"input": "user@example", "valid": False, "desc": "TLDなし"},
        {"input": "@example.com", "valid": False, "desc": "ローカル部なし"},
        {"input": "user@", "valid": False, "desc": "ドメインなし"},
        {"input": "", "valid": False, "desc": "空文字列"},
        {"input": "a" * 255 + "@example.com", "valid": False, "desc": "長すぎる"},
        {"input": "user+tag@example.com", "valid": True, "desc": "プラスタグ"},
        {"input": "user.name@example.com", "valid": True, "desc": "ドット入り"},
    ]

    @pytest.mark.parametrize("case", test_cases, ids=lambda c: c["desc"])
    def test_email_validation(self, case):
        result = validate_email(case["input"])
        assert result.is_valid == case["valid"], f"Failed: {case['desc']}"
```

---

## 7. テスト品質の維持と改善

### 7.1 テストカバレッジの適切な運用

```
テストカバレッジの種類
────────────────────────────────────
1. 行カバレッジ (Line Coverage)
   → 実行された行の割合。最も基本的だが浅い

2. 分岐カバレッジ (Branch Coverage)
   → if/else の各分岐が実行されたか。行より深い

3. 条件カバレッジ (Condition Coverage)
   → 複合条件(A && B)の各条件が true/false の両方を経験したか

4. ミューテーションスコア (Mutation Score)
   → コードを意図的に壊した場合にテストが検出できる割合
   → テストの「品質」を測る最も精度の高い指標
────────────────────────────────────
```

```python
# カバレッジの限界を示す例
def calculate_discount(amount: int, is_premium: bool) -> int:
    if amount > 10000 and is_premium:
        return int(amount * 0.15)
    return 0

# このテストはLine Coverage 100% だがバグを見逃す
def test_discount():
    assert calculate_discount(20000, True) == 3000  # カバレッジ100%
    # だが (20000, False) → 0 のケースをテストしていない
    # もし条件が or に変更されてもテストは通ってしまう
```

### 7.2 ミューテーションテスト

コードを意図的に「壊して」（ミュータント）、テストがそれを検出できるかを確認する。

```python
# 元のコード
def is_adult(age: int) -> bool:
    return age >= 18

# ミュータント1: >= を > に変更
def is_adult_mutant1(age: int) -> bool:
    return age > 18  # age == 18 のテストがないと検出できない

# ミュータント2: 18 を 17 に変更
def is_adult_mutant2(age: int) -> bool:
    return age >= 17  # 境界値テストがないと検出できない

# ミュータント3: True/False を反転
def is_adult_mutant3(age: int) -> bool:
    return not (age >= 18)  # 基本テストがあれば検出可能
```

```
ミューテーションテストの実行 (mutmut)
────────────────────────────────────
$ pip install mutmut
$ mutmut run --paths-to-mutate=src/ --tests-dir=tests/

結果の読み方:
  Killed: テストがミュータントを検出（良い）
  Survived: テストがミュータントを見逃した（テスト不足）
  Timeout: テストが終了しなかった

ミューテーションスコア = Killed / (Killed + Survived) * 100
目標: 80%以上
────────────────────────────────────
```

### 7.3 Flaky テスト対策

Flaky テスト（不安定なテスト）は、テストスイート全体への信頼を毀損する。「テストが失敗しても、まあ Flaky だよね」という文化が定着すると、真のバグも見逃すようになる。

```
Flaky テストの主な原因と対策
────────────────────────────────────
原因1: テスト間の順序依存
  対策: → pytest-randomly でランダム順序で実行
       → 各テストで状態をリセット

原因2: タイミング依存（非同期処理）
  対策: → sleep() ではなく明示的な待機（ポーリング）
       → awaitility パターンの使用

原因3: 外部サービスへの依存
  対策: → テストダブルで置換
       → WireMock 等でAPIをスタブ化

原因4: リソース競合（ポート、ファイル）
  対策: → ランダムポートの使用
       → テンポラリディレクトリの使用

原因5: 浮動小数点の比較
  対策: → pytest.approx() の使用
       → Decimal の使用
────────────────────────────────────
```

**コード例14: Flaky テストの修正**

```python
# BAD: タイミング依存
def test_async_job_completion():
    start_background_job("process-data")
    time.sleep(5)  # 5秒で終わるはず... Flaky!
    assert job_status("process-data") == "completed"

# GOOD: ポーリングで待機
def test_async_job_completion():
    start_background_job("process-data")

    # 最大30秒、1秒間隔でポーリング
    for _ in range(30):
        if job_status("process-data") == "completed":
            return  # テスト成功
        time.sleep(1)

    pytest.fail("ジョブが30秒以内に完了しなかった")


# BETTER: tenacity ライブラリで待機
from tenacity import retry, stop_after_delay, wait_fixed

@retry(stop=stop_after_delay(30), wait=wait_fixed(1))
def wait_for_job_completion(job_id: str):
    assert job_status(job_id) == "completed"

def test_async_job_completion():
    start_background_job("process-data")
    wait_for_job_completion("process-data")
```

---

## 8. テストの構造化とフィクスチャ

### 8.1 pytest フィクスチャ

**コード例15: フィクスチャの活用**

```python
import pytest

# セッションスコープ: テストスイート全体で1回だけ実行
@pytest.fixture(scope="session")
def database():
    """テスト用データベースの作成と破棄。"""
    db = create_test_database()
    create_schema(db)
    yield db
    drop_database(db)

# 関数スコープ: 各テスト関数ごとに実行
@pytest.fixture
def clean_db(database):
    """各テスト前にトランザクション開始、後にロールバック。"""
    database.begin()
    yield database
    database.rollback()

# カスタムファクトリフィクスチャ
@pytest.fixture
def create_user(clean_db):
    """テスト用ユーザーのファクトリ。"""
    def _create_user(name="Alice", email="alice@test.com", role="user"):
        user = User(name=name, email=email, role=role)
        clean_db.save(user)
        return user
    return _create_user

# フィクスチャの使用
def test_user_can_place_order(create_user, clean_db):
    user = create_user(name="Bob")
    order = Order(user_id=user.id, items=[...])
    clean_db.save(order)

    assert order.user_id == user.id
```

### 8.2 テストの分類とマーカー

```python
# conftest.py でマーカーを登録
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: 実行が遅いテスト")
    config.addinivalue_line("markers", "integration: 統合テスト")
    config.addinivalue_line("markers", "e2e: E2Eテスト")

# テストにマーカーを付与
@pytest.mark.slow
def test_full_data_migration():
    ...

@pytest.mark.integration
def test_database_connection():
    ...

# 特定のマーカーのテストのみ実行
# $ pytest -m "not slow"           # 遅いテスト以外を実行
# $ pytest -m "integration"        # 統合テストのみ実行
# $ pytest -m "not e2e"            # E2E 以外を実行
```

---

## 9. CI/CD パイプラインとの統合

### 9.1 テスト戦略の自動化

```yaml
# .github/workflows/test.yml
name: Test Pipeline

on:
  pull_request:
    branches: [main]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install -r requirements-dev.txt
      - run: |
          pytest tests/unit/ \
            --cov=src \
            --cov-report=xml \
            --cov-fail-under=80 \
            -x \
            --timeout=10

  integration-tests:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_DB: test
          POSTGRES_PASSWORD: test
        ports:
          - 5432:5432
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install -r requirements-dev.txt
      - run: pytest tests/integration/ -v --timeout=60
```

### 9.2 テストの並列実行

```
テスト実行の高速化戦略
────────────────────────────────────
1. pytest-xdist で並列実行
   $ pytest -n auto  # CPU コア数に応じて自動並列化

2. テストの分割
   # CI で複数ジョブに分割
   $ pytest --splits 4 --group 1  # 4分割の1番目
   $ pytest --splits 4 --group 2  # 4分割の2番目

3. 変更ファイルに関連するテストのみ実行
   $ pytest --picked  # git diff のファイルに関連するテスト

4. キャッシュの活用
   $ pytest --lf  # 前回失敗したテストのみ再実行
   $ pytest --ff  # 前回失敗したテストを先に実行
────────────────────────────────────
```

---

## 10. 比較表

### テスト手法の比較

| 手法 | 粒度 | 速度 | 保守性 | 適用場面 |
|------|------|------|--------|---------|
| ユニットテスト | メソッド/関数 | 最速 | 高 | ビジネスロジック、純関数 |
| 統合テスト | モジュール間連携 | 中速 | 中 | DB操作、API連携 |
| E2Eテスト | 全体フロー | 低速 | 低 | クリティカルパス |
| スナップショット | UI出力 | 高速 | 低 | UIコンポーネント |
| プロパティベース | ランダム入力 | 中速 | 高 | アルゴリズム、パーサー |
| ミューテーション | テスト品質 | 低速 | ── | テストスイートの品質検証 |

### テストダブルの使い分け

| テストダブル | 使う場面 | 避ける場面 |
|------------|---------|-----------|
| Stub | 外部サービスの応答を固定したい | ロジックが単純で不要 |
| Mock | 副作用（メール送信等）の発生を検証したい | 実装詳細に結合してしまう場合 |
| Fake | 完全なインメモリ代替が欲しい | 実装コストが高すぎる場合 |
| Spy | 実際の処理を行いつつ記録したい | Mock で十分な場合 |
| Dummy | 引数を埋めたいだけ | テスト対象が実際に使う場合 |

---

## 11. アンチパターン

### アンチパターン 1: 実装詳細をテストする

```python
# BAD: 内部実装（メソッド呼び出し順序）をテスト
def test_order_creation_calls_methods_in_order():
    mock = Mock()
    service = OrderService(repo=mock)
    service.create_order(...)
    assert mock.method_calls == [
        call.validate(),        # 内部の呼び出し順序に結合
        call.calculate_tax(),
        call.save(),
    ]
    # リファクタリングで内部実装を変えただけでテストが壊れる

# GOOD: 振る舞い（入力 → 出力）をテスト
def test_order_creation_returns_valid_order():
    repo = FakeOrderRepository()
    service = OrderService(repo=repo)
    result = service.create_order(user_id="u1", items=[...])
    assert result.status == "created"
    assert result.total > 0
```

**なぜダメか:** 実装詳細に結合したテストは、コードをリファクタリングするたびに壊れる。テストがリファクタリングを妨害するようになると、テストの最大の価値（安心して変更できること）が失われる。

### アンチパターン 2: テストが遅い

```python
# BAD: 各テストで DB を初期化
def test_user_query(self):
    db.create_all()          # 毎回スキーマ作成 (遅い)
    seed_test_data(1000)     # 毎回1000件投入 (遅い)
    result = query_users()
    assert len(result) > 0
    db.drop_all()

# GOOD: フィクスチャで共有、トランザクションロールバック
@pytest.fixture(scope="session")
def db():
    create_schema()
    yield database
    drop_schema()

@pytest.fixture(autouse=True)
def transaction(db):
    db.begin()
    yield
    db.rollback()    # 各テスト後にロールバック (高速)
```

**なぜダメか:** テストが遅いと、開発者はテストの実行を避けるようになる。「テストを実行するのが面倒」→「テストを書かない」→「品質低下」の悪循環に陥る。

### アンチパターン 3: テストが複数の概念を検証する

```python
# BAD: 1つのテストで複数の概念を検証
def test_user_creation():
    user = service.create_user("Alice", "alice@example.com")
    assert user.id is not None                    # 概念1: ID生成
    assert user.name == "Alice"                   # 概念2: 名前保存
    assert user.email == "alice@example.com"      # 概念3: メール保存
    assert user.created_at is not None            # 概念4: タイムスタンプ
    assert user.status == "active"                # 概念5: 初期ステータス
    assert email_was_sent("alice@example.com")    # 概念6: メール送信
    assert audit_log_exists("user_created")       # 概念7: 監査ログ

# GOOD: 概念ごとにテストを分割
def test_create_user_generates_unique_id():
    user = service.create_user("Alice", "alice@example.com")
    assert user.id is not None

def test_create_user_saves_name_and_email():
    user = service.create_user("Alice", "alice@example.com")
    assert user.name == "Alice"
    assert user.email == "alice@example.com"

def test_create_user_sets_initial_status_to_active():
    user = service.create_user("Alice", "alice@example.com")
    assert user.status == "active"

def test_create_user_sends_welcome_email():
    service.create_user("Alice", "alice@example.com")
    assert email_was_sent("alice@example.com")
```

### アンチパターン 4: 条件分岐のあるテスト

```python
# BAD: テスト内に条件分岐がある
def test_discount(user_type):
    calculator = DiscountCalculator()
    if user_type == "premium":
        assert calculator.calculate(10000, user_type) == 1500
    elif user_type == "regular":
        assert calculator.calculate(10000, user_type) == 1000
    else:
        assert calculator.calculate(10000, user_type) == 0

# GOOD: パラメタライズテストで条件分岐を排除
@pytest.mark.parametrize("user_type, expected", [
    ("premium", 1500),
    ("regular", 1000),
    ("guest", 0),
])
def test_discount(user_type, expected):
    calculator = DiscountCalculator()
    assert calculator.calculate(10000, user_type) == expected
```

---

## 12. 実践演習

### 演習1（基礎）: AAA パターンでテストを書く

以下の `PasswordValidator` クラスに対して、AAA パターンに従ったユニットテストを5つ以上書いてください。

```python
class PasswordValidator:
    MIN_LENGTH = 8
    MAX_LENGTH = 64

    def validate(self, password: str) -> ValidationResult:
        errors = []
        if len(password) < self.MIN_LENGTH:
            errors.append("最低8文字必要です")
        if len(password) > self.MAX_LENGTH:
            errors.append("最大64文字までです")
        if not any(c.isupper() for c in password):
            errors.append("大文字を1文字以上含めてください")
        if not any(c.isdigit() for c in password):
            errors.append("数字を1文字以上含めてください")
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)
```

**期待される出力:**

```python
class TestPasswordValidator:
    def setup_method(self):
        self.validator = PasswordValidator()

    def test_valid_password_returns_success(self):
        result = self.validator.validate("SecurePass1")
        assert result.is_valid is True
        assert result.errors == []

    def test_short_password_returns_error(self):
        result = self.validator.validate("Short1A")
        assert result.is_valid is False
        assert "最低8文字必要です" in result.errors

    def test_too_long_password_returns_error(self):
        result = self.validator.validate("A1" + "a" * 63)
        assert result.is_valid is False
        assert "最大64文字までです" in result.errors

    def test_no_uppercase_returns_error(self):
        result = self.validator.validate("lowercase123")
        assert result.is_valid is False
        assert "大文字を1文字以上含めてください" in result.errors

    def test_no_digit_returns_error(self):
        result = self.validator.validate("NoDigitsHere")
        assert result.is_valid is False
        assert "数字を1文字以上含めてください" in result.errors

    def test_multiple_violations_returns_all_errors(self):
        result = self.validator.validate("short")
        assert result.is_valid is False
        assert len(result.errors) >= 2
```

### 演習2（応用）: テストダブルを使ったテスト

以下の `OrderService` に対して、Stub と Mock を使い分けたテストを書いてください。

```python
class OrderService:
    def __init__(self, product_repo, payment_gateway, notifier):
        self._product_repo = product_repo
        self._payment = payment_gateway
        self._notifier = notifier

    def place_order(self, user_id, items):
        products = [self._product_repo.find_by_id(i["id"]) for i in items]
        total = sum(p.price * i["qty"] for p, i in zip(products, items))

        payment_result = self._payment.charge(user_id, total)
        if not payment_result.success:
            raise PaymentError(payment_result.error_message)

        order = Order(user_id=user_id, items=items, total=total)
        self._notifier.send_confirmation(user_id, order)
        return order
```

**期待される出力:**

```python
def test_place_order_calculates_correct_total():
    # Stub: 商品リポジトリから固定の商品を返す
    product_repo = StubProductRepo({
        "p1": Product(id="p1", price=1000),
        "p2": Product(id="p2", price=2000),
    })
    payment = StubPaymentGateway(always_succeeds=True)
    notifier = Mock()

    service = OrderService(product_repo, payment, notifier)
    order = service.place_order("u1", [
        {"id": "p1", "qty": 2},
        {"id": "p2", "qty": 1},
    ])

    assert order.total == 4000  # 1000*2 + 2000*1

def test_place_order_sends_confirmation():
    product_repo = StubProductRepo({"p1": Product(id="p1", price=1000)})
    payment = StubPaymentGateway(always_succeeds=True)
    notifier = Mock()  # Mock: 通知が呼ばれたか検証

    service = OrderService(product_repo, payment, notifier)
    service.place_order("u1", [{"id": "p1", "qty": 1}])

    # 通知が正しく呼ばれたか検証
    notifier.send_confirmation.assert_called_once()

def test_place_order_raises_on_payment_failure():
    product_repo = StubProductRepo({"p1": Product(id="p1", price=1000)})
    payment = StubPaymentGateway(always_fails=True, error="カード拒否")
    notifier = Mock()

    service = OrderService(product_repo, payment, notifier)

    with pytest.raises(PaymentError, match="カード拒否"):
        service.place_order("u1", [{"id": "p1", "qty": 1}])

    # 決済失敗時は通知が送られないこと
    notifier.send_confirmation.assert_not_called()
```

### 演習3（発展）: プロパティベーステストの設計

以下の `Money` クラスに対して、hypothesis を使ったプロパティベーステストを設計してください。「任意の金額に対して成り立つ性質」を3つ以上テストしてください。

```python
class Money:
    def __init__(self, amount: int, currency: str = "JPY"):
        if amount < 0:
            raise ValueError("金額は0以上")
        self.amount = amount
        self.currency = currency

    def add(self, other: "Money") -> "Money":
        if self.currency != other.currency:
            raise CurrencyMismatchError()
        return Money(self.amount + other.amount, self.currency)

    def multiply(self, factor: int) -> "Money":
        return Money(self.amount * factor, self.currency)
```

**期待される出力:**

```python
from hypothesis import given, strategies as st

jpy_amount = st.integers(min_value=0, max_value=10**9)

# 性質1: 加算の可換性 (a + b == b + a)
@given(a=jpy_amount, b=jpy_amount)
def test_addition_is_commutative(a, b):
    m1 = Money(a).add(Money(b))
    m2 = Money(b).add(Money(a))
    assert m1.amount == m2.amount

# 性質2: 加算の結合性 ((a + b) + c == a + (b + c))
@given(a=jpy_amount, b=jpy_amount, c=jpy_amount)
def test_addition_is_associative(a, b, c):
    left = Money(a).add(Money(b)).add(Money(c))
    right = Money(a).add(Money(b).add(Money(c)))
    assert left.amount == right.amount

# 性質3: ゼロの加算は恒等 (a + 0 == a)
@given(a=jpy_amount)
def test_adding_zero_is_identity(a):
    result = Money(a).add(Money(0))
    assert result.amount == a

# 性質4: 乗算の分配法則 ((a + b) * n == a*n + b*n)
@given(a=jpy_amount, b=jpy_amount, n=st.integers(min_value=0, max_value=100))
def test_multiplication_distributes_over_addition(a, b, n):
    left = Money(a).add(Money(b)).multiply(n)
    right = Money(a).multiply(n).add(Money(b).multiply(n))
    assert left.amount == right.amount
```

---

## 13. FAQ

### Q1. テストカバレッジは何%を目指すべきか？

**A.** カバレッジ数値は目標ではなく指標。80%前後が現実的な目安だが、重要なのは「クリティカルパスが網羅されているか」。100%を目指すと getter/setter のような価値の低いテストが増え、保守コストが上がる。カバレッジが低い箇所を可視化し、ビジネスリスクの高い部分から優先的にテストを追加するのが効果的。

ミューテーションテストを併用すると、カバレッジ100%でもバグを見逃すテストを発見できる。カバレッジは「テストされていない箇所を見つけるツール」として使い、「テストの品質指標」としては使わないこと。

### Q2. テストの実行が遅い場合の対策は？

**A.** 以下の優先順で対策する:
1. テストピラミッドを守り、ユニットテストの割合を増やす
2. テストの並列実行（`pytest-xdist -n auto`）
3. DB テストはトランザクションロールバックで高速化
4. 外部 API はテストダブルで置換
5. CI ではテストを分割して並列ジョブで実行
6. 変更されたファイルに関連するテストのみ実行（`pytest --picked`）
7. Docker レイヤーキャッシュ、pip キャッシュの活用

目標は「ユニットテスト全体 < 10秒、全テスト < 5分」。

### Q3. Flaky テスト（不安定なテスト）の対処法は？

**A.** Flaky テストの主な原因は (1) テスト間の順序依存、(2) タイミング依存（非同期処理の完了待ち不足）、(3) 外部サービスへの依存。対策として、独立性の確認（ランダム順序で実行）、明示的な待機（ポーリング + タイムアウト）、外部依存のモック化を行う。根本解決できない場合は quarantine（隔離）して個別に対処する。

Flaky テストは必ずトラッキングすること。「このテストは時々失敗する」を放置すると、チーム全体がテスト結果を信用しなくなる。

### Q4. TDD は必ず実践すべきか？

**A.** TDD は強力な技法だが、全てのコードに適用すべきではない。以下の場面で特に効果的:
- **ビジネスロジック**: 入出力が明確で、テストファーストが自然
- **バグ修正**: バグを再現するテストを先に書き、修正して緑にする
- **API設計**: テストがAPIの利用者視点を提供する

一方、以下の場面では TDD が非効率な場合がある:
- **プロトタイピング**: 仕様が流動的で、テストが無駄になりやすい
- **UIレイアウト**: 見た目のテストは TDD に不向き
- **探索的な実装**: 何を作るか自体が不明確な場合

### Q5. テストコードにもコードレビューは必要か？

**A.** 必要である。テストコードもプロダクションコードの一部であり、保守性が重要。レビューのポイント:
- テスト名が意図を表現しているか
- AAA パターンに従っているか
- テストが実装詳細ではなく振る舞いを検証しているか
- 境界値やエッジケースが考慮されているか
- テストダブルの使い方が適切か

---

## まとめ

| 項目 | ポイント |
|------|---------|
| テストの役割 | 回帰防止、設計フィードバック、ドキュメンテーション |
| AAA パターン | Arrange → Act → Assert の3段階で構造化 |
| FIRST 原則 | Fast, Independent, Repeatable, Self-Validating, Timely |
| テストピラミッド | Unit 70% : Integration 20% : E2E 10% |
| テストダブル | Stub (入力), Mock (出力検証), Fake (簡易実装) を適切に使い分け |
| 命名規則 | [対象]_[状況]_[期待結果] で意図を明示 |
| 振る舞いテスト | 実装詳細ではなく入出力をテスト |
| カバレッジ | 80%目安。ミューテーションテストで品質も検証 |
| Flaky 対策 | ランダム順序実行、ポーリング待機、外部依存のモック化 |

---

## 次に読むべきガイド

- [レガシーコード](../02-refactoring/02-legacy-code.md) ── テストのない既存コードへのテスト追加手法（特性テスト、Seam の発見）
- [継続的改善](../02-refactoring/04-continuous-improvement.md) ── CI/CD でのテスト自動化と品質ゲートの設定
- [コードスメル](../02-refactoring/00-code-smells.md) ── テストが書きにくいコードの改善指針
- [リファクタリング技法](../02-refactoring/01-refactoring-techniques.md) ── テストで保護しながらコードを改善する手法
- [API設計](../03-practices-advanced/03-api-design.md) ── API テストの設計と契約テスト
- [コメント](./03-comments.md) ── テストコードにおけるドキュメンテーション

---

## 参考文献

1. **Vladimir Khorikov** 『Unit Testing Principles, Practices, and Patterns』 Manning, 2020 ── テスト設計の決定版。Mock の過剰使用の弊害と、振る舞いテストの重要性を解説
2. **Kent Beck** 『Test Driven Development: By Example』 Addison-Wesley, 2002 ── TDD の原典。Red-Green-Refactor サイクルの考案者による解説
3. **Gerard Meszaros** 『xUnit Test Patterns: Refactoring Test Code』 Addison-Wesley, 2007 ── テストパターンの百科事典。テストダブルの分類の原典
4. **Steve Freeman & Nat Pryce** 『Growing Object-Oriented Software, Guided by Tests』 Addison-Wesley, 2009 ── Mock を活用した Outside-In TDD の手法
5. **Martin Fowler** "TestPyramid" (Blog, 2012) ── https://martinfowler.com/bliki/TestPyramid.html ── テストピラミッドの解説と実践的なガイドライン
