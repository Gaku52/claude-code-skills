# テスト原則

> テストはコードの品質を保証する安全網であり、設計を改善するフィードバックメカニズムである。AAA パターン・FIRST 原則・テストダブルの使い分けを理解し、信頼性の高いテストスイートを構築する手法を解説する

## この章で学ぶこと

1. **テストの基本原則** — AAA パターン、FIRST 原則、テストピラミッドによる構造化
2. **テストダブルの使い分け** — Stub、Mock、Spy、Fake の役割と適切な選択基準
3. **実践的なテスト設計** — 境界値テスト、パラメタライズテスト、プロパティベーステスト

---

## 1. テストピラミッド

### 1.1 全体構造

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

### 1.2 各レベルの特性

```
レベル        速度      信頼性    保守コスト    フィードバック
-----------------------------------------------------------------
Unit         < 1ms     高        低           即座
Integration  < 1sec    中        中           数秒
E2E          < 30sec   低(Flaky) 高           数分

  ★ ユニットテストが最も費用対効果が高い
  ★ E2E は「煙テスト」レベルの最低限に留める
```

---

## 2. AAA パターン

### 2.1 Arrange-Act-Assert

```python
# AAA パターンの基本形
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

### 2.2 テスト名の命名規則

```python
# 命名パターン: [対象]_[状況]_[期待結果]
def test_order_when_empty_items_raises_validation_error():
    ...

def test_discount_when_total_exceeds_10000_applies_10_percent():
    ...

def test_user_registration_with_duplicate_email_returns_conflict():
    ...

# BDD スタイル
class TestOrder:
    class TestPlace:
        def test_changes_status_to_placed(self): ...
        def test_raises_error_when_items_empty(self): ...
        def test_emits_order_placed_event(self): ...
```

---

## 3. FIRST 原則

| 原則 | 意味 | 具体的なガイドライン |
|------|------|------------------|
| **F**ast | 高速 | ユニットテスト1件 < 10ms。全スイート < 10秒 |
| **I**ndependent | 独立 | テスト間に依存なし。順序を変えても結果が同じ |
| **R**epeatable | 再現可能 | 環境・時刻に依存しない。CI でも同じ結果 |
| **S**elf-Validating | 自己検証 | Pass/Fail が自動判定。手動確認不要 |
| **T**imely | 適時 | プロダクションコードの直前・直後に書く |

```python
# FIRST 原則の違反例と修正

# BAD (Independent 違反): テスト間で状態を共有
class TestUserService:
    user_id = None

    def test_create_user(self):
        TestUserService.user_id = service.create_user("Alice")  # 状態を保存

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

```python
# BAD (Repeatable 違反): 現在時刻に依存
def test_is_expired():
    token = Token(expires_at=datetime(2026, 3, 1))
    assert token.is_expired()  # 2026年3月以降にしか通らない

# GOOD: 時刻を注入可能にする
def test_is_expired():
    token = Token(expires_at=datetime(2026, 3, 1))
    now = datetime(2026, 3, 2)  # テスト用の固定時刻
    assert token.is_expired(now=now)
```

---

## 4. テストダブル

### 4.1 種類と使い分け

| 種類 | 目的 | 検証対象 | 例 |
|------|------|---------|-----|
| **Stub** | 固定値を返す | 戻り値 | `find_by_id()` が固定のユーザーを返す |
| **Mock** | 呼び出しを検証 | メソッド呼び出し | `send_email()` が正しい引数で呼ばれたか |
| **Spy** | 実際の処理 + 記録 | 呼び出し回数・引数 | 実際にメール送信し、何回呼ばれたか記録 |
| **Fake** | 簡易実装 | ロジック全体 | In-memory DB で Repository を代替 |

### 4.2 実装例

```python
# Stub: 固定値を返す
class StubProductRepository:
    def find_by_id(self, product_id: str):
        return Product(id=product_id, name="テスト商品", price=1000)

def test_create_order_calculates_total():
    product_repo = StubProductRepository()
    use_case = CreateOrderUseCase(product_repo=product_repo, ...)
    result = use_case.execute(CreateOrderInput(
        items=[{"product_id": "p1", "quantity": 3}]
    ))
    assert result.total_amount == 3000


# Mock: 呼び出しを検証
from unittest.mock import Mock, call

def test_order_placement_sends_notification():
    notifier = Mock()
    service = OrderService(notifier=notifier)

    service.place_order(order_id="order-1")

    notifier.send.assert_called_once_with(
        recipient="customer@example.com",
        subject="注文確定のお知らせ",
    )


# Fake: 簡易実装
class FakeOrderRepository:
    def __init__(self):
        self._store = {}

    def save(self, order):
        self._store[order.id] = order

    def find_by_id(self, order_id):
        return self._store.get(order_id)

    def find_by_user(self, user_id):
        return [o for o in self._store.values() if o.user_id == user_id]

def test_order_persistence_and_retrieval():
    repo = FakeOrderRepository()
    order = Order(id="o1", user_id="u1", items=[...])

    repo.save(order)
    found = repo.find_by_id("o1")

    assert found.id == "o1"
    assert found.user_id == "u1"
```

---

## 5. 高度なテスト技法

### 5.1 パラメタライズテスト

```python
# pytest.mark.parametrize で複数ケースを効率的にテスト
import pytest

@pytest.mark.parametrize("total, expected_discount", [
    (5000,  0),         # 5000円未満: 割引なし
    (10000, 0),         # 10000円: 割引なし
    (10001, 1000),      # 10001円: 10%割引
    (20000, 2000),      # 20000円: 10%割引
    (50000, 7500),      # 50000円: 15%割引
    (100000, 20000),    # 100000円: 20%割引
])
def test_discount_calculation(total, expected_discount):
    calculator = DiscountCalculator()
    assert calculator.calculate(total) == expected_discount
```

### 5.2 境界値テスト

```python
# 境界値分析: ちょうどの値、±1 をテスト
class TestPasswordValidation:
    @pytest.mark.parametrize("password, is_valid", [
        ("1234567", False),     # 7文字: 最小-1 → 無効
        ("12345678", True),     # 8文字: 最小境界 → 有効
        ("12345678901234567890", True),   # 20文字: 最大境界 → 有効
        ("123456789012345678901", False), # 21文字: 最大+1 → 無効
    ])
    def test_length_boundary(self, password, is_valid):
        result = validate_password(password)
        assert result.is_valid == is_valid
```

---

## 6. テスト手法比較表

| 手法 | 粒度 | 速度 | 保守性 | 適用場面 |
|------|------|------|--------|---------|
| ユニットテスト | メソッド/関数 | 最速 | 高 | ビジネスロジック、純関数 |
| 統合テスト | モジュール間連携 | 中速 | 中 | DB操作、API連携 |
| E2Eテスト | 全体フロー | 低速 | 低 | クリティカルパス |
| スナップショット | UI出力 | 高速 | 低 | UIコンポーネント |
| プロパティベース | ランダム入力 | 中速 | 高 | アルゴリズム、パーサー |

| テストダブル | 使う場面 | 避ける場面 |
|------------|---------|-----------|
| Stub | 外部サービスの応答を固定したい | ロジックが単純で不要 |
| Mock | 副作用の発生を検証したい | 実装詳細に結合してしまう場合 |
| Fake | 完全なインメモリ代替が欲しい | 実装コストが高すぎる場合 |
| Spy | 実際の処理を行いつつ記録したい | Mock で十分な場合 |

---

## 7. アンチパターン

### アンチパターン 1: 実装詳細をテストする

```python
# BAD: 内部実装（メソッド呼び出し順序）をテスト
def test_order_creation_calls_methods_in_order():
    mock = Mock()
    service = OrderService(repo=mock)
    service.create_order(...)
    assert mock.method_calls == [
        call.validate(),        # ← 内部の呼び出し順序に結合
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

---

## 8. FAQ

### Q1. テストカバレッジは何%を目指すべきか？

**A.** カバレッジ数値は目標ではなく指標。80%前後が現実的な目安だが、重要なのは「クリティカルパスが網羅されているか」。100%を目指すと getter/setter のような価値の低いテストが増え、保守コストが上がる。カバレッジが低い箇所を可視化し、ビジネスリスクの高い部分から優先的にテストを追加するのが効果的。

### Q2. テストの実行が遅い場合の対策は？

**A.** (1) テストピラミッドを守り、ユニットテストの割合を増やす。(2) テストの並列実行（`pytest-xdist`）。(3) DB テストはトランザクションロールバックで高速化。(4) 外部 API はテストダブルで置換。(5) CI ではテストを分割して並列ジョブで実行。(6) 変更されたファイルに関連するテストのみ実行する選択実行。

### Q3. Flaky テスト（不安定なテスト）の対処法は？

**A.** Flaky テストの主な原因は (1) テスト間の順序依存、(2) タイミング依存（非同期処理の完了待ち不足）、(3) 外部サービスへの依存。対策として、独立性の確認（ランダム順序で実行）、明示的な待機（ポーリング + タイムアウト）、外部依存のモック化を行う。根本解決できない場合は quarantine（隔離）して個別に対処する。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| AAA パターン | Arrange → Act → Assert の3段階で構造化 |
| FIRST 原則 | Fast, Independent, Repeatable, Self-Validating, Timely |
| テストピラミッド | Unit 70% : Integration 20% : E2E 10% |
| テストダブル | Stub (値)、Mock (検証)、Fake (簡易実装) を適切に使い分け |
| 命名規則 | [対象]_[状況]_[期待結果] で意図を明示 |
| 振る舞いテスト | 実装詳細ではなく入出力をテスト |
| カバレッジ | 80%目安。クリティカルパスの網羅を優先 |

---

## 次に読むべきガイド

- [レガシーコード](../02-refactoring/02-legacy-code.md) — テストのない既存コードへのテスト追加手法
- [継続的改善](../02-refactoring/04-continuous-improvement.md) — CI/CD でのテスト自動化
- [API設計](../03-practices-advanced/03-api-design.md) — API テストの設計

---

## 参考文献

1. **Unit Testing Principles, Practices, and Patterns** — Vladimir Khorikov (Manning, 2020) — テスト設計の決定版
2. **Test Driven Development: By Example** — Kent Beck (Addison-Wesley, 2002) — TDD の原典
3. **xUnit Test Patterns** — Gerard Meszaros (Addison-Wesley, 2007) — テストパターンの百科事典
4. **Growing Object-Oriented Software, Guided by Tests** — Steve Freeman & Nat Pryce (Addison-Wesley, 2009)
