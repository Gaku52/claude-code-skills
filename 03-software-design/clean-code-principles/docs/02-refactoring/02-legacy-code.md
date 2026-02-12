# レガシーコード

> レガシーコードとは「テストのないコード」である（Michael Feathers）。何年も保守され、全体像を誰も把握していないコードベースと向き合い、安全に変更を加えるための体系的な技法を、依存性の切断・特性テスト・Strangler Fig パターンを通じて解説する。Feathers は『Working Effectively with Legacy Code』で「レガシーコードは恐怖の源である ── テストがなければ、変更するたびに何が壊れるか分からない」と述べた。本章では、恐怖を取り除き、レガシーコードを計画的に改善するための実践的な技法を深掘りする。

---

## 前提知識

| 前提 | 参照先 |
|------|--------|
| コードスメルの分類 | [00-code-smells.md](./00-code-smells.md) |
| リファクタリング技法 | [01-refactoring-techniques.md](./01-refactoring-techniques.md) |
| テストの基礎（AAA、テストダブル） | [01-practices/04-testing-principles.md](../01-practices/04-testing-principles.md) |
| クリーンコードの基本原則 | [00-principles/](../00-principles/) |

---

## この章で学ぶこと

1. **レガシーコードの定義と特徴** ── 「テストのないコード」としてのレガシーコードの本質と、変更リスクの定量評価方法を理解する
2. **Seam（継ぎ目）の発見** ── コードを変更せずに振る舞いを差し替えられるポイントを特定する技法を習得する
3. **特性テスト（Characterization Test）** ── 現在の振る舞いを記録し、リファクタリングの安全網を構築する方法を身につける
4. **Sprout / Wrap パターン** ── 既存コードを変更せず、新機能を安全に追加する技法を習得する
5. **Strangler Fig パターン** ── 大規模レガシーシステムの段階的な近代化戦略を設計できるようになる

---

## 1. レガシーコードの特徴と評価

### 1.1 レガシーコードの定義

Michael Feathers の定義が最も広く受け入れられている:

```
  レガシーコード = テストのないコード

  ┌─────────────────────────────────────────────────────┐
  │  なぜ「テストがない」ことが問題なのか？               │
  │                                                     │
  │  テストがない                                        │
  │    → 変更による影響を検証できない                    │
  │    → 変更するのが怖い                               │
  │    → 変更を避ける                                   │
  │    → コードが腐敗していく                            │
  │    → さらに変更が怖くなる                            │
  │    → ★ 恐怖のスパイラル                             │
  │                                                     │
  │  テストがある                                        │
  │    → 変更による影響を即座に検証できる                │
  │    → 安心して変更できる                              │
  │    → 積極的に改善できる                              │
  │    → コードが健全に保たれる                          │
  │    → ★ 改善のスパイラル                             │
  └─────────────────────────────────────────────────────┘
```

### 1.2 典型的なレガシーコードの兆候

```
  レガシーコードの兆候チェックリスト

  □ テストがない（またはほぼない）
  □ ドキュメントが古いか存在しない
  □ ビルドに15分以上かかる
  □ 1ファイル5000行以上の God Class が存在
  □ グローバル状態（static 変数、シングルトン）が多用されている
  □ new で直接依存オブジェクトを生成している（DI されていない）
  □ 「触ると壊れる」という暗黙の合意がある
  □ 全体像を把握している人がチームにいない
  □ 変更のたびに予想外の箇所でバグが発生する
  □ 本番デプロイが恐怖のイベントである

  ┌─────────────────────────────────────┐
  │  典型的なレガシーコードの構造       │
  │                                     │
  │  [God Class (5000行)]               │
  │     |                               │
  │     +-- static config (グローバル)  │
  │     +-- static dbConn (グローバル)  │
  │     +-- new DBConnection()  (直接) │
  │     +-- new HttpClient()    (直接) │
  │     +-- new EmailSender()   (直接) │
  │                                     │
  │  テスト: なし (またはほぼなし)       │
  │  ドキュメント: 3年前のもの          │
  │  ビルド: 15分                       │
  └─────────────────────────────────────┘
```

### 1.3 変更のリスクマトリクス

全てのレガシーコードを同等に扱う必要はない。変更頻度と複雑度のマトリクスで優先度を判断する。

```
            変更頻度が高い
                 |
   +-------------+-------------+
   |  Zone A:    |  Zone B:    |
   |  低リスク・  |  高リスク・  |  ← 最優先改善
   |  高頻度     |  高頻度     |
   |             |             |
   |  監視のみ   |  テスト追加  |
   |  問題なし   |  リファクタ  |
   +-------------+-------------+
   |  Zone C:    |  Zone D:    |
   |  低リスク・  |  高リスク・  |  ← 触る必要が出るまで放置
   |  低頻度     |  低頻度     |
   |             |             |
   |  放置可     |  次フェーズ  |
   +-------------+-------------+
                 |
            変更頻度が低い
   複雑度が低い ----+---- 複雑度が高い

  ★ Zone B (高リスク・高頻度) から着手するのが最も効率的
```

### 1.4 依存関係の可視化

**コード例1: 依存関係分析スクリプト（Python）**

```python
#!/usr/bin/env python3
"""
レガシーコードの依存関係を可視化するスクリプト。
各モジュールの依存数・被依存数を分析し、
リファクタリングの優先度を判断する。
"""
import ast
import sys
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class ModuleDependency:
    """モジュールの依存関係情報"""
    module: str
    imports: list[str] = field(default_factory=list)
    imported_by: list[str] = field(default_factory=list)

    @property
    def afferent_coupling(self) -> int:
        """求心性結合度: このモジュールに依存しているモジュール数"""
        return len(self.imported_by)

    @property
    def efferent_coupling(self) -> int:
        """遠心性結合度: このモジュールが依存しているモジュール数"""
        return len(self.imports)

    @property
    def instability(self) -> float:
        """不安定度: 0.0 (安定) ~ 1.0 (不安定)"""
        total = self.afferent_coupling + self.efferent_coupling
        if total == 0:
            return 0.0
        return self.efferent_coupling / total


def analyze_dependencies(src_path: str) -> dict[str, ModuleDependency]:
    """ソースコードの依存関係を分析"""
    modules: dict[str, ModuleDependency] = {}

    for py_file in Path(src_path).rglob("*.py"):
        module_name = str(py_file.relative_to(src_path)).replace("/", ".").rstrip(".py")
        tree = ast.parse(py_file.read_text())

        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)

        modules[module_name] = ModuleDependency(
            module=module_name, imports=imports
        )

    # 被依存関係を逆引き
    for mod_name, mod_dep in modules.items():
        for imp in mod_dep.imports:
            if imp in modules:
                modules[imp].imported_by.append(mod_name)

    return modules


def print_dependency_report(modules: dict[str, ModuleDependency]) -> None:
    """依存関係レポートを出力"""
    print("=" * 70)
    print("  依存関係分析レポート")
    print("=" * 70)
    print(f"{'モジュール':<30} {'依存数':>6} {'被依存':>6} {'不安定度':>8}")
    print("-" * 70)

    sorted_modules = sorted(
        modules.values(),
        key=lambda m: m.efferent_coupling,
        reverse=True
    )
    for mod in sorted_modules[:20]:
        stability = "安定" if mod.instability < 0.3 else (
            "中間" if mod.instability < 0.7 else "不安定"
        )
        print(f"{mod.module:<30} {mod.efferent_coupling:>6} "
              f"{mod.afferent_coupling:>6} {mod.instability:>7.2f} ({stability})")

    # 循環依存の検出
    print("\n--- 循環依存の検出 ---")
    for mod_name, mod_dep in modules.items():
        for imp in mod_dep.imports:
            if imp in modules and mod_name in modules[imp].imports:
                print(f"  ⚠ {mod_name} <-> {imp}")
```

```
  変更対象の特定: 影響波及分析

  [OrderProcessor]
       |
       +-- depends on --> [PriceCalculator]
       |                       |
       +-- depends on --> [InventoryChecker]
       |                       |
       +-- depends on --> [DatabaseHelper] ← 静的メソッド (テスト困難)
       |                       |
       +-- depends on --> [EmailSender]    ← 外部サービス (テスト困難)

  安全に変更するための優先順位:
  1. DatabaseHelper, EmailSender の依存を切断
  2. OrderProcessor に特性テストを追加
  3. 変更を加える
  4. テストが通ることを確認
```

---

## 2. Seam（継ぎ目）の発見

### 2.1 Seam とは

Seam とは、Michael Feathers が定義した概念で「コードを編集せずにプログラムの振る舞いを変更できるポイント」のこと。テスト時に依存をテストダブルに差し替えるために使う。

**コード例2: Seam の種類と適用（Python）**

```python
# ────────────────────────────────────────
# Seam がない状態: テスト不可能
# ────────────────────────────────────────
class OrderProcessor:
    """依存を直接生成 → テスト時にDBやメールサーバが必要"""

    def process(self, order):
        # 静的メソッド → テスト時に差し替え不可
        db = DatabaseHelper.get_connection()
        result = db.execute("SELECT stock FROM products ...", order.product_id)

        if result.stock < order.quantity:
            raise InsufficientStockError()

        # 直接生成 → テスト時に差し替え不可
        inventory = InventoryChecker()
        inventory.reserve(order.product_id, order.quantity)

        # 静的メソッド → テスト時にメール送信が実行される
        EmailSender.send(
            to=order.customer_email,
            subject="注文確定",
            body=f"注文 {order.id} が確定しました"
        )


# ────────────────────────────────────────
# Object Seam: コンストラクタインジェクション
# ────────────────────────────────────────
class OrderProcessor:
    """依存を注入 → テスト時にテストダブルに差し替え可能"""

    def __init__(self, db_connection, inventory_checker, email_sender):
        self._db = db_connection              # 注入 → Stub に差し替え可能
        self._inventory = inventory_checker   # 注入 → Mock に差し替え可能
        self._email = email_sender            # 注入 → Fake に差し替え可能

    def process(self, order):
        result = self._db.execute("SELECT stock FROM products ...", order.product_id)

        if result.stock < order.quantity:
            raise InsufficientStockError()

        self._inventory.reserve(order.product_id, order.quantity)
        self._email.send(
            to=order.customer_email,
            subject="注文確定",
            body=f"注文 {order.id} が確定しました"
        )


# テスト: テストダブルを注入
def test_order_process_sends_email():
    fake_db = FakeDatabase(stock=10)
    mock_inventory = Mock()
    mock_email = Mock()

    processor = OrderProcessor(fake_db, mock_inventory, mock_email)
    processor.process(Order(product_id="P001", quantity=2, customer_email="a@b.com"))

    mock_email.send.assert_called_once()
    mock_inventory.reserve.assert_called_once_with("P001", 2)
```

### 2.2 Seam の種類

| Seam の種類 | 仕組み | 適用場面 | 安全性 |
|------------|--------|---------|:------:|
| Object Seam | コンストラクタ/セッターインジェクション | 最も一般的。DI コンテナと相性良い | 高 |
| Preprocessing Seam | マクロ/条件付きコンパイル | C/C++ レガシーコード | 低 |
| Link Seam | リンク時にライブラリを差し替え | バイナリレベルの差し替え | 中 |
| Subclass Seam | Extract & Override | テストクラスでオーバーライド | 中 |

### 2.3 Extract and Override（抽出とオーバーライド）

最も安全に Seam を作る技法の1つ。テスト困難な部分をメソッドに抽出し、テスト用サブクラスでオーバーライドする。

**コード例3: Extract and Override（Python）**

```python
# ────────────────────────────────────────
# Step 1: テスト困難な部分をメソッドに抽出
# ────────────────────────────────────────
class OrderProcessor:
    def process(self, order):
        price = self._calculate_price(order)
        self._save_to_database(order, price)    # 抽出したメソッド
        self._send_notification(order)          # 抽出したメソッド
        return price

    def _calculate_price(self, order):
        """純粋な計算ロジック ── テスト可能"""
        base_price = order.unit_price * order.quantity
        if order.quantity >= 10:
            return int(base_price * 0.9)  # 10個以上で10%割引
        return base_price

    def _save_to_database(self, order, price):
        """DB保存 ── テスト困難な外部依存"""
        db = DatabaseHelper.get_connection()
        db.execute("INSERT INTO orders VALUES (%s, %s)", order.id, price)

    def _send_notification(self, order):
        """メール送信 ── テスト困難な外部依存"""
        EmailSender.send(order.customer_email, "注文確定",
                         f"合計: {order.total}円")


# ────────────────────────────────────────
# Step 2: テスト用サブクラスでオーバーライド
# ────────────────────────────────────────
class TestableOrderProcessor(OrderProcessor):
    """テスト用: DB とメールを差し替え"""

    def __init__(self):
        self.saved_orders: list[tuple] = []
        self.sent_emails: list[str] = []

    def _save_to_database(self, order, price):
        """DB を使わず、保存内容を記録"""
        self.saved_orders.append((order.id, price))

    def _send_notification(self, order):
        """メール送信せず、送信先を記録"""
        self.sent_emails.append(order.customer_email)


# ────────────────────────────────────────
# Step 3: テスト
# ────────────────────────────────────────
import pytest

class TestOrderProcessor:
    def test_process_calculates_correct_price(self):
        """ビジネスロジック（価格計算）のテスト"""
        processor = TestableOrderProcessor()
        order = Order(id="O001", unit_price=1000, quantity=2,
                      customer_email="test@example.com")

        result = processor.process(order)

        assert result == 2000

    def test_process_applies_bulk_discount(self):
        """10個以上で10%割引のテスト"""
        processor = TestableOrderProcessor()
        order = Order(id="O002", unit_price=1000, quantity=10,
                      customer_email="test@example.com")

        result = processor.process(order)

        assert result == 9000  # 10000 * 0.9

    def test_process_saves_to_database(self):
        """DB保存が呼ばれることを確認"""
        processor = TestableOrderProcessor()
        order = Order(id="O003", unit_price=500, quantity=3,
                      customer_email="test@example.com")

        processor.process(order)

        assert len(processor.saved_orders) == 1
        assert processor.saved_orders[0] == ("O003", 1500)

    def test_process_sends_notification(self):
        """メール送信が呼ばれることを確認"""
        processor = TestableOrderProcessor()
        order = Order(id="O004", unit_price=500, quantity=1,
                      customer_email="customer@example.com")

        processor.process(order)

        assert processor.sent_emails == ["customer@example.com"]
```

---

## 3. 特性テスト (Characterization Test)

### 3.1 特性テストとは

特性テストは、現在の振る舞いを「記録」するテスト。「正しい」振る舞いではなく、「実際の」振る舞いをテストする。リファクタリング後も同じ値が返ることを保証するのが目的。

```
  特性テストの考え方

  ┌─────────────────────────────────────────┐
  │  通常のテスト:                          │
  │  「仕様に基づいて期待値を設定」          │
  │  → assert calculate(100, 10) == 110    │
  │  (仕様: 100 + 10% = 110)               │
  │                                         │
  │  特性テスト:                            │
  │  「実際に実行して結果を記録」            │
  │  → assert calculate(100, 10) == 108    │
  │  (現実: なぜか108。バグかも? 仕様かも?)  │
  │                                         │
  │  ★ 特性テストでは「正しいか」は問わない │
  │  ★ 「リファクタリング後も同じか」を保証 │
  └─────────────────────────────────────────┘
```

**コード例4: 特性テストの作成（Python）**

```python
# ────────────────────────────────────────
# 特性テスト: 現在の振る舞いを「記録」する
# ────────────────────────────────────────
class TestLegacyPriceCalculatorCharacterization:
    """
    特性テスト: LegacyPriceCalculator の現在の振る舞いを記録。
    このテストが通る限り、リファクタリングは安全。
    """

    def setup_method(self):
        self.calculator = LegacyPriceCalculator()

    def test_single_item_basic_price(self):
        """単品の基本価格"""
        result = self.calculator.calculate(
            items=[{"price": 100, "qty": 1}]
        )
        assert result == 100

    def test_multiple_items_basic_price(self):
        """複数品の基本価格（割引なし）"""
        result = self.calculator.calculate(
            items=[{"price": 100, "qty": 5}]
        )
        assert result == 500

    def test_bulk_discount_at_10(self):
        """10個以上で割引が適用される"""
        result = self.calculator.calculate(
            items=[{"price": 100, "qty": 10}]
        )
        assert result == 900  # 10%割引? 仕様不明だが記録

    def test_empty_items(self):
        """空リストの場合"""
        result = self.calculator.calculate(items=[])
        assert result == 0

    def test_zero_price(self):
        """価格0の商品"""
        result = self.calculator.calculate(
            items=[{"price": 0, "qty": 5}]
        )
        assert result == 0

    def test_negative_price_boundary(self):
        """負の価格 ── バグの可能性あるが現状の振る舞いを記録"""
        result = self.calculator.calculate(
            items=[{"price": -100, "qty": 1}]
        )
        assert result == -100  # ★ バグかもしれないが、現状の記録が目的

    def test_large_quantity(self):
        """大量注文"""
        result = self.calculator.calculate(
            items=[{"price": 100, "qty": 1000}]
        )
        assert result == 80000  # 20%割引? 仕様不明


# ────────────────────────────────────────
# 特性テストの自動生成（大量ケース）
# ────────────────────────────────────────
import json
import itertools

def generate_characterization_tests(output_path: str = "characterization_tests.json"):
    """
    特性テストを自動生成して JSON に保存。
    レガシーコードの振る舞いを網羅的に記録する。
    """
    calculator = LegacyPriceCalculator()
    test_cases = []

    # 境界値を含む入力の組み合わせを生成
    prices = [0, 1, 50, 100, 500, 999, 1000, 5000, 9999, 10000]
    quantities = [0, 1, 5, 9, 10, 11, 50, 100, 500, 1000]

    for price, qty in itertools.product(prices, quantities):
        try:
            result = calculator.calculate(
                items=[{"price": price, "qty": qty}]
            )
            test_cases.append({
                "input": {"price": price, "qty": qty},
                "expected": result,
                "error": None
            })
        except Exception as e:
            test_cases.append({
                "input": {"price": price, "qty": qty},
                "expected": None,
                "error": str(e)
            })

    with open(output_path, "w") as f:
        json.dump(test_cases, f, indent=2, ensure_ascii=False)

    print(f"生成されたテストケース: {len(test_cases)}件")
    print(f"保存先: {output_path}")
    return test_cases


# ────────────────────────────────────────
# 保存したケースを使ったパラメタライズテスト
# ────────────────────────────────────────
def load_test_cases(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


@pytest.mark.parametrize("case", load_test_cases("characterization_tests.json"))
def test_refactored_matches_legacy(case):
    """リファクタリング後のコードがレガシーと同じ結果を返すことを確認"""
    calculator = RefactoredPriceCalculator()  # 新しい実装

    if case["error"]:
        with pytest.raises(Exception):
            calculator.calculate(items=[case["input"]])
    else:
        result = calculator.calculate(items=[case["input"]])
        assert result == case["expected"], (
            f"入力: {case['input']}, "
            f"期待: {case['expected']}, 実際: {result}"
        )
```

### 3.2 Golden Master テスト

特性テストの発展形で、大量の入出力をファイルに保存し、リファクタリング後の出力と比較する。

**コード例5: Golden Master テスト（Python）**

```python
import hashlib
from pathlib import Path

class GoldenMasterTest:
    """
    Golden Master テスト:
    レガシーシステムの出力をスナップショットとして保存し、
    リファクタリング後も同じ出力が得られることを検証する。
    """
    GOLDEN_DIR = Path("tests/golden_masters")

    def __init__(self, system_under_test):
        self.sut = system_under_test
        self.GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    def capture(self, test_name: str, inputs: list[dict]) -> None:
        """ゴールデンマスターを生成・保存"""
        outputs = []
        for inp in inputs:
            try:
                result = self.sut.process(**inp)
                outputs.append({"input": inp, "output": result, "error": None})
            except Exception as e:
                outputs.append({"input": inp, "output": None, "error": str(e)})

        golden_path = self.GOLDEN_DIR / f"{test_name}.json"
        with open(golden_path, "w") as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False, default=str)

        # チェックサムも保存
        checksum = hashlib.sha256(
            json.dumps(outputs, sort_keys=True).encode()
        ).hexdigest()
        (self.GOLDEN_DIR / f"{test_name}.sha256").write_text(checksum)

        print(f"ゴールデンマスター保存: {golden_path} ({len(outputs)} ケース)")

    def verify(self, test_name: str) -> bool:
        """現在の出力がゴールデンマスターと一致するか検証"""
        golden_path = self.GOLDEN_DIR / f"{test_name}.json"
        with open(golden_path) as f:
            golden = json.load(f)

        mismatches = []
        for case in golden:
            try:
                actual = self.sut.process(**case["input"])
                if actual != case["output"]:
                    mismatches.append({
                        "input": case["input"],
                        "expected": case["output"],
                        "actual": actual,
                    })
            except Exception as e:
                if case["error"] is None or str(e) != case["error"]:
                    mismatches.append({
                        "input": case["input"],
                        "expected_error": case["error"],
                        "actual_error": str(e),
                    })

        if mismatches:
            print(f"不一致: {len(mismatches)} 件")
            for m in mismatches[:5]:
                print(f"  {m}")
            return False

        print(f"OK: {len(golden)} ケース全て一致")
        return True
```

---

## 4. Sprout / Wrap パターン

### 4.1 Sprout Method（芽生えメソッド）

既存コードを変更せず、新しい機能を「芽生え」として新しいテスト済みメソッドに追加する。

**コード例6: Sprout Method（Python）**

```python
# ────────────────────────────────────────
# 状況: 巨大な process() メソッドに「ロイヤルティ割引」を追加したい
# しかし、process() にはテストがなく、構造が複雑で変更が怖い
# ────────────────────────────────────────

# BEFORE: 変更したい巨大メソッド (テストなし)
class OrderProcessor:
    def process(self, order):
        # ... 200行の複雑な処理 (理解が難しい) ...
        total = self._legacy_calculate(order)
        # ... さらに100行の複雑な処理 ...
        return total


# AFTER: 新機能は新しいテスト済みメソッドとして「芽生え」させる
class OrderProcessor:
    def process(self, order):
        # ... 200行の複雑な処理 (変更なし) ...
        total = self._legacy_calculate(order)

        # ★ Sprout: 新機能を独立したメソッドとして追加
        discount = self._calculate_loyalty_discount(order, total)
        total = total - discount

        # ... さらに100行の複雑な処理 (変更なし) ...
        return total

    def _calculate_loyalty_discount(self, order, total: int) -> int:
        """
        新機能: ロイヤルティ割引（テスト付き）

        3年以上の顧客は5%割引、5年以上は10%割引。
        ★ この新メソッドにはテストがある → 安全
        """
        years = order.customer.loyalty_years
        if years >= 5:
            return int(total * 0.10)
        elif years >= 3:
            return int(total * 0.05)
        return 0


# ────────────────────────────────────────
# テスト: 新しい Sprout メソッドのみテスト
# ────────────────────────────────────────
class TestLoyaltyDiscount:
    def setup_method(self):
        self.processor = OrderProcessor()

    def test_no_discount_for_new_customer(self):
        order = create_order(loyalty_years=1)
        assert self.processor._calculate_loyalty_discount(order, 10000) == 0

    def test_5_percent_for_3_year_customer(self):
        order = create_order(loyalty_years=3)
        assert self.processor._calculate_loyalty_discount(order, 10000) == 500

    def test_10_percent_for_5_year_customer(self):
        order = create_order(loyalty_years=5)
        assert self.processor._calculate_loyalty_discount(order, 10000) == 1000

    def test_10_percent_for_veteran_customer(self):
        order = create_order(loyalty_years=10)
        assert self.processor._calculate_loyalty_discount(order, 10000) == 1000
```

### 4.2 Sprout Class（芽生えクラス）

新機能がまとまった責任を持つ場合、メソッドではなくクラスとして追加する。

**コード例7: Sprout Class（Python）**

```python
# 新しい機能を独立したクラスとして追加
class LoyaltyDiscountCalculator:
    """
    ロイヤルティ割引計算 ── Sprout Class
    完全にテスト済みの独立したクラス。
    """
    TIERS = [
        (10, Decimal("0.15")),  # 10年以上: 15%
        (5, Decimal("0.10")),   # 5年以上: 10%
        (3, Decimal("0.05")),   # 3年以上: 5%
    ]

    def calculate(self, customer: Customer, amount: Decimal) -> Decimal:
        """ロイヤルティ割引額を計算"""
        rate = self._get_discount_rate(customer.loyalty_years)
        return (amount * rate).quantize(Decimal("1"))

    def _get_discount_rate(self, years: int) -> Decimal:
        """顧客の忠誠年数に基づく割引率"""
        for min_years, rate in self.TIERS:
            if years >= min_years:
                return rate
        return Decimal("0")


# 既存コードは最小限の変更で統合
class OrderProcessor:
    def __init__(self):
        self._loyalty_calculator = LoyaltyDiscountCalculator()  # Sprout

    def process(self, order):
        # ... 既存の複雑な処理 (変更なし) ...
        total = self._legacy_calculate(order)

        # Sprout Class の呼び出しを1行追加
        discount = self._loyalty_calculator.calculate(order.customer, total)
        total -= discount

        # ... 既存の複雑な処理 (変更なし) ...
        return total
```

### 4.3 Wrap Method（ラップメソッド）

既存メソッドをラップして前後に処理を追加する。既存のメソッド名を維持したまま、内部を新旧に分離する。

**コード例8: Wrap Method（Python）**

```python
# BEFORE: 複雑なレポート生成ロジック (変更したくない)
class ReportGenerator:
    def generate(self, data):
        # ... 複雑なレポート生成ロジック (200行) ...
        return report


# AFTER: 既存メソッドをラップ
class ReportGenerator:
    def generate(self, data):
        """ラッパー: 前後処理を追加"""
        self._log_generation_start(data)           # ← 新: ラップ (前処理)
        self._validate_input(data)                 # ← 新: ラップ (前処理)
        report = self._generate_legacy(data)       # ← 旧: リネーム
        self._record_metrics(report)               # ← 新: ラップ (後処理)
        self._log_generation_complete(report)       # ← 新: ラップ (後処理)
        return report

    def _generate_legacy(self, data):
        """元の複雑なロジック ── 変更なし"""
        # ... 200行のレガシーコード ...
        return report

    def _log_generation_start(self, data):
        """新: レポート生成開始のログ"""
        logger.info(f"レポート生成開始: {data.get('report_type')}")

    def _validate_input(self, data):
        """新: 入力データの事前バリデーション"""
        if not data:
            raise ValueError("データが空です")
        if 'report_type' not in data:
            raise ValueError("report_type が指定されていません")

    def _record_metrics(self, report):
        """新: メトリクス記録"""
        metrics.increment('reports_generated')
        metrics.histogram('report_size', len(str(report)))

    def _log_generation_complete(self, report):
        """新: レポート生成完了のログ"""
        logger.info(f"レポート生成完了: {len(str(report))} 文字")
```

---

## 5. Strangler Fig パターン

### 5.1 概念

Strangler Fig（絞め殺しの木）パターンは、レガシーシステムを段階的に新システムに置き換える手法。Martin Fowler がオーストラリアの絞め殺しの木にちなんで命名した。

```
  Strangler Fig パターンの4フェーズ

  Phase 1: ファサードを配置
  ┌──────┐    ┌─────────────┐    ┌─────────────────┐
  │Client│ -> │ Facade/Proxy│ -> │ Legacy System   │
  └──────┘    └─────────────┘    └─────────────────┘

  Phase 2: 新機能を新システムに実装
  ┌──────┐    ┌─────────┐  ┌──> │ Legacy System   │ (既存機能)
  │Client│ -> │ Facade  │──┤    └─────────────────┘
  └──────┘    └─────────┘  └──> │ New System      │ (新機能)
                                └─────────────────┘

  Phase 3: 既存機能を段階的に移行
  ┌──────┐    ┌─────────┐  ┌──> │ Legacy (残り)   │
  │Client│ -> │ Facade  │──┤    └─────────────────┘
  └──────┘    └─────────┘  └──> │ New System      │ (大部分)
                                └─────────────────┘

  Phase 4: レガシーを完全に置換
  ┌──────┐    ┌─────────┐       ┌─────────────────┐
  │Client│ -> │ Facade  │ ----> │ New System      │ (全機能)
  └──────┘    └─────────┘       └─────────────────┘
                                Legacy は廃止
```

### 5.2 Feature Flag によるルーティング

**コード例9: Strangler Fig の実装（Python）**

```python
from enum import Enum
from typing import Protocol


class FeatureFlag(Enum):
    """Feature Flag の一元管理"""
    NEW_ORDER_CREATION = "new_order_creation"
    NEW_ORDER_RETRIEVAL = "new_order_retrieval"
    NEW_PAYMENT_PROCESSING = "new_payment_processing"
    NEW_NOTIFICATION = "new_notification"


class FeatureFlagService:
    """Feature Flag サービス ── 環境変数・DB・設定ファイルから取得"""

    def __init__(self, config: dict[str, bool]):
        self._config = config

    def is_enabled(self, flag: FeatureFlag) -> bool:
        """指定されたフラグが有効かどうかを返す"""
        return self._config.get(flag.value, False)

    @classmethod
    def from_env(cls) -> "FeatureFlagService":
        """環境変数からフラグを読み込む"""
        import os
        config = {}
        for flag in FeatureFlag:
            config[flag.value] = os.getenv(
                f"FF_{flag.value.upper()}", "false"
            ).lower() == "true"
        return cls(config)


class OrderService(Protocol):
    """注文サービスのインターフェース"""
    def create_order(self, order_data: dict) -> Order: ...
    def get_order(self, order_id: str) -> Order: ...


class OrderFacade:
    """
    Strangler Fig Facade:
    Feature Flag に基づいてレガシーと新システムをルーティング。
    """

    def __init__(self, legacy: OrderService, new: OrderService,
                 flags: FeatureFlagService):
        self._legacy = legacy
        self._new = new
        self._flags = flags

    def create_order(self, order_data: dict) -> Order:
        if self._flags.is_enabled(FeatureFlag.NEW_ORDER_CREATION):
            return self._new.create_order(order_data)
        return self._legacy.create_order(order_data)

    def get_order(self, order_id: str) -> Order:
        if self._flags.is_enabled(FeatureFlag.NEW_ORDER_RETRIEVAL):
            return self._new.get_order(order_id)
        return self._legacy.get_order(order_id)


# ────────────────────────────────────────
# 使用例: 段階的な移行
# ────────────────────────────────────────

# Phase 2: 新機能のみ新システムで処理
flags = FeatureFlagService({
    "new_order_creation": True,     # 新システム
    "new_order_retrieval": False,   # まだレガシー
    "new_payment_processing": False,
    "new_notification": False,
})

facade = OrderFacade(
    legacy=LegacyOrderService(),
    new=NewOrderService(),
    flags=flags
)

# Phase 3: 既存機能も段階的に移行
flags = FeatureFlagService({
    "new_order_creation": True,     # 移行済み
    "new_order_retrieval": True,    # 移行済み ← 新たに有効化
    "new_payment_processing": False,
    "new_notification": False,
})
```

### 5.3 Strangler Fig の安全なロールバック

```python
# ロールバック可能な Facade
class SafeOrderFacade:
    """
    安全な Strangler Fig Facade:
    新システムでエラーが発生した場合、自動的にレガシーにフォールバック。
    """

    def __init__(self, legacy, new, flags, metrics):
        self._legacy = legacy
        self._new = new
        self._flags = flags
        self._metrics = metrics

    def create_order(self, order_data: dict) -> Order:
        if self._flags.is_enabled(FeatureFlag.NEW_ORDER_CREATION):
            try:
                result = self._new.create_order(order_data)
                self._metrics.increment("new_system.success")
                return result
            except Exception as e:
                self._metrics.increment("new_system.fallback")
                logger.warning(
                    f"新システムでエラー、レガシーにフォールバック: {e}"
                )
                # 自動フォールバック
                return self._legacy.create_order(order_data)
        return self._legacy.create_order(order_data)
```

### 5.4 並行実行による検証

**コード例10: Shadow Mode / Dark Launch（Python）**

```python
class ParallelVerificationFacade:
    """
    並行実行 Facade:
    新旧両方のシステムで処理を実行し、結果を比較。
    レスポンスはレガシーを返しつつ、不一致をログに記録。
    """

    def __init__(self, legacy, new, comparator, metrics):
        self._legacy = legacy
        self._new = new
        self._comparator = comparator
        self._metrics = metrics

    def create_order(self, order_data: dict) -> Order:
        # レガシーの結果を取得（これが正式なレスポンス）
        legacy_result = self._legacy.create_order(order_data)

        # 新システムの結果を非同期で取得（Shadow Mode）
        try:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self._new.create_order, order_data)
                new_result = future.result(timeout=5)

            # 結果を比較
            if self._comparator.are_equivalent(legacy_result, new_result):
                self._metrics.increment("parallel.match")
            else:
                self._metrics.increment("parallel.mismatch")
                logger.warning(
                    f"新旧の結果が不一致: "
                    f"legacy={legacy_result}, new={new_result}"
                )
        except Exception as e:
            self._metrics.increment("parallel.new_system_error")
            logger.error(f"新システムでエラー (Shadow Mode): {e}")

        # ★ 常にレガシーの結果を返す
        return legacy_result
```

---

## 6. 段階的な近代化戦略

### 6.1 改善のロードマップ

```
  レガシーコード改善の5段階ロードマップ

  Stage 1 (1-2週間): 可視化
  ├── 依存関係の分析と可視化
  ├── 変更頻度 x 複雑度のホットスポット分析
  ├── テストカバレッジの現状把握
  └── 技術的負債バックログの作成

  Stage 2 (2-4週間): 安全網の構築
  ├── ホットスポットに特性テストを追加
  ├── CI パイプラインの構築
  ├── テストカバレッジのベースライン設定
  └── デプロイの自動化

  Stage 3 (継続的): 段階的リファクタリング
  ├── Seam の発見と依存性注入の導入
  ├── Extract Method / Extract Class で構造改善
  ├── Sprout/Wrap で新機能を安全に追加
  └── ボーイスカウトルールの実践

  Stage 4 (四半期ごと): 大規模改善
  ├── Strangler Fig で大きなモジュールを移行
  ├── アーキテクチャの段階的な改善
  └── フレームワーク/ライブラリの更新

  Stage 5 (年次): 評価と計画
  ├── 技術的負債の残高レビュー
  ├── 改善のROI評価
  └── 次年度の改善計画策定
```

### 6.2 Git を使ったコードの考古学

**コード例11: レガシーコードの考古学スクリプト（Bash）**

```bash
#!/bin/bash
# レガシーコードの考古学: Git 履歴から改善の優先度を分析

echo "=== レガシーコード考古学レポート ==="

# 1. 変更頻度が最も高いファイル（過去6ヶ月）
echo ""
echo "--- 変更頻度 Top 20 (過去6ヶ月) ---"
git log --format=format: --name-only --since="6 months ago" \
  | sort | uniq -c | sort -rn | head -20

# 2. 最も多くの開発者が触ったファイル（知識が分散）
echo ""
echo "--- 最多開発者ファイル Top 10 ---"
git log --format='%aN' --name-only --since="1 year ago" \
  | awk '/^$/{next} /^[^\/]/{author=$0; next} {print author, $0}' \
  | sort -u | awk '{print $NF}' | sort | uniq -c | sort -rn | head -10

# 3. 最近変更されていないが大きなファイル（忘れられたレガシー）
echo ""
echo "--- 大きいが最近変更されていないファイル ---"
find src/ -name "*.py" -exec wc -l {} \; 2>/dev/null \
  | sort -rn | head -10

# 4. TODO / FIXME / HACK の分布
echo ""
echo "--- TODO/FIXME/HACK の数 ---"
grep -rn "TODO\|FIXME\|HACK\|XXX" src/ 2>/dev/null | wc -l

# 5. バグ修正コミットが多いファイル
echo ""
echo "--- バグ修正が多いファイル Top 10 ---"
git log --oneline --grep="fix\|bug\|hotfix" --name-only --since="1 year ago" \
  | grep -v "^[a-f0-9]" | sort | uniq -c | sort -rn | head -10
```

---

## 7. 比較表

### 7.1 レガシーコード改善手法の比較

| 手法 | 適用場面 | リスク | コスト | 効果 |
|------|---------|:------:|:------:|------|
| 特性テスト追加 | リファクタリング前の安全網構築 | 低 | 低 | 回帰バグ防止 |
| Sprout Method | 既存コードへの新機能追加 | 低 | 低 | レガシーへの影響最小化 |
| Sprout Class | まとまった新機能の追加 | 低 | 低-中 | テスト可能な新コード |
| Wrap Method | 既存機能の前後に処理追加 | 低 | 低 | ロギング・メトリクス追加 |
| Extract & Override | テスト困難な依存の切断 | 中 | 中 | テスタビリティ向上 |
| DI の導入 | 依存関係の明示化 | 中 | 中 | 長期的なテスタビリティ |
| Strangler Fig | 大規模システム置換 | 中 | 高 | 根本的な近代化 |
| ビッグバンリライト | 全面書き直し | 極高 | 極高 | 非推奨 |

### 7.2 優先度判断マトリクス

| 優先度 | アクション | 効果 | 目安期間 |
|:------:|-----------|------|---------|
| 最優先 | 変更頻度の高いモジュールに特性テスト追加 | 回帰バグ防止 | 1-2週間 |
| 高 | 依存性注入によるテスタビリティ向上 | テスト追加が容易に | 2-4週間 |
| 中 | 新機能は Sprout/Wrap で追加 | レガシーへの影響最小化 | 継続的 |
| 中 | CI/CD パイプラインの構築 | 品質の自動チェック | 1-2週間 |
| 低 | 段階的な Strangler Fig 移行 | 長期的な技術的負債解消 | 数ヶ月-年 |

---

## 8. アンチパターン

### アンチパターン 1: ビッグバンリライト

```
  BAD: 「全部書き直そう」

  「このレガシーコードはもう限界。1から書き直す。」
    → 数ヶ月〜数年の開発期間
    → その間、旧システムも並行保守が必要
    → 新チームは旧システムの暗黙知を持たない
    → 完成時にはビジネス要件が変わっている
    → 旧システムの「奇妙な仕様」が実はビジネス上の理由がある
    → Joel Spolsky: 「ソフトウェアでやってはいけない最悪のこと」

  歴史的教訓:
  - Netscape 6: 全面リライトに3年、市場シェアを失った
  - Borland dBase → Quattro Pro: リライト中に市場が変化

  GOOD: 段階的な移行

  Phase 1: ファサード配置 (1週間)
  Phase 2: 新機能は新システムに (継続的)
  Phase 3: 既存機能を段階的に移行 (機能単位で数週間ずつ)
  Phase 4: レガシー廃止

  ★ 各段階でリリース可能
  ★ ロールバック可能
  ★ 新旧の並行運用で安全性確保
  ★ ビジネス価値を継続的に提供
```

### アンチパターン 2: テストなしのリファクタリング

```
  BAD: テストなしで構造を変更

  1. 「このコード汚いからリファクタリングしよう」
  2. テストなしで構造を変更
  3. 「見た目はきれいになった」
  4. 1週間後: 本番で回帰バグ発生
  5. 修正に追われ、さらにコードが複雑化
  6. 「リファクタリングは危険」という誤った教訓

  GOOD: テストファーストのリファクタリング

  1. まず特性テストを書いて現状の振る舞いを記録
  2. テストが通ることを確認 (GREEN)
  3. 小さなステップでリファクタリング
  4. 各ステップ後にテスト実行 (GREEN を維持)
  5. テストが通り続けることを確認
  6. コミット
  7. 次のステップへ
```

### アンチパターン 3: 全てを一度にモダン化しようとする

```
  BAD: 「フレームワークも、ライブラリも、アーキテクチャも全部更新する」

  Sprint 1: React 16 → 18, Express → Fastify,
            MongoDB → PostgreSQL, モノリス → マイクロサービス
  → 全てが壊れる
  → デバッグが不可能（何が原因か特定できない）
  → チームが疲弊

  GOOD: 1つずつ段階的に

  Sprint N:   React 16 → 18 (UI層のみ)
  Sprint N+1: テストカバレッジを60% → 80%
  Sprint N+2: Express → Fastify (API層のみ)
  Sprint N+3: モノリスの一部をサービスとして切り出し
  ...

  ★ 各ステップで「リリース可能」を維持
  ★ 問題が出ても原因が特定しやすい
```

### アンチパターン 4: レガシーコードの「暗黙知」を無視する

```
  BAD: 「この条件分岐は意味がないから削除しよう」

  if customer.region == "EU" and order.total > 150:
      order.add_customs_declaration()  # なぜ 150? なぜ EU のみ?

  → 実はEU関税規則: 150ユーロ以上の輸入に関税申告が必要
  → 削除すると法令違反に

  GOOD: 特性テストで振る舞いを保存してから変更

  1. まず特性テストで現在の振る舞いを記録
  2. 「なぜこうなっているか」を調査（Git blame, 関係者ヒアリング）
  3. ビジネス上の理由があれば、コメントで理由を記録
  4. 不要と判断できたら、テストを更新してから削除

  # リファクタリング後:
  if order.requires_customs_declaration():
      # EU関税規則: 150ユーロ超の輸入には申告が必要
      # See: https://ec.europa.eu/taxation_customs/...
      order.add_customs_declaration()
```

---

## 9. 演習問題

### 演習1（基本）: Seam の発見

以下のコードにテストを追加するために、Seam を作成せよ。

```python
class NotificationService:
    def send_alert(self, user_id: str, message: str) -> bool:
        # DB から直接ユーザー情報を取得
        import sqlite3
        conn = sqlite3.connect("/var/db/production.db")
        cursor = conn.execute(
            "SELECT email, phone FROM users WHERE id = ?", (user_id,)
        )
        user = cursor.fetchone()
        if not user:
            return False

        # メール送信
        import smtplib
        smtp = smtplib.SMTP("mail.production.com", 587)
        smtp.send_message(create_email(user[0], message))

        # SMS送信
        import requests
        requests.post("https://api.sms-provider.com/send",
                      json={"phone": user[1], "text": message})

        return True
```

**期待される回答**: (1) Extract & Override: DB取得、メール送信、SMS送信をそれぞれ protected メソッドに抽出, (2) テスト用サブクラスでオーバーライド, または (3) コンストラクタインジェクションで `UserRepository`, `EmailSender`, `SmsSender` を注入可能にする。

---

### 演習2（応用）: 特性テストの作成

以下のレガシー関数に対して、特性テストを少なくとも10ケース作成せよ。

```python
def calculate_shipping(weight, destination, is_member, order_total):
    """配送料金計算（レガシー: 仕様書なし）"""
    base = weight * 100
    if destination == "overseas":
        base *= 3
    if is_member:
        base *= 0.8
    if order_total > 10000:
        base = 0
    if weight > 30:
        base += 2000
    return int(base)
```

**期待される回答**: 通常国内配送、海外配送、会員割引、1万円以上無料、重量超過サーチャージ、組み合わせ条件（海外+会員+重量超過など）のケースを網羅的にテスト。

---

### 演習3（上級）: Strangler Fig 移行計画

以下の状況で、6ヶ月間の Strangler Fig 移行計画を立案せよ。

```
レガシーシステムの状況:
- PHP 5.6 のモノリシック Web アプリケーション
- MySQL 5.5 データベース
- テストなし（カバレッジ 0%）
- 月間 PV: 100万
- 主要機能: ユーザー管理、商品カタログ、注文処理、決済、レポート
- 開発チーム: 5名
- 1日のデプロイ: 0回（月次手動デプロイ）
```

**期待される回答（概要）**:

```
Month 1: 可視化と安全網
  - 依存関係分析
  - CI/CD パイプライン構築
  - ホットスポットに特性テスト追加
  - Feature Flag 基盤の構築

Month 2: ファサード配置
  - API ゲートウェイ（Nginx reverse proxy）の配置
  - 新 API サーバー（Python/FastAPI）のセットアップ
  - 認証トークンの共有基盤

Month 3-4: 段階的移行（優先度順）
  - ユーザー管理 API を新システムに移行
  - 商品カタログ API を新システムに移行
  - Shadow Mode で結果を比較

Month 5-6: 核心機能の移行
  - 注文処理を新システムに移行
  - 決済を新システムに移行
  - レポートは最後（変更頻度が低い）

★ 各月で「リリース可能」を維持
★ 問題があれば Feature Flag でロールバック
```

---

## 10. FAQ

### Q1. レガシーコードのどこから手をつけるべきか？

**A.** 「変更頻度が高く、バグが多い箇所」から着手する。以下の手順で科学的にアプローチ:

1. **ホットスポット分析**: `git log` で変更頻度を分析し、`radon` で複雑度を測定。変更頻度 x 複雑度 のスコアが高いファイルが最優先。
2. **バグ追跡**: バグチケットが多く紐づくモジュールを特定する。
3. **チーム知識**: 「触るのが怖い」とチームメンバーが感じるモジュールをリストアップ。
4. **全体を均一に改善しようとしない** ── ホットスポットに集中投資する。

### Q2. テストがないコードに安全にテストを追加するには？

**A.** 段階的なアプローチ:

1. **特性テスト** で現在の振る舞いを記録する（正しいかは問わない）
2. **Seam** を見つけて依存を切断する（Extract & Override が最も安全）
3. 切断した依存を **テストダブル** に差し替えてユニットテストを書く
4. 十分なテストが揃ったら **リファクタリング** を開始する

最初から「正しい」テストを書く必要はない。現状の振る舞いの記録が最優先。

### Q3. レガシーコードの改善をチームに説得するには？

**A.** ビジネス指標で語る。技術的な話は避け、以下のように伝える:

- 「このモジュールの変更に平均3日かかっており、年間60日の工数がかかっている」
- 「過去6ヶ月で本番障害が5件発生し、顧客影響があった」
- 「新メンバーのオンボーディングに2週間余計にかかっている」
- 「改善に300万円投資すれば、翌年から年500万円のコスト削減になる」

技術的負債の利子を定量化し、改善によるROI（投資対効果）を示す。詳細は [技術的負債](./03-technical-debt.md) を参照。

### Q4. Strangler Fig パターンで新旧のデータ整合性をどう保つか？

**A.** 以下の戦略を組み合わせる:

1. **単一データベース**: 移行初期は新旧で同じDBを共有
2. **Change Data Capture (CDC)**: 旧システムのDB変更を新システムに同期
3. **Event Sourcing**: イベントを共通バスに発行し、新旧で消費
4. **Shadow Mode**: 新システムの結果をログに記録し、レガシーとの差分を分析

### Q5. レガシーコードの改善にどれくらいの時間を割くべきか？

**A.** Martin Fowler と Kent Beck の推奨:

- **日常**: ボーイスカウトルール（触ったファイルを少し改善: 工数の5-10%）
- **スプリント**: 20% ルール（各スプリントの20%を改善に充てる）
- **四半期**: 技術的負債スプリント（1スプリント分を集中改善に充てる）

重要なのは「改善を日常に組み込む」こと。改善を特別なイベントにすると、ビジネス要求に押し出されて実施されない。

### Q6. Extract & Override と DI のどちらを選ぶべきか？

**A.** 状況による:

| 観点 | Extract & Override | DI (依存性注入) |
|------|:-----------------:|:--------------:|
| 変更の少なさ | 少ない | やや多い |
| 長期的な設計 | 一時的な解決策 | 恒久的な改善 |
| テスタビリティ | テスト用サブクラスが必要 | テストダブルを直接注入 |
| 初期コスト | 低い | 中程度 |
| 推奨場面 | まずテストを追加したい段階 | 本格的にリファクタリングする段階 |

一般的な進め方: まず Extract & Override でテストを追加し、その後 DI に移行する。

---

## 11. まとめ

| 項目 | ポイント |
|------|---------|
| レガシーコードの定義 | テストのないコード（Michael Feathers） |
| 最初のアクション | 特性テストで現在の振る舞いを記録 |
| Seam の発見 | 依存性注入、Extract & Override でテスタビリティを確保 |
| Sprout / Wrap | 既存コードを変更せず新機能を安全に追加 |
| Strangler Fig | 大規模システムの段階的な置換 |
| ビッグバンリライトの回避 | 段階的な移行で各ステップをリリース可能に |
| 優先度の判断 | 変更頻度 x 複雑度のホットスポットから着手 |
| データ整合性 | Shadow Mode + 並行実行で新旧の結果を比較検証 |
| 文化の構築 | ボーイスカウトルール + 20%ルール + 四半期集中改善 |

| 技法 | 安全性 | コスト | 効果 |
|------|:------:|:------:|------|
| 特性テスト | 高 | 低 | リファクタリングの前提条件 |
| Extract & Override | 中 | 低 | 最も安全な Seam 作成法 |
| Sprout Method | 高 | 低 | 新機能の安全な追加 |
| Wrap Method | 高 | 低 | 前後処理の追加 |
| DI の導入 | 中 | 中 | 長期的なテスタビリティ |
| Strangler Fig | 中 | 高 | 根本的な近代化 |
| ビッグバンリライト | 極高 | 極高 | 非推奨 |

---

## 次に読むべきガイド

- [技術的負債](./03-technical-debt.md) ── 負債の分類・可視化・返済戦略
- [継続的改善](./04-continuous-improvement.md) ── CI/CD による品質の継続的向上
- [テスト原則](../01-practices/04-testing-principles.md) ── テスト設計の基礎（AAA パターン、テストダブル）
- [コードスメル](./00-code-smells.md) ── レガシーコードに潜むスメルの検出
- [リファクタリング技法](./01-refactoring-techniques.md) ── Extract Method、Move Method 等の具体的手法
- [デザインパターン概要](../../design-patterns-guide/docs/00-creational/) ── Facade, Strategy 等のパターン
- [システム設計の基礎](../../system-design-guide/docs/00-fundamentals/) ── アーキテクチャレベルの近代化

---

## 参考文献

1. **Michael Feathers** 『Working Effectively with Legacy Code』 Prentice Hall, 2004 ── レガシーコード改善の原典。Seam、Extract & Override、Characterization Test、Sprout/Wrap パターンの全てがここに記述されている。
2. **Martin Fowler** 『Refactoring: Improving the Design of Existing Code』 Addison-Wesley, 2018 (2nd Ed.) ── リファクタリングカタログ。Extract Method、Move Method 等の基本技法。Strangler Fig パターンの命名者。
3. **Marianne Bellotti** 『Kill It with Fire: Manage Aging Computer Systems (and Future Proof Modern Ones)』 No Starch Press, 2021 ── レガシーシステムの近代化戦略を組織論の観点から論じた良書。技術的な手法だけでなく、チームのモチベーションや組織文化の変革についても深く言及。
4. **Sam Newman** 『Building Microservices』 O'Reilly, 2021 (2nd Ed.) ── モノリスからマイクロサービスへの段階的な移行戦略。Strangler Fig パターンの実践例が豊富。
5. **Adam Tornhill** 『Your Code as a Crime Scene: Use Forensic Techniques to Arrest Defects, Bottlenecks, and Bad Design in Your Programs』 Pragmatic Bookshelf, 2015 ── Git 履歴を使ったコードの「犯罪現場調査」。ホットスポット分析、知識マップ、組織分析の手法を実践的に解説。
