# レガシーコード

> レガシーコードとは「テストのないコード」である（Michael Feathers）。何年も保守され、全体像を誰も把握していないコードベースと向き合い、安全に変更を加えるための体系的な技法を、依存性の切断・特性テスト・Strangler Fig パターンを通じて解説する

## この章で学ぶこと

1. **レガシーコードの理解** — コードの考古学、依存関係の可視化、変更の影響範囲の特定
2. **安全な変更技法** — Seam（継ぎ目）の発見、Sprout/Wrap パターン、特性テスト
3. **段階的な近代化戦略** — Strangler Fig パターン、段階的リファクタリング、マイクロサービス分離

---

## 1. レガシーコードの特徴

### 1.1 典型的な状態

```
レガシーコードの兆候

  +-------------------+
  | God Class         |  ← 1ファイル 5000行
  | - 全てを知っている  |
  | - 全てに依存       |
  +--------+----------+
           |
  +--------v----------+
  | グローバル状態      |  ← シングルトン、静的変数
  | static config     |
  | static dbConn     |
  +--------+----------+
           |
  +--------v----------+
  | 隠れた依存         |  ← new で直接生成
  | new DBConnection() |
  | new HttpClient()   |
  +-------------------+

  テスト: なし (またはほぼなし)
  ドキュメント: 古いか存在しない
  ビルド: 15分以上かかる
```

### 1.2 変更のリスクマトリクス

```
            変更頻度が高い
                 |
   +-------------+-------------+
   |  要注意ゾーン |  最優先改善  |  ← テスト追加・リファクタリング
   | (低リスク・   | (高リスク・  |
   |  高頻度)     |  高頻度)    |
   +-------------+-------------+
   |  放置可能    |  次フェーズ  |  ← 触る必要が出るまで放置
   | (低リスク・   | (高リスク・  |
   |  低頻度)     |  低頻度)    |
   +-------------+-------------+
                 |
            変更頻度が低い
   複雑度が低い ----+---- 複雑度が高い
```

### 1.3 依存関係の可視化

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

```python
# Seam = プログラムの振る舞いを変更できるポイント（コードを変更せずに）

# BAD: Seam がない（直接 new / 静的呼び出し）
class OrderProcessor:
    def process(self, order):
        db = DatabaseHelper.get_connection()      # 静的メソッド → テスト不可
        inventory = InventoryChecker()             # 直接生成 → テスト不可
        EmailSender.send(order.customer_email)     # 静的メソッド → テスト不可

# GOOD: コンストラクタインジェクションで Seam を作る
class OrderProcessor:
    def __init__(self, db_connection, inventory_checker, email_sender):
        self._db = db_connection              # 注入 → テスト時に差し替え可能
        self._inventory = inventory_checker   # 注入 → Stub に差し替え可能
        self._email = email_sender            # 注入 → Mock に差し替え可能

    def process(self, order):
        self._db.execute(...)
        self._inventory.check(order.items)
        self._email.send(order.customer_email)
```

### 2.2 Extract and Override（抽出とオーバーライド）

```python
# Step 1: テスト困難な部分をメソッドに抽出
class OrderProcessor:
    def process(self, order):
        price = self._calculate_price(order)
        self._save_to_database(order, price)    # 抽出したメソッド
        self._send_notification(order)          # 抽出したメソッド
        return price

    def _save_to_database(self, order, price):
        db = DatabaseHelper.get_connection()
        db.execute("INSERT INTO orders ...", order, price)

    def _send_notification(self, order):
        EmailSender.send(order.customer_email, "注文確定")

# Step 2: テスト用サブクラスでオーバーライド
class TestableOrderProcessor(OrderProcessor):
    def __init__(self):
        self.saved_orders = []
        self.sent_emails = []

    def _save_to_database(self, order, price):
        self.saved_orders.append((order, price))    # DB を使わない

    def _send_notification(self, order):
        self.sent_emails.append(order.customer_email)  # メール送信しない

# Step 3: テスト
def test_order_processing():
    processor = TestableOrderProcessor()
    order = Order(items=[Item(price=1000, qty=2)])

    result = processor.process(order)

    assert result == 2000
    assert len(processor.saved_orders) == 1
    assert len(processor.sent_emails) == 1
```

---

## 3. 特性テスト (Characterization Test)

```python
# 特性テスト: 現在の振る舞いを「記録」するテスト
# 正しい振る舞いではなく、実際の振る舞いをテストする

def test_characterization_price_calculation():
    """現在の料金計算の振る舞いを記録"""
    calculator = LegacyPriceCalculator()

    # 既知の入力に対する現在の出力を記録
    assert calculator.calculate(items=[{"price": 100, "qty": 1}]) == 100
    assert calculator.calculate(items=[{"price": 100, "qty": 10}]) == 900   # 10%割引?
    assert calculator.calculate(items=[{"price": 0, "qty": 5}]) == 0
    assert calculator.calculate(items=[]) == 0

    # 境界値ケース
    assert calculator.calculate(items=[{"price": -100, "qty": 1}]) == -100  # バグ? 仕様?

    # ★ この値が「正しいか」は問わない
    # ★ リファクタリング後も同じ値が返ることを保証するのが目的


# 特性テストの自動生成（大量ケース）
import json

def generate_characterization_tests():
    """特性テストを自動生成して JSON に保存"""
    calculator = LegacyPriceCalculator()
    test_cases = []

    for price in [0, 1, 100, 999, 1000, 9999, 10000]:
        for qty in [0, 1, 5, 10, 50, 100]:
            result = calculator.calculate(items=[{"price": price, "qty": qty}])
            test_cases.append({
                "input": {"price": price, "qty": qty},
                "expected": result,
            })

    with open("characterization_tests.json", "w") as f:
        json.dump(test_cases, f, indent=2)

# 保存したケースを使ったパラメタライズテスト
@pytest.mark.parametrize("case", load_test_cases("characterization_tests.json"))
def test_price_calculation_matches_legacy(case):
    calculator = RefactoredPriceCalculator()
    result = calculator.calculate(items=[case["input"]])
    assert result == case["expected"]
```

---

## 4. Sprout / Wrap パターン

### 4.1 Sprout Method（芽生えメソッド）

```python
# 既存コードを変更せず、新しい機能を「芽生え」として追加

# BEFORE: 変更したい巨大メソッド
class OrderProcessor:
    def process(self, order):
        # ... 200行の複雑な処理 ...
        total = self._legacy_calculate(order)
        # ... さらに100行 ...
        return total

# AFTER: 新機能は新しいテスト済みメソッドとして追加
class OrderProcessor:
    def process(self, order):
        # ... 200行の複雑な処理 (変更なし) ...
        total = self._legacy_calculate(order)
        discount = self._calculate_loyalty_discount(order, total)  # ← 新メソッド
        total = total - discount
        # ... さらに100行 (変更なし) ...
        return total

    def _calculate_loyalty_discount(self, order, total):
        """新機能: ロイヤルティ割引（テスト付き）"""
        if order.customer.loyalty_years >= 3:
            return int(total * 0.05)
        return 0
```

### 4.2 Wrap Method（ラップメソッド）

```python
# 既存メソッドをラップして前後に処理を追加

# BEFORE
class ReportGenerator:
    def generate(self, data):
        # ... 複雑なレポート生成ロジック (変更したくない) ...
        return report

# AFTER: 既存メソッドをラップ
class ReportGenerator:
    def generate(self, data):
        self._log_generation_start(data)          # ← 新: ラップ (前処理)
        report = self._generate_legacy(data)       # ← 旧: リネーム
        self._log_generation_complete(report)      # ← 新: ラップ (後処理)
        return report

    def _generate_legacy(self, data):
        # ... 元の複雑なロジック (変更なし) ...
        return report
```

---

## 5. Strangler Fig パターン

```
Strangler Fig (絞め殺しの木) パターン

Phase 1: ファサードを配置
  Client --> [Facade/Proxy] --> [Legacy System]

Phase 2: 新機能を新システムに実装
  Client --> [Facade] --+---> [Legacy System] (既存機能)
                        |
                        +---> [New System]    (新機能)

Phase 3: 既存機能を段階的に移行
  Client --> [Facade] --+---> [Legacy System] (残りの機能)
                        |
                        +---> [New System]    (移行済み + 新機能)

Phase 4: レガシーを完全に置換
  Client --> [Facade] ------> [New System]    (全機能)
             Legacy System は廃止
```

```python
# Strangler Fig の実装例: Feature Flag によるルーティング
class OrderFacade:
    def __init__(self, legacy_service, new_service, feature_flags):
        self._legacy = legacy_service
        self._new = new_service
        self._flags = feature_flags

    def create_order(self, order_data):
        if self._flags.is_enabled('new_order_creation'):
            return self._new.create_order(order_data)
        return self._legacy.create_order(order_data)

    def get_order(self, order_id):
        if self._flags.is_enabled('new_order_retrieval'):
            return self._new.get_order(order_id)
        return self._legacy.get_order(order_id)
```

---

## 6. 比較表

| 手法 | 適用場面 | リスク | コスト |
|------|---------|--------|-------|
| 特性テスト追加 | リファクタリング前の安全網構築 | 低 | 低 |
| Sprout Method | 新機能追加 | 低 | 低 |
| Wrap Method | 既存機能の前後に処理追加 | 低 | 低 |
| Extract & Override | テスト困難な依存の切断 | 中 | 中 |
| Strangler Fig | 大規模システム置換 | 中 | 高 |
| ビッグバンリライト | 全面書き直し | 極高 | 極高 |

| 優先度 | アクション | 効果 |
|--------|-----------|------|
| 最優先 | 変更頻度の高いモジュールに特性テスト追加 | 回帰バグ防止 |
| 高 | 依存性注入によるテスタビリティ向上 | テスト追加が容易に |
| 中 | 新機能は Sprout/Wrap で追加 | レガシーコードへの影響最小化 |
| 低 | 段階的な Strangler Fig 移行 | 長期的な技術的負債解消 |

---

## 7. アンチパターン

### アンチパターン 1: ビッグバンリライト

```
BAD: 「全部書き直そう」
  - 数ヶ月〜数年の開発期間
  - その間、旧システムも並行保守が必要
  - 完成時にはビジネス要件が変わっている
  - 旧システムの暗黙知が失われる
  - Joel Spolsky: 「ソフトウェアでやってはいけない最悪のこと」

GOOD: 段階的な移行
  - Strangler Fig で機能単位で移行
  - 各段階でリリース可能
  - ロールバック可能
  - 新旧の並行運用で安全性確保
```

### アンチパターン 2: テストなしのリファクタリング

```
BAD:
  1. 「このコード汚いからリファクタリングしよう」
  2. テストなしで構造を変更
  3. 本番で回帰バグ発生
  4. 修正に追われる

GOOD:
  1. まず特性テストを書いて現状の振る舞いを記録
  2. テストが通ることを確認
  3. 小さなステップでリファクタリング
  4. 各ステップ後にテスト実行
  5. テストが通り続けることを確認
```

---

## 8. FAQ

### Q1. レガシーコードのどこから手をつけるべきか？

**A.** 「変更頻度が高く、バグが多い箇所」から着手する。Git の履歴から変更頻度を分析し（`git log --format='%H' --follow -- [file] | wc -l`）、バグチケットが多く紐づくモジュールを特定する。全体を均一に改善しようとせず、ホットスポットに集中投資する。

### Q2. テストがないコードに安全にテストを追加するには？

**A.** (1) まず特性テストで現在の振る舞いを記録する。(2) Seam を見つけて依存を切断する（Extract & Override が最も安全）。(3) 切断した依存をテストダブルに差し替えてユニットテストを書く。(4) 十分なテストが揃ったらリファクタリングを開始する。最初から「正しい」テストを書く必要はない。現状の振る舞いの記録が最優先。

### Q3. レガシーコードの改善をチームに説得するには？

**A.** ビジネス指標で語る。「このモジュールの変更に平均3日かかっており、年間60日の工数がかかっている」「過去6ヶ月で本番障害が5件発生し、顧客影響があった」など。技術的負債の利子を定量化し、改善によるROI（投資対効果）を示す。改善をスプリントの20%ルールとして組み込むなど、小さく始めることも有効。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| レガシーコードの定義 | テストのないコード（Michael Feathers） |
| 最初のアクション | 特性テストで現在の振る舞いを記録 |
| Seam の発見 | 依存性注入、Extract & Override でテスタビリティを確保 |
| Sprout / Wrap | 既存コードを変更せず新機能を安全に追加 |
| Strangler Fig | 大規模システムの段階的な置換 |
| ビッグバンリライトの回避 | 段階的な移行で各ステップをリリース可能に |
| 優先度の判断 | 変更頻度 x 複雑度のホットスポットから着手 |

---

## 次に読むべきガイド

- [技術的負債](./03-technical-debt.md) — 負債の分類・可視化・返済戦略
- [継続的改善](./04-continuous-improvement.md) — CI/CD による品質の継続的向上
- [テスト原則](../01-practices/04-testing-principles.md) — テスト設計の基礎

---

## 参考文献

1. **Working Effectively with Legacy Code** — Michael Feathers (Prentice Hall, 2004) — レガシーコード改善の原典
2. **Refactoring** — Martin Fowler (Addison-Wesley, 2018, 2nd Ed.) — リファクタリングカタログ
3. **Kill It with Fire: Manage Aging Computer Systems** — Marianne Bellotti (No Starch Press, 2021) — レガシーシステムの近代化戦略
