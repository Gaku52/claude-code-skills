# リファクタリング技法 ── Extract Method、Move、その他の基本技法

> リファクタリングとは、外部から見たプログラムの振る舞いを変えずに、内部構造を改善する規律あるプロセスである。小さなステップを積み重ね、テストで安全性を確認しながら進める。Martin Fowler は『Refactoring』第2版で60以上の技法を体系化したが、本章では実務で最も頻繁に使われる主要技法を、安全な適用手順・コード例・IDE支援・判断基準とともに深掘りする。

---

## 前提知識

| 前提 | 参照先 |
|------|--------|
| コードスメルの分類 | [00-code-smells.md](./00-code-smells.md) |
| テストの基礎（AAA パターン、テストダブル） | [01-practices/04-testing-principles.md](../01-practices/04-testing-principles.md) |
| クリーンコードの基本原則 | [00-principles/](../00-principles/) |
| 命名と関数設計 | [01-practices/01-naming.md](../01-practices/01-naming.md), [01-practices/02-functions.md](../01-practices/02-functions.md) |

---

## この章で学ぶこと

1. **リファクタリングの5つの鉄則** ── 安全なリファクタリングの前提条件と進め方を理解する
2. **主要な10+の技法** ── Extract Method、Move Method、Rename、Replace Conditional 等を手順付きで習得する
3. **技法の選択基準** ── コードスメルから適切な技法を選ぶ判断フレームワークを身につける
4. **IDE によるリファクタリング支援** ── IntelliJ IDEA、VS Code、PyCharm の自動リファクタリング機能を使いこなす
5. **安全なリファクタリングワークフロー** ── テスト→変更→テスト→コミットのサイクルを実践できるようになる

---

## 1. リファクタリングの基本原則

### 1.1 5つの鉄則

```
+-------------------------------------------------------------------+
|  リファクタリングの5つの鉄則                                        |
|  ───────────────────────────────────────────                       |
|  1. 振る舞いを変えない（外部仕様は不変）                            |
|  2. テストを先に書く/確認する（安全網の確保）                       |
|  3. 小さなステップで進める（1回の変更は1種類の改善のみ）            |
|  4. 頻繁にコミットする（いつでも直前の状態に戻れるように）          |
|  5. リファクタリングと機能追加を同時にしない（2つの帽子の原則）     |
+-------------------------------------------------------------------+

Martin Fowler の「2つの帽子」比喩:
- 帽子A: 機能追加 → テストを追加し、新しい振る舞いを実装
- 帽子B: リファクタリング → 構造を改善するが、テストは変更しない
- ★ 同時に両方の帽子をかぶってはいけない
```

### 1.2 リファクタリングの安全なサイクル

```
  リファクタリングのマイクロサイクル

  ┌──────────────┐
  │ 1. テスト実行 │     全て通過していることを確認
  │   (GREEN)    │     → 通過しなければ先にバグ修正
  └──────┬───────┘
         v
  ┌──────────────┐
  │ 2. 小さな変更 │     1つの技法を1ステップだけ適用
  │   (1 step)   │     例: メソッド名の変更のみ
  └──────┬───────┘
         v
  ┌──────────────┐      失敗 → 変更を取り消す (git checkout)
  │ 3. テスト実行 │──────→ 何が壊れたか確認 → 修正して再試行
  │              │
  └──────┬───────┘
    成功  v
  ┌──────────────┐
  │ 4. コミット   │     「リファクタリング: XXXを改善」
  └──────┬───────┘     → いつでもこの時点に戻れる
         v
  ┌──────────────┐
  │ 5. 次の変更へ │──→ ステップ1に戻る
  └──────────────┘

  ★ 1サイクルの所要時間: 2-10分が理想
  ★ サイクルが長くなるほどリスクが高まる
```

### 1.3 リファクタリングのタイミング

```
  いつリファクタリングすべきか？

  ┌─────────────────────────────────────────────────┐
  │  Rule of Three (3回ルール)                       │
  │  ─────────────────────                          │
  │  1回目: そのまま書く                             │
  │  2回目: 重複に気づくが、まだ我慢する             │
  │  3回目: リファクタリングする                      │
  └─────────────────────────────────────────────────┘

  効果的なタイミング:
  ┌───────────────────┬───────────────────────────────┐
  │ タイミング         │ 理由                          │
  ├───────────────────┼───────────────────────────────┤
  │ 機能追加の前       │ 変更しやすい構造に整える       │
  │ バグ修正の後       │ 根本原因の設計問題を解消       │
  │ コードレビュー時   │ レビュー指摘への対応として     │
  │ 理解のため         │ コードを読む過程で構造を改善   │
  └───────────────────┴───────────────────────────────┘

  リファクタリングすべきでないタイミング:
  - デッドライン直前
  - テストが不十分な状態
  - 大規模な機能開発の最中
  - パフォーマンスが問題になっている場合（まず計測）
```

---

## 2. 主要なリファクタリング技法

### 2.1 Extract Method（メソッドの抽出）

**用途**: Long Method を分割し、各部分に意図を表す名前を与える。最も頻繁に使用される基本技法。

**安全な手順**:
1. 抽出したいコードブロックを特定する
2. 新しいメソッドを作成し、意図を表す名前を付ける
3. 元のコードを新しいメソッドの呼び出しに置き換える
4. 必要な変数をパラメータとして渡す
5. テストを実行して振る舞いが変わっていないことを確認する

**コード例1: Extract Method（Python）**

```python
# Before: 長いメソッドに複数の関心が混在
class Order:
    def print_invoice(self):
        print("========== 請求書 ==========")
        print(f"顧客: {self.customer.name}")
        print(f"日付: {self.date}")
        print()

        # 明細の出力
        total = 0
        for item in self.items:
            line_total = item.price * item.quantity
            total += line_total
            print(f"  {item.name}: {item.price} x {item.quantity} = {line_total}")

        # 合計の計算と出力
        tax = total * 0.10
        grand_total = total + tax
        print()
        print(f"小計: {total}")
        print(f"消費税(10%): {tax}")
        print(f"合計: {grand_total}")
        print("============================")


# After: 意図ごとにメソッドを抽出
class Order:
    def print_invoice(self):
        """請求書の出力 ── 高レベルの流れが一目で分かる"""
        self._print_header()
        self._print_line_items()
        self._print_totals()

    def _print_header(self):
        """ヘッダー部分の出力"""
        print("========== 請求書 ==========")
        print(f"顧客: {self.customer.name}")
        print(f"日付: {self.date}")
        print()

    def _print_line_items(self):
        """明細の出力"""
        for item in self.items:
            line_total = item.price * item.quantity
            print(f"  {item.name}: {item.price} x {item.quantity} = {line_total}")

    def _print_totals(self):
        """合計部分の出力"""
        subtotal = self.calculate_subtotal()
        tax = subtotal * Decimal("0.10")
        print()
        print(f"小計: {subtotal}")
        print(f"消費税(10%): {tax}")
        print(f"合計: {subtotal + tax}")
        print("============================")

    def calculate_subtotal(self) -> Decimal:
        """小計の計算 ── 他のメソッドからも再利用可能"""
        return sum(item.price * item.quantity for item in self.items)
```

**コード例2: Extract Method ── 変数の取り扱い（Python）**

```python
# Before: ローカル変数が多く、抽出が難しいケース
def generate_report(orders: list[Order], start_date: date, end_date: date) -> str:
    # フィルタリング
    filtered = [o for o in orders
                if start_date <= o.date <= end_date]

    # 集計
    total_revenue = sum(o.total for o in filtered)
    total_orders = len(filtered)
    avg_order_value = total_revenue / total_orders if total_orders > 0 else 0

    # 上位商品の集計
    product_counts = {}
    for order in filtered:
        for item in order.items:
            product_counts[item.name] = (
                product_counts.get(item.name, 0) + item.quantity
            )
    top_products = sorted(
        product_counts.items(), key=lambda x: x[1], reverse=True
    )[:10]

    # レポート文字列の生成
    lines = [
        f"=== レポート ({start_date} ~ {end_date}) ===",
        f"総売上: {total_revenue:,.0f}円",
        f"注文数: {total_orders}件",
        f"平均注文額: {avg_order_value:,.0f}円",
        "",
        "--- 上位商品 ---",
    ]
    for name, count in top_products:
        lines.append(f"  {name}: {count}個")

    return "\n".join(lines)


# After: 各段階を独立したメソッドに抽出
@dataclass
class ReportStatistics:
    """レポートの集計結果を保持するデータクラス"""
    total_revenue: Decimal
    total_orders: int
    avg_order_value: Decimal
    top_products: list[tuple[str, int]]


class ReportGenerator:
    def generate(self, orders: list[Order],
                 start_date: date, end_date: date) -> str:
        """レポート生成のオーケストレーション"""
        filtered = self._filter_by_date(orders, start_date, end_date)
        stats = self._calculate_statistics(filtered)
        return self._format_report(stats, start_date, end_date)

    def _filter_by_date(self, orders: list[Order],
                        start: date, end: date) -> list[Order]:
        """期間でフィルタリング"""
        return [o for o in orders if start <= o.date <= end]

    def _calculate_statistics(self, orders: list[Order]) -> ReportStatistics:
        """注文リストから統計情報を計算"""
        total_revenue = sum(o.total for o in orders)
        total_orders = len(orders)
        avg = total_revenue / total_orders if total_orders > 0 else Decimal("0")
        top_products = self._aggregate_products(orders)
        return ReportStatistics(total_revenue, total_orders, avg, top_products)

    def _aggregate_products(self, orders: list[Order]) -> list[tuple[str, int]]:
        """商品ごとの販売数を集計"""
        counts: dict[str, int] = {}
        for order in orders:
            for item in order.items:
                counts[item.name] = counts.get(item.name, 0) + item.quantity
        return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]

    def _format_report(self, stats: ReportStatistics,
                       start: date, end: date) -> str:
        """統計情報をレポート文字列に整形"""
        lines = [
            f"=== レポート ({start} ~ {end}) ===",
            f"総売上: {stats.total_revenue:,.0f}円",
            f"注文数: {stats.total_orders}件",
            f"平均注文額: {stats.avg_order_value:,.0f}円",
            "",
            "--- 上位商品 ---",
        ]
        for name, count in stats.top_products:
            lines.append(f"  {name}: {count}個")
        return "\n".join(lines)
```

### 2.2 Move Method / Move Field（メソッド・フィールドの移動）

**用途**: Feature Envy（他クラスのデータを過度に使用するメソッド）を解消する。メソッドをそのデータを持つクラスに移動する。

**安全な手順**:
1. 移動先クラスに同じメソッドのコピーを作成する
2. 移動先で正しく動作するようにパラメータを調整する
3. 元のメソッドを移動先のメソッドへの委譲に変更する
4. テストを実行する
5. 元のメソッドの全ての呼び出し元を移動先に変更する
6. 元のメソッドを削除する
7. テストを実行する

**コード例3: Move Method（Java）**

```java
// Before: calculateDiscount() は Order のデータを使っている (Feature Envy)
class Customer {
    private Address address;
    private List<Order> orders;

    // この計算は Order の責任であるべき
    public double calculateOrderDiscount(Order order) {
        double total = order.getTotal();
        int itemCount = order.getItems().size();

        if (total > 10000 && itemCount > 5) return 0.15;
        if (total > 10000) return 0.10;
        if (total > 5000) return 0.05;
        return 0;
    }

    // この計算も Order の内部データに依存
    public String formatOrderSummary(Order order) {
        return String.format("注文 #%s: %d品目, 合計 %.0f円",
            order.getId(),
            order.getItems().size(),
            order.getTotal());
    }
}


// After: メソッドを Order に移動
class Order {
    private String id;
    private List<OrderItem> items;
    private double total;

    public double calculateDiscount() {
        if (this.total > 10000 && this.items.size() > 5) return 0.15;
        if (this.total > 10000) return 0.10;
        if (this.total > 5000) return 0.05;
        return 0;
    }

    public String formatSummary() {
        return String.format("注文 #%s: %d品目, 合計 %.0f円",
            this.id, this.items.size(), this.total);
    }

    public double getDiscountedTotal() {
        return this.total * (1 - calculateDiscount());
    }
}

// Customer は Order のメソッドを呼ぶだけ
class Customer {
    public double getOrderDiscount(Order order) {
        return order.calculateDiscount();  // 委譲
    }
}
```

### 2.3 Rename（名前の変更）

**用途**: 不明確な名前を意図が伝わる名前に改善する。最も安全で効果の高い技法の1つ。

**コード例4: Rename ── 段階的な改善（TypeScript）**

```typescript
// Before: 意味不明な名前
class Mgr {
  proc(d: any[]): any[] {
    return d.filter(i => i.s === 1).map(i => ({ ...i, t: Date.now() }));
  }

  chk(v: string): boolean {
    return v.length > 0 && v.length <= 100;
  }

  calc(a: number, b: number, c: number): number {
    return a * b - c;
  }
}


// After: 意図が明確な名前
class ActiveUserProcessor {
  /**
   * アクティブなユーザーをフィルタリングし、処理タイムスタンプを付与する。
   */
  filterAndTimestamp(users: User[]): TimestampedUser[] {
    return users
      .filter(user => user.status === UserStatus.ACTIVE)
      .map(user => ({
        ...user,
        processedAt: Date.now(),
      }));
  }

  /**
   * 表示名が有効な長さであるかを検証する。
   */
  isValidDisplayName(name: string): boolean {
    return name.length > 0 && name.length <= 100;
  }

  /**
   * 販売利益を計算する: 単価 x 数量 - 割引額
   */
  calculateProfit(unitPrice: number, quantity: number, discount: number): number {
    return unitPrice * quantity - discount;
  }
}
```

### 2.4 Replace Conditional with Polymorphism（条件分岐のポリモーフィズム置換）

**用途**: Switch Statements スメルを解消する。型による分岐を各サブクラスのメソッドオーバーライドに置き換える。

**安全な手順**:
1. 分岐条件の各ケースに対応するサブクラスを作成する
2. 基底クラスに抽象メソッドを定義する
3. 各サブクラスでメソッドを実装する
4. ファクトリメソッドまたはファクトリクラスを作成する
5. 元の条件分岐をポリモーフィック呼び出しに置き換える
6. テストを実行する

**コード例5: Replace Conditional with Polymorphism（Python）**

```python
# Before: 型による条件分岐が複数メソッドに散在
class Employee:
    def __init__(self, employee_type: str, **kwargs):
        self.type = employee_type
        self.salary = kwargs.get('salary', 0)
        self.hourly_rate = kwargs.get('hourly_rate', 0)
        self.hours_worked = kwargs.get('hours_worked', 0)
        self.daily_rate = kwargs.get('daily_rate', 0)
        self.days_worked = kwargs.get('days_worked', 0)

    def calculate_pay(self) -> Decimal:
        if self.type == 'full_time':
            return self.salary
        elif self.type == 'part_time':
            return self.hourly_rate * self.hours_worked
        elif self.type == 'contractor':
            return self.daily_rate * self.days_worked
        elif self.type == 'intern':
            return Decimal("0")  # 無給インターン
        else:
            raise ValueError(f"Unknown type: {self.type}")

    def calculate_benefits(self) -> Decimal:
        if self.type == 'full_time':
            return self.salary * Decimal("0.2")
        elif self.type == 'part_time':
            return self.hourly_rate * self.hours_worked * Decimal("0.05")
        elif self.type == 'contractor':
            return Decimal("0")
        elif self.type == 'intern':
            return Decimal("0")
        else:
            raise ValueError(f"Unknown type: {self.type}")

    def get_title(self) -> str:
        if self.type == 'full_time':
            return "正社員"
        elif self.type == 'part_time':
            return "パートタイム"
        elif self.type == 'contractor':
            return "業務委託"
        elif self.type == 'intern':
            return "インターン"
        else:
            return "不明"


# After: ポリモーフィズムで各型が自分の振る舞いを持つ
from abc import ABC, abstractmethod
from decimal import Decimal

class Employee(ABC):
    """従業員の基底クラス"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def calculate_pay(self) -> Decimal:
        """給与を計算する"""

    @abstractmethod
    def calculate_benefits(self) -> Decimal:
        """福利厚生費を計算する"""

    @abstractmethod
    def get_title(self) -> str:
        """職種名を返す"""

    def total_cost(self) -> Decimal:
        """会社の総コスト = 給与 + 福利厚生"""
        return self.calculate_pay() + self.calculate_benefits()


class FullTimeEmployee(Employee):
    def __init__(self, name: str, salary: Decimal):
        super().__init__(name)
        self.salary = salary

    def calculate_pay(self) -> Decimal:
        return self.salary

    def calculate_benefits(self) -> Decimal:
        return self.salary * Decimal("0.2")

    def get_title(self) -> str:
        return "正社員"


class PartTimeEmployee(Employee):
    def __init__(self, name: str, hourly_rate: Decimal, hours_worked: int):
        super().__init__(name)
        self.hourly_rate = hourly_rate
        self.hours_worked = hours_worked

    def calculate_pay(self) -> Decimal:
        return self.hourly_rate * self.hours_worked

    def calculate_benefits(self) -> Decimal:
        return self.calculate_pay() * Decimal("0.05")

    def get_title(self) -> str:
        return "パートタイム"


class Contractor(Employee):
    def __init__(self, name: str, daily_rate: Decimal, days_worked: int):
        super().__init__(name)
        self.daily_rate = daily_rate
        self.days_worked = days_worked

    def calculate_pay(self) -> Decimal:
        return self.daily_rate * self.days_worked

    def calculate_benefits(self) -> Decimal:
        return Decimal("0")  # 福利厚生なし

    def get_title(self) -> str:
        return "業務委託"


class Intern(Employee):
    def __init__(self, name: str, stipend: Decimal = Decimal("0")):
        super().__init__(name)
        self.stipend = stipend

    def calculate_pay(self) -> Decimal:
        return self.stipend

    def calculate_benefits(self) -> Decimal:
        return Decimal("0")

    def get_title(self) -> str:
        return "インターン"


# ファクトリ関数: 旧コードとの互換性を維持
def create_employee(employee_type: str, name: str, **kwargs) -> Employee:
    """型文字列からEmployeeを生成するファクトリ"""
    factories = {
        'full_time': lambda: FullTimeEmployee(name, Decimal(str(kwargs['salary']))),
        'part_time': lambda: PartTimeEmployee(
            name, Decimal(str(kwargs['hourly_rate'])), kwargs['hours_worked']),
        'contractor': lambda: Contractor(
            name, Decimal(str(kwargs['daily_rate'])), kwargs['days_worked']),
        'intern': lambda: Intern(name, Decimal(str(kwargs.get('stipend', 0)))),
    }
    factory = factories.get(employee_type)
    if factory is None:
        raise ValueError(f"Unknown employee type: {employee_type}")
    return factory()
```

### 2.5 Introduce Parameter Object（パラメータオブジェクトの導入）

**用途**: Data Clumps（同じパラメータ群の繰り返し）や Long Parameter List を解消する。

**コード例6: Introduce Parameter Object（TypeScript）**

```typescript
// Before: 関連するパラメータの群れが繰り返し登場
function searchProducts(
  query: string,
  minPrice: number,
  maxPrice: number,
  category: string,
  sortBy: string,
  sortOrder: 'asc' | 'desc',
  page: number,
  pageSize: number
): Product[] { /* ... */ }

function countProducts(
  query: string,
  minPrice: number,
  maxPrice: number,
  category: string
): number { /* ... */ }

function exportProducts(
  query: string,
  minPrice: number,
  maxPrice: number,
  category: string,
  format: 'csv' | 'json'
): Buffer { /* ... */ }


// After: パラメータオブジェクトに集約
interface ProductSearchCriteria {
  query: string;
  priceRange: PriceRange;
  category: string;
}

class PriceRange {
  constructor(
    readonly min: number,
    readonly max: number
  ) {
    if (min < 0) throw new Error("最小価格は0以上");
    if (max < min) throw new Error("最大価格は最小価格以上");
  }

  contains(price: number): boolean {
    return this.min <= price && price <= this.max;
  }

  static unbounded(): PriceRange {
    return new PriceRange(0, Number.MAX_SAFE_INTEGER);
  }
}

interface SortOptions {
  field: string;
  order: 'asc' | 'desc';
}

interface Pagination {
  page: number;
  pageSize: number;

  get offset(): number;
  get limit(): number;
}

function searchProducts(
  criteria: ProductSearchCriteria,
  sort: SortOptions,
  pagination: Pagination
): Product[] { /* ... */ }

function countProducts(criteria: ProductSearchCriteria): number { /* ... */ }

function exportProducts(
  criteria: ProductSearchCriteria,
  format: 'csv' | 'json'
): Buffer { /* ... */ }
```

### 2.6 Inline Method（メソッドのインライン化）

**用途**: 不要な間接参照を排除する。メソッドの本体が名前と同じくらい明確な場合に適用する。Extract Method の逆操作。

**コード例7: Inline Method（Python）**

```python
# Before: 過度に細分化されたメソッド
class OrderValidator:
    def validate(self, order):
        if not self._has_items(order):
            raise ValidationError("商品なし")
        if not self._has_customer(order):
            raise ValidationError("顧客なし")
        if not self._is_positive_total(order):
            raise ValidationError("合計が不正")

    def _has_items(self, order) -> bool:
        return len(order.items) > 0        # メソッド名と本体が同じ意味

    def _has_customer(self, order) -> bool:
        return order.customer is not None   # メソッド名と本体が同じ意味

    def _is_positive_total(self, order) -> bool:
        return order.total > 0             # メソッド名と本体が同じ意味


# After: 不要な間接参照を排除
class OrderValidator:
    def validate(self, order):
        """注文の基本バリデーション"""
        if not order.items:
            raise ValidationError("商品が選択されていません")
        if order.customer is None:
            raise ValidationError("顧客情報がありません")
        if order.total <= 0:
            raise ValidationError("合計金額が不正です")

# ★ インライン化の判断基準:
#   - メソッドの本体がメソッド名と同じくらい明確
#   - メソッドが1箇所からしか呼ばれていない
#   - メソッドがただの委譲で、追加のロジックがない
```

### 2.7 Extract Class（クラスの抽出）

**用途**: God Class を責任ごとに分割する。クラスが2つ以上の明確に異なる責任を持つ場合に適用。

**コード例8: Extract Class（Python）**

```python
# Before: User クラスが認証と住所の両方の責任を持つ
class User:
    def __init__(self, name, email, password_hash,
                 street, city, zip_code, country,
                 login_attempts, last_login, is_locked):
        self.name = name
        self.email = email
        self.password_hash = password_hash
        self.street = street
        self.city = city
        self.zip_code = zip_code
        self.country = country
        self.login_attempts = login_attempts
        self.last_login = last_login
        self.is_locked = is_locked

    def verify_password(self, password):
        return bcrypt.checkpw(password.encode(), self.password_hash)

    def record_login_attempt(self, success: bool):
        if success:
            self.login_attempts = 0
            self.last_login = datetime.now()
        else:
            self.login_attempts += 1
            if self.login_attempts >= 5:
                self.is_locked = True

    def format_address(self) -> str:
        return f"〒{self.zip_code} {self.city}{self.street}"

    def is_domestic(self) -> bool:
        return self.country == "日本"

    def calculate_shipping(self) -> Decimal:
        if self.is_domestic():
            return Decimal("500")
        return Decimal("2000")


# After: 責任ごとにクラスを分離
@dataclass(frozen=True)
class Address:
    """住所 ── 値オブジェクト"""
    street: str
    city: str
    zip_code: str
    country: str

    def format(self) -> str:
        return f"〒{self.zip_code} {self.city}{self.street}"

    def is_domestic(self) -> bool:
        return self.country == "日本"

    def calculate_shipping(self) -> Decimal:
        return Decimal("500") if self.is_domestic() else Decimal("2000")


class LoginSecurity:
    """認証セキュリティ ── ログイン試行の管理"""
    MAX_ATTEMPTS = 5

    def __init__(self, password_hash: bytes):
        self._password_hash = password_hash
        self._login_attempts = 0
        self._last_login: datetime | None = None
        self._is_locked = False

    def verify_password(self, password: str) -> bool:
        return bcrypt.checkpw(password.encode(), self._password_hash)

    def record_attempt(self, success: bool) -> None:
        if success:
            self._login_attempts = 0
            self._last_login = datetime.now()
        else:
            self._login_attempts += 1
            if self._login_attempts >= self.MAX_ATTEMPTS:
                self._is_locked = True

    @property
    def is_locked(self) -> bool:
        return self._is_locked


class User:
    """ユーザー ── 住所と認証を組み合わせる"""
    def __init__(self, name: str, email: str,
                 address: Address, security: LoginSecurity):
        self.name = name
        self.email = email
        self.address = address
        self.security = security
```

### 2.8 Replace Temp with Query（一時変数のクエリ置換）

**用途**: 一時変数を問い合わせメソッドに置き換える。計算結果を他のメソッドからも利用可能にする。

**コード例9: Replace Temp with Query（Python）**

```python
# Before: 一時変数で中間結果を保持
class ShoppingCart:
    def checkout_summary(self) -> str:
        subtotal = sum(item.price * item.quantity for item in self.items)
        discount = subtotal * Decimal("0.1") if subtotal > 10000 else Decimal("0")
        tax = (subtotal - discount) * Decimal("0.10")
        total = subtotal - discount + tax

        return (f"小計: {subtotal}, 割引: {discount}, "
                f"税: {tax}, 合計: {total}")


# After: 一時変数をメソッドに変換
class ShoppingCart:
    @property
    def subtotal(self) -> Decimal:
        """商品合計"""
        return sum(item.price * item.quantity for item in self.items)

    @property
    def discount(self) -> Decimal:
        """割引額: 1万円以上で10%"""
        return self.subtotal * Decimal("0.1") if self.subtotal > 10000 else Decimal("0")

    @property
    def tax(self) -> Decimal:
        """消費税"""
        return (self.subtotal - self.discount) * Decimal("0.10")

    @property
    def total(self) -> Decimal:
        """支払い合計"""
        return self.subtotal - self.discount + self.tax

    def checkout_summary(self) -> str:
        return (f"小計: {self.subtotal}, 割引: {self.discount}, "
                f"税: {self.tax}, 合計: {self.total}")

    # ★ 他のメソッドからも self.total 等を利用可能に
    def can_apply_coupon(self, min_amount: Decimal) -> bool:
        return self.subtotal >= min_amount
```

### 2.9 Decompose Conditional（条件分岐の分解）

**用途**: 複雑な条件式を意図が明確なメソッドに分解する。

**コード例10: Decompose Conditional（Python）**

```python
# Before: 複雑な条件式
def calculate_charge(date: date, quantity: int, rate: Decimal) -> Decimal:
    if (date.month >= 6 and date.month <= 9) or \
       (date.weekday() >= 5) or \
       (date in get_holidays(date.year)):
        # 夏季・週末・祝日料金
        charge = quantity * rate * Decimal("1.5")
        if quantity > 100:
            charge *= Decimal("0.9")
    else:
        # 通常料金
        charge = quantity * rate
        if quantity > 200:
            charge *= Decimal("0.95")
    return charge


# After: 条件と各ブランチを意図が明確なメソッドに分解
def calculate_charge(date: date, quantity: int, rate: Decimal) -> Decimal:
    if is_peak_season(date):
        return calculate_peak_charge(quantity, rate)
    return calculate_regular_charge(quantity, rate)

def is_peak_season(date: date) -> bool:
    """繁忙期: 夏季 or 週末 or 祝日"""
    return is_summer(date) or is_weekend(date) or is_holiday(date)

def is_summer(date: date) -> bool:
    return 6 <= date.month <= 9

def is_weekend(date: date) -> bool:
    return date.weekday() >= 5

def is_holiday(date: date) -> bool:
    return date in get_holidays(date.year)

def calculate_peak_charge(quantity: int, rate: Decimal) -> Decimal:
    """繁忙期料金: 1.5倍、100個超で10%割引"""
    charge = quantity * rate * Decimal("1.5")
    if quantity > 100:
        charge *= Decimal("0.9")
    return charge

def calculate_regular_charge(quantity: int, rate: Decimal) -> Decimal:
    """通常料金: 200個超で5%割引"""
    charge = quantity * rate
    if quantity > 200:
        charge *= Decimal("0.95")
    return charge
```

### 2.10 Pull Up Method / Push Down Method（メソッドの引き上げ / 押し下げ）

**用途**: 継承階層内でメソッドを適切なレベルに移動する。重複コードの排除や、特殊なロジックの分離に使用。

**コード例11: Pull Up Method（Java）**

```java
// Before: 同じメソッドが複数のサブクラスに重複
class SavingsAccount extends Account {
    public double calculateInterest() {
        return this.balance * this.interestRate / 12;  // 重複
    }
}

class CheckingAccount extends Account {
    public double calculateInterest() {
        return this.balance * this.interestRate / 12;  // 重複
    }
}

class MoneyMarketAccount extends Account {
    public double calculateInterest() {
        return this.balance * this.interestRate / 12;  // 重複
    }
}


// After: 共通メソッドを親クラスに引き上げ
abstract class Account {
    protected double balance;
    protected double interestRate;

    // Pull Up: 共通の計算を親クラスに
    public double calculateInterest() {
        return this.balance * this.interestRate / 12;
    }
}

// 特殊な計算が必要なサブクラスのみオーバーライド
class HighYieldAccount extends Account {
    @Override
    public double calculateInterest() {
        // 特殊なロジック: 段階金利
        if (this.balance > 1_000_000) {
            return this.balance * (this.interestRate * 1.5) / 12;
        }
        return super.calculateInterest();
    }
}
```

---

## 3. IDE によるリファクタリング支援

### 3.1 IDE の自動リファクタリング機能比較

| 技法 | IntelliJ IDEA | VS Code | PyCharm | Eclipse |
|------|:------------:|:-------:|:-------:|:-------:|
| Extract Method | Ctrl+Alt+M | Ctrl+Shift+R | Ctrl+Alt+M | Alt+Shift+M |
| Rename | Shift+F6 | F2 | Shift+F6 | Alt+Shift+R |
| Move | F6 | ― | F6 | Alt+Shift+V |
| Inline | Ctrl+Alt+N | ― | Ctrl+Alt+N | Alt+Shift+I |
| Extract Variable | Ctrl+Alt+V | ― | Ctrl+Alt+V | Alt+Shift+L |
| Extract Parameter | Ctrl+Alt+P | ― | Ctrl+Alt+P | ― |
| Change Signature | Ctrl+F6 | ― | Ctrl+F6 | Alt+Shift+C |
| Safe Delete | Alt+Del | ― | Alt+Del | ― |

### 3.2 IDE リファクタリングの使い方

```
  IntelliJ IDEA / PyCharm でのリファクタリング手順

  1. Extract Method:
     a. 抽出したいコードブロックを選択
     b. Ctrl+Alt+M (Mac: Cmd+Option+M)
     c. メソッド名を入力
     d. パラメータと戻り値を確認
     e. Enter で確定
     → IDE が自動的に変数の受け渡しを処理

  2. Rename:
     a. 変更したいシンボルにカーソルを置く
     b. Shift+F6
     c. 新しい名前を入力
     d. Enter で確定
     → 全ての参照箇所（テスト含む）が自動更新

  3. Move:
     a. 移動したいメソッド/クラスにカーソルを置く
     b. F6
     c. 移動先のクラス/パッケージを選択
     d. Enter で確定
     → インポート文も自動調整

  ★ IDE の自動リファクタリングは「コンパイラレベルの正確さ」で
    全ての参照を更新する。手動よりも安全。
```

---

## 4. リファクタリング技法の選択ガイド

### 4.1 スメルから技法へのフローチャート

```
  コードスメルからリファクタリング技法への選択フロー

  スメルを発見
      |
      v
  メソッドが長い？ ──Yes──> Extract Method
      |
      No
      v
  メソッドが間違った ──Yes──> Move Method
  クラスにある？
      |
      No
      v
  同じパラメータが ──Yes──> Introduce Parameter Object
  繰り返される？                 または Extract Class
      |
      No
      v
  型による条件分岐？ ──Yes──> Replace Conditional
      |                        with Polymorphism
      No
      v
  名前が不明確？ ──Yes──> Rename
      |
      No
      v
  不要な間接参照？ ──Yes──> Inline Method
      |
      No
      v
  クラスが大きすぎる？ ──Yes──> Extract Class
      |
      No
      v
  条件式が複雑？ ──Yes──> Decompose Conditional
      |
      No
      v
  一時変数が多い？ ──Yes──> Replace Temp with Query
```

### 4.2 技法の選択比較表

| スメル | 第一選択の技法 | 代替技法 | 効果 |
|--------|------------|---------|------|
| Long Method | Extract Method | Decompose Conditional | 可読性向上、再利用性 |
| Feature Envy | Move Method/Field | Extract Class | 凝集度向上 |
| Data Clumps | Introduce Parameter Object | Extract Class | パラメータ削減 |
| Switch Statements | Replace Conditional with Polymorphism | Strategy パターン | 拡張性向上 (OCP) |
| 不明確な名前 | Rename | ― | 可読性向上 |
| 重複コード | Extract Method + Pull Up Method | Template Method パターン | DRY 化 |
| God Class | Extract Class | Facade パターン | SRP 達成 |
| 不要な間接参照 | Inline Method | Inline Class | 簡潔化 |
| Shotgun Surgery | Move Method + Inline Class | ― | 変更の局所化 |
| 複雑な条件式 | Decompose Conditional | Replace Conditional with Polymorphism | 可読性向上 |

### 4.3 リファクタリングの安全性レベル

| 安全性 | 技法 | 理由 |
|:------:|------|------|
| 高 | Rename | IDE が全参照を自動更新。意味を変えない |
| 高 | Extract Method | 振る舞いを変えず構造のみ変更。IDE 支援あり |
| 高 | Inline Method | Extract Method の逆。IDE 支援あり |
| 中 | Move Method | 参照元の更新が必要。IDE が大部分を自動化 |
| 中 | Replace Temp with Query | 副作用がある場合は注意 |
| 中 | Introduce Parameter Object | 呼び出し元の変更が広範に及ぶ可能性 |
| 中 | Extract Class | 依存関係の整理が必要 |
| 低 | Replace Conditional with Polymorphism | 設計の大幅な変更。十分なテストが必須 |
| 低 | Pull Up / Push Down | 継承階層の変更。テストの再構成が必要 |

---

## 5. 実践的なリファクタリングワークフロー

### 5.1 マイクロリファクタリングセッション（15分）

**コード例12: 実際のリファクタリング手順（Git 操作付き）**

```bash
# Step 1: 現在のテスト状態を確認
$ pytest tests/ -q
42 passed in 3.2s

# Step 2: リファクタリング開始のコミット
$ git stash  # 作業中の変更を退避（もしあれば）

# Step 3: Extract Method を適用
# (IDE でコードを選択 → Ctrl+Alt+M → メソッド名入力)

# Step 4: テスト実行
$ pytest tests/ -q
42 passed in 3.1s  # OK: テスト数が変わっていない

# Step 5: コミット
$ git add -p  # 変更を確認しながらステージング
$ git commit -m "refactor: extract _calculate_discount from process_order"

# Step 6: 次の変更（Rename）
# (IDE でシンボルにカーソル → Shift+F6 → 新名前入力)

# Step 7: テスト実行
$ pytest tests/ -q
42 passed in 3.1s  # OK

# Step 8: コミット
$ git commit -am "refactor: rename 'calc' to 'calculate_monthly_revenue'"

# Step 9: 次の変更（Move Method）...
# 以降、同じサイクルを繰り返す
```

### 5.2 リファクタリングのコミットメッセージ規約

```
  推奨するコミットメッセージのフォーマット

  refactor: <技法名> <対象> from <元の場所>

  例:
  refactor: extract calculate_discount from process_order
  refactor: move format_address from User to Address
  refactor: rename 'proc' to 'process_payment'
  refactor: replace conditional with polymorphism in Employee
  refactor: inline get_is_valid into validate
  refactor: introduce OrderCriteria parameter object

  ★ 「refactor:」プレフィックスにより、機能変更との区別が明確になる
  ★ git log --grep="refactor:" でリファクタリング履歴を一覧できる
```

---

## 6. アンチパターン

### アンチパターン1: テストなしのリファクタリング

```
  BAD: テストなしで構造を変更

  「このコードの構造が気に入らない」
    → テストを書かずにリファクタリング開始
    → 「動いてるように見える」
    → 1週間後に本番でバグ発覚
    → どの変更がバグを引き起こしたか不明
    → デバッグに数日かかる

  GOOD: テストで安全網を確保してからリファクタリング

  1. まずテストを書く（または既存テストを確認）
  2. テストが GREEN であることを確認
  3. 小さな変更を1つ適用
  4. テストが GREEN であることを確認
  5. コミット
  6. 次の変更へ

  ★ テストがない場合は、まず特性テスト（Characterization Test）を書く
  → 詳細は [レガシーコード](./02-legacy-code.md) を参照
```

### アンチパターン2: Big Bang リファクタリング

```
  BAD: 「全部一度に書き直そう！」

  「週末でこのモジュール全体をリファクタリングする」
    → 500行の差分が1つの PR に
    → レビュアーが理解不能
    → マージコンフリクト多発
    → テストが壊れても原因が特定困難
    → 結局 revert

  GOOD: Strangler Fig パターンで段階的に移行

  PR #1: [Extract Class] Address を User から分離 (差分: 40行)
  PR #2: [Move Method] format_address を Address に移動 (差分: 20行)
  PR #3: [Rename] User.addr を User.address に (差分: 15行)
  PR #4: [Inline] 不要になった User.get_city を削除 (差分: 10行)

  各 PR は:
  - 300行以下の差分
  - レビュー可能なサイズ
  - テストが常に通る
  - 独立してマージ/リバート可能
```

### アンチパターン3: リファクタリングと機能追加の同時実行

```
  BAD: 「ついでに機能も追加しよう」

  コミットメッセージ: "refactor user service and add email verification"
    → 構造変更と機能追加が混在
    → バグが出たとき、原因がリファクタリングか新機能か不明
    → revert すると新機能も失われる

  GOOD: 2つの帽子を分ける

  Phase 1: リファクタリング
    PR: "refactor: extract EmailValidator from UserService"
    → テストは変更しない（振る舞いが変わらないから）

  Phase 2: 機能追加
    PR: "feat: add email verification on registration"
    → テストを追加する（新しい振る舞いだから）

  ★ Martin Fowler の「2つの帽子」原則を厳守
```

### アンチパターン4: 過度な抽象化リファクタリング

```
  BAD: 「将来の拡張のために抽象化しよう」

  class Serializer(ABC): ...
  class JsonSerializer(Serializer): ...    # ← 唯一の実装
  class SerializerFactory: ...             # ← 1パターンのみ
  class SerializerStrategy: ...            # ← 不要な中間層

  → 3クラスと2インターフェースが1つの JSON 変換のために存在
  → 「抽象化の地獄」: コードを追うのに5ファイルのジャンプが必要

  GOOD: 必要になるまで抽象化しない (YAGNI)

  class JsonSerializer:
      def serialize(self, data): ...
      def deserialize(self, text): ...

  → 将来 XML が必要になったら、そのときに抽象化を導入
  → "Rule of Three": 3回目のパターン出現まで待つ
```

---

## 7. 演習問題

### 演習1（基本）: Extract Method の適用

以下のコードから適切なメソッドを抽出し、可読性を向上させよ。

```python
# 問題コード
def process_user_registration(data: dict) -> dict:
    # バリデーション
    errors = []
    if not data.get('name') or len(data['name']) < 2:
        errors.append("名前は2文字以上必須")
    if not data.get('email') or '@' not in data['email']:
        errors.append("有効なメールアドレスが必要")
    if not data.get('password') or len(data['password']) < 8:
        errors.append("パスワードは8文字以上必須")
    if data.get('password') and not any(c.isupper() for c in data['password']):
        errors.append("パスワードに大文字を含める必要")
    if data.get('password') and not any(c.isdigit() for c in data['password']):
        errors.append("パスワードに数字を含める必要")
    if errors:
        return {"success": False, "errors": errors}

    # ユーザー作成
    import hashlib
    salt = os.urandom(32)
    password_hash = hashlib.pbkdf2_hmac(
        'sha256', data['password'].encode(), salt, 100000
    )
    user = {
        "name": data['name'].strip(),
        "email": data['email'].lower().strip(),
        "password_hash": password_hash,
        "salt": salt,
        "created_at": datetime.now().isoformat(),
    }

    # DB保存
    db.execute("INSERT INTO users ...", user)

    # 歓迎メール送信
    email_body = f"""
    {user['name']} 様
    ご登録ありがとうございます。
    """
    send_email(user['email'], "ようこそ", email_body)

    return {"success": True, "user_id": user.get('id')}
```

**期待される回答**: 4つのメソッドに分離: `validate_registration`, `create_user_record`, `save_user`, `send_welcome_email`。メインメソッドは各メソッドの呼び出しのみ。

---

### 演習2（応用）: Move Method + Replace Conditional

以下のコードで Feature Envy と Switch Statements を解消せよ。

```python
class ShippingCalculator:
    def calculate(self, order) -> Decimal:
        # Feature Envy: order の内部データに過度に依存
        weight = sum(item.weight * item.quantity for item in order.items)
        destination = order.customer.address.country

        # Switch Statements: 配送方法による分岐
        if order.shipping_method == "standard":
            if destination == "日本":
                return Decimal("500") if weight < 5 else Decimal("1000")
            else:
                return Decimal("2000") if weight < 5 else Decimal("4000")
        elif order.shipping_method == "express":
            if destination == "日本":
                return Decimal("1200") if weight < 5 else Decimal("2000")
            else:
                return Decimal("5000") if weight < 5 else Decimal("8000")
        elif order.shipping_method == "overnight":
            if destination == "日本":
                return Decimal("2500")
            else:
                return Decimal("10000")
```

**期待される回答**: (1) `Order.total_weight()` メソッドの追加, (2) `ShippingMethod` 基底クラス + `StandardShipping`, `ExpressShipping`, `OvernightShipping` サブクラスの作成, (3) 各サブクラスに `calculate(weight, destination)` メソッドを実装。

---

### 演習3（上級）: レガシーコードのリファクタリング計画

以下のコードについて、テスト→リファクタリング→テストの安全なサイクルを設計せよ。各ステップのコミットメッセージも含めること。

```python
class ReportEngine:
    """500行のモノリシックなレポート生成エンジン"""
    def generate(self, report_type, data, format_type,
                 start_date, end_date, filters, sort_by,
                 include_charts, email_to, save_path):
        # Step 1: データフィルタリング (50行)
        # Step 2: 集計計算 (80行)
        # Step 3: チャート生成 (60行, include_charts の場合のみ)
        # Step 4: フォーマット変換 (100行, format_type による分岐)
        # Step 5: メール送信 (30行, email_to の場合のみ)
        # Step 6: ファイル保存 (30行, save_path の場合のみ)
        ...
```

**期待される回答（概要）**:

```
リファクタリング計画 (10 PR, 推定3スプリント):

PR #1: test: add characterization tests for ReportEngine
  - 既存の振る舞いを記録するテストを追加
  コミット: "test: add characterization tests for generate()"

PR #2: refactor: introduce ReportRequest parameter object
  - 11パラメータを ReportRequest に集約
  コミット: "refactor: introduce ReportRequest parameter object"

PR #3: refactor: extract _filter_data from generate
  コミット: "refactor: extract _filter_data from generate"

PR #4: refactor: extract _calculate_aggregates from generate
  コミット: "refactor: extract _calculate_aggregates from generate"

PR #5: refactor: extract ChartGenerator class
  コミット: "refactor: extract ChartGenerator from ReportEngine"

PR #6: refactor: replace format_type conditional with polymorphism
  - ReportFormatter 基底 + HtmlFormatter, PdfFormatter, CsvFormatter
  コミット: "refactor: replace conditional with polymorphism for formatters"

PR #7: refactor: extract EmailNotifier class
  コミット: "refactor: extract EmailNotifier from ReportEngine"

PR #8: refactor: extract FileExporter class
  コミット: "refactor: extract FileExporter from ReportEngine"

PR #9: refactor: ReportEngine を orchestrator に変換
  - generate() は各コンポーネントの呼び出しのみに
  コミット: "refactor: simplify ReportEngine to orchestrator"

PR #10: refactor: rename and cleanup
  コミット: "refactor: final cleanup and documentation"
```

---

## 8. FAQ

### Q1: リファクタリングのタイミングはいつが最適か？

**Rule of Three**: 同じパターンが3回現れたらリファクタリング。また、以下のタイミングが効果的:
- **機能追加の前**: 変更しやすくする準備として（「まず庭を整えてから種をまく」）
- **バグ修正の後**: 根本原因の設計問題を解消（「穴を塞いだら壁を補強する」）
- **コードレビュー時**: レビュー指摘への対応として
- **理解のため**: コードを読む過程で構造を改善し、理解を深める

### Q2: リファクタリングの成果をどう測定するか？

定量的指標:
- **サイクロマティック複雑度の減少**: `radon cc` で測定
- **テストカバレッジの向上**: `pytest --cov` で測定
- **コード重複率の低下**: `jscpd` で測定
- **変更に要する時間の短縮**: DORA メトリクスの Lead Time で測定
- **プルリクエストのサイズ縮小**: 1 PR の平均差分行数
- **バグ発生率の低下**: 変更あたりのバグ報告数

### Q3: マネジメントにリファクタリングの必要性をどう説明するか？

「リファクタリング」という言葉は避け、以下のように伝える:
- 「新機能の追加速度を維持するための構造改善」
- 「バグ発生率を下げるための予防的保守」
- 「開発チームの生産性投資」

数値で示す: 変更あたりのバグ率、機能追加にかかる時間の推移。「このモジュールの変更に平均3日かかっており、リファクタリングにより1日に短縮できる見込み」のように具体的なROIを示す。

### Q4: リファクタリング中にバグを見つけた場合はどうするか？

**リファクタリングとバグ修正を混ぜない**。以下の手順で対処:

1. リファクタリングの現在の変更をコミット（または stash）
2. バグ修正用のブランチを作成
3. バグを修正し、テストを追加
4. バグ修正をコミット・マージ
5. リファクタリングブランチに戻り、バグ修正をマージ
6. リファクタリングを再開

### Q5: 大規模なリファクタリングをどう進めるか？

**Strangler Fig パターン**を適用する:
1. 旧コードと新コードの間に **Facade（ファサード）** を配置
2. 新しいコードを **テスト付き** で追加
3. Facade のルーティングを段階的に新コードに切り替え
4. 旧コードが不要になったら削除

詳細は [レガシーコード](./02-legacy-code.md) の Strangler Fig パターンの章を参照。

### Q6: リファクタリングとパフォーマンス最適化の違いは？

| 観点 | リファクタリング | パフォーマンス最適化 |
|------|--------------|-----------------|
| 目的 | 可読性・保守性の向上 | 実行速度・リソース効率の向上 |
| 振る舞い | 変えない | 変えない（はず） |
| 可読性 | 向上する | 低下することがある |
| タイミング | コードを理解する過程で | プロファイリングの結果に基づき |
| 優先順位 | まずリファクタリング | 次にパフォーマンス最適化 |

Donald Knuth の名言: 「早まった最適化は諸悪の根源である」。まず readable なコードを書き、プロファイリングでボトルネックを特定してから最適化する。

---

## 9. まとめ

| 技法 | 用途 | 安全性 | IDE支援 |
|------|------|:------:|:------:|
| Extract Method | 長い関数の分割 | 高 | 可 |
| Move Method | 責任の移動（Feature Envy 解消） | 中 | 可 |
| Rename | 名前の改善 | 高 | 可 |
| Replace Conditional with Polymorphism | 分岐の排除 | 低 | 部分的 |
| Introduce Parameter Object | 引数の整理 | 中 | 部分的 |
| Inline Method | 不要な間接参照の排除 | 高 | 可 |
| Extract Class | God Class の分割 | 中 | 部分的 |
| Replace Temp with Query | 一時変数の排除 | 中 | 不可 |
| Decompose Conditional | 条件式の分解 | 高 | 不可 |
| Pull Up / Push Down | 継承階層の整理 | 低 | 可 |

| 原則 | 内容 |
|------|------|
| 振る舞い不変 | 外部から見た動作は変えない |
| テスト先行 | テストが GREEN でないならリファクタリングしない |
| 小さなステップ | 1回の変更は1種類の改善のみ |
| 頻繁なコミット | いつでも直前の安全な状態に戻れるように |
| 2つの帽子 | リファクタリングと機能追加を同時にしない |
| Rule of Three | 3回目のパターン出現でリファクタリング |

---

## 次に読むべきガイド

- [コードスメル](./00-code-smells.md) ── リファクタリングのトリガーとなるスメル分類
- [レガシーコード](./02-legacy-code.md) ── テストがないコードのリファクタリング（Seam, Characterization Test）
- [技術的負債](./03-technical-debt.md) ── リファクタリングの投資対効果の定量化
- [継続的改善](./04-continuous-improvement.md) ── リファクタリングを日常に組み込む CI/CD
- [テスト原則](../01-practices/04-testing-principles.md) ── リファクタリングの安全網としてのテスト設計
- [デザインパターン概要](../../design-patterns-guide/docs/00-creational/) ── スメル解消のためのパターン適用
- [関数設計](../01-practices/02-functions.md) ── Extract Method 後の関数設計指針

---

## 参考文献

1. **Martin Fowler** 『Refactoring: Improving the Design of Existing Code』 Addison-Wesley, 2018 (2nd Edition) ── リファクタリングの決定版。60以上の技法をカタログ化。JavaScript ベースのコード例で、第1版(1999, Java)から大幅に刷新。
2. **Joshua Kerievsky** 『Refactoring to Patterns』 Addison-Wesley, 2004 ── リファクタリングとデザインパターンの橋渡し。スメルからパターンへの段階的な変換手順を示す。
3. **Michael Feathers** 『Working Effectively with Legacy Code』 Prentice Hall, 2004 ── テストのないコードに対する安全なリファクタリング技法。Seam、Extract & Override、Characterization Test の原典。
4. **Kent Beck** 『Implementation Patterns』 Addison-Wesley, 2007 ── コードレベルの設計判断。命名、メソッド分割、状態管理のパターンを体系化。
5. **Sandi Metz** 『99 Bottles of OOP』 (2nd Edition, 2018) ── ポリモーフィズムへのリファクタリングを段階的に進める実践例。Open-Closed Principle の体得に最適。
