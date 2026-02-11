# リファクタリング技法 ── Extract Method、Move、その他の基本技法

> リファクタリングとは、外部から見たプログラムの振る舞いを変えずに、内部構造を改善する規律あるプロセスである。小さなステップを積み重ね、テストで安全性を確認しながら進める。

---

## この章で学ぶこと

1. **リファクタリングの基本原則** ── 安全なリファクタリングの進め方を理解する
2. **主要なリファクタリング技法** ── Extract Method、Move Method、Rename等の基本技法を身につける
3. **リファクタリングのワークフロー** ── テスト→リファクタリング→テストのサイクルを習得する

---

## 1. リファクタリングの基本原則

```
+-----------------------------------------------------------+
|  リファクタリングの鉄則                                    |
|  ─────────────────────────────────────                    |
|  1. 振る舞いを変えない（外部仕様は不変）                  |
|  2. テストを先に書く/確認する                             |
|  3. 小さなステップで進める                                |
|  4. 頻繁にコミットする                                    |
|  5. リファクタリングと機能追加を同時にしない              |
+-----------------------------------------------------------+
```

```
  リファクタリングのサイクル

  ┌──────────────┐
  │ 1. テスト実行 │
  │   (全て通過)  │
  └──────┬───────┘
         v
  ┌──────────────┐
  │ 2. 小さな変更 │
  │   (1ステップ) │
  └──────┬───────┘
         v
  ┌──────────────┐      失敗
  │ 3. テスト実行 │──────────→ 変更を取り消す
  │              │
  └──────┬───────┘
    成功  v
  ┌──────────────┐
  │ 4. コミット   │
  └──────┬───────┘
         v
  ┌──────────────┐
  │ 5. 次の変更へ │──→ ステップ2に戻る
  └──────────────┘
```

---

## 2. 主要なリファクタリング技法

### 2.1 Extract Method（メソッドの抽出）

**コード例1: Extract Method**

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
        self._print_header()
        self._print_line_items()
        self._print_totals()

    def _print_header(self):
        print("========== 請求書 ==========")
        print(f"顧客: {self.customer.name}")
        print(f"日付: {self.date}")
        print()

    def _print_line_items(self):
        for item in self.items:
            line_total = item.price * item.quantity
            print(f"  {item.name}: {item.price} x {item.quantity} = {line_total}")

    def _print_totals(self):
        subtotal = self.calculate_subtotal()
        tax = subtotal * Decimal("0.10")
        print()
        print(f"小計: {subtotal}")
        print(f"消費税(10%): {tax}")
        print(f"合計: {subtotal + tax}")
        print("============================")

    def calculate_subtotal(self) -> Decimal:
        return sum(item.price * item.quantity for item in self.items)
```

### 2.2 Move Method / Move Field

**コード例2: Move Method**

```java
// Before: メソッドが間違ったクラスにある（Feature Envy）
class Customer {
    private Address address;
    private List<Order> orders;

    // この計算は Order の責任
    public double calculateOrderDiscount(Order order) {
        if (order.getTotal() > 10000) return 0.10;
        if (order.getTotal() > 5000) return 0.05;
        return 0;
    }
}

// After: Order に移動
class Order {
    private double total;

    public double calculateDiscount() {
        if (this.total > 10000) return 0.10;
        if (this.total > 5000) return 0.05;
        return 0;
    }
}
```

### 2.3 Rename（名前の変更）

**コード例3: Rename**

```typescript
// Before: 不明確な名前
class Mgr {
  proc(d: any[]): any[] {
    return d.filter(i => i.s === 1).map(i => ({ ...i, t: Date.now() }));
  }
}

// After: 意図が明確な名前
class ActiveUserProcessor {
  filterAndTimestamp(users: User[]): TimestampedUser[] {
    return users
      .filter(user => user.status === UserStatus.ACTIVE)
      .map(user => ({ ...user, processedAt: Date.now() }));
  }
}
```

### 2.4 Replace Conditional with Polymorphism

**コード例4: 条件分岐のポリモーフィズム置換**

```python
# Before: 型による条件分岐
class Employee:
    def calculate_pay(self) -> Decimal:
        if self.type == 'full_time':
            return self.salary
        elif self.type == 'part_time':
            return self.hourly_rate * self.hours_worked
        elif self.type == 'contractor':
            return self.daily_rate * self.days_worked
        elif self.type == 'intern':
            return self.stipend
        else:
            raise ValueError(f"Unknown type: {self.type}")


# After: ポリモーフィズムで各型が自分の計算を持つ
class Employee(ABC):
    @abstractmethod
    def calculate_pay(self) -> Decimal:
        pass

class FullTimeEmployee(Employee):
    def __init__(self, salary: Decimal):
        self.salary = salary

    def calculate_pay(self) -> Decimal:
        return self.salary

class PartTimeEmployee(Employee):
    def __init__(self, hourly_rate: Decimal, hours_worked: int):
        self.hourly_rate = hourly_rate
        self.hours_worked = hours_worked

    def calculate_pay(self) -> Decimal:
        return self.hourly_rate * self.hours_worked

class Contractor(Employee):
    def __init__(self, daily_rate: Decimal, days_worked: int):
        self.daily_rate = daily_rate
        self.days_worked = days_worked

    def calculate_pay(self) -> Decimal:
        return self.daily_rate * self.days_worked
```

### 2.5 Introduce Parameter Object

**コード例5: パラメータオブジェクトの導入**

```typescript
// Before: 関連するパラメータの群れ
function searchProducts(
  query: string,
  minPrice: number,
  maxPrice: number,
  category: string,
  sortBy: string,
  sortOrder: 'asc' | 'desc',
  page: number,
  pageSize: number
): Product[] { ... }

// After: パラメータオブジェクトに集約
interface ProductSearchCriteria {
  query: string;
  priceRange: { min: number; max: number };
  category: string;
}

interface SortOptions {
  field: string;
  order: 'asc' | 'desc';
}

interface Pagination {
  page: number;
  pageSize: number;
}

function searchProducts(
  criteria: ProductSearchCriteria,
  sort: SortOptions,
  pagination: Pagination
): Product[] { ... }
```

---

## 3. リファクタリング技法の選択ガイド

| スメル | 適用する技法 | 効果 |
|--------|------------|------|
| Long Method | Extract Method | 可読性向上、再利用性 |
| Feature Envy | Move Method/Field | 凝集度向上 |
| Data Clumps | Introduce Parameter Object | パラメータ削減 |
| Switch Statements | Replace Conditional with Polymorphism | 拡張性向上 |
| 不明確な名前 | Rename | 可読性向上 |
| 重複コード | Extract Method + Pull Up Method | DRY化 |

| ツール | 自動リファクタリング機能 |
|--------|----------------------|
| IntelliJ IDEA | Extract, Move, Rename, Inline 等 |
| VS Code | Rename, Extract Function |
| PyCharm | Python専用の豊富なリファクタリング |
| Eclipse | Java向けリファクタリングメニュー |

---

## 4. アンチパターン

### アンチパターン1: テストなしのリファクタリング

```
変更 → テストなし → 動いてるように見える → 後でバグ発覚
→ 原因: どの変更でバグが入ったか特定不能

正しい手順:
テスト確認 → 小さな変更 → テスト確認 → コミット → 繰り返し
```

### アンチパターン2: Big Bang リファクタリング

```
「全部一度に書き直そう！」
→ 数週間の並行開発 → マージ地獄 → 新しいバグ大量発生
→ 結局、旧コードに戻す

正しいアプローチ:
Strangler Fig パターン: 古い機能を少しずつ新しい実装に置き換え
```

---

## 5. FAQ

### Q1: リファクタリングのタイミングはいつが最適か？

**Rule of Three**: 同じパターンが3回現れたらリファクタリング。また、以下のタイミングが効果的:
- 機能追加の前（変更しやすくする準備として）
- バグ修正の後（根本原因の設計問題を解消）
- コードレビュー時（レビュー指摘への対応として）

### Q2: リファクタリングの成果をどう測定するか？

- サイクロマティック複雑度の減少
- テストカバレッジの向上
- コード重複率の低下
- 変更に要する時間の短縮
- プルリクエストのサイズ縮小

### Q3: マネジメントにリファクタリングの必要性をどう説明するか？

「リファクタリング」という言葉は避け、以下のように伝える:
- 「新機能の追加速度を維持するための構造改善」
- 「バグ発生率を下げるための予防的保守」
- 「開発チームの生産性投資」
数値で示す: 変更あたりのバグ率、機能追加にかかる時間の推移。

---

## まとめ

| 技法 | 用途 | 安全性 |
|------|------|--------|
| Extract Method | 長い関数の分割 | 高（自動化可能） |
| Move Method | 責任の移動 | 中 |
| Rename | 名前の改善 | 高（自動化可能） |
| Replace Conditional | 分岐の排除 | 中 |
| Introduce Parameter Object | 引数の整理 | 高 |
| Inline Method | 不要な間接参照の排除 | 高 |

---

## 次に読むべきガイド

- [コードスメル](./00-code-smells.md) ── リファクタリングのトリガーとなるスメル
- [レガシーコード](./02-legacy-code.md) ── テストがないコードのリファクタリング
- [継続的改善](./04-continuous-improvement.md) ── リファクタリングを習慣化する

---

## 参考文献

1. **Martin Fowler** 『Refactoring: Improving the Design of Existing Code』 Addison-Wesley, 2018 (2nd Edition)
2. **Joshua Kerievsky** 『Refactoring to Patterns』 Addison-Wesley, 2004
3. **Michael Feathers** 『Working Effectively with Legacy Code』 Prentice Hall, 2004
