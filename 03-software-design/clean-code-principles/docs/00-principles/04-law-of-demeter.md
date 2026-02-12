# デメテルの法則 ── 最小知識の原則

> デメテルの法則（Law of Demeter / LoD）は「直接の友人とだけ話せ」という設計原則である。オブジェクトが知るべき範囲を最小限に保つことで、結合度を下げ、変更に強いシステムを構築する。

---

## この章で学ぶこと

1. **デメテルの法則の定義と目的** ── 「ドットの連鎖を避けよ」の本質を理解する
2. **違反パターンの検出** ── Train Wreck（列車事故）コードを見抜く目を養う
3. **適用のバランス** ── 過度な適用を避け、適切な範囲で活用する判断力を身につける
4. **実践的なリファクタリング手法** ── Tell, Don't Ask 原則との関係を理解し適用する
5. **言語別の適用方法** ── OOP、関数型、データ指向での適用の違いを把握する

---

## 前提知識

このガイドを最大限に活用するには、以下の知識が必要です。

| 前提知識 | 必要レベル | 参照先 |
|---------|----------|--------|
| 結合度と凝集度 | 基本を理解 | [結合度と凝集度](./03-coupling-cohesion.md) |
| SOLID原則（特にSRP, ISP） | 基本を理解 | [SOLID原則](./01-solid.md) |
| クリーンコードの概要 | 読了推奨 | [クリーンコード概論](./00-clean-code-overview.md) |
| オブジェクト指向の基本 | クラス設計の経験あり | -- |
| リファクタリングの基礎 | 概要を把握 | [コードスメル](../02-refactoring/00-code-smells.md) |

---

## 1. デメテルの法則とは

### 1.1 歴史的背景

デメテルの法則は1987年にNortheastern UniversityのKarl Lieberherrらが「Demeter Project」の研究で提唱した。プロジェクト名はギリシャ神話の女神デメテル（農業と収穫の神）に由来する。「必要なもの（知識）だけを収穫せよ」という意味が込められている。

彼らの研究は、オブジェクト指向プログラムにおいて「メソッドが呼び出すオブジェクトの範囲を制限すること」で、プログラムの保守性が大幅に向上するという実証データに基づいている。

```
  デメテルの法則が解決する問題

  高結合コード（LoD違反）:
  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ Client   │───→│ Object A │───→│ Object B │───→│ Object C │
  └──────────┘    └──────────┘    └──────────┘    └──────────┘
  client.a.b.c.doSomething()
  → Client は A, B, C すべての内部構造を知っている
  → A, B, C のいずれかが変更されると Client が壊れる

  低結合コード（LoD準拠）:
  ┌──────────┐    ┌──────────┐
  │ Client   │───→│ Object A │  （A が内部で B, C を管理）
  └──────────┘    └──────────┘
  client.a.doSomething()
  → Client は A の公開メソッドだけを知っている
  → B, C の変更は Client に影響しない
```

### 1.2 正式な定義

```
+-----------------------------------------------------------+
|  デメテルの法則 (Law of Demeter, 1987)                    |
|  ─────────────────────────────────────────────────         |
|  あるメソッド M 内で、M が呼び出してよいのは:             |
|                                                           |
|  1. M が属するオブジェクト自身のメソッド          (self)  |
|  2. M の引数として渡されたオブジェクトのメソッド  (param) |
|  3. M 内で生成されたオブジェクトのメソッド        (local) |
|  4. M が属するオブジェクトのフィールドのメソッド   (field) |
|                                                           |
|  つまり「友人の友人」のメソッドを呼んではいけない          |
+-----------------------------------------------------------+
```

**コード例1: 4つの許可されたメソッド呼び出し**

```python
class OrderProcessor:
    def __init__(self, validator: OrderValidator, logger: Logger):
        self.validator = validator  # フィールド
        self.logger = logger       # フィールド

    def process(self, order: Order) -> ProcessResult:
        # ルール1: 自身のメソッド（self）
        self._log_processing_start(order)

        # ルール2: 引数のメソッド（param）
        if not order.is_valid():
            return ProcessResult.invalid()

        # ルール3: ローカルで生成したオブジェクトのメソッド（local）
        receipt = Receipt.create(order)
        receipt.finalize()

        # ルール4: フィールドのメソッド（field）
        self.validator.validate(order)
        self.logger.info(f"Order {order.id} processed")

        return ProcessResult.success(receipt)

    def _log_processing_start(self, order: Order):
        self.logger.info(f"Processing order {order.id}")

    # 以下は NG（友人の友人のメソッドを呼んでいる）
    # order.customer.address.city.name  ← NG！
    # self.validator.config.timeout      ← NG！
```

### 1.3 直感的な理解

```
  LoD違反: 友人の友人に直接話しかける

  自分 ──→ 友人 ──→ 友人の友人 ──→ そのまた友人
  obj.getA().getB().getC().doSomething()
       ↑       ↑       ↑
       OK     NG!     NG!

  LoD準拠: 友人に頼む

  自分 ──→ 友人
  obj.doSomethingThroughChain()
       ↑
       OK (友人が内部で責任を持つ)
```

**現実世界のアナロジー:**

```
  LoD違反（現実世界の例）:
  あなたがピザを注文するとき...

  ✗ 店員の財布からクレジットカードの暗証番号を見て
    決済システムに直接入力する
    → 店員の内部（財布）を知りすぎている

  ✓ 店員にお金を渡して「ピザください」と言う
    → 店員が内部的にどう処理するかは知らなくてよい


  LoD違反（コードの例）:
  order.getCustomer().getWallet().getCreditCard().charge(amount)
  → 注文オブジェクトが顧客の財布の中身まで知っている

  LoD準拠:
  order.charge(amount)
  → 注文オブジェクトが内部的に処理を委譲する
```

### 1.4 なぜデメテルの法則が重要か

デメテルの法則が防ぐ問題を定量的に示す。

| 問題 | LoD違反時 | LoD準拠時 |
|------|----------|----------|
| 変更影響範囲 | チェーン上のどのクラスが変更されても壊れる（N箇所） | 直接の友人のインターフェースが変わった時だけ（1箇所） |
| テスト準備コスト | チェーン上のすべてのオブジェクトをモック（N個） | 直接の友人のみモック（1個） |
| 可読性 | 長いドット連鎖の意味を解読する必要がある | メソッド名から意図が読める |
| NullPointerException | チェーンのどこでnullか特定困難 | 1段のみなのでデバッグ容易 |

```
  変更の波及効果の比較

  LoD違反: customer.getAddress().getCity().getZipCode()
  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐
  │Customer│→│Address │→│ City   │→│ZipCode │
  └────────┘  └────────┘  └────────┘  └────────┘
  ↑ City の構造変更が Client に波及（結合度: 高）
  影響ファイル: Client + Customer + Address + City = 4

  LoD準拠: customer.getShippingZipCode()
  ┌────────┐  ┌────────┐
  │ Client │→│Customer│  （Customer が内部で管理）
  └────────┘  └────────┘
  ↑ City の構造変更は Customer 内で吸収
  影響ファイル: Customer のみ = 1
```

---

## 2. 違反パターンと改善

### 2.1 Train Wreck（列車事故）コード

```
  LoD違反のコード構造（ドット連鎖）

  customer.getAddress().getCity().getZipCode().format()
  ────────┬──────────┬─────────┬────────────┬────────
          │          │         │            │
    1段目の │    2段目の│   3段目の │      4段目の│
    取得     取得      取得       操作

  → 4つのクラスの内部構造に依存している
  → いずれかのクラスが変更されると呼び出し側が壊れる
```

**コード例2: Train Wreck の修正**

```python
# === LoD違反: 深いドット連鎖 ===
class OrderProcessor:
    def calculate_shipping(self, order):
        # Order → Customer → Address → City → ZipCode と4段掘り下げ
        zip_code = order.customer.address.city.zip_code
        if zip_code.startswith('100'):
            return 500  # 都内は500円
        return 1000


# === LoD準拠: 各オブジェクトが自分の責任範囲で応答 ===
class Order:
    def __init__(self, customer: Customer):
        self.customer = customer

    def get_shipping_zip_code(self) -> str:
        """配送先の郵便番号を返す（内部構造を隠蔽）"""
        return self.customer.get_zip_code()

class Customer:
    def __init__(self, address: Address):
        self.address = address

    def get_zip_code(self) -> str:
        """顧客の郵便番号を返す"""
        return self.address.get_zip_code()

class Address:
    def __init__(self, city: City):
        self.city = city

    def get_zip_code(self) -> str:
        """住所の郵便番号を返す"""
        return self.city.zip_code

class OrderProcessor:
    def calculate_shipping(self, order):
        zip_code = order.get_shipping_zip_code()  # 1ドットで完結
        if zip_code.startswith('100'):
            return 500
        return 1000
```

**コード例3: コレクションの操作**

```java
// === LoD違反: 内部コレクションを外部に公開 ===
class Department {
    private List<Employee> employees;

    public List<Employee> getEmployees() {
        return employees;  // 内部構造を暴露
    }
}

// 呼び出し側が内部構造に依存
int count = department.getEmployees().stream()
    .filter(e -> e.getSalary() > 500000)
    .count();

// 問題点:
// 1. Department が内部のデータ構造（List）を公開している
// 2. 呼び出し側が Employee の salary フィールドを知っている
// 3. フィルタリングロジックが呼び出し側に漏れている


// === LoD準拠: 問い合わせメソッドを提供 ===
class Department {
    private List<Employee> employees;

    public int countEmployeesWithSalaryAbove(int threshold) {
        return (int) employees.stream()
            .filter(e -> e.getSalary() > threshold)
            .count();
    }

    public List<String> getEmployeeNames() {
        return employees.stream()
            .map(Employee::getName)
            .collect(Collectors.toUnmodifiableList());
    }

    public Optional<Employee> findHighestPaid() {
        return employees.stream()
            .max(Comparator.comparingInt(Employee::getSalary));
    }

    // 必要に応じて防御的コピーを返す
    public List<Employee> getEmployeesSnapshot() {
        return List.copyOf(employees);
    }
}

// 呼び出し側はシンプル
int count = department.countEmployeesWithSalaryAbove(500000);
```

**コード例4: 設定オブジェクトのアクセス**

```typescript
// === LoD違反: 設定の深い階層に直接アクセス ===
function connectDatabase(config: AppConfig) {
  const host = config.database.connection.primary.host;
  const port = config.database.connection.primary.port;
  const timeout = config.database.connection.primary.timeoutMs;
  return new Database(`${host}:${port}`, { timeout });
}
// → config の内部構造に3段階依存
// → database, connection, primary のいずれかの構造変更で壊れる


// === LoD準拠: 必要なデータだけを受け取る ===
interface DatabaseConnectionInfo {
  host: string;
  port: number;
  timeoutMs: number;
}

function connectDatabase(connection: DatabaseConnectionInfo) {
  return new Database(`${connection.host}:${connection.port}`, {
    timeout: connection.timeoutMs
  });
}

// 呼び出し側で必要な情報を抽出して渡す
// config の構造変更は呼び出し側（1箇所）のみに影響
const db = connectDatabase(config.getDatabaseConnection());
```

**コード例5: null チェックの連鎖**

```python
# === LoD違反 + null地獄 ===
def get_manager_email(employee):
    if employee is not None:
        dept = employee.department
        if dept is not None:
            manager = dept.manager
            if manager is not None:
                contact = manager.contact
                if contact is not None:
                    return contact.email
    return "unknown@example.com"
# → 4段のnullチェック = 4つのクラスの内部構造を知っている
# → 追跡が困難、テスト組み合わせが 2^4 = 16パターン


# === LoD準拠: 各オブジェクトが責任を持つ ===
class Employee:
    def get_manager_email(self) -> str:
        """マネージャーのメールアドレスを取得（部門に委譲）"""
        if self.department is None:
            return "unknown@example.com"
        return self.department.get_manager_email()

class Department:
    def get_manager_email(self) -> str:
        """マネージャーのメールを取得（マネージャーに委譲）"""
        if self.manager is None:
            return "unknown@example.com"
        return self.manager.get_email()

class Manager:
    def get_email(self) -> str:
        """メールアドレスを取得"""
        return self.contact.email if self.contact else "unknown@example.com"

# 呼び出し: シンプルかつnull安全
email = employee.get_manager_email()
# → nullチェックは各クラスが自身の責任範囲で処理
# → テストは各クラスごとに独立して記述可能
```

### 2.2 Tell, Don't Ask 原則との関係

デメテルの法則は「Tell, Don't Ask（命令せよ、質問するな）」原則と密接に関連する。

```
  Ask（質問する）→ LoD違反しやすい
  ┌────────┐      ┌────────┐
  │ Caller │ ask →│ Object │
  └────┬───┘      └────────┘
       │ 質問の結果を元に判断して...
       │
  ┌────▼───┐
  │ 処理   │ ← 判断ロジックが呼び出し側に漏れる
  └────────┘

  Tell（命令する）→ LoD準拠しやすい
  ┌────────┐       ┌────────┐
  │ Caller │ tell →│ Object │
  └────────┘       └────┬───┘
                         │ 自分で判断して処理する
                    ┌────▼───┐
                    │ 処理   │ ← 判断ロジックがオブジェクト内に閉じる
                    └────────┘
```

**コード例6: Tell, Don't Ask の適用**

```python
# === Ask パターン（LoD違反しやすい） ===
class OrderProcessor:
    def process_discount(self, order: Order):
        # 顧客の情報を「聞いて」自分で判断する
        customer = order.customer
        if customer.membership_level == 'gold':
            if customer.total_purchases > 100000:
                discount = 0.15
            else:
                discount = 0.10
        elif customer.membership_level == 'silver':
            discount = 0.05
        else:
            discount = 0.0

        order.apply_discount(discount)
        # → 割引ロジックが OrderProcessor に漏れている
        # → Customer の membership_level の仕様変更で OrderProcessor が壊れる


# === Tell パターン（LoD準拠） ===
class OrderProcessor:
    def process_discount(self, order: Order):
        # 注文に「割引を計算して適用して」と命令する
        order.apply_membership_discount()
        # → OrderProcessor は割引の詳細を知らない

class Order:
    def apply_membership_discount(self):
        discount = self.customer.calculate_discount()
        self.total *= (1 - discount)

class Customer:
    def calculate_discount(self) -> float:
        """顧客自身が自分の割引率を知っている"""
        if self.membership_level == 'gold':
            return 0.15 if self.total_purchases > 100000 else 0.10
        elif self.membership_level == 'silver':
            return 0.05
        return 0.0
    # → 割引ロジックが Customer クラスに閉じている
    # → 仕様変更は Customer のみに影響
```

**コード例7: 複雑なビジネスロジックの委譲**

```typescript
// === LoD違反: 呼び出し側がすべての判断を行う ===
class ShippingService {
  calculateShippingCost(order: Order): number {
    // 注文の内部構造を深く知りすぎている
    const weight = order.items.reduce(
      (sum, item) => sum + item.product.weight * item.quantity, 0
    );
    const isOversized = order.items.some(
      item => item.product.dimensions.width > 120 ||
              item.product.dimensions.height > 120
    );
    const address = order.customer.shippingAddress;
    const isRemoteArea = address.prefecture === '沖縄' ||
                         address.prefecture === '北海道';

    let cost = weight * 10; // 基本運賃
    if (isOversized) cost *= 1.5;
    if (isRemoteArea) cost += 500;
    if (order.customer.membershipLevel === 'premium') cost *= 0.8;

    return cost;
  }
}
// → ShippingService が Order, Item, Product, Dimensions,
//   Customer, Address, MembershipLevel を知っている


// === LoD準拠: 各オブジェクトに判断を委譲 ===
class ShippingService {
  calculateShippingCost(order: Order): number {
    const shippingInfo = order.getShippingInfo();
    return this.calculateBaseCost(shippingInfo)
         * shippingInfo.sizeMultiplier
         + shippingInfo.remoteAreaSurcharge
         * shippingInfo.membershipDiscount;
  }

  private calculateBaseCost(info: ShippingInfo): number {
    return info.totalWeight * 10;
  }
}

// Order が必要な情報をまとめて提供
class Order {
  getShippingInfo(): ShippingInfo {
    return {
      totalWeight: this.calculateTotalWeight(),
      sizeMultiplier: this.hasOversizedItems() ? 1.5 : 1.0,
      remoteAreaSurcharge: this.customer.getRemoteAreaSurcharge(),
      membershipDiscount: this.customer.getMembershipDiscount(),
    };
  }

  private calculateTotalWeight(): number {
    return this.items.reduce(
      (sum, item) => sum + item.getWeight(), 0
    );
  }

  private hasOversizedItems(): boolean {
    return this.items.some(item => item.isOversized());
  }
}

// 各オブジェクトが自身の判断を担当
class OrderItem {
  getWeight(): number { return this.product.weight * this.quantity; }
  isOversized(): boolean { return this.product.isOversized(); }
}

class Product {
  isOversized(): boolean {
    return this.dimensions.width > 120 || this.dimensions.height > 120;
  }
}
```

---

## 3. 適用すべき場面と適用しない場面

### 3.1 適用判断マトリクス

| 適用すべき場面 | 理由 | 例 |
|--------------|------|-----|
| ドメインオブジェクト間のナビゲーション | 内部構造への依存を防ぐ | `order.customer.address.city` |
| 外部API/ライブラリの利用 | 変更影響を局所化 | `lib.getConfig().getModule().getSetting()` |
| レイヤー間の通信 | アーキテクチャ境界の維持 | Controller → Service → Repository |
| テストコードの安定性 | モック対象を最小化 | テストで5段のモックチェーンは危険 |

| 適用しない場面 | 理由 | 例 |
|--------------|------|-----|
| Fluent Interface / Builder | 同一オブジェクトの操作 | `builder.setName("x").setAge(20).build()` |
| データ転送オブジェクト (DTO) | 構造が契約の一部 | `response.data.user.name` |
| 内部DSL | ドット連鎖が表現力の源泉 | `select("name").from("users").where(...)` |
| Stream/LINQ 操作 | 関数型のパイプライン | `list.filter(...).map(...).reduce(...)` |
| Optional/Maybe チェーン | nullsafe なナビゲーション | `user?.address?.city?.name` |

**コード例8: Fluent Interface との区別**

```python
# これはLoD違反ではない！（Fluent Interface / メソッドチェーン）
query = (
    QueryBuilder()
    .select("name", "email")
    .from_table("users")
    .where("age > 18")
    .order_by("name")
    .limit(10)
    .build()
)
# 各メソッドは同じオブジェクト（または同じ型）を返す
# → 「友人の友人」ではなく「友人自身」に繰り返し話しかけている

# これもLoD違反ではない（Stream パイプライン）
result = (
    users
    .stream()
    .filter(lambda u: u.is_active())
    .map(lambda u: u.name)
    .sorted()
    .collect(to_list())
)
# データ変換のパイプラインであり、内部構造の探索ではない
```

### 3.2 判断フローチャート

```
  LoD適用の判断フロー

  ドット連鎖がある
    │
    ▼
  同一オブジェクトを返すか？ ─── Yes ──→ LoD違反ではない
  (Fluent/Builder)                       (Fluent Interface)
    │ No
    ▼
  DTOの構造アクセスか？ ─── Yes ──→ LoD違反ではない
                                    (DTOの構造は契約)
    │ No
    ▼
  Stream/LINQ操作か？ ─── Yes ──→ LoD違反ではない
                                   (関数型パイプライン)
    │ No
    ▼
  ドメインオブジェクトの
  内部構造を探索しているか？ ─── Yes ──→ LoD違反！
    │ No                                 委譲メソッドを追加
    ▼
  問題なし
```

---

## 4. デメテルの法則と関連原則

### 4.1 結合度の7段階との関係

デメテルの法則違反は、結合度の分類では「内容結合」または「スタンプ結合」に相当する。

| LoD状態 | 結合度レベル | 例 |
|---------|------------|-----|
| LoD違反（深いチェーン） | 内容結合に近い | `order.customer.address.city.zipCode` |
| LoD違反（1段の内部アクセス） | スタンプ結合 | `order.customer.name` |
| LoD準拠（委譲メソッド） | データ結合 | `order.getCustomerName()` |
| LoD準拠（イベント駆動） | メッセージ結合 | `eventBus.publish(event)` |

### 4.2 SOLID原則との関係

| SOLID原則 | デメテルの法則との関連 |
|-----------|---------------------|
| SRP | LoD準拠すると各クラスが自分の責任に関する判断を持つ |
| OCP | 委譲メソッドにより内部構造の変更が外部に波及しない |
| LSP | LoD準拠のインターフェースは置換可能性を高める |
| ISP | LoD準拠は最小限のインターフェースを提供することに繋がる |
| DIP | LoD準拠は抽象への依存を促進する |

### 4.3 情報隠蔽（Information Hiding）との関係

David Parnas（1972）が提唱した情報隠蔽の原則と、デメテルの法則は密接に関連する。

```
  情報隠蔽とデメテルの法則

  情報隠蔽: モジュールの設計判断（内部構造）を外部から隠す
  デメテルの法則: オブジェクトの内部構造をメソッド呼び出しで探索しない

  情報隠蔽が「何を隠すか」を定義し、
  デメテルの法則が「どう隠すか」の具体的ルールを提供する

  ┌─────────────────────────────────────────────┐
  │  情報隠蔽の原則（Parnas, 1972）              │
  │  「変更されやすい設計判断をモジュール内に    │
  │   隠蔽し、安定したインターフェースを提供する」│
  └──────────────────┬──────────────────────────┘
                      │ 具体化
                      ▼
  ┌─────────────────────────────────────────────┐
  │  デメテルの法則（Lieberherr, 1987）          │
  │  「メソッド内で呼び出してよいオブジェクトを  │
  │   直接の友人に制限する」                      │
  └─────────────────────────────────────────────┘
```

---

## 5. 高度な適用パターン

### 5.1 Wrapper/Facade による情報隠蔽

**コード例9: サードパーティライブラリのラッピング**

```python
# === LoD違反: サードパーティAPIの内部構造に依存 ===
class PaymentProcessor:
    def __init__(self):
        self.stripe = stripe

    def charge(self, customer_id: str, amount: int):
        # Stripe APIの内部構造に直接依存
        customer = self.stripe.Customer.retrieve(customer_id)
        default_source = customer.sources.data[0]  # 内部構造の探索
        charge = self.stripe.Charge.create(
            amount=amount,
            currency='jpy',
            source=default_source.id,
            customer=customer.id,
        )
        return charge.status == 'succeeded'
        # → Stripe API の構造変更で壊れる


# === LoD準拠: Adapterでラッピング ===
class StripePaymentAdapter:
    """Stripe APIとの結合をこのクラスに閉じ込める"""
    def __init__(self, api_key: str):
        stripe.api_key = api_key

    def charge_default_method(self, customer_id: str, amount: int) -> PaymentResult:
        """顧客のデフォルト決済方法で課金する"""
        try:
            intent = stripe.PaymentIntent.create(
                amount=amount,
                currency='jpy',
                customer=customer_id,
                confirm=True,
            )
            return PaymentResult(
                success=intent.status == 'succeeded',
                transaction_id=intent.id,
            )
        except stripe.error.CardError as e:
            return PaymentResult(success=False, error=str(e))


class PaymentProcessor:
    """決済処理 - Stripeの内部構造を知らない"""
    def __init__(self, payment_gateway: PaymentGateway):
        self.gateway = payment_gateway  # インターフェースに依存

    def charge(self, customer_id: str, amount: int) -> PaymentResult:
        return self.gateway.charge_default_method(customer_id, amount)
```

### 5.2 コンテキストオブジェクトパターン

**コード例10: 必要な情報をまとめて渡す**

```typescript
// === LoD違反: 長いチェーンで情報を収集 ===
function generateInvoice(order: Order): Invoice {
  const customerName = order.customer.profile.displayName;
  const billingAddress = order.customer.addresses.find(a => a.type === 'billing');
  const taxId = order.customer.taxInfo.registrationNumber;
  const items = order.items.map(i => ({
    name: i.product.name,
    price: i.product.price * i.quantity,
    taxRate: i.product.category.taxRate,
  }));
  // → order, customer, profile, addresses, taxInfo, items,
  //   product, category すべてを知っている
}


// === LoD準拠: コンテキストオブジェクトで必要情報をまとめる ===

// 請求書生成に必要な情報のみを集めた型
interface InvoiceContext {
  customerName: string;
  billingAddress: Address;
  taxRegistrationNumber: string;
  lineItems: InvoiceLineItem[];
}

interface InvoiceLineItem {
  productName: string;
  totalPrice: number;
  taxRate: number;
}

// Order がコンテキストを構築する責任を持つ
class Order {
  toInvoiceContext(): InvoiceContext {
    return {
      customerName: this.customer.getDisplayName(),
      billingAddress: this.customer.getBillingAddress(),
      taxRegistrationNumber: this.customer.getTaxRegistrationNumber(),
      lineItems: this.items.map(item => item.toInvoiceLineItem()),
    };
  }
}

class OrderItem {
  toInvoiceLineItem(): InvoiceLineItem {
    return {
      productName: this.product.getName(),
      totalPrice: this.product.getPrice() * this.quantity,
      taxRate: this.product.getTaxRate(),
    };
  }
}

// 請求書生成関数は InvoiceContext だけを知る
function generateInvoice(context: InvoiceContext): Invoice {
  // Order の内部構造を一切知らない
  return new Invoice(context);
}

// 使用
const invoice = generateInvoice(order.toInvoiceContext());
```

---

## 6. アンチパターン

### アンチパターン1: Middle Man の過剰生成

```java
// NG: LoD を過度に適用するとラッパーメソッドだらけになる
class Customer {
    private Address address;

    // 委譲メソッドが爆発的に増える
    public String getStreet() { return address.getStreet(); }
    public String getCity() { return address.getCity(); }
    public String getState() { return address.getState(); }
    public String getZipCode() { return address.getZipCode(); }
    public String getCountry() { return address.getCountry(); }
    public void setStreet(String s) { address.setStreet(s); }
    public void setCity(String c) { address.setCity(c); }
    // ... 延々と続く → Middle Man コードスメル

// OK: 意味のある抽象レベルで委譲する
class Customer {
    private Address address;

    // 全フィールドを個別に委譲するのではなく、
    // 意味のある操作を提供する
    public String getFormattedAddress() {
        return address.format();
    }

    public boolean isInDeliveryArea(DeliveryZone zone) {
        return zone.includes(address);
    }

    // Address 自体が必要な場合は、防御的コピーを返す
    public Address getAddressCopy() {
        return new Address(address); // 不変コピー
    }
}
```

### アンチパターン2: Feature Envy の裏返し

```python
# NG: LoD に従おうとして、無関係な責任を押し付ける
class Order:
    def format_customer_address_for_shipping_label(self) -> str:
        # これは Order の責任ではない！
        return f"{self.customer.name}\n{self.customer.address.full_address()}"

# OK: 適切なオブジェクトに責任を持たせる
class ShippingLabel:
    """配送ラベルの生成を担当するクラス"""
    def format(self, customer: Customer) -> str:
        return f"{customer.name}\n{customer.get_formatted_address()}"

# 使用
label = ShippingLabel()
formatted = label.format(order.customer)
# → Order は配送ラベルの知識を持たず、Customer へのアクセスのみ提供
```

### アンチパターン3: 過度なラッピングによるパフォーマンス低下

```python
# NG: 不必要な委譲レイヤーが多すぎる
class WidgetA:
    def get_value(self):
        return self.widget_b.get_value()

class WidgetB:
    def get_value(self):
        return self.widget_c.get_value()

class WidgetC:
    def get_value(self):
        return self.widget_d.get_value()

class WidgetD:
    def get_value(self):
        return self._actual_value
# → 4段の委譲。LoD準拠だが設計が歪んでいる

# OK: 設計を見直し、適切な粒度にする
# 過度な委譲は「設計の歪み」のサイン
# 本来の責任の所在を再考する
```

---

## 7. 静的解析によるLoD違反の検出

### 7.1 メトリクスと検出ルール

| メトリクス | 説明 | しきい値 |
|-----------|------|---------|
| ドットの連鎖数 | メソッドチェーンの段数 | 2段以上で警告 |
| Message Chain (Martin Fowler) | コードスメルとしての検出 | 3段以上で要検討 |
| Feature Envy | 他クラスのメソッドを過度に呼ぶ | 自クラス以上で警告 |
| CBO (Coupling Between Objects) | 依存先クラス数 | 10以上で警告 |

### 7.2 Linter ルールの設定例

```python
# === Python: pylint でのLoD違反検出 ===
# .pylintrc
# [DESIGN]
# max-args=5              # 引数が多いとLoD違反の可能性
# max-attributes=7        # フィールドが多いとLoD違反の可能性

# === カスタムルール: ドット連鎖の検出 ===
import ast

class LoDChecker(ast.NodeVisitor):
    """LoD違反の候補を検出するAST解析"""
    MAX_CHAIN_LENGTH = 2

    def visit_Attribute(self, node):
        chain_length = self._count_chain(node)
        if chain_length > self.MAX_CHAIN_LENGTH:
            print(
                f"Line {node.lineno}: "
                f"ドット連鎖が{chain_length}段 "
                f"(しきい値: {self.MAX_CHAIN_LENGTH}段)"
            )
        self.generic_visit(node)

    def _count_chain(self, node, depth=1):
        if isinstance(node.value, ast.Attribute):
            return self._count_chain(node.value, depth + 1)
        return depth
```

```typescript
// === TypeScript: ESLint カスタムルール ===
// eslint-plugin-demeter
module.exports = {
  rules: {
    'no-deep-chain': {
      create(context) {
        return {
          MemberExpression(node) {
            let depth = 0;
            let current = node;
            while (current.type === 'MemberExpression') {
              depth++;
              current = current.object;
            }
            if (depth > 2) {
              context.report({
                node,
                message: `ドット連鎖が${depth}段です。デメテルの法則に違反する可能性があります。`,
              });
            }
          },
        };
      },
    },
  },
};
```

---

## 8. 演習問題

### 演習1（基礎）: LoD違反の識別

以下のコードからLoD違反を見つけ、それぞれの理由を説明してください。

```python
class ReportGenerator:
    def generate(self, company):
        # (1)
        ceo_name = company.board.ceo.personal_info.full_name

        # (2)
        total = sum(d.budget for d in company.departments)

        # (3)
        report = Report()
        report.set_title(f"Annual Report for {company.name}")

        # (4)
        formatted = (
            ReportFormatter()
            .set_font("Arial")
            .set_size(12)
            .format(report)
        )

        # (5)
        company.departments[0].employees[0].salary

        return formatted
```

**期待される回答:**

```
(1) LoD違反: company → board → ceo → personal_info → full_name
    4段のドット連鎖。company.getCeoName() に委譲すべき。

(2) LoD違反: company.departments に直接アクセスし、
    さらに各departmentの budget にアクセス。
    company.getTotalBudget() に委譲すべき。

(3) LoD準拠: report はローカルで生成したオブジェクト（ルール3）。
    company.name は引数の直接のプロパティ（ルール2）。

(4) LoD準拠: ReportFormatter はローカルで生成（ルール3）。
    Fluent Interface なのでドット連鎖はOK。

(5) LoD違反: company → departments[0] → employees[0] → salary
    内部コレクションの内容に直接アクセス。
```

### 演習2（応用）: Tell, Don't Ask でリファクタリング

以下の Ask パターンのコードを Tell パターンにリファクタリングしてください。

```typescript
class NotificationSender {
  sendReminder(user: User) {
    if (user.preferences.notifications.email.enabled) {
      const email = user.contactInfo.primaryEmail;
      if (user.subscription.plan === 'premium') {
        this.emailService.sendPriority(email, 'Reminder', '...');
      } else {
        this.emailService.send(email, 'Reminder', '...');
      }
    }

    if (user.preferences.notifications.push.enabled) {
      const deviceToken = user.devices[0].pushToken;
      this.pushService.send(deviceToken, 'Reminder', '...');
    }
  }
}
```

**期待される回答:**

```typescript
// Tell パターン: 各オブジェクトに判断を委譲
class NotificationSender {
  sendReminder(user: User) {
    user.sendNotification({
      type: 'reminder',
      subject: 'Reminder',
      body: '...',
      emailService: this.emailService,
      pushService: this.pushService,
    });
  }
}

class User {
  sendNotification(params: NotificationParams) {
    if (this.shouldSendEmail()) {
      this.sendEmailNotification(params);
    }
    if (this.shouldSendPush()) {
      this.sendPushNotification(params);
    }
  }

  private shouldSendEmail(): boolean {
    return this.preferences.isEmailEnabled();
  }

  private sendEmailNotification(params: NotificationParams) {
    const email = this.contactInfo.getPrimaryEmail();
    if (this.subscription.isPremium()) {
      params.emailService.sendPriority(email, params.subject, params.body);
    } else {
      params.emailService.send(email, params.subject, params.body);
    }
  }

  private shouldSendPush(): boolean {
    return this.preferences.isPushEnabled();
  }

  private sendPushNotification(params: NotificationParams) {
    const token = this.getPreferredPushToken();
    params.pushService.send(token, params.subject, params.body);
  }
}
```

### 演習3（発展）: 設計判断の実践

以下のシナリオで、デメテルの法則を適用すべきか判断し、理由を述べてください。

```
シナリオA: GraphQLのリゾルバ
query {
  user(id: "123") {
    profile { displayName }
    orders {
      items { product { name, price } }
    }
  }
}

シナリオB: ORMのリレーション
user = User.find(123)
orders = user.orders.where(status: "active").includes(:items)

シナリオC: ドメインロジック
def calculate_bonus(employee):
    base_salary = employee.contract.compensation.base_salary
    department_budget = employee.department.budget_allocation.bonus_pool
    return min(base_salary * 0.1, department_budget / len(employee.department.members))
```

**期待される回答:**

```
シナリオA: LoD適用しない
→ GraphQLはクエリ言語であり、クライアントがデータの
  構造を宣言的に指定する。これはLoD違反ではなく
  APIの設計パターンとして適切。

シナリオB: LoD適用しない（条件付き）
→ ORMのリレーションアクセスはDSLとして許容される。
  ただし、ドメインロジック内でuser.orders.items.productの
  ようなチェーンが出現したら、専用のクエリメソッドを提供すべき。

シナリオC: LoD適用する（違反あり）
→ 3段のドット連鎖が2箇所。リファクタリング:
  employee.get_base_salary() + employee.get_available_bonus_pool()
  各オブジェクトが自身の責任範囲で値を提供する。
```

---

## 9. FAQ

### Q1: DTOやデータクラスにもデメテルの法則を適用すべきか？

DTOは「行動」を持たないデータの入れ物であり、その構造自体が契約の一部。DTO内部のフィールドにドットでアクセスするのはLoD違反とはみなさない。ただし、DTOを受け取る側のドメインロジックでは、必要なデータだけを引数で受け取る設計にすべき。

```python
# DTOのアクセスはOK
user_dto = api_response.data.user
name = user_dto.profile.display_name  # DTOなのでOK

# ドメインロジックではDTOの構造に依存しない
def greet_user(display_name: str) -> str:  # プリミティブ値で受け取る
    return f"Hello, {display_name}!"

greet_user(user_dto.profile.display_name)  # 呼び出し側でDTOを展開
```

### Q2: デメテルの法則に「法則」と名付けるのは大げさではないか？

実際に多くの開発者が「法則」ではなく「ガイドライン」として扱っている。盲目的に従うのではなく、**結合度を下げる**という目的を理解した上で判断することが重要。

Robert C. Martinも「LoD は厳密なルールではなく、有用なヒューリスティック（経験則）だ」と述べている。重要なのはドット連鎖の数を数えることではなく、「このコードは他のオブジェクトの内部構造を知りすぎていないか？」を常に問うことである。

### Q3: 関数型プログラミングではデメテルの法則はどう適用するか？

関数型では、パイプライン演算子やmap/filterチェーンがドット連鎖に見えるが、これらはデータ変換のパイプラインでありLoD違反ではない。関数型で重要なのは「関数が知るべき型を最小限にする」こと。パラメトリック多相（ジェネリクス）を活用して、関数が具体的な型に依存しない設計を心がける。

```haskell
-- 関数型: パイプラインはLoD違反ではない
result = users
  |> filter isActive
  |> map getName
  |> sort
  |> take 10
-- 各関数はデータを変換するだけで、内部構造を探索していない

-- LoD違反に相当するもの
getUserCity :: User -> String
getUserCity user = city (address (profile user))
-- → User の内部構造（profile → address → city）を知りすぎている

-- LoD準拠
getUserCity :: User -> String
getUserCity = getCity . getAddress . getProfile
-- → ただし、これは関数合成として許容される場合が多い
-- → 重要なのは、getUserCity が User モジュールから公開されること
```

### Q4: マイクロサービスのAPI設計でもLoDは適用すべきか？

マイクロサービスでは、API設計にLoDの精神を適用すべきだが、実装方法は異なる。

```
良いAPI設計（LoD準拠の精神）:
GET /orders/{id}/shipping-cost
→ クライアントは注文の内部構造を知らずに配送費を取得

悪いAPI設計（LoD違反の精神）:
GET /orders/{id} → customer_id を取得
GET /customers/{customer_id} → address_id を取得
GET /addresses/{address_id} → zip_code を取得
→ クライアントが複数APIを順序依存で呼ぶ必要がある
```

### Q5: LoDを導入する際の優先度は？

以下の順序で段階的に導入することを推奨する。

1. **サービス層のAPI境界**: 最も効果が高い。外部に公開するインターフェースを最小限にする
2. **外部ライブラリとの結合**: Adapterパターンで隔離
3. **ドメインオブジェクト間のナビゲーション**: Train Wreckを委譲メソッドに変換
4. **テストコードの安定化**: モックチェーンの削減
5. **レガシーコードの段階的改善**: 新規コードから適用し、変更のあったコードを順次改善

---

## まとめ

| 項目 | 内容 |
|------|------|
| 核心 | 「直接の友人とだけ話せ」 |
| 目的 | 結合度の低減、変更影響の局所化 |
| 正式な定義 | メソッドM内で呼べるのは: self, param, local, field のメソッドのみ |
| 違反の兆候 | ドット連鎖（Train Wreck）、深いnullチェック |
| 改善方法 | 委譲メソッド、Tell Don't Ask、コンテキストオブジェクト、Adapter |
| 適用しない場面 | Fluent Interface、DTO、Stream操作、内部DSL |
| 過度な適用の弊害 | Middle Man、ラッパーメソッドの爆発 |
| 関連原則 | Tell Don't Ask、情報隠蔽、SRP、DIP |

| 判断基準 | 質問 |
|---------|------|
| LoD違反か？ | 「このコードは友人の友人の内部構造を知っているか？」 |
| 改善すべきか？ | 「チェーン上のクラスが変更された時、このコードも壊れるか？」 |
| 委譲すべきか？ | 「この判断ロジックは、どのオブジェクトの責任か？」 |
| 過度な適用か？ | 「委譲メソッドが爆発的に増えていないか？」 |

---

## 次に読むべきガイド

- [結合度と凝集度](./03-coupling-cohesion.md) ── デメテルの法則が解決する問題の背景
- [関数設計](../01-practices/01-functions.md) ── 引数設計でのLoD適用
- [コードスメル](../02-refactoring/00-code-smells.md) ── Feature Envy、Middle Man との関係
- [リファクタリング技法](../02-refactoring/01-refactoring-catalog.md) ── Extract Method、Move Method の手順
- [デザインパターン: Facade](../../design-patterns-guide/docs/01-structural/02-facade.md) ── 結合度を下げるパターン

---

## 参考文献

1. **Karl J. Lieberherr, Ian M. Holland** "Assuring Good Style for Object-Oriented Programs" IEEE Software, 1989 ── デメテルの法則の原論文
2. **Robert C. Martin** 『Clean Code: A Handbook of Agile Software Craftsmanship』 Prentice Hall, 2008 (Chapter 6: Objects and Data Structures) ── Train Wreck の解説
3. **Martin Fowler** 『Refactoring: Improving the Design of Existing Code』 Addison-Wesley, 2018 ── Middle Man smell、Feature Envy の改善手法
4. **David Parnas** "On the criteria to be used in decomposing systems into modules" Communications of the ACM, 1972 ── 情報隠蔽の原則
5. **Andrew Hunt, David Thomas** 『The Pragmatic Programmer: From Journeyman to Master』 Addison-Wesley, 1999 ── "Don't talk to strangers" の実践的解説
6. **Karl Lieberherr** "Demeter Project" Northeastern University ── プロジェクトの公式ドキュメント
7. **Pragmatic Dave Thomas** "Tell, Don't Ask" ── Tell Don't Ask 原則の解説
8. **Eric Evans** 『Domain-Driven Design: Tackling Complexity in the Heart of Software』 Addison-Wesley, 2003 ── Bounded Context と情報隠蔽
