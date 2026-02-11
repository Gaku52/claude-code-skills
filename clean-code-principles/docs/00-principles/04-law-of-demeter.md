# デメテルの法則 ── 最小知識の原則

> デメテルの法則（Law of Demeter / LoD）は「直接の友人とだけ話せ」という設計原則である。オブジェクトが知るべき範囲を最小限に保つことで、結合度を下げ、変更に強いシステムを構築する。

---

## この章で学ぶこと

1. **デメテルの法則の定義と目的** ── 「ドットの連鎖を避けよ」の本質を理解する
2. **違反パターンの検出** ── Train Wreck（列車事故）コードを見抜く目を養う
3. **適用のバランス** ── 過度な適用を避け、適切な範囲で活用する判断力を身につける

---

## 1. デメテルの法則とは

### 1.1 正式な定義

```
+-----------------------------------------------------------+
|  デメテルの法則 (Law of Demeter, 1987)                    |
|  ─────────────────────────────────────────────────         |
|  あるメソッド M 内で、M が呼び出してよいのは:             |
|                                                           |
|  1. M が属するオブジェクト自身のメソッド                  |
|  2. M の引数として渡されたオブジェクトのメソッド          |
|  3. M 内で生成されたオブジェクトのメソッド                |
|  4. M が属するオブジェクトのフィールドのメソッド          |
|                                                           |
|  つまり「友人の友人」のメソッドを呼んではいけない          |
+-----------------------------------------------------------+
```

### 1.2 直感的な理解

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

**コード例1: Train Wreck の修正**

```python
# LoD違反: 深いドット連鎖
class OrderProcessor:
    def calculate_shipping(self, order):
        # Order → Customer → Address → City → ZipCode と4段掘り下げ
        zip_code = order.customer.address.city.zip_code
        if zip_code.startswith('100'):
            return 500  # 都内は500円
        return 1000

# LoD準拠: 各オブジェクトが自分の責任範囲で応答
class Order:
    def get_shipping_zip_code(self) -> str:
        return self.customer.get_zip_code()

class Customer:
    def get_zip_code(self) -> str:
        return self.address.get_zip_code()

class Address:
    def get_zip_code(self) -> str:
        return self.city.zip_code

class OrderProcessor:
    def calculate_shipping(self, order):
        zip_code = order.get_shipping_zip_code()  # 1ドットで完結
        if zip_code.startswith('100'):
            return 500
        return 1000
```

**コード例2: コレクションの操作**

```java
// LoD違反: 内部コレクションを外部に公開
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

// LoD準拠: 問い合わせメソッドを提供
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
}

// 呼び出し側はシンプル
int count = department.countEmployeesWithSalaryAbove(500000);
```

**コード例3: 設定オブジェクトのアクセス**

```typescript
// LoD違反: 設定の深い階層に直接アクセス
function connectDatabase(config: AppConfig) {
  const host = config.database.connection.primary.host;
  const port = config.database.connection.primary.port;
  return new Database(`${host}:${port}`);
}

// LoD準拠: 必要なデータだけを受け取る
interface DatabaseConnectionInfo {
  host: string;
  port: number;
}

function connectDatabase(connection: DatabaseConnectionInfo) {
  return new Database(`${connection.host}:${connection.port}`);
}

// 呼び出し側で必要な情報を抽出して渡す
const db = connectDatabase(config.getDatabaseConnection());
```

**コード例4: null チェックの連鎖**

```python
# LoD違反 + null地獄
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

# LoD準拠: 各オブジェクトが責任を持つ
class Employee:
    def get_manager_email(self) -> str:
        return self.department.get_manager_email()

class Department:
    def get_manager_email(self) -> str:
        if self.manager is None:
            return "unknown@example.com"
        return self.manager.get_email()

class Manager:
    def get_email(self) -> str:
        return self.contact.email if self.contact else "unknown@example.com"

# 呼び出し: シンプルかつnull安全
email = employee.get_manager_email()
```

**コード例5: Fluent Interface との区別**

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
```

---

## 3. 適用すべき場面と適用しない場面

| 適用すべき場面 | 理由 |
|--------------|------|
| ドメインオブジェクト間のナビゲーション | 内部構造への依存を防ぐ |
| 外部API/ライブラリの利用 | 変更影響を局所化 |
| レイヤー間の通信 | アーキテクチャ境界の維持 |

| 適用しない場面 | 理由 |
|--------------|------|
| Fluent Interface / Builder | 同一オブジェクトの操作 |
| データ転送オブジェクト (DTO) | 構造が契約の一部 |
| 内部DSL | ドット連鎖が表現力の源泉 |
| Stream/LINQ 操作 | 関数型のパイプライン |

---

## 4. アンチパターン

### アンチパターン1: Middle Man の過剰生成

```java
// LoD を過度に適用するとラッパーメソッドだらけになる
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
```

### アンチパターン2: Feature Envy の裏返し

```python
# LoD に従おうとして、無関係な責任を押し付ける
class Order:
    def format_customer_address_for_shipping_label(self) -> str:
        # これは Order の責任ではない
        return f"{self.customer.name}\n{self.customer.address.full_address()}"

# 改善: 適切なオブジェクトに責任を持たせる
class ShippingLabel:
    def format(self, customer: Customer) -> str:
        return f"{customer.name}\n{customer.get_formatted_address()}"
```

---

## 5. FAQ

### Q1: DTOやデータクラスにもデメテルの法則を適用すべきか？

DTOは「行動」を持たないデータの入れ物であり、その構造自体が契約の一部。DTO内部のフィールドにドットでアクセスするのはLoD違反とはみなさない。ただし、DTOを受け取る側のドメインロジックでは、必要なデータだけを引数で受け取る設計にすべき。

### Q2: デメテルの法則に「法則」と名付けるのは大げさではないか？

実際に多くの開発者が「法則」ではなく「ガイドライン」として扱っている。盲目的に従うのではなく、**結合度を下げる**という目的を理解した上で判断することが重要。

### Q3: 関数型プログラミングではデメテルの法則はどう適用するか？

関数型では、パイプライン演算子やmap/filterチェーンがドット連鎖に見えるが、これらはデータ変換のパイプラインでありLoD違反ではない。関数型で重要なのは「関数が知るべき型を最小限にする」こと。パラメトリック多相（ジェネリクス）を活用して、関数が具体的な型に依存しない設計を心がける。

---

## まとめ

| 項目 | 内容 |
|------|------|
| 核心 | 「直接の友人とだけ話せ」 |
| 目的 | 結合度の低減、変更影響の局所化 |
| 違反の兆候 | ドット連鎖（Train Wreck）、深いnullチェック |
| 改善方法 | 委譲メソッド、Tell Don't Ask、データの直接渡し |
| 適用しない場面 | Fluent Interface、DTO、Stream操作 |
| 過度な適用の弊害 | Middle Man、ラッパーメソッドの爆発 |

---

## 次に読むべきガイド

- [結合度と凝集度](./03-coupling-cohesion.md) ── デメテルの法則が解決する問題の背景
- [関数設計](../01-practices/01-functions.md) ── 引数設計でのLoD適用
- [コードスメル](../02-refactoring/00-code-smells.md) ── Feature Envy、Middle Man との関係

---

## 参考文献

1. **Karl J. Lieberherr, Ian M. Holland** "Assuring Good Style for Object-Oriented Programs" IEEE Software, 1989
2. **Robert C. Martin** 『Clean Code: A Handbook of Agile Software Craftsmanship』 Prentice Hall, 2008 (Chapter 6: Objects and Data Structures)
3. **Martin Fowler** 『Refactoring: Improving the Design of Existing Code』 Addison-Wesley, 2018 (Middle Man smell)
