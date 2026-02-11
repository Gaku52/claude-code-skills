# 関数設計 ── 単一責任・引数・副作用

> 関数はプログラムの基本構成要素である。小さく、1つのことだけを行い、名前が意図を伝える関数は、システム全体の可読性と保守性を決定づける。

---

## この章で学ぶこと

1. **関数の単一責任** ── 1つの関数が1つのことだけを行う設計を理解する
2. **引数設計の原則** ── 引数の数・順序・型の最適化方法を身につける
3. **副作用の管理** ── 予測可能で安全な関数を書くための技法を習得する

---

## 1. 関数の基本原則

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
# 悪い: 抽象レベルが混在
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


# 良い: 各関数が同じ抽象レベル
def process_order(order: Order) -> OrderResult:
    validate_order(order)
    total = calculate_total(order.items)
    payment_result = charge_payment(order.customer, total)
    return create_order_result(order, payment_result)

def calculate_total(items: list[OrderItem]) -> Money:
    subtotal = sum(calculate_item_price(item) for item in items)
    tax = calculate_tax(subtotal)
    return subtotal + tax

def calculate_item_price(item: OrderItem) -> Money:
    base_price = item.unit_price * item.quantity
    discount = get_discount_rate(item.discount_code)
    return base_price * (1 - discount)
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

**コード例2: 引数オブジェクトパターン**

```typescript
// 悪い: 引数が多すぎる
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

// 良い: 引数オブジェクトで構造化
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

**コード例3: フラグ引数の排除**

```python
# 悪い: ブール引数で動作を切り替え（SRP違反の兆候）
def create_report(data, is_pdf: bool):
    if is_pdf:
        return generate_pdf_report(data)
    else:
        return generate_html_report(data)

# 良い: 関数を分離
def create_pdf_report(data: ReportData) -> bytes:
    return generate_pdf(data)

def create_html_report(data: ReportData) -> str:
    return generate_html(data)

# さらに良い: 戦略パターンで拡張性を確保
class ReportGenerator(Protocol):
    def generate(self, data: ReportData) -> Report: ...

class PdfReportGenerator:
    def generate(self, data: ReportData) -> PdfReport: ...

class HtmlReportGenerator:
    def generate(self, data: ReportData) -> HtmlReport: ...
```

---

## 3. 副作用の管理

**コード例4: 隠れた副作用の排除**

```java
// 悪い: 名前からは読み取れない副作用がある
public boolean checkPassword(String userName, String password) {
    User user = userRepository.findByName(userName);
    if (user == null) return false;

    if (user.getPassword().equals(encrypt(password))) {
        Session.initialize();  // 隠れた副作用！パスワードチェックなのにセッションを初期化
        return true;
    }
    return false;
}

// 良い: 副作用を分離し、名前で意図を明示
public boolean isPasswordValid(String userName, String password) {
    User user = userRepository.findByName(userName);
    if (user == null) return false;
    return user.getPassword().equals(encrypt(password));
}

public AuthResult authenticateAndCreateSession(String userName, String password) {
    if (!isPasswordValid(userName, password)) {
        return AuthResult.failure("認証失敗");
    }
    Session session = sessionManager.createSession(userName);
    return AuthResult.success(session);
}
```

### 3.1 コマンド/クエリ分離 (CQS)

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

**コード例5: CQSの適用**

```python
# CQS違反: 1つのメソッドが変更と取得を同時に行う
class Stack:
    def pop(self):
        """要素を削除して返す（変更+取得の混在）"""
        item = self.items[-1]
        self.items.pop()
        return item

# CQS準拠: コマンドとクエリを分離
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
```

---

## 4. ガード節と早期リターン

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

## 5. アンチパターン

### アンチパターン1: God Function（巨大関数）

```python
# アンチパターン: 1つの関数で全てを処理（200行超）
def process_everything(request):
    # バリデーション (30行)
    # データ変換 (40行)
    # ビジネスロジック (60行)
    # DB操作 (30行)
    # 外部API呼び出し (20行)
    # レスポンス生成 (20行)
    pass
```

### アンチパターン2: Output引数（出力パラメータ）

```java
// アンチパターン: 引数を変更して結果を返す
void calculateTotal(Order order, Money result) {
    result.setAmount(order.getSubtotal() + order.getTax());
}

// 改善: 戻り値で返す
Money calculateTotal(Order order) {
    return order.getSubtotal().add(order.getTax());
}
```

---

## 6. FAQ

### Q1: 関数は何行以下にすべきか？

Robert C. Martin は「4〜5行が理想」と述べているが、実践的には**20行以下**が目安。重要なのは行数ではなく「1つのことだけをしているか」。20行でも1つの責任なら問題なく、5行でも複数の責任が混在していればリファクタリング対象。

### Q2: 引数にnullを渡すのは許容されるか？

原則として**null引数は避けるべき**。Optional型、デフォルト値、メソッドオーバーロードで代替する。null引数はNullPointerExceptionの温床であり、呼び出し側にnullチェックの責任を押し付ける。

### Q3: privateメソッドが増えすぎた場合はどうすべきか？

privateメソッドが多いクラスは、**隠れたクラスが存在する兆候**。関連するprivateメソッド群を新しいクラスに抽出（Extract Class）することで、各クラスの凝集度が上がる。

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

---

## 次に読むべきガイド

- [エラーハンドリング](./02-error-handling.md) ── 関数のエラー処理設計
- [テスト原則](./04-testing-principles.md) ── テストしやすい関数の条件
- [関数型原則](../03-practices-advanced/02-functional-principles.md) ── 純粋関数と参照透過性

---

## 参考文献

1. **Robert C. Martin** 『Clean Code: A Handbook of Agile Software Craftsmanship』 Prentice Hall, 2008 (Chapter 3: Functions)
2. **Bertrand Meyer** 『Object-Oriented Software Construction』 Prentice Hall, 1997 (Command-Query Separation)
3. **Martin Fowler** 『Refactoring: Improving the Design of Existing Code』 Addison-Wesley, 2018 (Extract Function, Introduce Parameter Object)
