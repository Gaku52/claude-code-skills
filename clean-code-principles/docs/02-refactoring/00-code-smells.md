# コードスメル ── Long Method、God Class、その他の警告サイン

> コードスメルとは、コードの表面に現れる「何かがおかしい」という兆候である。バグではないが、設計上の問題を示唆する。スメルを素早く検出し、適切なリファクタリングに繋げる能力は、ソフトウェア品質を維持する上で不可欠である。

---

## この章で学ぶこと

1. **主要なコードスメルの分類** ── Martin Fowler の分類に基づくスメル一覧を理解する
2. **スメルの検出方法** ── 自動ツールとコードレビューでの検出技法を身につける
3. **各スメルに対応するリファクタリング** ── スメルから適切なリファクタリングへの対応表を習得する

---

## 1. コードスメルの分類体系

```
+-----------------------------------------------------------+
|  コードスメルの5分類 (Martin Fowler)                       |
+-----------------------------------------------------------+
| 1. 肥大化 (Bloaters)                                      |
|    Long Method, Large Class, Long Parameter List           |
+-----------------------------------------------------------+
| 2. 乱用 (Object-Orientation Abusers)                      |
|    Switch Statements, Refused Bequest, Parallel Hierarchy  |
+-----------------------------------------------------------+
| 3. 変更妨害 (Change Preventers)                            |
|    Divergent Change, Shotgun Surgery, Parallel Hierarchy   |
+-----------------------------------------------------------+
| 4. 不要物 (Dispensables)                                   |
|    Dead Code, Speculative Generality, Lazy Class           |
+-----------------------------------------------------------+
| 5. 結合過多 (Couplers)                                     |
|    Feature Envy, Inappropriate Intimacy, Message Chains    |
+-----------------------------------------------------------+
```

```
  スメルの影響度マトリクス

  影響度: 大
    ^
    | God      Shotgun
    | Class    Surgery
    |
    | Feature  Long
    | Envy     Method
    |
    | Dead     Comments
    | Code     (excessive)
    +──────────────────────> 検出しやすさ
   難しい                  簡単
```

---

## 2. 主要なコードスメル

### 2.1 Long Method（長すぎるメソッド）

**コード例1: Long Method の検出と改善**

```python
# スメル: 1つのメソッドが50行以上
def process_order(order_data: dict) -> OrderResult:
    # バリデーション (10行)
    if not order_data.get('items'):
        raise ValidationError("商品が選択されていません")
    if not order_data.get('customer_id'):
        raise ValidationError("顧客情報がありません")
    for item in order_data['items']:
        if item['quantity'] <= 0:
            raise ValidationError(f"数量が不正: {item['name']}")
        product = db.query("SELECT stock FROM products WHERE id = %s", item['id'])
        if product.stock < item['quantity']:
            raise ValidationError(f"在庫不足: {item['name']}")

    # 合計計算 (15行)
    subtotal = 0
    for item in order_data['items']:
        price = item['unit_price'] * item['quantity']
        if item.get('discount_code'):
            discount = db.query("SELECT rate FROM discounts ...")
            price *= (1 - discount.rate)
        subtotal += price
    tax = subtotal * 0.10
    shipping = 500 if subtotal < 5000 else 0
    total = subtotal + tax + shipping

    # DB保存 (10行) ...
    # メール送信 (10行) ...
    # 分析トラッキング (5行) ...


# 改善: 意図ごとに関数を分離
def process_order(order_data: dict) -> OrderResult:
    validate_order(order_data)
    pricing = calculate_order_pricing(order_data['items'])
    order = save_order(order_data, pricing)
    send_order_confirmation(order)
    track_order_analytics(order)
    return OrderResult.success(order)
```

### 2.2 God Class（神クラス）

```
  God Class の症状

  ┌────────────────────────────────────────────┐
  │  ApplicationManager                        │
  │  ────────────────────────────              │
  │  - userRepository                          │
  │  - orderRepository                         │
  │  - paymentGateway                          │
  │  - emailService                            │
  │  - cacheManager                            │
  │  - logger                                  │
  │  ────────────────────────────              │
  │  + authenticateUser()                      │
  │  + createOrder()                           │
  │  + processPayment()                        │
  │  + sendEmail()                             │
  │  + clearCache()                            │
  │  + generateReport()                        │
  │  + ... (50+ methods)                       │
  └────────────────────────────────────────────┘
       ↓ Extract Class で分離 ↓
  ┌──────────┐ ┌──────────┐ ┌──────────┐
  │ AuthSvc   │ │ OrderSvc  │ │ EmailSvc  │
  │ ────────  │ │ ────────  │ │ ────────  │
  │ auth()    │ │ create()  │ │ send()    │
  │ logout()  │ │ cancel()  │ │ template()│
  └──────────┘ └──────────┘ └──────────┘
```

### 2.3 Feature Envy（他クラスへの羨望）

**コード例2: Feature Envy の検出と改善**

```java
// スメル: OrderPrinter が Order の内部データを過度に使用
class OrderPrinter {
    String formatOrder(Order order) {
        StringBuilder sb = new StringBuilder();
        sb.append("注文番号: ").append(order.getId()).append("\n");
        sb.append("顧客名: ").append(order.getCustomer().getName()).append("\n");
        sb.append("日付: ").append(order.getDate().format(DateTimeFormatter.ISO_DATE)).append("\n");

        double subtotal = 0;
        for (OrderItem item : order.getItems()) {
            double lineTotal = item.getPrice() * item.getQuantity();
            subtotal += lineTotal;
            sb.append(String.format("  %s x%d = %.0f円\n",
                item.getName(), item.getQuantity(), lineTotal));
        }
        double tax = subtotal * 0.10;
        sb.append(String.format("小計: %.0f円\n税: %.0f円\n合計: %.0f円\n",
            subtotal, tax, subtotal + tax));
        return sb.toString();
    }
}

// 改善: ロジックを Order に移動
class Order {
    public String format() {
        StringBuilder sb = new StringBuilder();
        sb.append("注文番号: ").append(this.id).append("\n");
        sb.append("顧客名: ").append(this.customer.getName()).append("\n");
        sb.append("日付: ").append(this.date.format(DateTimeFormatter.ISO_DATE)).append("\n");
        this.items.forEach(item -> sb.append(item.formatLine()));
        sb.append(this.formatTotals());
        return sb.toString();
    }
}
```

**コード例3: Data Clumps（データの群れ）**

```python
# スメル: 同じパラメータ群が繰り返し登場
def create_user(first_name, last_name, street, city, zip_code, country): ...
def update_address(user_id, street, city, zip_code, country): ...
def validate_address(street, city, zip_code, country): ...
def format_address(street, city, zip_code, country): ...

# 改善: データクラスに抽出
@dataclass
class Address:
    street: str
    city: str
    zip_code: str
    country: str

    def validate(self) -> bool: ...
    def format(self) -> str: ...

def create_user(first_name: str, last_name: str, address: Address): ...
def update_address(user_id: str, address: Address): ...
```

**コード例4: Primitive Obsession（プリミティブ偏執）**

```typescript
// スメル: ドメイン概念をプリミティブ型で表現
function processPayment(
  amount: number,       // 円？ドル？
  currency: string,     // "JPY"? "jpy"? "円"?
  email: string,        // バリデーション済み？
  cardNumber: string    // マスク済み？
): boolean { ... }

// 改善: 値オブジェクトで型安全性を確保
class Money {
  constructor(
    readonly amount: number,
    readonly currency: Currency
  ) {
    if (amount < 0) throw new Error("金額は正の数");
  }

  add(other: Money): Money {
    if (this.currency !== other.currency)
      throw new CurrencyMismatchError();
    return new Money(this.amount + other.amount, this.currency);
  }
}

class Email {
  readonly value: string;
  constructor(value: string) {
    if (!EMAIL_REGEX.test(value))
      throw new InvalidEmailError(value);
    this.value = value;
  }
}

function processPayment(amount: Money, email: Email, card: CreditCard): PaymentResult { ... }
```

**コード例5: Switch Statements（条件分岐の増殖）**

```python
# スメル: 型による分岐が複数箇所に散在
def calculate_area(shape):
    if shape.type == 'circle':
        return math.pi * shape.radius ** 2
    elif shape.type == 'rectangle':
        return shape.width * shape.height
    elif shape.type == 'triangle':
        return shape.base * shape.height / 2

def draw(shape):
    if shape.type == 'circle':
        draw_circle(shape)
    elif shape.type == 'rectangle':
        draw_rectangle(shape)
    elif shape.type == 'triangle':
        draw_triangle(shape)

# 改善: ポリモーフィズムで置換
class Shape(ABC):
    @abstractmethod
    def area(self) -> float: ...
    @abstractmethod
    def draw(self) -> None: ...

class Circle(Shape):
    def area(self) -> float:
        return math.pi * self.radius ** 2
    def draw(self) -> None:
        draw_circle(self)
```

---

## 3. スメルとリファクタリングの対応表

| コードスメル | 対応するリファクタリング |
|------------|----------------------|
| Long Method | Extract Method |
| God Class | Extract Class, Move Method |
| Feature Envy | Move Method, Move Field |
| Data Clumps | Extract Class, Introduce Parameter Object |
| Primitive Obsession | Replace Primitive with Object |
| Switch Statements | Replace Conditional with Polymorphism |
| Shotgun Surgery | Move Method, Inline Class |
| Dead Code | Safe Delete |
| Long Parameter List | Introduce Parameter Object |
| Message Chains | Hide Delegate |

| 自動検出ツール | 対応言語 |
|--------------|---------|
| SonarQube | 多言語対応 |
| PMD | Java |
| pylint / flake8 | Python |
| ESLint | JavaScript/TypeScript |
| RuboCop | Ruby |

---

## 4. アンチパターン

### アンチパターン1: スメルの放置（割れ窓理論）

```
最初の放置 → 「まあいいか」→ 追加の放置 → 「前からこうだし」
→ 品質の雪崩的低下 → リファクタリング不能状態
```

### アンチパターン2: 一度にすべてを直す

```
全スメルを一括修正しようとする
→ 巨大な変更セット → レビュー不能 → バグ混入リスク
→ 改善: 1スメル1プルリクエストで段階的に修正
```

---

## 5. FAQ

### Q1: コードスメルは必ず修正すべきか？

いいえ。スメルは「調査すべき兆候」であり、必ずしもリファクタリングが必要とは限らない。変更頻度が低い箇所や使い捨てコードは放置してもよい。修正の優先度は「変更頻度 x 影響範囲」で判断する。

### Q2: チームでスメルの基準が異なる場合はどうするか？

1. SonarQube等の静的解析ツールで客観的基準を設定
2. コーディング規約に閾値を明記（例: メソッド30行以下）
3. コードレビューのチェックリストに含める
4. 定期的な「テクニカルレビュー会」で認識を統一

### Q3: レガシーコードのスメルはどこから手をつけるべきか？

**変更頻度が高いファイルから**。git logで変更回数を分析し、頻繁に触るファイルのスメルを優先的に解消する。触らないファイルのスメルは放置してよい。

---

## まとめ

| 分類 | 代表的スメル | 危険度 | 検出しやすさ |
|------|------------|--------|------------|
| 肥大化 | Long Method, God Class | 高 | 高 |
| 結合過多 | Feature Envy, Message Chains | 高 | 中 |
| 変更妨害 | Shotgun Surgery | 高 | 低 |
| 乱用 | Switch Statements | 中 | 高 |
| 不要物 | Dead Code | 低 | 高 |

---

## 次に読むべきガイド

- [リファクタリング技法](./01-refactoring-techniques.md) ── スメルを解消する具体的手法
- [レガシーコード](./02-legacy-code.md) ── スメルだらけのコードとの向き合い方
- [技術的負債](./03-technical-debt.md) ── スメルの放置がもたらす負債

---

## 参考文献

1. **Martin Fowler** 『Refactoring: Improving the Design of Existing Code』 Addison-Wesley, 2018 (2nd Edition)
2. **Robert C. Martin** 『Clean Code: A Handbook of Agile Software Craftsmanship』 Prentice Hall, 2008 (Chapter 17: Smells and Heuristics)
3. **Joshua Kerievsky** 『Refactoring to Patterns』 Addison-Wesley, 2004
