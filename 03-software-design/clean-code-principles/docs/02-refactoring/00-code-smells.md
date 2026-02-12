# コードスメル ── Long Method、God Class、その他の警告サイン

> コードスメルとは、コードの表面に現れる「何かがおかしい」という兆候である。バグではないが、設計上の問題を示唆する。スメルを素早く検出し、適切なリファクタリングに繋げる能力は、ソフトウェア品質を維持する上で不可欠である。Martin Fowler は『Refactoring』第2版で「スメルは設計品質の温度計」であると述べ、Kent Beck は「スメルが発する "ここを見て" というシグナルに耳を傾けよ」と説いた。本章では Martin Fowler の5分類を出発点に、22種以上のスメルを体系的に解説し、自動検出ツール・レビュー技法・段階的なリファクタリング戦略を深掘りする。

---

## 前提知識

| 前提 | 参照先 |
|------|--------|
| クリーンコードの基本原則 | [00-principles/](../00-principles/) |
| 命名と関数設計 | [01-practices/01-naming.md](../01-practices/01-naming.md), [01-practices/02-functions.md](../01-practices/02-functions.md) |
| テストの基礎 | [01-practices/04-testing-principles.md](../01-practices/04-testing-principles.md) |
| オブジェクト指向の基本（クラス、継承、ポリモーフィズム） | ― |

---

## この章で学ぶこと

1. **主要なコードスメルの5分類** ── Martin Fowler の分類に基づく22種以上のスメルを理解する
2. **スメルの重症度・影響度マトリクス** ── スメルの深刻さを定量的に評価する方法を身につける
3. **スメルの自動検出と手動検出** ── ツールとコードレビューを組み合わせた検出技法を習得する
4. **各スメルに対応するリファクタリング** ── スメルから適切なリファクタリングへの対応表を習得する
5. **検出ワークフローの設計** ── CI/CD・レビュー・静的解析を統合した組織的な検出体制を構築できるようになる

---

## 1. コードスメルとは何か

### 1.1 定義と歴史

「コードスメル」という用語は、Kent Beck が Martin Fowler との会話で使い始めたもので、1999年の『Refactoring: Improving the Design of Existing Code』初版で広く知られるようになった。

```
スメルの本質

  +-----------------------------------------------------------------+
  |  コードスメル ≠ バグ                                              |
  |  コードスメル ≠ コーディング規約違反                              |
  |  コードスメル  = 「ここに設計上の問題が潜んでいる可能性がある」   |
  |                   というヒューリスティックな手がかり               |
  +-----------------------------------------------------------------+

  スメルは以下の特徴を持つ:
  1. 主観的 ── 文脈によって許容される場合もある
  2. 累積的 ── 単体では軽微でも複合すると致命的になる
  3. 検出可能 ── 多くはツールまたはレビューで発見できる
  4. 対処可能 ── 各スメルに対応するリファクタリング技法が存在する
```

### 1.2 スメルの「匂いの強さ」

全てのスメルが同じ深刻度ではない。Robert C. Martin は『Clean Code』で「匂いの強さ」という概念を導入し、修正の緊急度を判断する指針とした。

```
  匂いの強さレベル

  Level 5 (腐敗臭) ── 即座に修正すべき
    例: 循環依存、テストなしの God Class
    → 放置すると開発停止レベルの障害を引き起こす

  Level 4 (悪臭)   ── 次のスプリントで修正
    例: Shotgun Surgery、Feature Envy の多発
    → 変更のたびに不要な工数が発生する

  Level 3 (異臭)   ── 計画的に修正
    例: Long Method、Primitive Obsession
    → 可読性・保守性が徐々に悪化する

  Level 2 (微臭)   ── 気づいたときに修正
    例: コメントの過剰、不要なパラメータ
    → ボーイスカウトルールで対処

  Level 1 (無臭)   ── 許容範囲
    例: 文脈上やむを得ない複雑性
    → ドキュメントで意図を記録して放置
```

---

## 2. コードスメルの5分類体系

Martin Fowler は『Refactoring』でスメルを5つのカテゴリに分類した。各カテゴリには複数のスメルが含まれ、それぞれに対応するリファクタリング技法が存在する。

```
+-------------------------------------------------------------------+
|  コードスメルの5分類 (Martin Fowler, 2018 2nd Edition)              |
+-------------------------------------------------------------------+
| 1. 肥大化 (Bloaters)                                              |
|    Long Method, Large Class, Long Parameter List,                 |
|    Data Clumps, Primitive Obsession                               |
+-------------------------------------------------------------------+
| 2. 乱用 (Object-Orientation Abusers)                              |
|    Switch Statements, Refused Bequest,                            |
|    Parallel Inheritance Hierarchy, Alternative Classes with       |
|    Different Interfaces, Temporary Field                          |
+-------------------------------------------------------------------+
| 3. 変更妨害 (Change Preventers)                                   |
|    Divergent Change, Shotgun Surgery,                             |
|    Parallel Inheritance Hierarchy                                 |
+-------------------------------------------------------------------+
| 4. 不要物 (Dispensables)                                          |
|    Dead Code, Speculative Generality, Lazy Class,                 |
|    Data Class, Duplicate Code, Comments (過剰な)                  |
+-------------------------------------------------------------------+
| 5. 結合過多 (Couplers)                                            |
|    Feature Envy, Inappropriate Intimacy, Message Chains,          |
|    Middle Man, Incomplete Library Class                            |
+-------------------------------------------------------------------+
```

### 2.1 スメル関連図 ── スメル間の因果関係

スメルは単独で存在するのではなく、相互に関連し、連鎖的に発生することが多い。

```
  スメル間の因果関係マップ

  [Long Method] ──引き起こす──> [Duplicated Code]
       |                              |
       v                              v
  [Feature Envy] ──引き起こす──> [Shotgun Surgery]
       |                              |
       v                              v
  [God Class] ──引き起こす──> [Divergent Change]
       |
       v
  [Inappropriate Intimacy]
       |
       v
  [Message Chains]

  典型的な連鎖パターン:
  1. Long Method → 重複コード → Shotgun Surgery
  2. God Class → Divergent Change → Feature Envy
  3. Primitive Obsession → Data Clumps → Long Parameter List
```

---

## 3. 肥大化スメル (Bloaters) の詳細

### 3.1 Long Method（長すぎるメソッド）

**定義**: メソッドの行数が多すぎて、1つの画面に収まらない。一般的な閾値は20-30行以上。

**検出指標**:
- 行数: 20行以上で警告、50行以上で危険
- サイクロマティック複雑度: 10以上
- ネスト深度: 3レベル以上
- コメントで区切られたブロックが存在

**コード例1: Long Method の検出と改善（Python）**

```python
# NG: 1つのメソッドに複数の責任が混在 (50行以上)
class OrderProcessor:
    def process_order(self, order_data: dict) -> OrderResult:
        # ── バリデーション ──
        if not order_data.get('items'):
            raise ValidationError("商品が選択されていません")
        if not order_data.get('customer_id'):
            raise ValidationError("顧客情報がありません")
        for item in order_data['items']:
            if item['quantity'] <= 0:
                raise ValidationError(f"数量が不正: {item['name']}")
            product = self.db.query(
                "SELECT stock FROM products WHERE id = %s", item['id']
            )
            if product.stock < item['quantity']:
                raise ValidationError(f"在庫不足: {item['name']}")

        # ── 合計計算 ──
        subtotal = 0
        for item in order_data['items']:
            price = item['unit_price'] * item['quantity']
            if item.get('discount_code'):
                discount = self.db.query(
                    "SELECT rate FROM discounts WHERE code = %s",
                    item['discount_code']
                )
                price *= (1 - discount.rate)
            subtotal += price
        tax = subtotal * Decimal("0.10")
        shipping = Decimal("500") if subtotal < Decimal("5000") else Decimal("0")
        total = subtotal + tax + shipping

        # ── DB保存 ──
        order = Order(
            customer_id=order_data['customer_id'],
            items=order_data['items'],
            subtotal=subtotal, tax=tax, shipping=shipping, total=total
        )
        self.db.save(order)

        # ── メール送信 ──
        customer = self.db.query(
            "SELECT * FROM customers WHERE id = %s",
            order_data['customer_id']
        )
        self.email_service.send(
            to=customer.email,
            subject="ご注文ありがとうございます",
            body=self._render_confirmation(order)
        )

        # ── 分析トラッキング ──
        self.analytics.track('order_completed', {
            'order_id': order.id,
            'total': float(total),
            'item_count': len(order_data['items'])
        })

        return OrderResult.success(order)


# OK: 意図ごとに関数を分離 ── 各メソッドは5-10行
class OrderProcessor:
    def process_order(self, order_data: dict) -> OrderResult:
        """注文処理のオーケストレーション"""
        self._validate_order(order_data)
        pricing = self._calculate_pricing(order_data['items'])
        order = self._save_order(order_data, pricing)
        self._send_confirmation(order)
        self._track_analytics(order)
        return OrderResult.success(order)

    def _validate_order(self, order_data: dict) -> None:
        """注文データのバリデーション"""
        validator = OrderValidator(self.db)
        validator.validate(order_data)

    def _calculate_pricing(self, items: list[dict]) -> OrderPricing:
        """価格計算"""
        calculator = PricingCalculator(self.db)
        return calculator.calculate(items)

    def _save_order(self, data: dict, pricing: OrderPricing) -> Order:
        """注文のDB保存"""
        order = Order.from_data(data, pricing)
        self.db.save(order)
        return order

    def _send_confirmation(self, order: Order) -> None:
        """確認メールの送信"""
        customer = self.db.find_customer(order.customer_id)
        self.email_service.send_order_confirmation(customer, order)

    def _track_analytics(self, order: Order) -> None:
        """分析トラッキング"""
        self.analytics.track('order_completed', order.to_analytics_dict())
```

### 3.2 God Class（神クラス / Large Class）

**定義**: あまりにも多くの責任を持つクラス。Single Responsibility Principle (SRP) に違反している。

**検出指標**:
- メソッド数: 20以上
- フィールド数: 15以上
- 行数: 500行以上
- 依存するクラス数: 10以上

```
  God Class の症状と分解

  ┌────────────────────────────────────────────┐
  │  ApplicationManager                        │
  │  ────────────────────────────              │
  │  - userRepository                          │
  │  - orderRepository                         │
  │  - paymentGateway                          │
  │  - emailService                            │
  │  - cacheManager                            │
  │  - logger                                  │
  │  - configService                           │
  │  - analyticsTracker                        │
  │  ────────────────────────────              │
  │  + authenticateUser()                      │
  │  + createOrder()                           │
  │  + processPayment()                        │
  │  + sendEmail()                             │
  │  + clearCache()                            │
  │  + generateReport()                        │
  │  + validateConfig()                        │
  │  + trackEvent()                            │
  │  + ... (50+ methods)                       │
  └────────────────────────────────────────────┘
         ↓ Extract Class で責任ごとに分離 ↓
  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │ AuthSvc   │  │ OrderSvc  │  │ PaymentSvc│  │ EmailSvc  │
  │ ────────  │  │ ────────  │  │ ────────  │  │ ────────  │
  │ login()   │  │ create()  │  │ charge()  │  │ send()    │
  │ logout()  │  │ cancel()  │  │ refund()  │  │ template()│
  │ verify()  │  │ update()  │  │ verify()  │  │ queue()   │
  └──────────┘  └──────────┘  └──────────┘  └──────────┘
```

**コード例2: God Class の段階的分解（Java）**

```java
// NG: God Class ── 複数の責任が1つのクラスに集中
public class ApplicationManager {
    private UserRepository userRepo;
    private OrderRepository orderRepo;
    private PaymentGateway paymentGw;
    private EmailService emailSvc;

    // 認証の責任
    public User authenticateUser(String email, String password) {
        User user = userRepo.findByEmail(email);
        if (user == null) throw new AuthException("ユーザーが見つかりません");
        if (!BCrypt.checkpw(password, user.getPasswordHash())) {
            throw new AuthException("パスワードが一致しません");
        }
        user.setLastLogin(Instant.now());
        userRepo.save(user);
        return user;
    }

    // 注文の責任
    public Order createOrder(User user, List<OrderItem> items) {
        Order order = new Order(user, items);
        order.setTotal(items.stream()
            .mapToDouble(i -> i.getPrice() * i.getQuantity())
            .sum());
        orderRepo.save(order);
        return order;
    }

    // 決済の責任
    public PaymentResult processPayment(Order order, CreditCard card) {
        return paymentGw.charge(card, order.getTotal());
    }

    // メールの責任
    public void sendConfirmation(Order order) {
        emailSvc.send(order.getUser().getEmail(), "注文確認", "...");
    }
    // ... さらに数十のメソッドが続く
}


// OK: 責任ごとにクラスを分離
public class AuthenticationService {
    private final UserRepository userRepo;
    private final PasswordEncoder encoder;

    public AuthenticationService(UserRepository userRepo, PasswordEncoder encoder) {
        this.userRepo = userRepo;
        this.encoder = encoder;
    }

    public User authenticate(String email, String password) {
        User user = userRepo.findByEmail(email)
            .orElseThrow(() -> new AuthException("ユーザーが見つかりません"));
        if (!encoder.matches(password, user.getPasswordHash())) {
            throw new AuthException("パスワードが一致しません");
        }
        user.recordLogin();
        userRepo.save(user);
        return user;
    }
}

public class OrderService {
    private final OrderRepository orderRepo;
    private final PricingCalculator calculator;

    public Order create(User user, List<OrderItem> items) {
        Order order = Order.create(user, items, calculator);
        orderRepo.save(order);
        return order;
    }
}
```

### 3.3 Long Parameter List（長すぎるパラメータリスト）

**定義**: メソッドのパラメータが多すぎる。一般的に4つ以上で警告。

**コード例3: Long Parameter List の改善（TypeScript）**

```typescript
// NG: パラメータが8つ ── 順序を間違えやすい
function createUser(
  firstName: string,
  lastName: string,
  email: string,
  phone: string,
  street: string,
  city: string,
  zipCode: string,
  country: string
): User {
  // ...
}

// 呼び出し側: どのパラメータが何か分からない
createUser("太郎", "山田", "taro@example.com", "090-1234-5678",
           "丸の内1-1-1", "千代田区", "100-0001", "日本");


// OK: パラメータオブジェクトに集約
interface PersonalInfo {
  firstName: string;
  lastName: string;
  email: string;
  phone: string;
}

interface Address {
  street: string;
  city: string;
  zipCode: string;
  country: string;

  // Address 自身がバリデーションを持つ
  validate(): ValidationResult;
  format(): string;
}

interface CreateUserRequest {
  personal: PersonalInfo;
  address: Address;
}

function createUser(request: CreateUserRequest): User {
  request.address.validate();
  return new User(request);
}

// 呼び出し側: 構造が明確
createUser({
  personal: { firstName: "太郎", lastName: "山田",
              email: "taro@example.com", phone: "090-1234-5678" },
  address:  { street: "丸の内1-1-1", city: "千代田区",
              zipCode: "100-0001", country: "日本",
              validate() { /* ... */ }, format() { /* ... */ } },
});
```

### 3.4 Data Clumps（データの群れ）

**定義**: 同じパラメータの組み合わせが複数の場所に繰り返し登場する。

**コード例4: Data Clumps の抽出（Python）**

```python
# NG: 同じパラメータ群が繰り返し登場
def create_user(first_name, last_name, street, city, zip_code, country): ...
def update_address(user_id, street, city, zip_code, country): ...
def validate_address(street, city, zip_code, country): ...
def format_address(street, city, zip_code, country): ...
def calculate_shipping(street, city, zip_code, country, weight): ...

# テスト: パラメータの組み合わせが多すぎてテストしづらい


# OK: データクラスに抽出
@dataclass(frozen=True)
class Address:
    """住所値オブジェクト ── 不変・バリデーション付き"""
    street: str
    city: str
    zip_code: str
    country: str

    def __post_init__(self):
        if not self.zip_code:
            raise ValueError("郵便番号は必須です")
        if not self.country:
            raise ValueError("国名は必須です")

    def validate(self) -> bool:
        """住所の妥当性を検証"""
        return bool(self.street and self.city and
                    self.zip_code and self.country)

    def format(self, style: str = "japanese") -> str:
        """住所をフォーマット"""
        if style == "japanese":
            return f"〒{self.zip_code} {self.city}{self.street}"
        return f"{self.street}, {self.city} {self.zip_code}, {self.country}"

    def is_domestic(self) -> bool:
        return self.country == "日本"


def create_user(first_name: str, last_name: str, address: Address): ...
def update_address(user_id: str, address: Address): ...
def calculate_shipping(address: Address, weight: float) -> Decimal: ...

# テスト: Address を1つ作れば全関数で再利用可能
```

### 3.5 Primitive Obsession（プリミティブ偏執）

**定義**: ドメイン概念を `str`、`int`、`float` 等のプリミティブ型で表現し、型安全性を失っている。

**コード例5: Primitive Obsession の改善（TypeScript）**

```typescript
// NG: ドメイン概念をプリミティブ型で表現
function processPayment(
  amount: number,       // 円？ドル？ マイナスは？
  currency: string,     // "JPY"? "jpy"? "円"?
  email: string,        // バリデーション済み？
  cardNumber: string    // マスク済み？ 有効期限は？
): boolean { /* ... */ }

// 呼び出し側: 負の金額や不正なメールを渡せてしまう
processPayment(-1000, "yen", "not-an-email", "1234");


// OK: 値オブジェクトで型安全性を確保
class Money {
  readonly amount: number;
  readonly currency: Currency;

  constructor(amount: number, currency: Currency) {
    if (amount < 0) throw new Error("金額は0以上でなければなりません");
    if (!Number.isFinite(amount)) throw new Error("金額は有限の数値です");
    this.amount = amount;
    this.currency = currency;
  }

  add(other: Money): Money {
    if (this.currency !== other.currency) {
      throw new CurrencyMismatchError(this.currency, other.currency);
    }
    return new Money(this.amount + other.amount, this.currency);
  }

  multiply(factor: number): Money {
    return new Money(Math.round(this.amount * factor), this.currency);
  }

  format(): string {
    return new Intl.NumberFormat('ja-JP', {
      style: 'currency', currency: this.currency
    }).format(this.amount);
  }
}

class Email {
  private static readonly PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  readonly value: string;

  constructor(value: string) {
    const normalized = value.trim().toLowerCase();
    if (!Email.PATTERN.test(normalized)) {
      throw new InvalidEmailError(value);
    }
    this.value = normalized;
  }
}

class CreditCard {
  readonly maskedNumber: string;
  readonly last4: string;

  constructor(number: string, expiry: Date) {
    if (!CreditCard.isValidLuhn(number)) {
      throw new InvalidCardError("カード番号が不正です");
    }
    if (expiry < new Date()) {
      throw new ExpiredCardError("有効期限切れです");
    }
    this.last4 = number.slice(-4);
    this.maskedNumber = `****-****-****-${this.last4}`;
  }

  private static isValidLuhn(number: string): boolean {
    // Luhn アルゴリズムによる検証
    let sum = 0;
    let isEven = false;
    for (let i = number.length - 1; i >= 0; i--) {
      let digit = parseInt(number[i], 10);
      if (isEven) { digit *= 2; if (digit > 9) digit -= 9; }
      sum += digit;
      isEven = !isEven;
    }
    return sum % 10 === 0;
  }
}

// 型安全な関数シグネチャ
function processPayment(
  amount: Money,
  email: Email,
  card: CreditCard
): PaymentResult { /* ... */ }
```

---

## 4. OO乱用スメル (Object-Orientation Abusers) の詳細

### 4.1 Switch Statements（条件分岐の増殖）

**定義**: 型や状態による条件分岐が複数箇所に散在している。新しい型や状態を追加するたびに、全ての分岐を修正する必要がある（Open-Closed Principle 違反）。

**コード例6: Switch Statements の改善（Python）**

```python
# NG: 型による分岐が複数箇所に散在
# 新しい Shape を追加するたびに全関数を修正する必要がある
def calculate_area(shape):
    if shape.type == 'circle':
        return math.pi * shape.radius ** 2
    elif shape.type == 'rectangle':
        return shape.width * shape.height
    elif shape.type == 'triangle':
        return shape.base * shape.height / 2
    else:
        raise ValueError(f"未知の図形: {shape.type}")

def calculate_perimeter(shape):
    if shape.type == 'circle':
        return 2 * math.pi * shape.radius
    elif shape.type == 'rectangle':
        return 2 * (shape.width + shape.height)
    elif shape.type == 'triangle':
        return shape.side_a + shape.side_b + shape.side_c
    else:
        raise ValueError(f"未知の図形: {shape.type}")

def draw(shape, canvas):
    if shape.type == 'circle':
        canvas.draw_circle(shape.center, shape.radius)
    elif shape.type == 'rectangle':
        canvas.draw_rect(shape.origin, shape.width, shape.height)
    elif shape.type == 'triangle':
        canvas.draw_polygon(shape.vertices)


# OK: ポリモーフィズムで置換 ── 新しい図形はクラスを追加するだけ
from abc import ABC, abstractmethod
import math

class Shape(ABC):
    """図形の基底クラス"""

    @abstractmethod
    def area(self) -> float:
        """面積を計算"""

    @abstractmethod
    def perimeter(self) -> float:
        """外周を計算"""

    @abstractmethod
    def draw(self, canvas: Canvas) -> None:
        """キャンバスに描画"""

class Circle(Shape):
    def __init__(self, center: Point, radius: float):
        if radius <= 0:
            raise ValueError("半径は正の数でなければなりません")
        self.center = center
        self.radius = radius

    def area(self) -> float:
        return math.pi * self.radius ** 2

    def perimeter(self) -> float:
        return 2 * math.pi * self.radius

    def draw(self, canvas: Canvas) -> None:
        canvas.draw_circle(self.center, self.radius)

class Rectangle(Shape):
    def __init__(self, origin: Point, width: float, height: float):
        self.origin = origin
        self.width = width
        self.height = height

    def area(self) -> float:
        return self.width * self.height

    def perimeter(self) -> float:
        return 2 * (self.width + self.height)

    def draw(self, canvas: Canvas) -> None:
        canvas.draw_rect(self.origin, self.width, self.height)

# 新しい図形を追加: 既存コードの修正は不要
class Pentagon(Shape):
    def __init__(self, center: Point, side_length: float):
        self.center = center
        self.side_length = side_length

    def area(self) -> float:
        return (math.sqrt(5 * (5 + 2 * math.sqrt(5))) / 4) * self.side_length ** 2

    def perimeter(self) -> float:
        return 5 * self.side_length

    def draw(self, canvas: Canvas) -> None:
        canvas.draw_polygon(self._calculate_vertices())
```

### 4.2 Refused Bequest（拒否された遺産）

**定義**: サブクラスが親クラスのメソッドやプロパティの大部分を使用しない。

```python
# NG: Stack は List のメソッドの大部分を使わない/使わせたくない
class Stack(list):
    """スタック ── だが list の insert, sort, reverse が使えてしまう"""
    def push(self, item):
        self.append(item)

    def peek(self):
        return self[-1] if self else None

# stack.insert(0, "割り込み")  ← スタックの不変条件を破壊！
# stack.sort()                  ← 意味がない操作が可能


# OK: 委譲（コンポジション）で必要なインターフェースだけを公開
class Stack:
    """スタック ── LIFO のみを公開"""
    def __init__(self):
        self._items: list = []

    def push(self, item) -> None:
        self._items.append(item)

    def pop(self):
        if not self._items:
            raise EmptyStackError("スタックが空です")
        return self._items.pop()

    def peek(self):
        if not self._items:
            raise EmptyStackError("スタックが空です")
        return self._items[-1]

    def is_empty(self) -> bool:
        return len(self._items) == 0

    def __len__(self) -> int:
        return len(self._items)
```

### 4.3 Temporary Field（一時フィールド）

**定義**: 特定のメソッドでのみ使用されるフィールドがクラスに存在する。

```java
// NG: totalSales と averagePrice は generateReport() でしか使わない
public class SalesAnalyzer {
    private List<Sale> sales;
    private double totalSales;     // ← 一時フィールド
    private double averagePrice;   // ← 一時フィールド

    public Report generateReport() {
        this.totalSales = sales.stream().mapToDouble(Sale::getAmount).sum();
        this.averagePrice = this.totalSales / sales.size();
        return new Report(this.totalSales, this.averagePrice);
    }
}


// OK: メソッドローカル変数またはデータクラスに移動
public class SalesAnalyzer {
    private final List<Sale> sales;

    public Report generateReport() {
        SalesStatistics stats = calculateStatistics();
        return new Report(stats);
    }

    private SalesStatistics calculateStatistics() {
        double total = sales.stream().mapToDouble(Sale::getAmount).sum();
        double average = total / sales.size();
        return new SalesStatistics(total, average);
    }
}

record SalesStatistics(double totalSales, double averagePrice) {}
```

---

## 5. 変更妨害スメル (Change Preventers) の詳細

### 5.1 Divergent Change（発散的変更）

**定義**: 1つのクラスが異なる理由で頻繁に変更される。

```
  Divergent Change の症状

  [UserService] が以下の全ての変更で修正される:
  ├── 認証方式の変更時     → authenticate() を修正
  ├── 注文ロジックの変更時 → processOrder() を修正
  ├── メール送信の変更時   → sendNotification() を修正
  └── レポート形式の変更時 → generateReport() を修正

  対策: Extract Class で責任を分離
  [AuthService]         ← 認証方式の変更のみ
  [OrderService]        ← 注文ロジックの変更のみ
  [NotificationService] ← メール送信の変更のみ
  [ReportService]       ← レポート形式の変更のみ
```

### 5.2 Shotgun Surgery（散弾銃手術）

**定義**: 1つの変更を行うために、多数のクラスを修正する必要がある。Divergent Change の逆。

**コード例7: Shotgun Surgery の検出と改善（Python）**

```python
# NG: 「税率の変更」で6箇所のファイルを修正する必要がある
# order_service.py
class OrderService:
    def calculate_total(self, order):
        return order.subtotal * 1.10  # ← 税率ハードコード

# invoice_service.py
class InvoiceService:
    def generate(self, order):
        tax = order.subtotal * 0.10   # ← 同じ税率
        return f"税額: {tax}"

# report_service.py
class ReportService:
    def monthly_tax(self, orders):
        return sum(o.subtotal * 0.10 for o in orders)  # ← また同じ

# receipt_printer.py, api_controller.py, test_helper.py ... 全て修正


# OK: 税率計算を1箇所に集約
class TaxCalculator:
    """税金計算の唯一の真実の源 (Single Source of Truth)"""
    STANDARD_RATE = Decimal("0.10")
    REDUCED_RATE = Decimal("0.08")

    @classmethod
    def calculate(cls, amount: Decimal,
                  rate_type: str = "standard") -> Decimal:
        rate = cls.REDUCED_RATE if rate_type == "reduced" else cls.STANDARD_RATE
        return (amount * rate).quantize(Decimal("1"))

    @classmethod
    def add_tax(cls, amount: Decimal,
                rate_type: str = "standard") -> Decimal:
        return amount + cls.calculate(amount, rate_type)


# 全てのサービスが TaxCalculator を使う
class OrderService:
    def calculate_total(self, order):
        return TaxCalculator.add_tax(order.subtotal)

class InvoiceService:
    def generate(self, order):
        tax = TaxCalculator.calculate(order.subtotal)
        return f"税額: {tax}"
```

---

## 6. 不要物スメル (Dispensables) の詳細

### 6.1 Dead Code（デッドコード）

**定義**: 実行されないコード、呼び出されない関数、使われないインポート・変数。

**コード例8: Dead Code の検出と除去（Python）**

```python
# NG: デッドコードが散在
import os                     # ← 未使用インポート
import json                   # ← 未使用インポート
from datetime import datetime

class UserManager:
    def __init__(self, db):
        self.db = db
        self.cache = {}        # ← 一度も読み書きされない

    def get_user(self, user_id: str) -> User:
        return self.db.find(user_id)

    def _old_get_user(self, user_id: str) -> User:
        """旧実装 ── いつか必要になるかもしれないので残す"""  # ← デッドコード
        result = self.db.execute(f"SELECT * FROM users WHERE id = '{user_id}'")
        return User.from_row(result)

    def update_user(self, user_id: str, data: dict) -> User:
        user = self.get_user(user_id)
        # if False:                              # ← 到達不能コード
        #     self._send_update_notification(user)
        user.update(data)
        self.db.save(user)
        return user

    def _send_update_notification(self, user):   # ← 呼び出されない
        pass


# OK: デッドコードを完全に除去
from datetime import datetime

class UserManager:
    def __init__(self, db):
        self.db = db

    def get_user(self, user_id: str) -> User:
        return self.db.find(user_id)

    def update_user(self, user_id: str, data: dict) -> User:
        user = self.get_user(user_id)
        user.update(data)
        self.db.save(user)
        return user

# 旧実装が必要なら Git の履歴から復元可能
# → デッドコードを「念のため」残す必要はない
```

### 6.2 Speculative Generality（投機的汎用化）

**定義**: 「将来必要になるかもしれない」と過度に抽象化されたコード。YAGNI (You Aren't Gonna Need It) 違反。

```python
# NG: 現在 JSON しか使わないのに過度に抽象化
class SerializerFactory:
    def create(self, format_type: str) -> Serializer:
        if format_type == "json":
            return JsonSerializer()
        elif format_type == "xml":          # ← 使われていない
            return XmlSerializer()
        elif format_type == "yaml":         # ← 使われていない
            return YamlSerializer()
        elif format_type == "msgpack":      # ← 使われていない
            return MsgpackSerializer()
        elif format_type == "protobuf":     # ← 使われていない
            return ProtobufSerializer()

class Serializer(ABC): ...
class JsonSerializer(Serializer): ...
class XmlSerializer(Serializer): ...        # ← 使われていない
class YamlSerializer(Serializer): ...       # ← 使われていない
class MsgpackSerializer(Serializer): ...    # ← 使われていない
class ProtobufSerializer(Serializer): ...   # ← 使われていない


# OK: 必要なものだけを実装
class JsonSerializer:
    """JSON シリアライゼーション ── 現在必要な唯一のフォーマット"""
    def serialize(self, data: dict) -> str:
        return json.dumps(data, ensure_ascii=False, default=str)

    def deserialize(self, text: str) -> dict:
        return json.loads(text)

# 将来 XML が必要になったら、そのときに抽象化を導入する
# → "Rule of Three": 3回目のパターン出現まで抽象化を待つ
```

### 6.3 Duplicate Code（重複コード）

**定義**: 同一または非常に類似したコードが複数箇所に存在する。

**コード例9: 重複コードの統合（Python）**

```python
# NG: ほぼ同じ検証ロジックが3箇所に重複
class UserRegistration:
    def register(self, data: dict):
        # メールバリデーション (重複1)
        if not data.get('email'):
            raise ValueError("メールは必須です")
        if '@' not in data['email']:
            raise ValueError("メール形式が不正です")
        if len(data['email']) > 254:
            raise ValueError("メールが長すぎます")
        # ...

class ProfileUpdate:
    def update_email(self, data: dict):
        # メールバリデーション (重複2 ── ほぼ同一)
        if not data.get('new_email'):
            raise ValueError("メールは必須です")
        if '@' not in data['new_email']:
            raise ValueError("メール形式が不正です")
        if len(data['new_email']) > 254:
            raise ValueError("メールが長すぎます")
        # ...

class InviteService:
    def send_invite(self, email: str):
        # メールバリデーション (重複3 ── 微妙に異なる)
        if not email or '@' not in email:
            raise ValueError("不正なメール")
        # ...


# OK: バリデーションロジックを値オブジェクトに統合
import re
from dataclasses import dataclass

@dataclass(frozen=True)
class Email:
    """メールアドレス値オブジェクト ── バリデーションは1箇所のみ"""
    value: str

    _PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    _MAX_LENGTH = 254

    def __post_init__(self):
        if not self.value:
            raise ValueError("メールアドレスは必須です")
        if len(self.value) > self._MAX_LENGTH:
            raise ValueError(f"メールアドレスは{self._MAX_LENGTH}文字以下です")
        if not self._PATTERN.match(self.value):
            raise ValueError(f"メール形式が不正です: {self.value}")
        # frozen=True なので object.__setattr__ で正規化
        object.__setattr__(self, 'value', self.value.strip().lower())

class UserRegistration:
    def register(self, data: dict):
        email = Email(data.get('email', ''))  # バリデーションは Email 内で実行
        # ...

class ProfileUpdate:
    def update_email(self, data: dict):
        new_email = Email(data.get('new_email', ''))
        # ...
```

---

## 7. 結合過多スメル (Couplers) の詳細

### 7.1 Feature Envy（他クラスへの羨望）

**定義**: あるメソッドが、自クラスよりも他クラスのデータを多く参照している。

**コード例10: Feature Envy の改善（Java）**

```java
// NG: OrderPrinter が Order の内部データを過度に使用
class OrderPrinter {
    String formatOrder(Order order) {
        StringBuilder sb = new StringBuilder();
        sb.append("注文番号: ").append(order.getId()).append("\n");
        sb.append("顧客名: ").append(order.getCustomer().getName()).append("\n");
        sb.append("日付: ").append(order.getDate()
            .format(DateTimeFormatter.ISO_DATE)).append("\n");

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


// OK: フォーマットロジックを Order に移動
class Order {
    public String format() {
        StringBuilder sb = new StringBuilder();
        sb.append("注文番号: ").append(this.id).append("\n");
        sb.append("顧客名: ").append(this.customer.getName()).append("\n");
        sb.append("日付: ").append(this.date
            .format(DateTimeFormatter.ISO_DATE)).append("\n");
        this.items.forEach(item -> sb.append(item.formatLine()));
        sb.append(formatTotals());
        return sb.toString();
    }

    private String formatTotals() {
        double subtotal = calculateSubtotal();
        double tax = subtotal * 0.10;
        return String.format("小計: %.0f円\n税: %.0f円\n合計: %.0f円\n",
            subtotal, tax, subtotal + tax);
    }

    public double calculateSubtotal() {
        return items.stream()
            .mapToDouble(item -> item.getPrice() * item.getQuantity())
            .sum();
    }
}

class OrderItem {
    public String formatLine() {
        double lineTotal = this.price * this.quantity;
        return String.format("  %s x%d = %.0f円\n",
            this.name, this.quantity, lineTotal);
    }
}
```

### 7.2 Message Chains（メッセージチェーン / Demeter の法則違反）

**定義**: `a.getB().getC().getD().doSomething()` のように、オブジェクトの連鎖的なアクセスが発生する。

**コード例11: Message Chains の改善（Python）**

```python
# NG: Demeter の法則に違反
class OrderReport:
    def get_customer_city(self, order):
        # order → customer → address → city の4段階チェーン
        return order.get_customer().get_address().get_city()

    def get_payment_bank(self, order):
        # order → payment → method → bank の4段階チェーン
        return order.get_payment().get_method().get_bank_name()


# OK: 必要な情報を直接取得できるメソッドを提供
class Order:
    def get_customer_city(self) -> str:
        """顧客の都市名を返す（内部構造を隠蔽）"""
        return self._customer.get_city()

    def get_payment_bank(self) -> str:
        """決済銀行名を返す"""
        return self._payment.get_bank_name()

class Customer:
    def get_city(self) -> str:
        """住所の都市名を返す（内部構造を隠蔽）"""
        return self._address.city

class Payment:
    def get_bank_name(self) -> str:
        return self._method.bank_name

# 呼び出し側は1段階のみ
class OrderReport:
    def get_customer_city(self, order):
        return order.get_customer_city()
```

### 7.3 Inappropriate Intimacy（不適切な親密さ）

**定義**: 2つのクラスが互いの内部実装に過度に依存している。

**コード例12: Inappropriate Intimacy の改善（TypeScript）**

```typescript
// NG: User と Profile が互いの内部を直接操作
class User {
  public name: string;
  public email: string;
  public profile: Profile;

  updateProfile(bio: string): void {
    // User が Profile の内部を直接操作
    this.profile._bio = bio;               // private を迂回
    this.profile._updatedAt = new Date();  // Profile の責任を侵害
  }
}

class Profile {
  _bio: string;          // 本来は private にすべき
  _updatedAt: Date;      // 本来は private にすべき

  getDisplayName(): string {
    // Profile が User の内部を直接参照
    return `${this.user.name} (${this.user.email})`;
  }
}


// OK: 明確なインターフェースを通じてのみ通信
class User {
  private readonly name: string;
  private readonly email: string;
  private readonly profile: Profile;

  updateProfile(bio: string): void {
    this.profile.updateBio(bio);  // Profile のメソッドを呼ぶ
  }

  getDisplayInfo(): UserDisplayInfo {
    return { name: this.name, email: this.email };
  }
}

class Profile {
  private bio: string;
  private updatedAt: Date;

  updateBio(bio: string): void {
    if (bio.length > 500) throw new Error("プロフィールは500文字以内");
    this.bio = bio;
    this.updatedAt = new Date();
  }

  formatDisplay(userInfo: UserDisplayInfo): string {
    return `${userInfo.name} - ${this.bio}`;
  }
}
```

---

## 8. スメルの影響度マトリクス

```
  スメルの影響度マトリクス ── 縦軸: 影響度, 横軸: 検出しやすさ

  影響度: 大
    ^
    |  [循環依存]    [Shotgun Surgery]
    |
    |  [God Class]   [Divergent Change]
    |
    |  [Feature      [Long Method]     [Duplicate Code]
    |   Envy]
    |
    |  [Message       [Dead Code]      [Long Parameter]
    |   Chains]
    |
    |  [Temporary     [Speculative     [過剰なコメント]
    |   Field]         Generality]
    +──────────────────────────────────────────────> 検出しやすさ
   難しい                中間                 簡単

  ★ 右上ゾーン (高影響・検出容易) から着手するのが最も効率的
```

---

## 9. スメルとリファクタリングの対応表

### 9.1 スメル → リファクタリング対応表

| コードスメル | 対応するリファクタリング | 自動化可否 | 優先度 |
|------------|----------------------|:--------:|:------:|
| Long Method | Extract Method | 可 | 高 |
| God Class | Extract Class, Move Method | 部分的 | 高 |
| Feature Envy | Move Method, Move Field | 部分的 | 高 |
| Data Clumps | Extract Class, Introduce Parameter Object | 部分的 | 中 |
| Primitive Obsession | Replace Primitive with Object | 不可 | 中 |
| Switch Statements | Replace Conditional with Polymorphism | 不可 | 中 |
| Shotgun Surgery | Move Method, Inline Class | 不可 | 高 |
| Divergent Change | Extract Class | 部分的 | 高 |
| Dead Code | Safe Delete | 可 | 低 |
| Duplicate Code | Extract Method, Pull Up Method | 部分的 | 中 |
| Long Parameter List | Introduce Parameter Object | 部分的 | 中 |
| Message Chains | Hide Delegate | 不可 | 中 |
| Refused Bequest | Replace Inheritance with Delegation | 不可 | 低 |
| Speculative Generality | Collapse Hierarchy, Remove Middle Man | 可 | 低 |
| Inappropriate Intimacy | Move Method, Extract Class | 不可 | 高 |
| Temporary Field | Extract Class | 部分的 | 低 |

### 9.2 自動検出ツール比較表

| ツール | 対応言語 | 検出可能なスメル | CI/CD統合 | ライセンス |
|--------|---------|---------------|:---------:|-----------|
| SonarQube | 多言語(30+) | 複雑度, 重複, デッドコード, セキュリティ | 可 | Community Edition: 無料 |
| PMD | Java, Apex | 複雑度, 命名, 設計問題 | 可 | BSD |
| pylint | Python | 複雑度, 命名, 未使用コード, スタイル | 可 | GPL |
| Ruff | Python | pylint + flake8 互換, 高速 | 可 | MIT |
| ESLint | JS/TS | 複雑度, 未使用変数, スタイル | 可 | MIT |
| RuboCop | Ruby | 複雑度, スタイル, 命名, Metrics | 可 | MIT |
| detekt | Kotlin | 複雑度, コードスメル, スタイル | 可 | Apache 2.0 |
| radon | Python | 複雑度(CC, MI)専用 | 可 | MIT |
| jscpd | 多言語 | 重複コード検出専用 | 可 | MIT |

---

## 10. スメル検出のワークフロー

### 10.1 3段階の検出体制

```
  スメル検出の3段階ワークフロー

  ┌──────────────────────────────────────────────┐
  │  Stage 1: 自動検出 (pre-commit / CI)         │
  │  ─────────────────────────────────           │
  │  - Ruff/ESLint でスタイル・複雑度を自動チェック │
  │  - radon で CC > 10 の関数を検出              │
  │  - jscpd で重複率 > 3% をブロック             │
  │  効果: 機械的に検出可能な60%のスメルを自動排除  │
  └──────────────┬───────────────────────────────┘
                 v
  ┌──────────────────────────────────────────────┐
  │  Stage 2: コードレビュー (PR レビュー)         │
  │  ─────────────────────────────────           │
  │  - Feature Envy, God Class の兆候を人間が判断 │
  │  - 設計意図との整合性を確認                    │
  │  - チェックリスト形式で見落としを防止           │
  │  効果: 文脈が必要な30%のスメルを検出           │
  └──────────────┬───────────────────────────────┘
                 v
  ┌──────────────────────────────────────────────┐
  │  Stage 3: 定期棚卸し (四半期レビュー)          │
  │  ─────────────────────────────────           │
  │  - SonarQube ダッシュボードでトレンド分析       │
  │  - ホットスポット分析 (変更頻度 x 複雑度)      │
  │  - 技術的負債バックログの優先度再評価           │
  │  効果: 蓄積した10%のスメルを組織的に解消        │
  └──────────────────────────────────────────────┘
```

### 10.2 CI/CD でのスメル検出設定（GitHub Actions）

**コード例13: スメル自動検出パイプライン**

```yaml
# .github/workflows/code-smell-detection.yml
name: Code Smell Detection

on:
  pull_request:
    branches: [main, develop]

jobs:
  detect-smells:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install analysis tools
        run: |
          pip install ruff radon jscpd

      # 1. Lint チェック (スタイル + 基本的なスメル)
      - name: Ruff Lint
        run: ruff check src/ --output-format=github

      # 2. サイクロマティック複雑度チェック
      - name: Complexity Check
        run: |
          echo "## 複雑度レポート" >> $GITHUB_STEP_SUMMARY
          radon cc src/ -a -nc --min C
          # C以上 (複雑度11+) がある場合に警告
          COMPLEX=$(radon cc src/ -nc --min C -j | python3 -c "
          import json, sys
          data = json.load(sys.stdin)
          count = sum(len(funcs) for funcs in data.values())
          print(count)
          ")
          if [ "$COMPLEX" -gt 0 ]; then
            echo "::warning::高複雑度の関数が${COMPLEX}個あります"
          fi

      # 3. コード重複チェック
      - name: Duplication Check
        run: |
          jscpd src/ --min-lines 6 --min-tokens 50 \
                --reporters "consoleFull" --threshold 3

      # 4. デッドコード検出
      - name: Dead Code Detection
        run: |
          pip install vulture
          vulture src/ --min-confidence 80

      # 5. 保守性指数
      - name: Maintainability Index
        run: |
          radon mi src/ -s --min B
```

### 10.3 コードレビューチェックリスト

```
  ┌─────────────────────────────────────────────────────────┐
  │  スメル検出チェックリスト (コードレビュー用)              │
  ├─────────────────────────────────────────────────────────┤
  │  □ メソッドは20行以下か？（Long Method）                 │
  │  □ クラスは単一責任か？（God Class）                     │
  │  □ 同じパラメータ群の繰り返しはないか？（Data Clumps）   │
  │  □ ドメイン概念にプリミティブ型を使っていないか？         │
  │  □ 条件分岐が複数箇所に散在していないか？                │
  │  □ メソッドは自クラスのデータを主に使用しているか？       │
  │  □ 1つの変更で複数ファイルの修正が必要にならないか？      │
  │  □ 使われていないコード・インポートはないか？             │
  │  □ 「将来のため」の過度な抽象化はないか？                 │
  │  □ オブジェクトのチェーンアクセス（a.b.c.d）はないか？   │
  └─────────────────────────────────────────────────────────┘
```

---

## 11. ホットスポット分析 ── 優先度の科学的な判断

### 11.1 変更頻度 x 複雑度 分析

**コード例14: ホットスポット分析スクリプト（Python）**

```python
#!/usr/bin/env python3
"""
ホットスポット分析: 変更頻度と複雑度を組み合わせて
リファクタリングの優先度を科学的に判断する。

使用法:
  python hotspot_analysis.py /path/to/repo --days 90
"""
import subprocess
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FileMetrics:
    """ファイルのスメル指標"""
    path: str
    change_count: int       # Git の変更回数
    complexity: float       # 平均サイクロマティック複雑度
    lines: int              # 行数
    hotspot_score: float    # 変更頻度 x 複雑度

    @property
    def priority(self) -> str:
        if self.hotspot_score > 100:
            return "最優先"
        elif self.hotspot_score > 50:
            return "高"
        elif self.hotspot_score > 20:
            return "中"
        return "低"


def analyze_hotspots(repo_path: str, days: int = 90) -> list[FileMetrics]:
    """変更頻度 x 複雑度のホットスポットを分析"""

    # 1. Git log から変更頻度を取得
    result = subprocess.run(
        ['git', 'log', '--format=format:', '--name-only',
         f'--since={days} days ago', '--diff-filter=M'],
        capture_output=True, text=True, cwd=repo_path
    )
    file_changes: dict[str, int] = {}
    for line in result.stdout.strip().split('\n'):
        line = line.strip()
        if line and line.endswith('.py'):
            file_changes[line] = file_changes.get(line, 0) + 1

    # 2. radon で複雑度を取得
    result = subprocess.run(
        ['radon', 'cc', repo_path, '-a', '-j'],
        capture_output=True, text=True
    )
    complexity_data = json.loads(result.stdout)

    # 3. ホットスポットスコアを計算
    metrics = []
    for filepath, change_count in file_changes.items():
        cc = get_file_complexity(complexity_data, filepath)
        score = change_count * cc
        metrics.append(FileMetrics(
            path=filepath,
            change_count=change_count,
            complexity=cc,
            lines=count_lines(Path(repo_path) / filepath),
            hotspot_score=score
        ))

    return sorted(metrics, key=lambda m: m.hotspot_score, reverse=True)


def print_hotspot_report(metrics: list[FileMetrics]) -> None:
    """ホットスポットレポートを出力"""
    print("=" * 75)
    print("  ホットスポット分析レポート")
    print("=" * 75)
    print(f"{'優先度':<8} {'スコア':<8} {'変更':>4} {'CC':>5} {'行数':>6}  {'ファイル'}")
    print("-" * 75)
    for m in metrics[:20]:
        print(f"{m.priority:<8} {m.hotspot_score:>6.1f} {m.change_count:>4} "
              f"{m.complexity:>5.1f} {m.lines:>6}  {m.path}")
```

### 11.2 ホットスポットマトリクス

```
  変更頻度 x 複雑度マトリクス

            変更頻度が高い (20+/四半期)
                 |
   +-------------+-------------+
   |  コードA     |  コードB     |
   |  低複雑度    |  高複雑度    |
   |  高変更      |  高変更      |
   |              |              |
   |  → 監視のみ  | → 最優先     |
   |    問題なし  |    改善対象  |
   +-------------+-------------+
   |  コードC     |  コードD     |
   |  低複雑度    |  高複雑度    |
   |  低変更      |  低変更      |
   |              |              |
   |  → 放置可    | → 次フェーズ |
   |             |    で改善    |
   +-------------+-------------+
                 |
            変更頻度が低い (< 5/四半期)
   低複雑度 (CC<5) ──+── 高複雑度 (CC>10)
```

---

## 12. アンチパターン

### アンチパターン1: スメルの放置（割れ窓理論）

```
  BAD: 放置の連鎖

  最初の放置
    → 「まあいいか、動いてるし」
    → 追加の放置
    → 「前からこうだし」
    → 品質の雪崩的低下
    → 「もう全部書き直すしかない」
    → リファクタリング不能状態

  なぜ危険か:
  - 1つの Long Method を放置すると、同僚も「この長さが許容される」と学習する
  - スメルは「割れた窓」と同じ ── 1つあると急速に増殖する
  - James Q. Wilson & George L. Kelling の「割れ窓理論」がソフトウェアにも適用される

  GOOD: ボーイスカウトルール
  「来た時よりも美しく」── 触ったファイルは少しでもきれいにして去る
  - 全てを一度に直す必要はない
  - 関連する変更のついでに、近くのスメルを1つ修正する
  - 小さな改善の積み重ねが大きな品質向上に繋がる
```

### アンチパターン2: 一度にすべてを直す（Big Bang リファクタリング）

```
  BAD: 全スメルを一括修正

  「週末にコード全体をきれいにしよう！」
    → 巨大な変更セット（500ファイル変更）
    → レビュー不能（差分が1万行）
    → テストが壊れる
    → マージコンフリクト地獄
    → バグ混入リスク
    → 結局 revert

  GOOD: 段階的な改善

  Sprint N:   [Long Method x 3] を修正 (PR #101, #102, #103)
  Sprint N+1: [God Class x 1] を分割  (PR #110)
  Sprint N+2: [Dead Code] を一掃     (PR #120)

  原則:
  - 1スメル1プルリクエスト
  - 各PRは300行以下の差分
  - レビュー可能なサイズ
  - テストが常に通る状態を維持
```

### アンチパターン3: スメルの過剰検出（ツール信仰）

```
  BAD: ツールの警告を全て修正しようとする

  SonarQube: 「Issue 1,247件」
    → 全件修正を目標にする
    → 実際には誤検出(false positive)が30%
    → 優先度の低いスメルにも同じ工数を投入
    → 重要なスメルが埋もれる

  GOOD: トリアージ（優先度分類）

  1. ツールの出力をホットスポット分析と組み合わせる
  2. false positive をルールから除外設定
  3. 影響度 x 変更頻度で優先度をつける
  4. 上位20%のスメルに集中する（パレートの法則）
```

### アンチパターン4: スメルを見つけたら即座にリファクタリング

```
  BAD: デッドライン直前にリファクタリングを始める

  「このコード汚い！リファクタリングしなきゃ！」（リリース2日前）
    → 予定外の変更
    → テストの修正に想定以上の時間
    → リリースが遅れる

  GOOD: 記録して計画的に対処

  1. スメルを発見 → バックログに記録（場所、種類、推定工数）
  2. 現在のタスクを完了
  3. 次のスプリントプランニングで優先度を評価
  4. テストが十分な状態でリファクタリングを実施
```

---

## 13. 演習問題

### 演習1（基本）: スメルの分類

以下のコードに含まれるスメルを全て特定し、分類せよ。

```python
# 問題コード: 以下のスメルを全て特定せよ
import os, sys, json, csv, re  # 使われていないインポートあり

class AppManager:
    """アプリケーション全体を管理するクラス"""
    def __init__(self):
        self.db = Database()
        self.mailer = Mailer()
        self.cache = {}
        self.temp_result = None   # 特定のメソッドでのみ使用

    def process(self, t, n, e, a, c, z, co):
        """ユーザー登録処理"""
        # バリデーション
        if not n:
            raise ValueError("名前が必要")
        if '@' not in e:
            raise ValueError("メールが不正")

        # ユーザー作成
        user = {"type": t, "name": n, "email": e,
                "address": a, "city": c, "zip": z, "country": co}

        # 料金計算
        if t == "premium":
            price = 9800
        elif t == "standard":
            price = 4980
        elif t == "free":
            price = 0

        # DB保存
        self.db.execute(f"INSERT INTO users VALUES ('{n}', '{e}')")

        # メール送信
        self.mailer.send(e, "登録完了", f"ようこそ {n} さん")

        # キャッシュクリア
        self.cache = {}

        self.temp_result = price
        return price
```

**期待される回答**:

| スメル | 分類 | 箇所 |
|--------|------|------|
| Dead Code (未使用インポート) | 不要物 | `os, sys, csv, re` |
| God Class | 肥大化 | `AppManager` が DB・メール・キャッシュ・計算を担当 |
| Long Parameter List | 肥大化 | `process(self, t, n, e, a, c, z, co)` 7パラメータ |
| Data Clumps | 肥大化 | `a, c, z, co` は住所として一塊 |
| Primitive Obsession | 肥大化 | ユーザー型を文字列で判定、メールが `str` |
| Switch Statements | OO乱用 | `if t == "premium"` の分岐 |
| Temporary Field | OO乱用 | `self.temp_result` |
| 不明確な命名 | ― | `t, n, e, a, c, z, co` |
| SQLインジェクション | セキュリティ | `f"INSERT INTO users VALUES ('{n}', '{e}')"` |

---

### 演習2（応用）: スメル除去リファクタリング

演習1のコードを以下の手順でリファクタリングせよ。

1. 未使用インポートを除去
2. Data Clumps を値オブジェクトに抽出
3. Primitive Obsession を型安全に改善
4. Switch Statements をポリモーフィズムで置換
5. God Class を責任ごとに分割
6. Long Parameter List をパラメータオブジェクトに集約

**期待される回答（概要）**:

```python
# 1. 必要なインポートのみ
from dataclasses import dataclass
from abc import ABC, abstractmethod
from decimal import Decimal

# 2. 値オブジェクト
@dataclass(frozen=True)
class Email:
    value: str
    def __post_init__(self):
        if '@' not in self.value:
            raise ValueError(f"不正なメール: {self.value}")

@dataclass(frozen=True)
class Address:
    street: str
    city: str
    zip_code: str
    country: str

# 3. ポリモーフィズムで料金計算
class UserPlan(ABC):
    @abstractmethod
    def monthly_price(self) -> Decimal: ...

class PremiumPlan(UserPlan):
    def monthly_price(self) -> Decimal:
        return Decimal("9800")

class StandardPlan(UserPlan):
    def monthly_price(self) -> Decimal:
        return Decimal("4980")

class FreePlan(UserPlan):
    def monthly_price(self) -> Decimal:
        return Decimal("0")

# 4. パラメータオブジェクト
@dataclass
class RegistrationRequest:
    name: str
    email: Email
    address: Address
    plan: UserPlan

# 5. 責任の分離
class UserRegistrationService:
    def __init__(self, repository, notifier):
        self.repository = repository
        self.notifier = notifier

    def register(self, request: RegistrationRequest) -> Decimal:
        user = User(request.name, request.email, request.address, request.plan)
        self.repository.save(user)
        self.notifier.send_welcome(user)
        return request.plan.monthly_price()
```

---

### 演習3（上級）: ホットスポット分析と改善計画

以下の分析結果を基に、3スプリント分の改善計画を立案せよ。

```
ホットスポット分析結果:
ファイル                      変更回数  CC   行数  スコア
------------------------------------------------------------
src/services/order_service.py    42    18   850   756
src/services/user_service.py     38    15   620   570
src/utils/helpers.py             35     4   200   140
src/api/endpoints.py             30    12   450   360
src/models/payment.py            25     8   300   200
src/config/settings.py           22     2   100    44
src/services/email_service.py    15    10   350   150
src/tests/test_helpers.py        10     3   150    30
```

**期待される回答（概要）**:

```
Sprint 1 (最優先 ── スコア500超):
  1. order_service.py (スコア756)
     - God Class の分割: OrderCreation, OrderPricing, OrderFulfillment
     - Long Method の Extract Method
     - テストカバレッジ確保後にリファクタリング
     推定工数: 8ストーリーポイント

  2. user_service.py (スコア570)
     - Feature Envy の Move Method
     - 複雑なバリデーションの Extract Class
     推定工数: 5ストーリーポイント

Sprint 2 (高優先 ── スコア200-500):
  3. endpoints.py (スコア360)
     - Long Method の Extract Method
     - Controller の薄型化 (ロジックを Service 層に移動)
     推定工数: 5ストーリーポイント

  4. payment.py (スコア200)
     - Primitive Obsession の改善 (金額を Money 値オブジェクトに)
     推定工数: 3ストーリーポイント

Sprint 3 (中優先):
  5. email_service.py (スコア150)
     - 複雑度の削減
  6. helpers.py (スコア140)
     - 高変更頻度だが低複雑度 → 監視継続

  注: settings.py (スコア44), test_helpers.py (スコア30) は放置可
```

---

## 14. FAQ

### Q1: コードスメルは必ず修正すべきか？

いいえ。スメルは「調査すべき兆候」であり、必ずしもリファクタリングが必要とは限らない。以下の判断基準を使う。

**修正すべき場合**:
- 変更頻度が高いファイルに存在する（ホットスポット）
- チーム内で同じスメルが繰り返し問題になっている
- テストの追加・変更が困難になっている
- 新メンバーのオンボーディングを妨げている

**放置してよい場合**:
- 変更頻度が低い（年に1-2回程度）
- 使い捨てコード（プロトタイプ、PoC）
- レガシーシステムで近い将来廃止予定
- 修正コストが得られる利益を大幅に上回る

### Q2: チームでスメルの基準が異なる場合はどうするか？

1. **客観的基準の設定**: SonarQube 等の静的解析ツールで数値基準を設定
   - メソッド行数: 20行以下
   - サイクロマティック複雑度: 10以下
   - クラスの行数: 300行以下
   - パラメータ数: 4つ以下
2. **コーディング規約への明記**: チームで合意した閾値をドキュメント化
3. **コードレビューチェックリスト**: スメル検出項目をレビューの必須確認事項に
4. **定期的な「テクニカルレビュー会」**: 月1回、スメルの実例を共有して認識を統一
5. **新メンバー向けオンボーディング**: スメルの事例集を研修資料に含める

### Q3: レガシーコードのスメルはどこから手をつけるべきか？

**変更頻度が高いファイルから**。以下の手順で科学的にアプローチする。

```bash
# Step 1: 変更頻度の高いファイルを特定
git log --format=format: --name-only --since="6 months ago" \
  | sort | uniq -c | sort -rn | head -20

# Step 2: 複雑度が高いファイルを特定
radon cc src/ -a -nc --min C

# Step 3: 両方に含まれるファイルがホットスポット
```

### Q4: スメルの検出ツールが大量の警告を出す場合、どう対処するか？

1. **トリアージ**: ホットスポット分析で優先度をつける
2. **ベースライン設定**: 現時点の警告数をベースラインとし、「新たな追加を防ぐ」ことに注力
3. **品質ゲート**: 「新しいコードのスメル数が0であること」をPRマージ条件に
4. **段階的な閾値引き下げ**: 四半期ごとに許容される警告数を減らす
5. **false positive の除外**: 不要なルールを `.sonarqube-exclusions` 等で除外設定

### Q5: スメルとデザインパターンの関係は？

スメルとデザインパターンは表裏一体の関係にある。

| スメル | 対応するパターン | 参照 |
|--------|---------------|------|
| Switch Statements | Strategy, State, Factory Method | [design-patterns-guide/00-creational/](../../design-patterns-guide/docs/00-creational/) |
| Feature Envy | Mediator, Facade | [design-patterns-guide/02-behavioral/](../../design-patterns-guide/docs/02-behavioral/) |
| Parallel Hierarchy | Bridge, Abstract Factory | [design-patterns-guide/01-structural/](../../design-patterns-guide/docs/01-structural/) |
| God Class | Facade + 複数の小さなクラスに分割 | [design-patterns-guide/01-structural/](../../design-patterns-guide/docs/01-structural/) |
| Message Chains | Facade, Mediator | [design-patterns-guide/02-behavioral/](../../design-patterns-guide/docs/02-behavioral/) |

### Q6: テストコードにもスメルはあるか？

ある。テストスメル（Test Smell）は本番コードのスメルとは異なる独自のカテゴリを持つ。

| テストスメル | 説明 | 対策 |
|------------|------|------|
| Eager Test | 1つのテストが多くの機能を検証 | テストを分割 |
| Mystery Guest | テストが外部リソースに暗黙に依存 | テストデータをテスト内に |
| Assertion Roulette | 複数のアサーションにメッセージなし | 各アサーションにメッセージ追加 |
| Test Code Duplication | テスト間で同じセットアップが重複 | pytest fixture, @Before で共通化 |
| Slow Tests | テストの実行が遅い | テストダブルの活用、DB の in-memory 化 |

詳細は [テスト原則](../01-practices/04-testing-principles.md) を参照。

---

## 15. まとめ

| 分類 | 代表的スメル | 危険度 | 検出しやすさ | 主な対策 |
|------|------------|:------:|:----------:|---------|
| 肥大化 | Long Method, God Class | 高 | 高 | Extract Method/Class |
| OO乱用 | Switch Statements, Refused Bequest | 中 | 高 | ポリモーフィズム, 委譲 |
| 変更妨害 | Shotgun Surgery, Divergent Change | 高 | 低 | Move Method, Extract Class |
| 不要物 | Dead Code, Speculative Generality | 低 | 高 | Safe Delete |
| 結合過多 | Feature Envy, Message Chains | 高 | 中 | Move Method, Hide Delegate |

| 検出段階 | 手法 | 効果 |
|---------|------|------|
| 自動 (CI) | Ruff, ESLint, SonarQube | 機械的なスメルの60%を自動排除 |
| レビュー | チェックリスト + 人間の判断 | 文脈が必要なスメルの30%を検出 |
| 定期棚卸し | ホットスポット分析, ダッシュボード | 蓄積したスメルの10%を組織的に解消 |

| 原則 | 内容 |
|------|------|
| ボーイスカウトルール | 触ったファイルは少しでもきれいにして去る |
| Rule of Three | 3回目のパターン出現までリファクタリングを待つ |
| パレートの法則 | 上位20%のスメルに集中することで80%の効果を得る |
| ホットスポット分析 | 変更頻度 x 複雑度で科学的に優先度を判断 |

---

## 次に読むべきガイド

- [リファクタリング技法](./01-refactoring-techniques.md) ── スメルを解消する具体的手法（Extract Method, Move Method 等）
- [レガシーコード](./02-legacy-code.md) ── スメルだらけのコードとの向き合い方（Seam, Characterization Test）
- [技術的負債](./03-technical-debt.md) ── スメルの放置がもたらす負債とその返済戦略
- [テスト原則](../01-practices/04-testing-principles.md) ── リファクタリングの安全網としてのテスト設計
- [命名](../01-practices/01-naming.md) ── 名前の改善によるスメルの予防
- [デザインパターン概要](../../design-patterns-guide/docs/00-creational/) ── スメルの構造的な解決策としてのパターン
- [システム設計の基礎](../../system-design-guide/docs/00-fundamentals/) ── アーキテクチャレベルのスメル対策

---

## 参考文献

1. **Martin Fowler** 『Refactoring: Improving the Design of Existing Code』 Addison-Wesley, 2018 (2nd Edition) ── コードスメルの原典。22種のスメルとその対応リファクタリングを網羅。特に Chapter 3 "Bad Smells in Code" が本章の基礎。
2. **Robert C. Martin** 『Clean Code: A Handbook of Agile Software Craftsmanship』 Prentice Hall, 2008 ── Chapter 17 "Smells and Heuristics" で、スメルの嗅覚を磨くためのヒューリスティクスを体系化。
3. **Joshua Kerievsky** 『Refactoring to Patterns』 Addison-Wesley, 2004 ── スメルからデザインパターンへの対応を示した先駆的著作。スメルの解消手段としてパターンを位置づける。
4. **Michael Feathers** 『Working Effectively with Legacy Code』 Prentice Hall, 2004 ── レガシーコード（テストのないコード）に対するスメル検出と安全なリファクタリングの技法。
5. **Mika Mantyla & Casper Lassenius** "Subjective evaluation of software evolvability using code smells: An empirical study" (Empirical Software Engineering, 2006) ── コードスメルの客観的評価に関する学術研究。スメルの深刻度と保守性の相関を実証的に分析。
