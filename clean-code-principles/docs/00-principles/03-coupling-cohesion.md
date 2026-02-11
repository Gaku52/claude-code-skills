# 結合度と凝集度 ── モジュール設計の基盤原則

> 優れたモジュール設計は「低結合・高凝集」に集約される。結合度はモジュール間の依存関係の強さ、凝集度はモジュール内の要素の関連性の強さを示す。この2つの指標を意識することで、変更に強く理解しやすいシステムが構築できる。

---

## この章で学ぶこと

1. **結合度の7段階** ── 内容結合からデータ結合まで、依存関係の種類と危険度を理解する
2. **凝集度の7段階** ── 偶発的凝集から機能的凝集まで、モジュール内のまとまり方を理解する
3. **低結合・高凝集を達成する設計技法** ── 具体的なリファクタリング手法を身につける

---

## 1. 結合度（Coupling）── モジュール間の依存の強さ

### 1.1 結合度の7段階

```
  危険度: 高 ←──────────────────────────────→ 低

  ┌───────────┬───────────┬───────────┬───────────┐
  │ 内容結合   │ 共通結合   │ 制御結合   │ データ結合 │
  │ (Content)  │ (Common)  │ (Control) │ (Data)    │
  │            │           │           │           │
  │ 他モジュール│ グローバル │ フラグで   │ 必要な    │
  │ の内部を   │ 変数を    │ 相手の動作 │ データだけ │
  │ 直接参照   │ 共有      │ を切替     │ を渡す    │
  └───────────┴───────────┴───────────┴───────────┘
       避ける      避ける     最小限に     目指す
```

**コード例1: 結合度の段階別コード**

```python
# --- 内容結合（最悪）: 他モジュールの内部実装に依存 ---
class OrderProcessor:
    def process(self, cart):
        # Cart の内部実装を直接操作
        cart._items[0]._price = cart._items[0]._price * 0.9
        cart._total_cache = None  # キャッシュを手動リセット

# --- 共通結合（悪い）: グローバル変数を共有 ---
GLOBAL_CONFIG = {}

class ServiceA:
    def do_work(self):
        GLOBAL_CONFIG['last_run'] = datetime.now()

class ServiceB:
    def do_work(self):
        if GLOBAL_CONFIG.get('last_run'):  # ServiceA の副作用に依存
            pass

# --- 制御結合（要注意）: フラグで動作を切り替え ---
class ReportGenerator:
    def generate(self, data, format_type: str):
        if format_type == 'pdf':
            return self._generate_pdf(data)
        elif format_type == 'csv':
            return self._generate_csv(data)

# --- データ結合（理想）: 必要なデータのみ受け渡し ---
class TaxCalculator:
    def calculate(self, subtotal: float, tax_rate: float) -> float:
        return subtotal * tax_rate
```

### 1.2 結合度を下げるテクニック

```
  直接依存                     間接依存（抽象を介する）

  ┌───────┐                   ┌───────┐
  │ ModuleA │                  │ ModuleA │
  └───┬───┘                   └───┬───┘
      │ import & new               │ 抽象に依存
      v                           v
  ┌───────┐               ┌─────────────┐
  │ ModuleB │               │ <<interface>>│
  └───────┘               │  IModuleB    │
                            └──────┬──────┘
                                   │ 実装
                                   v
                            ┌───────────┐
                            │ ModuleBImpl│
                            └───────────┘
```

**コード例2: イベント駆動による疎結合化**

```python
# 強結合: OrderService が直接 InventoryService と NotificationService を呼ぶ
class OrderService:
    def __init__(self):
        self.inventory = InventoryService()
        self.notification = NotificationService()
        self.analytics = AnalyticsService()

    def place_order(self, order):
        self.inventory.reduce_stock(order.items)
        self.notification.send_confirmation(order)
        self.analytics.track_purchase(order)

# 疎結合: イベント経由で通知
class EventBus:
    def __init__(self):
        self._handlers: dict[str, list] = {}

    def subscribe(self, event_type: str, handler):
        self._handlers.setdefault(event_type, []).append(handler)

    def publish(self, event_type: str, data):
        for handler in self._handlers.get(event_type, []):
            handler(data)

class OrderService:
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus

    def place_order(self, order):
        # OrderService は他のサービスの存在を知らない
        order.confirm()
        self.event_bus.publish('order_placed', order)
```

---

## 2. 凝集度（Cohesion）── モジュール内の要素の関連性

### 2.1 凝集度の7段階

```
  品質: 低 ←──────────────────────────────→ 高

  ┌──────────┬──────────┬──────────┬──────────┐
  │ 偶発的    │ 論理的    │ 時間的    │ 機能的    │
  │ 凝集      │ 凝集      │ 凝集      │ 凝集      │
  │           │           │           │           │
  │ 無関係な  │ 似た種類  │ 同時に    │ 1つの     │
  │ 要素の    │ を集めた  │ 実行する  │ 明確な    │
  │ 寄せ集め  │ だけ      │ だけ      │ 責任      │
  └──────────┴──────────┴──────────┴──────────┘
     避ける     避ける     注意が必要   目指す
```

**コード例3: 凝集度の段階別**

```java
// --- 偶発的凝集（最低）: 無関係な機能の寄せ集め ---
class Utilities {
    public static String formatDate(Date d) { ... }
    public static double calculateTax(double amount) { ... }
    public static void sendEmail(String to, String body) { ... }
    public static Image resizeImage(Image img, int w, int h) { ... }
}

// --- 論理的凝集（低い）: 似た種類を集めただけ ---
class InputHandler {
    public void handleMouseInput(MouseEvent e) { ... }
    public void handleKeyboardInput(KeyEvent e) { ... }
    public void handleTouchInput(TouchEvent e) { ... }
    public void handleGamepadInput(GamepadEvent e) { ... }
}

// --- 時間的凝集（中程度）: 同じタイミングで実行するだけ ---
class AppInitializer {
    public void initialize() {
        loadConfig();
        initDatabase();
        startWebServer();
        registerShutdownHook();
    }
}

// --- 機能的凝集（最高）: 1つの明確な責任 ---
class PasswordHasher {
    private final int saltLength;
    private final int iterations;

    public PasswordHasher(int saltLength, int iterations) {
        this.saltLength = saltLength;
        this.iterations = iterations;
    }

    public String hash(String password) {
        byte[] salt = generateSalt();
        return pbkdf2(password, salt, iterations);
    }

    public boolean verify(String password, String hashedPassword) {
        byte[] salt = extractSalt(hashedPassword);
        String rehashed = pbkdf2(password, salt, iterations);
        return constantTimeEquals(rehashed, hashedPassword);
    }

    private byte[] generateSalt() { ... }
    private byte[] extractSalt(String hash) { ... }
    private String pbkdf2(String password, byte[] salt, int iterations) { ... }
}
```

---

## 3. 低結合・高凝集の実現パターン

**コード例4: Facadeパターンで結合度を管理**

```typescript
// 高結合: クライアントが複数のサブシステムに直接依存
class OrderPage {
  placeOrder(cart: Cart) {
    const inventory = new InventorySystem();
    const payment = new PaymentGateway();
    const shipping = new ShippingCalculator();
    const notification = new EmailService();

    inventory.check(cart.items);
    const total = shipping.calculate(cart);
    payment.charge(cart.customer, total);
    notification.sendConfirmation(cart.customer.email);
  }
}

// Facade で結合を集約
class OrderFacade {
  constructor(
    private inventory: InventorySystem,
    private payment: PaymentGateway,
    private shipping: ShippingCalculator,
    private notification: EmailService
  ) {}

  placeOrder(cart: Cart): OrderResult {
    this.inventory.check(cart.items);
    const total = this.shipping.calculate(cart);
    const paymentResult = this.payment.charge(cart.customer, total);
    this.notification.sendConfirmation(cart.customer.email);
    return new OrderResult(paymentResult.transactionId);
  }
}

// クライアントは Facade だけに依存
class OrderPage {
  constructor(private orderFacade: OrderFacade) {}

  placeOrder(cart: Cart) {
    return this.orderFacade.placeOrder(cart);
  }
}
```

**コード例5: パッケージ構造で凝集度を表現**

```
# 低凝集なパッケージ構成（技術レイヤー基準）
src/
  controllers/
    UserController.ts
    OrderController.ts
    ProductController.ts
  services/
    UserService.ts
    OrderService.ts
    ProductService.ts
  repositories/
    UserRepository.ts
    OrderRepository.ts
    ProductRepository.ts

# 高凝集なパッケージ構成（ドメイン基準）
src/
  user/
    UserController.ts
    UserService.ts
    UserRepository.ts
    User.ts
  order/
    OrderController.ts
    OrderService.ts
    OrderRepository.ts
    Order.ts
  product/
    ProductController.ts
    ProductService.ts
    ProductRepository.ts
    Product.ts
```

---

## 4. 結合度と凝集度の関係

| 組み合わせ | 低結合 | 高結合 |
|-----------|--------|--------|
| **高凝集** | 理想的。独立した明確なモジュール | 責任は明確だが依存が多い |
| **低凝集** | 依存は少ないがモジュールの意味が不明 | 最悪。スパゲッティコード |

| 改善アプローチ | 対象 | 具体的手法 |
|--------------|------|-----------|
| 結合度を下げる | モジュール間 | DI、インターフェース、イベント駆動 |
| 凝集度を上げる | モジュール内 | Extract Class、Move Method |
| 両方同時に改善 | アーキテクチャ | ドメイン駆動設計（DDD） |

---

## 5. アンチパターン

### アンチパターン1: God Module（低凝集の極致）

```python
# 1つのモジュールがシステム全体の機能を持つ
class ApplicationManager:
    def authenticate_user(self, ...): ...
    def process_payment(self, ...): ...
    def generate_invoice(self, ...): ...
    def send_notification(self, ...): ...
    def update_inventory(self, ...): ...
    def calculate_shipping(self, ...): ...
    def manage_cache(self, ...): ...
    # 全ドメインの知識がここに集中
```

### アンチパターン2: Shotgun Surgery（高結合の結果）

```python
# 1つの変更が多数のファイルに波及する
# 「消費税率を8%→10%に変更」で修正が必要なファイル:
# - cart.py (税計算)
# - invoice.py (請求書の税額)
# - receipt.py (領収書の税額)
# - report.py (レポートの税表示)
# - api.py (APIレスポンスの税額)
# - frontend/cart.js (フロントの税表示)
# → 税率がグローバル定数やDRY化されていない証拠
```

---

## 6. FAQ

### Q1: 結合度ゼロは目指すべきか？

結合度ゼロは不可能であり、目指すべきでもない。モジュール間の通信がなければシステムは機能しない。目指すのは**必要最小限の、明示的な結合**。暗黙の依存（グローバル変数、隠れた副作用）を排除し、明示的なインターフェースを通じた依存に置き換えることが重要。

### Q2: マイクロサービスにすれば自動的に低結合になるか？

ならない。分散システムでもサービス間の結合は存在する。共有データベース、同期的なAPI呼び出しチェーン、共有ライブラリによる結合は、モノリス以上に管理が困難になる場合がある。

### Q3: ユーティリティクラスは凝集度が低いので作るべきではないか？

「何でも入りのUtils」は避けるべきだが、**明確なテーマを持ったユーティリティ**（例: `StringUtils`, `DateUtils`）は論理的凝集であり実用上許容される。ただし、ドメインロジックがユーティリティに漏れていないか定期的に検証する。

---

## まとめ

| 指標 | 目標 | 達成手段 | 測定方法 |
|------|------|---------|---------|
| 結合度 | 低く保つ | DI、IF、イベント | 依存グラフの分析 |
| 凝集度 | 高く保つ | SRP、Extract Class | LCOM（凝集度メトリクス） |
| バランス | 低結合+高凝集 | ドメイン駆動設計 | 変更影響範囲の分析 |

---

## 次に読むべきガイド

- [デメテルの法則](./04-law-of-demeter.md) ── 結合度を下げるための具体的規則
- [SOLID原則](./01-solid.md) ── 特にSRPとDIPが結合度・凝集度に直結
- [合成 vs 継承](../03-practices-advanced/01-composition-over-inheritance.md) ── 結合度に影響する設計判断

---

## 参考文献

1. **Glenford J. Myers** 『Composite/Structured Design』 Van Nostrand Reinhold, 1978
2. **Larry Constantine, Edward Yourdon** 『Structured Design: Fundamentals of a Discipline of Computer Program and Systems Design』 Yourdon Press, 1979
3. **Robert C. Martin** 『Clean Architecture: A Craftsman's Guide to Software Structure and Design』 Prentice Hall, 2017
4. **Eric Evans** 『Domain-Driven Design: Tackling Complexity in the Heart of Software』 Addison-Wesley, 2003
