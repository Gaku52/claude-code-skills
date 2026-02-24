# OOPアンチパターン

> アンチパターンは「よくある悪い設計」。God Object、深い継承階層、Anemic Domain Model など、OOP で陥りがちな罠とその回避方法を解説する。アンチパターンを認識し、適切にリファクタリングできることは、中級から上級エンジニアへのステップアップに不可欠なスキルである。

## この章で学ぶこと

- [ ] 主要なOOPアンチパターンを認識できるようになる
- [ ] 各アンチパターンの問題点と発生原因を理解する
- [ ] リファクタリングによる解決方法を学ぶ
- [ ] コードレビューでアンチパターンを指摘できるようになる
- [ ] 予防的な設計手法を身につける

---

## 1. God Object（神オブジェクト）

### 1.1 概要と症状

God Object は最も一般的で最も有害なアンチパターンの一つである。一つのクラスがアプリケーションの多くの責任を担い、事実上の「全知全能」オブジェクトになってしまう現象を指す。

```
症状:
  → 1つのクラスが全てを知り、全てを行う
  → 1000行以上のクラス
  → 20以上のメソッド
  → あらゆるクラスがこのクラスに依存
  → メソッド名に統一感がない（create, process, handle, manage...）
  → 複数の異なるドメイン概念が混在

原因:
  → 「とりあえずここに追加」の積み重ね
  → 責任の分離を意識しない開発
  → 最初から設計せずにコードを追加し続けた結果
  → チームメンバー間でクラス設計の合意がない

問題:
  → 変更が困難（影響範囲が広大）
  → テスト困難（依存が多すぎる）
  → チーム開発でコンフリクト多発
  → コンパイル時間の増大
  → メモリ使用量の増大（不要なデータもロードされる）
```

### 1.2 God Object の発生パターン

```
時間経過による肥大化:

 Month 1:  [UserService: 100行]     ← 最初は適切
 Month 3:  [UserService: 300行]     ← 「ちょっと追加」
 Month 6:  [UserService: 800行]     ← 「関連するから」
 Month 12: [UserService: 2000行]    ← God Object 完成
 Month 18: [UserService: 5000行]    ← 誰も触りたくない

依存関係の増大:
                 ┌──────────┐
    ┌───────────►│          │◄───────────┐
    │            │   God    │            │
    │   ┌──────►│  Object  │◄──────┐   │
    │   │        │          │        │   │
    │   │        └──────────┘        │   │
    │   │         ▲  ▲  ▲           │   │
  [A] [B]       [C][D][E]         [F] [G]

  → 全てのクラスが God Object に依存
  → God Object の変更が全体に波及
```

### 1.3 具体例と問題

```typescript
// ❌ God Object の典型例
class ApplicationManager {
  // ユーザー管理
  private users: Map<string, User> = new Map();
  private sessions: Map<string, Session> = new Map();

  createUser(data: any) { /* ... */ }
  deleteUser(id: string) { /* ... */ }
  authenticateUser(email: string, password: string) { /* ... */ }
  updateUserProfile(id: string, profile: any) { /* ... */ }
  getUserPermissions(id: string) { /* ... */ }

  // 注文管理
  private orders: Map<string, Order> = new Map();

  createOrder(userId: string, items: any[]) { /* ... */ }
  cancelOrder(orderId: string) { /* ... */ }
  getOrderHistory(userId: string) { /* ... */ }
  updateOrderStatus(orderId: string, status: string) { /* ... */ }

  // 支払い管理
  processPayment(orderId: string, card: any) { /* ... */ }
  refundPayment(paymentId: string) { /* ... */ }
  getPaymentHistory(userId: string) { /* ... */ }

  // 通知管理
  sendEmail(to: string, body: string) { /* ... */ }
  sendSms(to: string, body: string) { /* ... */ }
  sendPushNotification(userId: string, message: string) { /* ... */ }

  // レポート
  generateMonthlyReport() { /* ... */ }
  generateUserReport(userId: string) { /* ... */ }
  generateSalesReport(from: Date, to: Date) { /* ... */ }

  // 設定管理
  getConfig(key: string) { /* ... */ }
  updateConfig(key: string, value: any) { /* ... */ }

  // キャッシュ管理
  clearCache() { /* ... */ }
  warmUpCache() { /* ... */ }

  // ... さらに100メソッド以上
}
```

### 1.4 段階的リファクタリング手順

God Object のリファクタリングは一度にやるのではなく、段階的に進めるのが重要である。

```
リファクタリング戦略:

Phase 1: 責任の識別
  → メソッドをカテゴリごとにグループ化
  → 依存関係を可視化
  → 分割境界を決定

Phase 2: インターフェース抽出
  → 各グループのインターフェースを定義
  → God Object にインターフェースを実装させる
  → 利用側をインターフェース経由に変更

Phase 3: クラス抽出
  → インターフェースごとに新しいクラスを作成
  → God Object からロジックを移動
  → God Object はファサードとして残す（一時的）

Phase 4: ファサード除去
  → 利用側を新しいクラスに直接接続
  → God Object を完全に削除
```

```typescript
// Phase 1: 責任の識別（コメントでグループ化）

// Phase 2: インターフェース抽出
interface IUserService {
  createUser(data: CreateUserDTO): Promise<User>;
  deleteUser(id: string): Promise<void>;
  authenticateUser(email: string, password: string): Promise<AuthResult>;
  updateUserProfile(id: string, profile: ProfileDTO): Promise<User>;
}

interface IOrderService {
  createOrder(userId: string, items: OrderItemDTO[]): Promise<Order>;
  cancelOrder(orderId: string): Promise<void>;
  getOrderHistory(userId: string): Promise<Order[]>;
}

interface IPaymentService {
  processPayment(orderId: string, card: CardDTO): Promise<PaymentResult>;
  refundPayment(paymentId: string): Promise<void>;
}

interface INotificationService {
  sendEmail(to: string, body: string): Promise<void>;
  sendSms(to: string, body: string): Promise<void>;
  sendPushNotification(userId: string, message: string): Promise<void>;
}

// Phase 3: クラス抽出
class UserService implements IUserService {
  constructor(
    private readonly userRepo: UserRepository,
    private readonly passwordHasher: PasswordHasher,
    private readonly eventBus: EventBus,
  ) {}

  async createUser(data: CreateUserDTO): Promise<User> {
    const hashedPassword = await this.passwordHasher.hash(data.password);
    const user = new User({
      email: Email.create(data.email),
      password: hashedPassword,
      name: data.name,
    });
    await this.userRepo.save(user);
    await this.eventBus.publish(new UserCreatedEvent(user.id));
    return user;
  }

  async deleteUser(id: string): Promise<void> {
    const user = await this.userRepo.findById(id);
    if (!user) throw new UserNotFoundError(id);
    await this.userRepo.delete(id);
    await this.eventBus.publish(new UserDeletedEvent(id));
  }

  async authenticateUser(email: string, password: string): Promise<AuthResult> {
    const user = await this.userRepo.findByEmail(email);
    if (!user) throw new AuthenticationError("Invalid credentials");
    const isValid = await this.passwordHasher.verify(password, user.password);
    if (!isValid) throw new AuthenticationError("Invalid credentials");
    return { userId: user.id, token: this.generateToken(user) };
  }

  async updateUserProfile(id: string, profile: ProfileDTO): Promise<User> {
    const user = await this.userRepo.findById(id);
    if (!user) throw new UserNotFoundError(id);
    user.updateProfile(profile);
    await this.userRepo.save(user);
    return user;
  }

  private generateToken(user: User): string {
    // JWT生成ロジック
    return "token";
  }
}

class OrderService implements IOrderService {
  constructor(
    private readonly orderRepo: OrderRepository,
    private readonly userService: IUserService,
    private readonly eventBus: EventBus,
  ) {}

  async createOrder(userId: string, items: OrderItemDTO[]): Promise<Order> {
    const order = Order.create(userId, items);
    await this.orderRepo.save(order);
    await this.eventBus.publish(new OrderCreatedEvent(order.id));
    return order;
  }

  async cancelOrder(orderId: string): Promise<void> {
    const order = await this.orderRepo.findById(orderId);
    if (!order) throw new OrderNotFoundError(orderId);
    order.cancel();
    await this.orderRepo.save(order);
    await this.eventBus.publish(new OrderCancelledEvent(orderId));
  }

  async getOrderHistory(userId: string): Promise<Order[]> {
    return this.orderRepo.findByUserId(userId);
  }
}

// Phase 4: 利用側の変更
// Before: const app = new ApplicationManager(); app.createUser(...);
// After:  直接各サービスを利用
class OrderController {
  constructor(
    private readonly orderService: IOrderService,
    private readonly paymentService: IPaymentService,
    private readonly notificationService: INotificationService,
  ) {}

  async placeOrder(req: Request): Promise<Response> {
    const order = await this.orderService.createOrder(req.userId, req.items);
    const payment = await this.paymentService.processPayment(order.id, req.card);
    await this.notificationService.sendEmail(req.userEmail, "注文完了");
    return { orderId: order.id, paymentId: payment.id };
  }
}
```

### 1.5 God Object を防ぐルール

```
予防策:
  1. 新しいメソッドを追加する前に「このクラスの責任か？」と問う
  2. クラスの命名で責任を明確化（Manager/Handler は危険信号）
  3. 200行を超えたら分割を検討する
  4. コードレビューで責任の混在を指摘する
  5. 静的解析ツールでクラスの複雑度をモニタリング

危険な命名パターン:
  ❌ ApplicationManager
  ❌ SystemHelper
  ❌ Utility
  ❌ CommonService
  ❌ MainController
  → これらの名前は「何でも入る」ことを暗示する
```

---

## 2. Anemic Domain Model（貧血ドメインモデル）

### 2.1 概要と症状

Anemic Domain Model は Martin Fowler が「ドメイン駆動設計の最大のアンチパターン」と呼んだ問題である。ドメインオブジェクトがデータの入れ物でしかなく、ビジネスロジックが全てサービスクラスに流出している状態を指す。

```
症状:
  → クラスがデータ（getter/setter）だけを持つ
  → ビジネスロジックが全てサービスクラスにある
  → ドメインオブジェクトが単なるデータの入れ物
  → 不変条件（invariant）がどこにも強制されない
  → 同じバリデーションロジックが複数箇所に散在

原因:
  → 「データクラス + サービスクラス」の分離が目的化
  → DTOとドメインモデルの混同
  → データベーステーブルと1:1でクラスを作る習慣
  → 手続き型プログラミングの思考が残っている

問題:
  → オブジェクト指向の利点が活かせない
  → ビジネスルールの重複
  → 不正な状態を防げない
  → テストでサービスとデータの両方を用意する必要
  → ドメイン知識がコードに表現されない
```

### 2.2 Anemic vs Rich の比較図

```
Anemic Domain Model:

  ┌─────────────┐         ┌─────────────────┐
  │   Order      │         │  OrderService    │
  │  (データだけ) │◄────────│  (ロジックだけ)   │
  │             │         │                 │
  │ id          │         │ calculateTotal()│
  │ items[]     │         │ canCancel()     │
  │ status      │         │ cancel()        │
  │ totalPrice  │         │ applyDiscount() │
  │ createdAt   │         │ validate()      │
  └─────────────┘         └─────────────────┘
  ※ データとロジックが分離 → 不正な状態が可能


Rich Domain Model:

  ┌──────────────────────────┐
  │   Order                   │
  │  (データ + ロジック)        │
  │                          │
  │ - id: OrderId            │
  │ - items: OrderItem[]     │
  │ - status: OrderStatus    │
  │                          │
  │ + addItem(item)          │
  │ + removeItem(itemId)     │
  │ + totalPrice: Money      │
  │ + confirm()              │
  │ + cancel()               │
  │ + applyDiscount(coupon)  │
  └──────────────────────────┘
  ※ データとロジックが一体 → 不正な状態を型で防ぐ
```

### 2.3 具体例：Before / After

```typescript
// ❌ Anemic Domain Model
class Order {
  id: string = "";
  items: OrderItem[] = [];
  status: string = "pending";
  totalPrice: number = 0;
  discount: number = 0;
  shippingAddress: string = "";
  createdAt: Date = new Date();
  // getter/setter だけ。ロジックなし
}

class OrderService {
  // 全てのロジックがサービスに
  calculateTotal(order: Order): number {
    order.totalPrice = order.items.reduce(
      (sum, item) => sum + item.price * item.quantity, 0
    );
    return order.totalPrice;
  }

  applyDiscount(order: Order, discountRate: number): void {
    // バリデーションがサービス側にある
    if (discountRate < 0 || discountRate > 0.5) {
      throw new Error("Invalid discount rate");
    }
    order.discount = order.totalPrice * discountRate;
    order.totalPrice -= order.discount;
  }

  canCancel(order: Order): boolean {
    return order.status === "pending" || order.status === "confirmed";
  }

  cancel(order: Order): void {
    if (!this.canCancel(order)) throw new Error("Cannot cancel");
    order.status = "cancelled";
  }

  validate(order: Order): string[] {
    const errors: string[] = [];
    if (order.items.length === 0) errors.push("商品が必要です");
    if (!order.shippingAddress) errors.push("配送先が必要です");
    if (order.totalPrice < 0) errors.push("合計金額が不正です");
    return errors;
  }
}

// 問題: 不正な状態を簡単に作れてしまう
const order = new Order();
order.status = "shipped";      // いきなり出荷済み？
order.totalPrice = -1000;      // 負の金額？
order.items = [];              // 商品なしで出荷済み？
// → 何のエラーも発生しない！
```

```typescript
// ✅ Rich Domain Model
type OrderStatus = "pending" | "confirmed" | "shipped" | "delivered" | "cancelled";

class OrderId {
  constructor(private readonly value: string) {
    if (!value || value.length === 0) {
      throw new Error("OrderId cannot be empty");
    }
  }
  toString(): string { return this.value; }
  equals(other: OrderId): boolean { return this.value === other.value; }
}

class Order {
  private _items: OrderItem[] = [];
  private _status: OrderStatus = "pending";
  private _discount: Money = Money.zero("JPY");
  private _shippingAddress?: Address;
  private readonly _createdAt: Date = new Date();

  constructor(private readonly _id: OrderId) {}

  // ビジネスルールを持つメソッド
  addItem(item: OrderItem): void {
    if (this._status !== "pending") {
      throw new DomainError("確定済みの注文には追加できません");
    }
    const existing = this._items.find(i => i.productId === item.productId);
    if (existing) {
      existing.increaseQuantity(item.quantity);
    } else {
      this._items.push(item);
    }
  }

  removeItem(productId: string): void {
    if (this._status !== "pending") {
      throw new DomainError("確定済みの注文からは削除できません");
    }
    const index = this._items.findIndex(i => i.productId === productId);
    if (index === -1) throw new DomainError("商品が見つかりません");
    this._items.splice(index, 1);
  }

  // 計算ロジックがドメインオブジェクト内
  get subtotal(): Money {
    return this._items.reduce(
      (sum, item) => sum.add(item.totalPrice),
      Money.zero("JPY")
    );
  }

  get totalPrice(): Money {
    return this.subtotal.subtract(this._discount);
  }

  applyDiscount(coupon: Coupon): void {
    if (this._status !== "pending") {
      throw new DomainError("確定済みの注文に割引は適用できません");
    }
    if (!coupon.isValid()) {
      throw new DomainError("無効なクーポンです");
    }
    this._discount = coupon.calculateDiscount(this.subtotal);
  }

  setShippingAddress(address: Address): void {
    if (this._status === "shipped" || this._status === "delivered") {
      throw new DomainError("発送済みの注文の配送先は変更できません");
    }
    this._shippingAddress = address;
  }

  // 状態遷移を管理するメソッド
  confirm(): void {
    if (this._items.length === 0) {
      throw new DomainError("商品がない注文は確定できません");
    }
    if (!this._shippingAddress) {
      throw new DomainError("配送先が設定されていません");
    }
    if (this._status !== "pending") {
      throw new DomainError(`${this._status} の注文は確定できません`);
    }
    this._status = "confirmed";
  }

  ship(): void {
    if (this._status !== "confirmed") {
      throw new DomainError("確定済みの注文のみ発送できます");
    }
    this._status = "shipped";
  }

  deliver(): void {
    if (this._status !== "shipped") {
      throw new DomainError("発送済みの注文のみ配達完了にできます");
    }
    this._status = "delivered";
  }

  cancel(): void {
    if (this._status === "shipped" || this._status === "delivered") {
      throw new DomainError("発送済みの注文はキャンセルできません");
    }
    this._status = "cancelled";
  }

  get id(): OrderId { return this._id; }
  get status(): OrderStatus { return this._status; }
  get items(): ReadonlyArray<OrderItem> { return [...this._items]; }
  get createdAt(): Date { return this._createdAt; }
}

// → 不正な状態を作ることができない
// → ビジネスルールがオブジェクト内に集約
// → テストが明確（Orderだけテストすれば良い）
```

### 2.4 状態遷移図で見る Rich Domain Model

```
Order の状態遷移:

  ┌─────────┐  confirm()  ┌───────────┐  ship()  ┌─────────┐  deliver()  ┌───────────┐
  │ pending  │───────────►│ confirmed │────────►│ shipped │───────────►│ delivered │
  └─────────┘             └───────────┘         └─────────┘            └───────────┘
       │                       │
       │     cancel()          │  cancel()
       │                       │
       ▼                       ▼
  ┌───────────┐          ┌───────────┐
  │ cancelled │          │ cancelled │
  └───────────┘          └───────────┘

  → 各状態遷移にビジネスルールが埋め込まれている
  → 不正な遷移（例: pending → delivered）は例外で防止
```

### 2.5 Java での Rich Domain Model 例

```java
// ✅ Java での Rich Domain Model
public class BankAccount {
    private final AccountId id;
    private Money balance;
    private AccountStatus status;
    private final List<Transaction> transactions;

    private BankAccount(AccountId id, Money initialDeposit) {
        if (initialDeposit.isLessThan(Money.of(1000, "JPY"))) {
            throw new DomainException("最低入金額は1000円です");
        }
        this.id = id;
        this.balance = initialDeposit;
        this.status = AccountStatus.ACTIVE;
        this.transactions = new ArrayList<>();
        this.transactions.add(Transaction.initialDeposit(initialDeposit));
    }

    public static BankAccount open(AccountId id, Money initialDeposit) {
        return new BankAccount(id, initialDeposit);
    }

    public void deposit(Money amount) {
        ensureActive();
        if (amount.isLessThanOrEqual(Money.zero("JPY"))) {
            throw new DomainException("入金額は正の数である必要があります");
        }
        this.balance = this.balance.add(amount);
        this.transactions.add(Transaction.deposit(amount));
    }

    public void withdraw(Money amount) {
        ensureActive();
        if (amount.isLessThanOrEqual(Money.zero("JPY"))) {
            throw new DomainException("出金額は正の数である必要があります");
        }
        if (this.balance.isLessThan(amount)) {
            throw new InsufficientFundsException(this.balance, amount);
        }
        this.balance = this.balance.subtract(amount);
        this.transactions.add(Transaction.withdrawal(amount));
    }

    public void freeze() {
        ensureActive();
        this.status = AccountStatus.FROZEN;
    }

    public void close() {
        if (this.balance.isGreaterThan(Money.zero("JPY"))) {
            throw new DomainException("残高がある口座は閉鎖できません");
        }
        this.status = AccountStatus.CLOSED;
    }

    private void ensureActive() {
        if (this.status != AccountStatus.ACTIVE) {
            throw new DomainException("アクティブな口座でのみ操作可能です");
        }
    }

    // 不変条件が常に保証される
    public Money getBalance() { return this.balance; }
    public AccountStatus getStatus() { return this.status; }
    public List<Transaction> getTransactionHistory() {
        return Collections.unmodifiableList(this.transactions);
    }
}
```

### 2.6 Anemic Model が適切な場面

全てのケースで Rich Domain Model が最適とは限らない。以下の場面では Anemic Model が合理的な場合もある。

```
Anemic Model が許容される場面:
  → 単純な CRUD アプリケーション（ビジネスロジックが少ない）
  → DTO（データ転送オブジェクト）としての利用
  → 外部 API との通信用オブジェクト
  → レポート/集計用の読み取り専用データ

Rich Domain Model が必須な場面:
  → 複雑なビジネスルールが存在する
  → 状態遷移の管理が重要
  → ドメインエキスパートとの共通言語が必要
  → 不正な状態を確実に防ぐ必要がある
```

---

## 3. 深い継承階層（Deep Inheritance Hierarchy）

### 3.1 概要と症状

```
症状:
  → 4段階以上の継承チェーン
  → 各レイヤーの変更が下位全てに影響
  → どのメソッドがどのレイヤーで定義されたか不明
  → super.super.method() のような呼び出し
  → 一つの変更で予想外の動作変更が発生

  Entity
  └── LivingEntity
      └── Animal
          └── Mammal
              └── Dog
                  └── GuideDog
                      └── TrainedGuideDog

  → 7段階！ TrainedGuideDog のバグを直すには
     全レイヤーを理解する必要がある

解決:
  → 継承は2-3段階まで
  → コンポジションで機能を組み合わせ
  → インターフェースで型の関係を表現
```

### 3.2 深い継承の問題を図解

```
メソッド解決の複雑さ:

  TrainedGuideDog.walk() はどこで定義されている？

  TrainedGuideDog   → walk() なし
  GuideDog          → walk() なし
  Dog               → walk() あり！ ... でも super.walk() を呼んでいる
  Mammal            → walk() あり！ ... でもこちらも super.move() を呼ぶ
  Animal            → move() あり！
  LivingEntity      → なし
  Entity            → なし

  → 「Yo-Yo 問題」: 上下に行ったり来たりしないと理解できない

脆い基底クラス問題（Fragile Base Class Problem）:

  Animal に sound() メソッドを追加:
    → Mammal: 問題なし
    → Dog: 問題なし
    → GuideDog: sound() が既に定義済み → 意図しないオーバーライド！
    → TrainedGuideDog: 動作が変わる → テスト失敗！
```

### 3.3 リファクタリング例

```typescript
// ❌ 深い継承
class BaseComponent { /* 基本機能 */ }
class StyledComponent extends BaseComponent { /* スタイル */ }
class InteractiveComponent extends StyledComponent { /* インタラクション */ }
class AnimatedComponent extends InteractiveComponent { /* アニメーション */ }
class AccessibleComponent extends AnimatedComponent { /* a11y */ }
class MyButton extends AccessibleComponent { /* ボタン */ }

// ✅ コンポジション + インターフェース
interface Stylable { applyStyles(styles: Styles): void; }
interface Interactive { onClick(handler: () => void): void; }
interface Animatable { animate(animation: Animation): void; }
interface Accessible { setAriaLabel(label: string): void; }

class StyleEngine implements Stylable {
  applyStyles(styles: Styles): void {
    // スタイル適用ロジック
  }
}

class AnimationEngine implements Animatable {
  animate(animation: Animation): void {
    // アニメーションロジック
  }
}

class AccessibilityManager implements Accessible {
  setAriaLabel(label: string): void {
    // ARIA属性設定ロジック
  }
}

class MyButton implements Stylable, Interactive, Animatable, Accessible {
  private styleEngine: StyleEngine;
  private animationEngine: AnimationEngine;
  private accessibilityManager: AccessibilityManager;

  constructor(deps: ButtonDeps) {
    this.styleEngine = deps.styleEngine;
    this.animationEngine = deps.animationEngine;
    this.accessibilityManager = deps.accessibilityManager;
  }

  applyStyles(styles: Styles): void {
    this.styleEngine.applyStyles(styles);
  }

  onClick(handler: () => void): void {
    // クリックイベント処理
  }

  animate(animation: Animation): void {
    this.animationEngine.animate(animation);
  }

  setAriaLabel(label: string): void {
    this.accessibilityManager.setAriaLabel(label);
  }
}
```

### 3.4 Python での Mixin パターン

Python では多重継承を利用した Mixin パターンが、深い単一継承の代替として使える。

```python
# ❌ 深い単一継承
class BaseModel:
    def save(self): pass

class TimestampedModel(BaseModel):
    created_at: datetime
    updated_at: datetime

class SoftDeletableModel(TimestampedModel):
    deleted_at: datetime | None = None

    def soft_delete(self):
        self.deleted_at = datetime.now()

class AuditableModel(SoftDeletableModel):
    created_by: str
    updated_by: str

class VersionedModel(AuditableModel):
    version: int = 1

class Product(VersionedModel):  # 5段階の継承！
    name: str
    price: float


# ✅ Mixin パターン
class TimestampMixin:
    """タイムスタンプ機能"""
    created_at: datetime
    updated_at: datetime

    def touch(self):
        self.updated_at = datetime.now()

class SoftDeleteMixin:
    """論理削除機能"""
    deleted_at: datetime | None = None

    def soft_delete(self):
        self.deleted_at = datetime.now()

    @property
    def is_deleted(self) -> bool:
        return self.deleted_at is not None

class AuditMixin:
    """監査ログ機能"""
    created_by: str
    updated_by: str

class VersionMixin:
    """バージョニング機能"""
    version: int = 1

    def increment_version(self):
        self.version += 1

class BaseModel:
    """基底モデル"""
    def save(self): pass

class Product(BaseModel, TimestampMixin, SoftDeleteMixin, AuditMixin, VersionMixin):
    """必要な機能だけをMixinで組み合わせ（継承は1段階）"""
    name: str
    price: float

# 別のモデルには異なる組み合わせ
class LogEntry(BaseModel, TimestampMixin):
    """ログエントリ: タイムスタンプだけ必要"""
    message: str
    level: str
```

### 3.5 継承とコンポジションの使い分け

```
判断基準:

  ┌─────────────────────┬────────────────────┬────────────────────┐
  │ 基準                │ 継承を使う          │ コンポジションを使う │
  ├─────────────────────┼────────────────────┼────────────────────┤
  │ 関係性              │ "is-a" 関係         │ "has-a" 関係        │
  │ 例                  │ Cat is an Animal   │ Car has an Engine  │
  │ 再利用の方向        │ 共通の振る舞い       │ 機能の組み合わせ     │
  │ 変更頻度            │ 基底クラスが安定     │ 機能が独立に変化     │
  │ 柔軟性              │ コンパイル時に決定   │ 実行時に変更可能     │
  │ 段階数              │ 2-3段階まで         │ 制限なし            │
  └─────────────────────┴────────────────────┴────────────────────┘

  原則: 迷ったらコンポジション（Composition over Inheritance）
```

---

## 4. Circular Dependency（循環依存）

### 4.1 概要と症状

```
循環依存の構造:

  直接循環:
    A → B → A

  間接循環:
    A → B → C → A

  複雑な循環:
    A → B → C → D
    ↑           │
    └───────────┘

症状:
  → コンパイルエラーまたは実行時エラー
  → モジュールのロード順序の問題
  → テストで一方をモックできない
  → 単独でのデプロイが不可能
  → 変更がカスケード的に伝播

原因:
  → 設計時にモジュール境界を考えていない
  → 便宜的に双方向の参照を追加
  → 共通モジュールの抽出不足
```

### 4.2 循環依存の具体例と解決

```typescript
// ❌ 循環依存の例
// user.ts
import { Order } from "./order";

class User {
  orders: Order[] = [];

  getActiveOrders(): Order[] {
    return this.orders.filter(o => o.isActive());
  }

  getTotalSpent(): number {
    return this.orders.reduce((sum, o) => sum + o.totalPrice, 0);
  }
}

// order.ts
import { User } from "./user";  // ← 循環依存！

class Order {
  user: User;
  items: OrderItem[] = [];

  isActive(): boolean {
    return this.status !== "cancelled";
  }

  get totalPrice(): number {
    return this.items.reduce((sum, i) => sum + i.price * i.quantity, 0);
  }

  // User の情報を使うメソッド
  getShippingAddress(): Address {
    return this.user.defaultAddress;  // User に依存
  }
}
```

```typescript
// ✅ 解決策1: インターフェースの導入（依存性逆転）
// interfaces.ts（共通モジュール）
interface IUser {
  id: string;
  defaultAddress: Address;
}

interface IOrder {
  isActive(): boolean;
  totalPrice: number;
}

// user.ts
import { IOrder } from "./interfaces";

class User implements IUser {
  private orders: IOrder[] = [];  // インターフェースに依存

  getActiveOrders(): IOrder[] {
    return this.orders.filter(o => o.isActive());
  }
}

// order.ts
import { IUser } from "./interfaces";

class Order implements IOrder {
  private userId: string;  // User オブジェクトではなく ID で参照

  constructor(private readonly userProvider: (id: string) => IUser) {}

  getShippingAddress(): Address {
    const user = this.userProvider(this.userId);
    return user.defaultAddress;
  }
}
```

```typescript
// ✅ 解決策2: 中間モジュールの導入
// user.ts - User は Order を知らない
class User {
  id: string;
  defaultAddress: Address;
}

// order.ts - Order は User を知らない
class Order {
  userId: string;
  items: OrderItem[] = [];

  get totalPrice(): number {
    return this.items.reduce((sum, i) => sum + i.price * i.quantity, 0);
  }
}

// user-order-service.ts - 両方を知る仲介者
import { User } from "./user";
import { Order } from "./order";

class UserOrderService {
  constructor(
    private userRepo: UserRepository,
    private orderRepo: OrderRepository,
  ) {}

  async getUserActiveOrders(userId: string): Promise<Order[]> {
    const orders = await this.orderRepo.findByUserId(userId);
    return orders.filter(o => o.isActive());
  }

  async getOrderShippingAddress(orderId: string): Promise<Address> {
    const order = await this.orderRepo.findById(orderId);
    const user = await this.userRepo.findById(order.userId);
    return user.defaultAddress;
  }
}
```

```typescript
// ✅ 解決策3: イベント駆動アーキテクチャ
// user.ts
class User {
  deactivate(eventBus: EventBus): void {
    this.status = "inactive";
    eventBus.publish(new UserDeactivatedEvent(this.id));
  }
}

// order.ts - User を直接参照しない
class OrderEventHandler {
  constructor(private orderRepo: OrderRepository) {}

  @EventListener(UserDeactivatedEvent)
  async onUserDeactivated(event: UserDeactivatedEvent): Promise<void> {
    const orders = await this.orderRepo.findActiveByUserId(event.userId);
    for (const order of orders) {
      order.cancel();
      await this.orderRepo.save(order);
    }
  }
}
```

### 4.3 依存方向の原則

```
依存の方向は常に安定な方向へ向ける:

  不安定（変更が多い）
     ↓
  ┌───────────┐
  │ Controller │ ─ → 変更頻度: 高
  └───────────┘
       │
       ▼
  ┌───────────┐
  │  Service   │ ─ → 変更頻度: 中
  └───────────┘
       │
       ▼
  ┌───────────┐
  │  Domain    │ ─ → 変更頻度: 低
  └───────────┘
       │
       ▼
  ┌───────────┐
  │ Interface  │ ─ → 変更頻度: 最低
  └───────────┘
     ↓
  安定（変更が少ない）

  依存性逆転の原則（DIP）:
    → 具象クラスではなくインターフェースに依存する
    → 上位モジュールは下位モジュールに依存しない
    → 両方とも抽象に依存する
```

---

## 5. Feature Envy（機能の羨望）

### 5.1 概要

Feature Envy は、あるクラスのメソッドが自分自身のデータよりも、他のクラスのデータを多く使っている状態を指す。

```
症状:
  → メソッド内で他のオブジェクトの getter を3回以上呼んでいる
  → 自クラスの this をほとんど使わない
  → 「このメソッドは本当にこのクラスにあるべきか？」と疑問に思う

原因:
  → データの場所とロジックの場所が一致していない
  → 責任の配置ミス
```

### 5.2 具体例

```typescript
// ❌ Feature Envy: InvoiceGenerator が Customer のデータばかり使っている
class InvoiceGenerator {
  generateGreeting(customer: Customer): string {
    // customer のデータを4回も使っている → Feature Envy
    if (customer.getType() === "premium") {
      return `${customer.getTitle()} ${customer.getLastName()} 様、` +
             `いつもご利用ありがとうございます。`;
    }
    return `${customer.getFirstName()} ${customer.getLastName()} 様`;
  }

  calculateDiscount(customer: Customer, amount: number): number {
    // またもや customer のデータばかり
    if (customer.getType() === "premium") {
      return amount * customer.getDiscountRate();
    }
    if (customer.getYearsAsCustomer() > 5) {
      return amount * 0.05;
    }
    return 0;
  }
}

// ✅ リファクタリング: ロジックをデータのあるクラスに移動
class Customer {
  private type: CustomerType;
  private title: string;
  private firstName: string;
  private lastName: string;
  private discountRate: number;
  private registeredAt: Date;

  getGreeting(): string {
    if (this.type === "premium") {
      return `${this.title} ${this.lastName} 様、いつもご利用ありがとうございます。`;
    }
    return `${this.firstName} ${this.lastName} 様`;
  }

  calculateDiscount(amount: number): number {
    if (this.type === "premium") {
      return amount * this.discountRate;
    }
    if (this.yearsAsCustomer > 5) {
      return amount * 0.05;
    }
    return 0;
  }

  private get yearsAsCustomer(): number {
    return new Date().getFullYear() - this.registeredAt.getFullYear();
  }
}

class InvoiceGenerator {
  generateInvoice(customer: Customer, items: InvoiceItem[]): Invoice {
    const subtotal = this.calculateSubtotal(items);
    const discount = customer.calculateDiscount(subtotal);
    return new Invoice({
      greeting: customer.getGreeting(),
      items,
      subtotal,
      discount,
      total: subtotal - discount,
    });
  }

  private calculateSubtotal(items: InvoiceItem[]): number {
    return items.reduce((sum, item) => sum + item.price * item.quantity, 0);
  }
}
```

---

## 6. Shotgun Surgery（散弾銃手術）

### 6.1 概要

Shotgun Surgery は、一つの変更を行うために多数のクラスやファイルを修正する必要がある状態を指す。機能が複数のクラスに分散しているために起きる。

```
症状:
  → 1つの仕様変更で5つ以上のファイルを修正
  → 「税率変更」で Order, Invoice, Cart, Report, Export を全部修正
  → 変更漏れによるバグが頻発
  → 「ここも修正が必要だった」が頻繁に起きる

  仕様変更: 消費税率 8% → 10%

  修正が必要なクラス:
    ┌────────┐ ┌───────┐ ┌──────┐ ┌────────┐ ┌────────┐
    │ Order  │ │Invoice│ │ Cart │ │ Report │ │ Export │
    │  8→10  │ │ 8→10  │ │ 8→10 │ │  8→10  │ │  8→10  │
    └────────┘ └───────┘ └──────┘ └────────┘ └────────┘
    → 5箇所修正！ 1つ忘れたらバグ！

解決:
  → 関連するロジックを1つのクラスに集約
  → 変更の影響を局所化する
```

### 6.2 具体例

```typescript
// ❌ 税率計算が散在
class Order {
  calculateTax(): number {
    return this.subtotal * 0.10;  // ← ハードコード
  }
}

class Invoice {
  getTaxAmount(): number {
    return this.amount * 0.10;  // ← 同じ値がここにも
  }
}

class Cart {
  estimateTax(): number {
    return this.total * 0.10;  // ← ここにも
  }
}

class SalesReport {
  calculateTaxTotal(): number {
    return this.sales.reduce((sum, s) => sum + s.amount * 0.10, 0);  // ← ここにも
  }
}


// ✅ 税率計算を一箇所に集約
class TaxCalculator {
  private static readonly STANDARD_RATE = 0.10;
  private static readonly REDUCED_RATE = 0.08;  // 軽減税率

  static calculate(amount: number, type: TaxType = "standard"): number {
    const rate = type === "reduced"
      ? TaxCalculator.REDUCED_RATE
      : TaxCalculator.STANDARD_RATE;
    return Math.floor(amount * rate);
  }

  static getRate(type: TaxType = "standard"): number {
    return type === "reduced"
      ? TaxCalculator.REDUCED_RATE
      : TaxCalculator.STANDARD_RATE;
  }
}

class Order {
  calculateTax(): number {
    return TaxCalculator.calculate(this.subtotal);
  }
}

class Invoice {
  getTaxAmount(): number {
    return TaxCalculator.calculate(this.amount);
  }
}

// → 税率変更時は TaxCalculator だけ修正すれば OK
```

---

## 7. Primitive Obsession（基本型への執着）

### 7.1 概要

Primitive Obsession は、ドメインの概念を string、number、boolean などの基本型で表現し続けるアンチパターンである。

```
症状:
  → メールアドレスが string
  → 金額が number
  → 電話番号が string
  → ステータスが string（"active", "inactive"...）
  → 同じバリデーションが複数箇所に散在

問題:
  → 型安全性がない（string と string を混同しやすい）
  → バリデーションの重複
  → 不正な値を防げない
  → ドメインの意味がコードに表現されない
```

### 7.2 値オブジェクト（Value Object）による解決

```typescript
// ❌ Primitive Obsession
function createUser(
  name: string,
  email: string,
  phone: string,
  age: number,
  zipCode: string,
): void {
  // name, email, phone, zipCode は全て string
  // 引数の順番を間違えてもコンパイルエラーにならない！
  // createUser("test@email.com", "John Doe", "100-0001", 25, "090-1234-5678")
  //            ↑ email と name が逆！でもコンパイル通る
}

function sendEmail(to: string, subject: string, body: string): void {
  // to が有効なメールアドレスか検証が必要...毎回
  if (!to.match(/^[^\s@]+@[^\s@]+\.[^\s@]+$/)) {
    throw new Error(`Invalid email: ${to}`);
  }
  // ...
}

// ✅ 値オブジェクトで型安全に
class Email {
  private constructor(private readonly value: string) {}

  static create(value: string): Email {
    const trimmed = value.trim().toLowerCase();
    if (!trimmed.match(/^[^\s@]+@[^\s@]+\.[^\s@]+$/)) {
      throw new InvalidEmailError(value);
    }
    return new Email(trimmed);
  }

  get domain(): string {
    return this.value.split("@")[1];
  }

  toString(): string { return this.value; }
  equals(other: Email): boolean { return this.value === other.value; }
}

class PhoneNumber {
  private constructor(private readonly value: string) {}

  static create(value: string): PhoneNumber {
    const normalized = value.replace(/[-\s()]/g, "");
    if (!normalized.match(/^0\d{9,10}$/)) {
      throw new InvalidPhoneNumberError(value);
    }
    return new PhoneNumber(normalized);
  }

  get formatted(): string {
    // 090-1234-5678 形式
    if (this.value.length === 11) {
      return `${this.value.slice(0,3)}-${this.value.slice(3,7)}-${this.value.slice(7)}`;
    }
    return this.value;
  }

  toString(): string { return this.value; }
  equals(other: PhoneNumber): boolean { return this.value === other.value; }
}

class Money {
  constructor(
    public readonly amount: number,
    public readonly currency: "JPY" | "USD" | "EUR",
  ) {
    if (!Number.isFinite(amount)) throw new Error("金額は有限の数値");
    if (amount < 0) throw new Error("金額は0以上");
    if (currency === "JPY" && !Number.isInteger(amount)) {
      throw new Error("日本円は整数のみ");
    }
  }

  add(other: Money): Money {
    this.ensureSameCurrency(other);
    return new Money(this.amount + other.amount, this.currency);
  }

  subtract(other: Money): Money {
    this.ensureSameCurrency(other);
    const result = this.amount - other.amount;
    if (result < 0) throw new Error("金額が負になります");
    return new Money(result, this.currency);
  }

  multiply(factor: number): Money {
    return new Money(
      this.currency === "JPY"
        ? Math.floor(this.amount * factor)
        : Math.round(this.amount * factor * 100) / 100,
      this.currency,
    );
  }

  isGreaterThan(other: Money): boolean {
    this.ensureSameCurrency(other);
    return this.amount > other.amount;
  }

  private ensureSameCurrency(other: Money): void {
    if (this.currency !== other.currency) {
      throw new CurrencyMismatchError(this.currency, other.currency);
    }
  }

  static zero(currency: "JPY" | "USD" | "EUR"): Money {
    return new Money(0, currency);
  }

  toString(): string {
    if (this.currency === "JPY") return `¥${this.amount.toLocaleString()}`;
    if (this.currency === "USD") return `$${this.amount.toFixed(2)}`;
    return `€${this.amount.toFixed(2)}`;
  }
}

class ZipCode {
  private constructor(private readonly value: string) {}

  static create(value: string): ZipCode {
    const normalized = value.replace(/-/g, "");
    if (!normalized.match(/^\d{7}$/)) {
      throw new Error(`Invalid zip code: ${value}`);
    }
    return new ZipCode(normalized);
  }

  get formatted(): string {
    return `${this.value.slice(0,3)}-${this.value.slice(3)}`;
  }

  toString(): string { return this.value; }
}

// 値オブジェクトを使った関数
function createUser(
  name: UserName,
  email: Email,
  phone: PhoneNumber,
  age: Age,
  zipCode: ZipCode,
): void {
  // 引数の順番を間違えたらコンパイルエラー！
  // 各値は生成時にバリデーション済み
}

function sendEmail(to: Email, subject: string, body: string): void {
  // to は必ず有効なメールアドレス。バリデーション不要！
}
```

### 7.3 Python での値オブジェクト

```python
from dataclasses import dataclass
from typing import Self
import re

# ✅ Python での値オブジェクト（dataclass + __post_init__）
@dataclass(frozen=True)  # frozen=True で不変に
class Email:
    value: str

    def __post_init__(self):
        normalized = self.value.strip().lower()
        if not re.match(r'^[^\s@]+@[^\s@]+\.[^\s@]+$', normalized):
            raise ValueError(f"Invalid email: {self.value}")
        # frozen=True でも __post_init__ では object.__setattr__ が使える
        object.__setattr__(self, 'value', normalized)

    @property
    def domain(self) -> str:
        return self.value.split('@')[1]

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Money:
    amount: int  # 最小単位（日本円なら1円単位）
    currency: str

    def __post_init__(self):
        if self.amount < 0:
            raise ValueError("金額は0以上である必要があります")
        if self.currency not in ("JPY", "USD", "EUR"):
            raise ValueError(f"未対応の通貨: {self.currency}")

    def add(self, other: Self) -> Self:
        if self.currency != other.currency:
            raise ValueError(f"通貨が異なります: {self.currency} vs {other.currency}")
        return Money(self.amount + other.amount, self.currency)

    def subtract(self, other: Self) -> Self:
        if self.currency != other.currency:
            raise ValueError(f"通貨が異なります")
        return Money(self.amount - other.amount, self.currency)

    def multiply(self, factor: float) -> Self:
        return Money(int(self.amount * factor), self.currency)

    @classmethod
    def zero(cls, currency: str) -> Self:
        return cls(0, currency)

    def __str__(self) -> str:
        if self.currency == "JPY":
            return f"¥{self.amount:,}"
        return f"{self.currency} {self.amount / 100:.2f}"


# 使用例
email = Email("User@Example.COM")
print(email)         # user@example.com
print(email.domain)  # example.com

price = Money(1000, "JPY")
tax = price.multiply(0.10)
total = price.add(tax)
print(total)  # ¥1,100
```

### 7.4 Java での値オブジェクト（record）

```java
// Java 16+ の record を使った値オブジェクト
public record Email(String value) {
    public Email {
        // Compact constructor でバリデーション
        value = value.trim().toLowerCase();
        if (!value.matches("^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$")) {
            throw new IllegalArgumentException("Invalid email: " + value);
        }
    }

    public String domain() {
        return value.split("@")[1];
    }
}

public record Money(long amount, Currency currency) {
    public Money {
        if (amount < 0) {
            throw new IllegalArgumentException("Amount must be non-negative");
        }
        Objects.requireNonNull(currency, "Currency must not be null");
    }

    public Money add(Money other) {
        ensureSameCurrency(other);
        return new Money(this.amount + other.amount, this.currency);
    }

    public Money subtract(Money other) {
        ensureSameCurrency(other);
        return new Money(this.amount - other.amount, this.currency);
    }

    private void ensureSameCurrency(Money other) {
        if (!this.currency.equals(other.currency)) {
            throw new IllegalArgumentException("Currency mismatch");
        }
    }

    public static Money zero(Currency currency) {
        return new Money(0, currency);
    }
}
```

---

## 8. Singleton の誤用

### 8.1 概要

Singleton パターン自体はデザインパターンだが、乱用するとアンチパターンになる。グローバルな可変状態を隠す手段として使われることが多い。

```
問題点:
  → グローバル状態の隠蔽（実質的なグローバル変数）
  → テスト困難（モック化が難しい）
  → 隠れた依存関係（コンストラクタに表れない）
  → 並行処理での問題
  → ライフサイクル管理の困難

Singleton の正当な使用:
  → ハードウェアリソースの管理（プリンタースプーラーなど）
  → 設定の読み取り専用キャッシュ
  → ログファクトリ
```

### 8.2 具体例

```typescript
// ❌ Singleton の誤用
class Database {
  private static instance: Database;
  private connection: Connection;

  private constructor() {
    this.connection = createConnection(/* ... */);
  }

  static getInstance(): Database {
    if (!Database.instance) {
      Database.instance = new Database();
    }
    return Database.instance;
  }

  query(sql: string): Promise<any[]> {
    return this.connection.query(sql);
  }
}

// 利用側 - 隠れた依存
class UserRepository {
  async findById(id: string): Promise<User | null> {
    // Database への依存がコンストラクタに表れない！
    const db = Database.getInstance();
    const rows = await db.query(`SELECT * FROM users WHERE id = '${id}'`);
    return rows[0] ? this.toUser(rows[0]) : null;
  }
}

// テスト時の問題
// UserRepository をテストするのに、実際のDBが必要になる
// Database.getInstance() をモック化するのが困難


// ✅ 依存性注入で解決
interface IDatabase {
  query(sql: string): Promise<any[]>;
}

class Database implements IDatabase {
  constructor(private connection: Connection) {}

  async query(sql: string): Promise<any[]> {
    return this.connection.query(sql);
  }
}

class UserRepository {
  // 依存が明示的
  constructor(private readonly db: IDatabase) {}

  async findById(id: string): Promise<User | null> {
    const rows = await this.db.query(`SELECT * FROM users WHERE id = ?`, [id]);
    return rows[0] ? this.toUser(rows[0]) : null;
  }
}

// テスト時 - モックを簡単に注入できる
class MockDatabase implements IDatabase {
  private mockData: Map<string, any[]> = new Map();

  setMockResult(sql: string, result: any[]): void {
    this.mockData.set(sql, result);
  }

  async query(sql: string): Promise<any[]> {
    return this.mockData.get(sql) || [];
  }
}

const mockDb = new MockDatabase();
const repo = new UserRepository(mockDb);  // テスト用DBを注入
```

---

## 9. Poltergeist（ポルターガイスト）

### 9.1 概要

Poltergeist は、存在意義の薄いクラスで、他のクラスのメソッドを呼ぶだけで自身は何も行わないオブジェクトである。

```
症状:
  → クラスが他のクラスのメソッドを呼ぶだけ
  → 自身の状態やロジックを持たない
  → 「マネージャー」「コントローラー」「ハンドラー」という名前
  → 短命で、一瞬だけ使われてすぐ捨てられる
  → メソッドが1-2行で、全て委譲（delegation）
```

### 9.2 具体例

```typescript
// ❌ Poltergeist: 何もしていないクラス
class OrderProcessor {
  private orderService: OrderService;
  private paymentService: PaymentService;
  private emailService: EmailService;

  processOrder(order: Order): void {
    this.orderService.validate(order);    // 委譲するだけ
    this.paymentService.charge(order);    // 委譲するだけ
    this.emailService.sendConfirmation(order);  // 委譲するだけ
  }
}

// ✅ 解決策1: Poltergeist を削除し、直接呼び出す
// （ワークフローが単純な場合）
async function processOrder(
  order: Order,
  orderService: OrderService,
  paymentService: PaymentService,
  emailService: EmailService,
): Promise<void> {
  await orderService.validate(order);
  await paymentService.charge(order);
  await emailService.sendConfirmation(order);
}

// ✅ 解決策2: 付加価値のあるオーケストレーター
// （エラーハンドリングやトランザクション管理など、自身のロジックがある場合は正当）
class OrderWorkflow {
  constructor(
    private orderService: OrderService,
    private paymentService: PaymentService,
    private emailService: EmailService,
    private logger: Logger,
  ) {}

  async execute(order: Order): Promise<OrderResult> {
    this.logger.info(`Processing order ${order.id}`);

    // バリデーション
    const validationResult = await this.orderService.validate(order);
    if (!validationResult.isValid) {
      this.logger.warn(`Order ${order.id} validation failed`, validationResult.errors);
      return OrderResult.failure(validationResult.errors);
    }

    // 支払い（リトライ付き）
    let paymentResult: PaymentResult;
    try {
      paymentResult = await this.retryWithBackoff(
        () => this.paymentService.charge(order),
        3,
      );
    } catch (error) {
      this.logger.error(`Payment failed for order ${order.id}`, error);
      await this.orderService.markAsFailed(order.id);
      return OrderResult.paymentFailed(error);
    }

    // 確認メール（失敗しても注文は成功）
    try {
      await this.emailService.sendConfirmation(order);
    } catch (error) {
      this.logger.warn(`Email sending failed for order ${order.id}`, error);
      // メール失敗は注文の失敗にはしない
    }

    return OrderResult.success(order.id, paymentResult.transactionId);
  }

  private async retryWithBackoff<T>(
    fn: () => Promise<T>,
    maxRetries: number,
  ): Promise<T> {
    // リトライロジック
    for (let i = 0; i < maxRetries; i++) {
      try {
        return await fn();
      } catch (error) {
        if (i === maxRetries - 1) throw error;
        await this.delay(Math.pow(2, i) * 1000);
      }
    }
    throw new Error("Unreachable");
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}
```

---

## 10. Lava Flow（溶岩流）

### 10.1 概要

Lava Flow は、「なぜそこにあるのか誰も分からないが、怖くて触れないコード」を指す。過去の実装が残骸として残り、固まった溶岩のように除去困難になっている状態。

```
症状:
  → 「これ何に使ってるの？」「分からないけど消したら壊れるかも」
  → コメントアウトされたコードが大量に存在
  → 未使用のクラスやメソッド
  → TODO コメントが何年も放置
  → 「レガシー」「旧」「temp」「test」という名前のファイル
  → 複数の方法で同じことを行う重複コード

原因:
  → プロトタイプコードがそのまま本番へ
  → リファクタリングが途中で中断
  → 担当者が退職して知識が失われた
  → 「動いてるなら触るな」の文化

対策:
  → テストカバレッジを上げてから削除
  → git log で最終更新日を確認（数年触られていなければ候補）
  → デッドコード検出ツールの活用
  → コードオーナーシップの明確化
  → 定期的な「技術的負債返済スプリント」
```

```typescript
// ❌ Lava Flow の例
class OrderProcessor {
  // TODO: refactor this (2019-03-15)
  process(order: Order): void {
    // ...
  }

  // 旧バージョン（移行完了後に削除予定）
  // processOld(order: Order): void {
  //   const tax = order.total * 0.08;  // 旧税率
  //   // ... 200行のコメントアウト
  // }

  // processV2 もある...
  processV2(order: Order): void {
    // "一時的な" 修正（3年前）
    if (order.type === "legacy") {
      return this.processLegacy(order);
    }
    // ...
  }

  // 誰が使っているか不明
  private processLegacy(order: Order): void {
    // 2020年のシステム移行時のコード
    // 今も使われている？ 不明...
  }
}
```

---

## 11. 検出と回避

### 11.1 定量的な検出指標

```
検出の指標:
  → クラスの行数 > 300行 → 分割を検討
  → メソッド数 > 15 → 責任の分離を検討
  → 継承の深さ > 3 → コンポジションに変更
  → 依存数 > 7 → ファサードの導入を検討
  → instanceof の使用 → ポリモーフィズムに変更
  → getter/setter だけのクラス → Rich Domain Model
  → 同じバリデーションが3箇所以上 → 値オブジェクト
  → 循環 import → インターフェース分離
  → メソッド内の他オブジェクト getter 3回以上 → Feature Envy

メトリクスの目安:
  ┌────────────────────────┬─────────┬──────────┬─────────┐
  │ メトリクス              │ 良好     │ 注意      │ 危険    │
  ├────────────────────────┼─────────┼──────────┼─────────┤
  │ クラス行数              │ < 200   │ 200-500  │ > 500   │
  │ メソッド数              │ < 10    │ 10-20    │ > 20    │
  │ 継承の深さ              │ 1-2     │ 3        │ > 3     │
  │ 循環的複雑度            │ < 10    │ 10-20    │ > 20    │
  │ 依存クラス数            │ < 5     │ 5-10     │ > 10    │
  │ メソッドのパラメータ数   │ < 4     │ 4-6      │ > 6     │
  │ コードカバレッジ        │ > 80%   │ 50-80%   │ < 50%   │
  └────────────────────────┴─────────┴──────────┴─────────┘
```

### 11.2 ツール

```
静的解析ツール:
  → SonarQube: コード品質メトリクス（コード臭い検出）
  → ESLint: complexity ルール（JavaScript/TypeScript）
  → Pylint: Python のコード品質チェック
  → SpotBugs / PMD: Java のバグパターン検出
  → IDE: クラス図の可視化（IntelliJ, VS Code）

TypeScript / JavaScript:
  → eslint-plugin-sonarjs: コード臭い検出
  → eslint-plugin-import: 循環依存検出
  → madge: モジュール依存の可視化
  → dependency-cruiser: 依存ルールの定義・検証

Python:
  → pylint: 全般的なコード品質
  → radon: 循環的複雑度の計測
  → vulture: デッドコード検出
  → pydeps: 依存グラフの可視化

Java:
  → Checkstyle: コーディングスタイル
  → ArchUnit: アーキテクチャテスト
  → JDepend: パッケージ依存の分析
```

### 11.3 コードレビューチェックリスト

```
コードレビュー時のアンチパターンチェック:

□ God Object
  - 新しいメソッドは既存クラスの責任に合っているか？
  - クラスの説明を1文で言えるか？

□ Anemic Domain Model
  - ドメインオブジェクトにビジネスロジックはあるか？
  - サービスクラスがデータクラスの setter を直接操作していないか？

□ 深い継承
  - 継承は3段階以内か？
  - 継承よりコンポジションが適切ではないか？

□ 循環依存
  - 新しい import が循環を作っていないか？
  - モジュール間の依存方向は適切か？

□ Feature Envy
  - メソッドが他のオブジェクトのデータを多用していないか？
  - ロジックはデータのあるクラスに配置されているか？

□ Shotgun Surgery
  - この変更で他に修正すべき場所はないか？
  - 同じロジックが複数箇所に存在しないか？

□ Primitive Obsession
  - ドメイン概念を基本型で表現していないか？
  - バリデーションが散在していないか？

□ Lava Flow
  - 不要なコメントアウトコードが残っていないか？
  - 未使用のメソッドやクラスが追加されていないか？
```

### 11.4 設計原則

```
原則:
  1. 小さいクラス（SRP: 単一責任原則）
     → 1つのクラスは1つの理由でのみ変更される

  2. 浅い継承（コンポジション優先）
     → "is-a" でないなら継承しない

  3. 豊かなドメインモデル（振る舞いを持つ）
     → データとロジックを一体にする

  4. 値オブジェクト（型で制約を表現）
     → ドメインの概念には専用の型を作る

  5. 依存の方向を一方向に（DIP: 依存性逆転の原則）
     → 具象ではなく抽象に依存する

  6. インターフェース分離の原則（ISP）
     → クライアントが使わないメソッドに依存させない

  7. 開放閉鎖の原則（OCP）
     → 拡張に開き、修正に閉じる

  SOLID 原則との対応:
  ┌──────────────────┬──────────────────────────┐
  │ アンチパターン    │ 違反している SOLID 原則    │
  ├──────────────────┼──────────────────────────┤
  │ God Object       │ SRP（単一責任）            │
  │ Anemic Model     │ OCP（開放閉鎖）            │
  │ 深い継承         │ LSP（リスコフの置換）       │
  │ 循環依存         │ DIP（依存性逆転）           │
  │ Feature Envy     │ SRP（単一責任）            │
  │ Shotgun Surgery  │ SRP（単一責任）            │
  │ Primitive Obsession │ ISP（インターフェース分離）│
  └──────────────────┴──────────────────────────┘
```

---

## まとめ

| アンチパターン | 症状 | 解決策 | 検出方法 |
|---------------|------|--------|----------|
| God Object | 全部入りクラス | SRPで分割 | 行数 > 300, メソッド数 > 15 |
| Anemic Model | データだけのクラス | Rich Domain Model | getter/setter のみ |
| 深い継承 | 4段階以上 | コンポジション | 継承の深さ計測 |
| 循環依存 | A→B→C→A | インターフェース導入 | 依存グラフの分析 |
| Feature Envy | 他クラスのデータ多用 | メソッドの移動 | 他オブジェクトの getter 回数 |
| Shotgun Surgery | 1変更で多数ファイル修正 | ロジックの集約 | 変更の影響範囲分析 |
| Primitive Obsession | string/number乱用 | 値オブジェクト | バリデーションの重複 |
| Singleton 誤用 | グローバル状態 | 依存性注入 | getInstance() の多用 |
| Poltergeist | 委譲するだけのクラス | クラス削除/統合 | 1-2行メソッドばかり |
| Lava Flow | 触れないコード | テスト追加後に削除 | 最終更新日の確認 |

---

## 演習問題

### 演習1: God Object の分割

以下の `AppController` は典型的な God Object である。5つ以上のクラスに分割し、各クラスの責任を明確にせよ。

```typescript
class AppController {
  // ユーザー認証
  login(email: string, password: string): Token { /* ... */ }
  logout(token: string): void { /* ... */ }
  refreshToken(token: string): Token { /* ... */ }

  // プロフィール管理
  getProfile(userId: string): Profile { /* ... */ }
  updateProfile(userId: string, data: any): void { /* ... */ }
  uploadAvatar(userId: string, file: File): string { /* ... */ }

  // 商品管理
  listProducts(filter: any): Product[] { /* ... */ }
  getProduct(id: string): Product { /* ... */ }
  searchProducts(query: string): Product[] { /* ... */ }

  // カート管理
  addToCart(userId: string, productId: string): void { /* ... */ }
  removeFromCart(userId: string, productId: string): void { /* ... */ }
  getCart(userId: string): Cart { /* ... */ }

  // 注文
  checkout(userId: string): Order { /* ... */ }
  getOrderHistory(userId: string): Order[] { /* ... */ }

  // 管理者機能
  getSystemStats(): Stats { /* ... */ }
  generateReport(type: string): Report { /* ... */ }
}
```

### 演習2: Anemic → Rich リファクタリング

以下の Anemic Domain Model を Rich Domain Model にリファクタリングせよ。ビジネスルールがサービス外に漏れないようにすること。

```typescript
class Subscription {
  plan: "free" | "basic" | "premium" = "free";
  startDate: Date = new Date();
  endDate: Date | null = null;
  isCancelled: boolean = false;
  paymentMethod: string | null = null;
}

class SubscriptionService {
  upgrade(sub: Subscription, newPlan: string): void {
    if (sub.isCancelled) throw new Error("Cancelled");
    if (newPlan === "free") throw new Error("Cannot downgrade to free");
    sub.plan = newPlan as any;
  }

  cancel(sub: Subscription): void {
    if (sub.isCancelled) throw new Error("Already cancelled");
    sub.isCancelled = true;
    sub.endDate = new Date();
  }

  isActive(sub: Subscription): boolean {
    return !sub.isCancelled &&
           (sub.endDate === null || sub.endDate > new Date());
  }
}
```

### 演習3: 循環依存の解消

以下のコードには循環依存がある。インターフェースを導入して循環を断ち切れ。

```typescript
// department.ts
import { Employee } from "./employee";
class Department {
  employees: Employee[] = [];
  manager: Employee;
  getHeadcount(): number { return this.employees.length; }
  getBudget(): number {
    return this.employees.reduce((sum, e) => sum + e.salary, 0);
  }
}

// employee.ts
import { Department } from "./department";  // 循環依存！
class Employee {
  salary: number;
  department: Department;
  getDepartmentName(): string { return this.department.name; }
  getColleagues(): Employee[] { return this.department.employees; }
}
```

### 演習4: Primitive Obsession の解消

以下のコードで使われている基本型を値オブジェクトに置き換えよ。最低3つの値オブジェクト（Email, PhoneNumber, Address のうち任意）を作成すること。

```typescript
function registerUser(
  name: string,
  email: string,
  phone: string,
  addressLine1: string,
  addressLine2: string,
  city: string,
  zipCode: string,
  country: string,
): User {
  // バリデーションが関数の先頭に長々と...
  if (!email.includes("@")) throw new Error("Invalid email");
  if (phone.length < 10) throw new Error("Invalid phone");
  if (zipCode.length !== 7) throw new Error("Invalid zip code");
  // ...
}
```

### 演習5: アンチパターンの識別

以下のコードにはどのアンチパターンが含まれているか、全て識別せよ。また、それぞれの修正方針を示せ。

```typescript
class SystemManager {
  private static instance: SystemManager;
  static getInstance() {
    if (!this.instance) this.instance = new SystemManager();
    return this.instance;
  }

  processUserOrder(userId: string, orderId: string) {
    const db = Database.getInstance();
    const user = db.query(`SELECT * FROM users WHERE id = ${userId}`);
    const order = db.query(`SELECT * FROM orders WHERE id = ${orderId}`);

    // Feature Envy: user のデータばかり使っている
    const discount = user.type === "premium" ? user.discountRate :
                     user.yearsActive > 5 ? 0.05 : 0;

    const tax = order.total * 0.10;  // Shotgun Surgery: 税率がハードコード
    const total = order.total - (order.total * discount) + tax;

    db.query(`UPDATE orders SET total = ${total} WHERE id = ${orderId}`);

    // さらに通知、ログ、レポート更新も全部ここで...
    this.sendEmail(user.email, `注文確定: ${total}円`);
    this.logActivity(userId, "order_processed");
    this.updateDashboard();
  }

  sendEmail(to: string, body: string) { /* ... */ }
  logActivity(userId: string, action: string) { /* ... */ }
  updateDashboard() { /* ... */ }
  // ... さらに50メソッド
}
```

**ヒント**: 少なくとも5つのアンチパターンが含まれている。

### 演習6: テスタビリティの改善

以下のクラスをアンチパターンを解消しつつ、ユニットテストが容易な設計にリファクタリングせよ。テストコードも1つ以上書くこと。

```typescript
class ReportGenerator {
  generate(type: string): string {
    const db = Database.getInstance();
    const data = db.query(`SELECT * FROM ${type}_data`);
    const now = new Date();

    let report = `Report: ${type}\nGenerated: ${now.toISOString()}\n\n`;

    for (const row of data) {
      report += `${row.name}: ${row.value}\n`;
    }

    // ファイルに直接書き込み
    const fs = require("fs");
    fs.writeFileSync(`/reports/${type}_${now.getTime()}.txt`, report);

    return report;
  }
}
```

---

## 参考文献
1. Fowler, M. "Refactoring: Improving the Design of Existing Code." 2nd Ed, Addison-Wesley, 2018.
2. Brown, W. "AntiPatterns: Refactoring Software, Architectures, and Projects in Crisis." Wiley, 1998.
3. Evans, E. "Domain-Driven Design: Tackling Complexity in the Heart of Software." Addison-Wesley, 2003.
4. Martin, R. C. "Clean Code: A Handbook of Agile Software Craftsmanship." Prentice Hall, 2008.
5. Martin, R. C. "Agile Software Development, Principles, Patterns, and Practices." Prentice Hall, 2002.
6. Gamma, E. et al. "Design Patterns: Elements of Reusable Object-Oriented Software." Addison-Wesley, 1994.
7. Fowler, M. "AnemicDomainModel." martinfowler.com (2003).
8. Vernon, V. "Implementing Domain-Driven Design." Addison-Wesley, 2013.
