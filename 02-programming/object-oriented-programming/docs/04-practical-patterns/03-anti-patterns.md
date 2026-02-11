# OOPアンチパターン

> アンチパターンは「よくある悪い設計」。God Object、深い継承階層、Anemic Domain Model など、OOP で陥りがちな罠とその回避方法を解説する。

## この章で学ぶこと

- [ ] 主要なOOPアンチパターンを認識できるようになる
- [ ] 各アンチパターンの問題点と発生原因を理解する
- [ ] リファクタリングによる解決方法を学ぶ

---

## 1. God Object（神オブジェクト）

```
症状:
  → 1つのクラスが全てを知り、全てを行う
  → 1000行以上のクラス
  → 20以上のメソッド
  → あらゆるクラスがこのクラスに依存

原因:
  → 「とりあえずここに追加」の積み重ね
  → 責任の分離を意識しない開発

問題:
  → 変更が困難（影響範囲が広大）
  → テスト困難（依存が多すぎる）
  → チーム開発でコンフリクト多発
```

```typescript
// ❌ God Object
class ApplicationManager {
  // ユーザー管理
  createUser(data: any) { /* ... */ }
  deleteUser(id: string) { /* ... */ }
  authenticateUser(email: string, password: string) { /* ... */ }
  // 注文管理
  createOrder(userId: string, items: any[]) { /* ... */ }
  cancelOrder(orderId: string) { /* ... */ }
  // 支払い管理
  processPayment(orderId: string, card: any) { /* ... */ }
  refundPayment(paymentId: string) { /* ... */ }
  // 通知管理
  sendEmail(to: string, body: string) { /* ... */ }
  sendSms(to: string, body: string) { /* ... */ }
  // レポート
  generateMonthlyReport() { /* ... */ }
  generateUserReport(userId: string) { /* ... */ }
  // ... さらに100メソッド以上
}

// ✅ 責任ごとに分割
class UserService { /* ユーザー管理のみ */ }
class OrderService { /* 注文管理のみ */ }
class PaymentService { /* 支払い管理のみ */ }
class NotificationService { /* 通知のみ */ }
class ReportService { /* レポートのみ */ }
```

---

## 2. Anemic Domain Model（貧血ドメインモデル）

```
症状:
  → クラスがデータ（getter/setter）だけを持つ
  → ビジネスロジックが全てサービスクラスにある
  → ドメインオブジェクトが単なるデータの入れ物

原因:
  → 「データクラス + サービスクラス」の分離が目的化
  → DTOとドメインモデルの混同

問題:
  → オブジェクト指向の利点が活かせない
  → ビジネスルールの重複
  → 不正な状態を防げない
```

```typescript
// ❌ Anemic Domain Model
class Order {
  id: string = "";
  items: OrderItem[] = [];
  status: string = "pending";
  totalPrice: number = 0;
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

  canCancel(order: Order): boolean {
    return order.status === "pending" || order.status === "confirmed";
  }

  cancel(order: Order): void {
    if (!this.canCancel(order)) throw new Error("Cannot cancel");
    order.status = "cancelled";
  }
}

// ✅ Rich Domain Model
class Order {
  private _items: OrderItem[] = [];
  private _status: OrderStatus = "pending";

  constructor(private readonly id: string) {}

  addItem(item: OrderItem): void {
    if (this._status !== "pending") {
      throw new Error("確定済みの注文には追加できません");
    }
    this._items.push(item);
  }

  get totalPrice(): number {
    return this._items.reduce(
      (sum, item) => sum + item.price * item.quantity, 0
    );
  }

  confirm(): void {
    if (this._items.length === 0) {
      throw new Error("商品がない注文は確定できません");
    }
    this._status = "confirmed";
  }

  cancel(): void {
    if (this._status === "shipped" || this._status === "delivered") {
      throw new Error("発送済みの注文はキャンセルできません");
    }
    this._status = "cancelled";
  }

  get status(): OrderStatus { return this._status; }
}
// → ビジネスルールがオブジェクト内に。不正な状態を型で防ぐ
```

---

## 3. 深い継承階層

```
症状:
  → 4段階以上の継承チェーン
  → 各レイヤーの変更が下位全てに影響
  → どのメソッドがどのレイヤーで定義されたか不明

  Entity
  └── LivingEntity
      └── Animal
          └── Mammal
              └── Dog
                  └── GuideDog
                      └── TrainedGuideDog

解決:
  → 継承は2-3段階まで
  → コンポジションで機能を組み合わせ
  → インターフェースで型の関係を表現
```

```typescript
// ❌ 深い継承
class BaseComponent { /* 基本機能 */ }
class StyledComponent extends BaseComponent { /* スタイル */ }
class InteractiveComponent extends StyledComponent { /* インタラクション */ }
class AnimatedComponent extends InteractiveComponent { /* アニメーション */ }
class AccessibleComponent extends AnimatedComponent { /* a11y */ }
class MyButton extends AccessibleComponent { /* ボタン */ }

// ✅ コンポジション
interface Stylable { applyStyles(styles: Styles): void; }
interface Interactive { onClick(handler: () => void): void; }
interface Animatable { animate(animation: Animation): void; }
interface Accessible { setAriaLabel(label: string): void; }

class MyButton implements Stylable, Interactive, Animatable, Accessible {
  private styleEngine: StyleEngine;
  private animationEngine: AnimationEngine;

  constructor(deps: ButtonDeps) {
    this.styleEngine = deps.styleEngine;
    this.animationEngine = deps.animationEngine;
  }
  // 各機能をコンポジションで実現
}
```

---

## 4. その他のアンチパターン

```
Yo-Yo 問題:
  → 継承チェーンを上下に行ったり来たりしないと理解できない
  → 解決: フラットな構造、コンポジション

Circular Dependency（循環依存）:
  → A → B → C → A
  → 解決: インターフェースの導入、依存の方向を整理

Feature Envy（機能の羨望）:
  → あるクラスが他のクラスのデータを頻繁に使う
  → 解決: メソッドをデータがあるクラスに移動

Shotgun Surgery（散弾銃手術）:
  → 1つの変更で多数のクラスを修正する必要
  → 解決: 関連する責任を1つのクラスに統合

Primitive Obsession（基本型への執着）:
  → Email を string で、Price を number で表現
  → 解決: 値オブジェクト（Value Object）を作る
```

```typescript
// Primitive Obsession の解決: 値オブジェクト
// ❌ string で表現
function sendEmail(to: string, subject: string) {
  // to が有効なメールアドレスか検証が必要...毎回
}

// ✅ 値オブジェクト
class Email {
  private constructor(private readonly value: string) {}

  static create(value: string): Email {
    if (!value.match(/^[^\s@]+@[^\s@]+\.[^\s@]+$/)) {
      throw new Error(`Invalid email: ${value}`);
    }
    return new Email(value);
  }

  toString(): string { return this.value; }
  equals(other: Email): boolean { return this.value === other.value; }
}

class Money {
  constructor(
    public readonly amount: number,
    public readonly currency: "JPY" | "USD" | "EUR",
  ) {
    if (amount < 0) throw new Error("金額は0以上");
  }

  add(other: Money): Money {
    if (this.currency !== other.currency) throw new Error("通貨が異なります");
    return new Money(this.amount + other.amount, this.currency);
  }
}
```

---

## 5. 検出と回避

```
検出の指標:
  → クラスの行数 > 300行 → 分割を検討
  → メソッド数 > 15 → 責任の分離を検討
  → 継承の深さ > 3 → コンポジションに変更
  → 依存数 > 7 → ファサードの導入を検討
  → instanceof の使用 → ポリモーフィズムに変更

ツール:
  → SonarQube: コード品質メトリクス
  → ESLint: complexity ルール
  → IDE: クラス図の可視化

原則:
  1. 小さいクラス（SRP）
  2. 浅い継承（コンポジション優先）
  3. 豊かなドメインモデル（振る舞いを持つ）
  4. 値オブジェクト（型で制約を表現）
  5. 依存の方向を一方向に（DIP）
```

---

## まとめ

| アンチパターン | 症状 | 解決策 |
|---------------|------|--------|
| God Object | 全部入りクラス | SRPで分割 |
| Anemic Model | データだけのクラス | Rich Domain Model |
| 深い継承 | 4段階以上 | コンポジション |
| 循環依存 | A→B→C→A | インターフェース導入 |
| Primitive Obsession | string/number乱用 | 値オブジェクト |

---

## 参考文献
1. Fowler, M. "Refactoring." 2nd Ed, Addison-Wesley, 2018.
2. Brown, W. "AntiPatterns." Wiley, 1998.
3. Evans, E. "Domain-Driven Design." Addison-Wesley, 2003.
