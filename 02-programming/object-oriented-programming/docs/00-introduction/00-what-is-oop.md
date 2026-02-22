# OOPとは何か

> オブジェクト指向プログラミング（OOP）は「データとそれを操作する手続きを一つの単位（オブジェクト）にまとめる」プログラミングパラダイム。現実世界のモデリングから大規模ソフトウェアの構造化まで、最も広く使われている設計手法。

## この章で学ぶこと

- [ ] OOPの本質的な考え方を理解する
- [ ] オブジェクトとメッセージパッシングの関係を把握する
- [ ] OOPが解決する問題と適用領域を理解する
- [ ] OOPの基本原則が実際のコードにどう反映されるかを体験する
- [ ] 複数言語でのOOP実装の違いを比較する

---

## 1. OOPの本質

```
プログラミングパラダイムの比較:

  手続き型:    データ + 関数（別々に管理）
  OOP:        データ + 関数 = オブジェクト（一体化）
  関数型:     関数（データ変換のパイプライン）

OOPの核心:
  「世界をオブジェクトの集まりとして捉え、
   オブジェクト間のメッセージのやり取りで処理を進める」

Alan Kayの定義（Smalltalk の設計者）:
  1. Everything is an object（すべてはオブジェクト）
  2. Objects communicate by sending messages（メッセージで通信）
  3. Objects have their own memory（独自のメモリを持つ）
  4. Every object is an instance of a class（クラスのインスタンス）
  5. The class holds shared behavior（クラスが共通の振る舞いを保持）
```

### 1.1 OOPの定義を深掘りする

OOPの定義は時代と論者によって異なる。大きく分けて2つの流派がある。

```
Scandinavian School（スカンジナビア学派）:
  → Simula から派生
  → クラス、継承、静的型付けを重視
  → C++, Java, C# に受け継がれる
  → 「OOP = クラスベースのプログラミング」

American School（アメリカ学派）:
  → Smalltalk から派生
  → メッセージパッシング、動的型付けを重視
  → Ruby, Python, Objective-C に受け継がれる
  → 「OOP = オブジェクト間のメッセージング」

現代の統合的理解:
  → どちらか一方ではなく、両方の要素を組み合わせる
  → TypeScript, Kotlin, Swift は両方の長所を取り入れている
```

### 1.2 OOPを支える4つの柱

OOPには4つの基本原則がある。これらは後続の章で詳しく扱うが、ここで概要を押さえておく。

```
4つの柱（Four Pillars of OOP）:

  1. カプセル化（Encapsulation）
     → データと振る舞いを1つの単位にまとめる
     → 内部実装を隠蔽し、公開APIのみを提供する
     → 変更の影響範囲を限定する

  2. 継承（Inheritance）
     → 既存クラスの機能を引き継いで新しいクラスを作る
     → コードの再利用を促進する
     → is-a 関係を表現する

  3. ポリモーフィズム（Polymorphism）
     → 同じインターフェースで異なる実装を呼び出す
     → 実行時に適切なメソッドが選択される
     → 柔軟で拡張可能なコードを実現する

  4. 抽象化（Abstraction）
     → 複雑な詳細を隠し、本質的な特徴のみを表現する
     → 抽象クラスやインターフェースで契約を定義する
     → 利用者が知る必要のない複雑さを隠す
```

```typescript
// TypeScript: 4つの柱を1つの例で示す

// 抽象化: 共通のインターフェースを定義
interface Shape {
  area(): number;
  perimeter(): number;
  describe(): string;
}

// カプセル化: 内部データを隠蔽
class Circle implements Shape {
  private readonly _radius: number;

  constructor(radius: number) {
    if (radius <= 0) throw new Error("半径は正の数である必要があります");
    this._radius = radius;
  }

  // ポリモーフィズム: Shape インターフェースの実装
  area(): number {
    return Math.PI * this._radius ** 2;
  }

  perimeter(): number {
    return 2 * Math.PI * this._radius;
  }

  describe(): string {
    return `円（半径: ${this._radius}）`;
  }

  get radius(): number {
    return this._radius;
  }
}

// 継承 + ポリモーフィズム
class Rectangle implements Shape {
  constructor(
    private readonly width: number,
    private readonly height: number,
  ) {
    if (width <= 0 || height <= 0) {
      throw new Error("幅と高さは正の数である必要があります");
    }
  }

  area(): number {
    return this.width * this.height;
  }

  perimeter(): number {
    return 2 * (this.width + this.height);
  }

  describe(): string {
    return `長方形（幅: ${this.width}, 高さ: ${this.height}）`;
  }
}

// 継承: Rectangle を拡張
class Square extends Rectangle {
  constructor(side: number) {
    super(side, side);
  }

  describe(): string {
    return `正方形（辺: ${this.perimeter() / 4}）`;
  }
}

// ポリモーフィズム: 異なる型を同じインターフェースで扱う
function printShapeInfo(shapes: Shape[]): void {
  for (const shape of shapes) {
    console.log(`${shape.describe()} - 面積: ${shape.area().toFixed(2)}`);
  }
}

const shapes: Shape[] = [
  new Circle(5),
  new Rectangle(4, 6),
  new Square(3),
];
printShapeInfo(shapes);
```

---

## 2. メンタルモデル

```
手続き型のメンタルモデル:
  「手順書」— 上から順に実行する命令の列

  1. ユーザー情報を取得する
  2. バリデーションする
  3. データベースに保存する
  4. メールを送信する

OOPのメンタルモデル:
  「役割を持つ人々の組織」— 各人が責任を持って仕事する

  ┌─────────┐    ┌──────────┐    ┌──────────┐
  │  User    │───→│ Validator│───→│ Database │
  │ (データ) │    │ (検証)   │    │ (保存)   │
  └─────────┘    └──────────┘    └──────────┘
       │                              │
       │         ┌──────────┐         │
       └────────→│ Mailer   │←────────┘
                 │ (通知)   │
                 └──────────┘

  各オブジェクトは:
    - 自分のデータ（状態）を管理
    - 自分の責任範囲の処理を実行
    - 他のオブジェクトにメッセージ（メソッド呼び出し）を送る
```

### 2.1 手続き型からOOPへの思考の転換

手続き型とOOPでは、問題に対するアプローチが根本的に異なる。以下に具体例で比較する。

```python
# === 手続き型アプローチ: ユーザー登録処理 ===

# データは辞書で管理
users_db = []

def validate_email(email: str) -> bool:
    """メールアドレスの妥当性チェック"""
    return "@" in email and "." in email.split("@")[1]

def validate_password(password: str) -> bool:
    """パスワードの妥当性チェック"""
    return len(password) >= 8

def hash_password(password: str) -> str:
    """パスワードのハッシュ化"""
    import hashlib
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(name: str, email: str, password: str) -> dict:
    """ユーザー登録の手順を逐次実行"""
    # 手順1: バリデーション
    if not validate_email(email):
        raise ValueError("無効なメールアドレス")
    if not validate_password(password):
        raise ValueError("パスワードは8文字以上")

    # 手順2: 重複チェック
    for user in users_db:
        if user["email"] == email:
            raise ValueError("既に登録済みのメールアドレス")

    # 手順3: ユーザー作成
    user = {
        "name": name,
        "email": email,
        "password_hash": hash_password(password),
    }

    # 手順4: 保存
    users_db.append(user)
    return user

# 問題点:
# - データ（users_db）と関数が分離している
# - グローバル状態に依存
# - テストが困難（users_db のリセットが必要）
# - 機能追加時に関数がどんどん増える
```

```python
# === OOPアプローチ: ユーザー登録処理 ===

import hashlib
from dataclasses import dataclass, field
from typing import Optional


class EmailAddress:
    """メールアドレス値オブジェクト: バリデーションを内包"""
    def __init__(self, value: str):
        if "@" not in value or "." not in value.split("@")[1]:
            raise ValueError(f"無効なメールアドレス: {value}")
        self._value = value

    @property
    def value(self) -> str:
        return self._value

    def __eq__(self, other: object) -> bool:
        if isinstance(other, EmailAddress):
            return self._value == other._value
        return False

    def __hash__(self) -> int:
        return hash(self._value)

    def __str__(self) -> str:
        return self._value


class Password:
    """パスワード値オブジェクト: ハッシュ化を内包"""
    def __init__(self, plain_text: str):
        if len(plain_text) < 8:
            raise ValueError("パスワードは8文字以上必要です")
        self._hash = hashlib.sha256(plain_text.encode()).hexdigest()

    @property
    def hashed(self) -> str:
        return self._hash

    def verify(self, plain_text: str) -> bool:
        return hashlib.sha256(plain_text.encode()).hexdigest() == self._hash


class User:
    """ユーザーエンティティ: データと振る舞いを統合"""
    def __init__(self, name: str, email: EmailAddress, password: Password):
        self._name = name
        self._email = email
        self._password = password

    @property
    def name(self) -> str:
        return self._name

    @property
    def email(self) -> EmailAddress:
        return self._email

    def authenticate(self, plain_password: str) -> bool:
        """認証はユーザー自身の責任"""
        return self._password.verify(plain_password)


class UserRepository:
    """ユーザーリポジトリ: データの永続化を担当"""
    def __init__(self):
        self._users: list[User] = []

    def find_by_email(self, email: EmailAddress) -> Optional[User]:
        for user in self._users:
            if user.email == email:
                return user
        return None

    def save(self, user: User) -> None:
        if self.find_by_email(user.email) is not None:
            raise ValueError("既に登録済みのメールアドレス")
        self._users.append(user)

    @property
    def count(self) -> int:
        return len(self._users)


class UserRegistrationService:
    """ユーザー登録サービス: ユースケースを調整"""
    def __init__(self, repository: UserRepository):
        self._repository = repository

    def register(self, name: str, email_str: str, password_str: str) -> User:
        # 各オブジェクトが自分の責任範囲のバリデーションを行う
        email = EmailAddress(email_str)        # メール形式チェック
        password = Password(password_str)      # パスワード強度チェック
        user = User(name, email, password)     # ユーザー生成
        self._repository.save(user)            # 永続化（重複チェック含む）
        return user


# 利点:
# - 各クラスが明確な責任を持つ
# - バリデーションがデータと一体化
# - テスト容易（リポジトリをモックに差し替え可能）
# - 機能追加がクラス単位で管理できる
```

### 2.2 現実世界のアナロジー

OOPの概念を理解する上で、現実世界のアナロジーが役立つ。

```
レストランのアナロジー:

  手続き型的思考:
    1. お客さんが来店する
    2. メニューを見せる
    3. 注文を取る
    4. 注文をキッチンに伝える
    5. 料理を作る
    6. 料理を運ぶ
    7. 会計する
    → すべての手順を1つのスクリプトで管理

  OOP的思考:
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ Customer │───→│ Waiter   │───→│ Kitchen  │
    │ 注文する │    │ 取り次ぐ │    │ 調理する │
    └──────────┘    └──────────┘    └──────────┘
         │                              │
         │         ┌──────────┐         │
         └────────→│ Cashier  │←────────┘
                   │ 会計する │
                   └──────────┘

    各オブジェクト（人）が自分の責任を遂行する:
    - Customer: メニューを選ぶ、食べる、支払う
    - Waiter: 注文を取る、料理を運ぶ
    - Kitchen: 注文に基づいて料理を作る
    - Cashier: 合計を計算して会計する

    → 新しいメニューが増えても Kitchen だけ変更
    → 支払い方法が変わっても Cashier だけ変更
    → 変更の影響範囲が限定される
```

```typescript
// TypeScript: レストランのアナロジーをコードで表現

interface MenuItem {
  name: string;
  price: number;
  category: "appetizer" | "main" | "dessert" | "drink";
}

class Order {
  private _items: MenuItem[] = [];
  private _status: "pending" | "preparing" | "ready" | "served" = "pending";

  addItem(item: MenuItem): void {
    if (this._status !== "pending") {
      throw new Error("注文確定後は追加できません");
    }
    this._items.push(item);
  }

  get total(): number {
    return this._items.reduce((sum, item) => sum + item.price, 0);
  }

  get items(): ReadonlyArray<MenuItem> {
    return [...this._items]; // 防衛的コピー
  }

  get status(): string {
    return this._status;
  }

  confirm(): void {
    if (this._items.length === 0) {
      throw new Error("空の注文は確定できません");
    }
    this._status = "preparing";
  }

  markReady(): void {
    this._status = "ready";
  }

  markServed(): void {
    this._status = "served";
  }
}

class Customer {
  private _currentOrder: Order | null = null;

  constructor(public readonly name: string) {}

  createOrder(): Order {
    this._currentOrder = new Order();
    return this._currentOrder;
  }

  get currentOrder(): Order | null {
    return this._currentOrder;
  }
}

class Kitchen {
  private _queue: Order[] = [];

  receiveOrder(order: Order): void {
    order.confirm();
    this._queue.push(order);
    console.log(`キッチン: 注文を受け付けました（${order.items.length}品）`);
  }

  prepareNext(): Order | null {
    const order = this._queue.shift();
    if (order) {
      order.markReady();
      console.log("キッチン: 料理が完成しました");
    }
    return order ?? null;
  }

  get pendingOrders(): number {
    return this._queue.length;
  }
}

class Waiter {
  constructor(
    private readonly name: string,
    private readonly kitchen: Kitchen,
  ) {}

  takeOrder(customer: Customer, items: MenuItem[]): void {
    const order = customer.createOrder();
    for (const item of items) {
      order.addItem(item);
    }
    this.kitchen.receiveOrder(order);
    console.log(`${this.name}: ${customer.name}様の注文を受け付けました`);
  }

  serveOrder(order: Order): void {
    order.markServed();
    console.log(`${this.name}: 料理をお持ちしました`);
  }
}

class Cashier {
  private _totalRevenue = 0;

  checkout(order: Order): number {
    const total = order.total;
    const tax = Math.floor(total * 0.1);
    const grandTotal = total + tax;
    this._totalRevenue += grandTotal;
    console.log(`会計: 小計 ${total}円 + 税 ${tax}円 = ${grandTotal}円`);
    return grandTotal;
  }

  get totalRevenue(): number {
    return this._totalRevenue;
  }
}
```

---

## 3. オブジェクトの3要素

```
オブジェクト = 状態（State）+ 振る舞い（Behavior）+ アイデンティティ（Identity）

  ┌─────────────────────────────────┐
  │        BankAccount              │
  ├─────────────────────────────────┤
  │ 状態（State）:                   │
  │   - owner: "田中太郎"           │
  │   - balance: 100000             │
  │   - accountNumber: "1234567"    │
  ├─────────────────────────────────┤
  │ 振る舞い（Behavior）:            │
  │   - deposit(amount)             │
  │   - withdraw(amount)            │
  │   - getBalance()                │
  ├─────────────────────────────────┤
  │ アイデンティティ（Identity）:     │
  │   - メモリアドレス: 0x7ff...     │
  │   - 同じ状態でも別のオブジェクト  │
  └─────────────────────────────────┘
```

### 3.1 状態（State）の管理

状態はオブジェクトが持つデータであり、時間とともに変化し得る。状態管理はOOPの最も重要な関心事の1つである。

```typescript
// TypeScript: 状態管理の実践例 - ECサイトのショッピングカート

interface Product {
  readonly id: string;
  readonly name: string;
  readonly price: number;
  readonly stock: number;
}

interface CartItem {
  readonly product: Product;
  quantity: number;
}

class ShoppingCart {
  // 状態: カート内の商品リスト
  private _items: Map<string, CartItem> = new Map();
  // 状態: カートの作成日時
  private readonly _createdAt: Date = new Date();
  // 状態: 最終更新日時
  private _updatedAt: Date = new Date();

  /**
   * 商品をカートに追加する
   * ビジネスルール: 在庫を超える数量は追加できない
   */
  addItem(product: Product, quantity: number = 1): void {
    if (quantity <= 0) {
      throw new Error("数量は1以上である必要があります");
    }

    const existing = this._items.get(product.id);
    const currentQty = existing?.quantity ?? 0;
    const newQty = currentQty + quantity;

    if (newQty > product.stock) {
      throw new Error(
        `在庫不足: ${product.name}の在庫は${product.stock}個です`
      );
    }

    this._items.set(product.id, { product, quantity: newQty });
    this._updatedAt = new Date();
  }

  /**
   * 商品の数量を変更する
   */
  updateQuantity(productId: string, quantity: number): void {
    if (quantity < 0) {
      throw new Error("数量は0以上である必要があります");
    }

    if (quantity === 0) {
      this._items.delete(productId);
    } else {
      const item = this._items.get(productId);
      if (!item) {
        throw new Error("カートに存在しない商品です");
      }
      if (quantity > item.product.stock) {
        throw new Error("在庫を超える数量は指定できません");
      }
      item.quantity = quantity;
    }
    this._updatedAt = new Date();
  }

  /**
   * カート内の商品を削除する
   */
  removeItem(productId: string): void {
    if (!this._items.has(productId)) {
      throw new Error("カートに存在しない商品です");
    }
    this._items.delete(productId);
    this._updatedAt = new Date();
  }

  /**
   * 小計を計算する
   */
  get subtotal(): number {
    let total = 0;
    for (const item of this._items.values()) {
      total += item.product.price * item.quantity;
    }
    return total;
  }

  /**
   * 税込合計を計算する
   */
  get totalWithTax(): number {
    return Math.floor(this.subtotal * 1.1);
  }

  /**
   * カート内の商品数を返す
   */
  get itemCount(): number {
    let count = 0;
    for (const item of this._items.values()) {
      count += item.quantity;
    }
    return count;
  }

  /**
   * カートが空かどうかを返す
   */
  get isEmpty(): boolean {
    return this._items.size === 0;
  }

  /**
   * カートの内容を表示する
   */
  display(): string {
    if (this.isEmpty) return "カートは空です";

    const lines: string[] = ["=== ショッピングカート ==="];
    for (const item of this._items.values()) {
      const lineTotal = item.product.price * item.quantity;
      lines.push(
        `${item.product.name} x${item.quantity} = ¥${lineTotal.toLocaleString()}`
      );
    }
    lines.push("─".repeat(30));
    lines.push(`小計: ¥${this.subtotal.toLocaleString()}`);
    lines.push(`合計（税込）: ¥${this.totalWithTax.toLocaleString()}`);
    return lines.join("\n");
  }
}
```

### 3.2 振る舞い（Behavior）の設計

振る舞いはオブジェクトが外部に提供する操作であり、オブジェクトの状態を安全に変更するための手段である。

```python
# Python: 振る舞いの設計 - タスク管理システム

from datetime import datetime, timedelta
from enum import Enum
from typing import Optional


class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class TaskStatus(Enum):
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    IN_REVIEW = "in_review"
    DONE = "done"
    CANCELLED = "cancelled"


class Task:
    """タスク: 状態遷移のルールを振る舞いとして内包"""

    # 許可される状態遷移を定義
    _VALID_TRANSITIONS = {
        TaskStatus.TODO: {TaskStatus.IN_PROGRESS, TaskStatus.CANCELLED},
        TaskStatus.IN_PROGRESS: {TaskStatus.IN_REVIEW, TaskStatus.TODO, TaskStatus.CANCELLED},
        TaskStatus.IN_REVIEW: {TaskStatus.DONE, TaskStatus.IN_PROGRESS},
        TaskStatus.DONE: set(),       # 完了後は変更不可
        TaskStatus.CANCELLED: set(),  # キャンセル後は変更不可
    }

    def __init__(
        self,
        title: str,
        description: str = "",
        priority: Priority = Priority.MEDIUM,
        due_date: Optional[datetime] = None,
    ):
        if not title.strip():
            raise ValueError("タイトルは必須です")

        self._title = title
        self._description = description
        self._priority = priority
        self._status = TaskStatus.TODO
        self._due_date = due_date
        self._created_at = datetime.now()
        self._updated_at = datetime.now()
        self._history: list[tuple[datetime, str]] = []
        self._history.append((self._created_at, f"タスク作成: {title}"))

    @property
    def title(self) -> str:
        return self._title

    @property
    def status(self) -> TaskStatus:
        return self._status

    @property
    def priority(self) -> Priority:
        return self._priority

    @property
    def is_overdue(self) -> bool:
        """期限切れかどうかを判定"""
        if self._due_date is None:
            return False
        if self._status in (TaskStatus.DONE, TaskStatus.CANCELLED):
            return False
        return datetime.now() > self._due_date

    def start(self) -> None:
        """タスクを開始する"""
        self._transition_to(TaskStatus.IN_PROGRESS, "タスク開始")

    def submit_for_review(self) -> None:
        """レビュー依頼する"""
        self._transition_to(TaskStatus.IN_REVIEW, "レビュー依頼")

    def complete(self) -> None:
        """タスクを完了する"""
        self._transition_to(TaskStatus.DONE, "タスク完了")

    def cancel(self, reason: str = "") -> None:
        """タスクをキャンセルする"""
        self._transition_to(TaskStatus.CANCELLED, f"キャンセル: {reason}")

    def send_back(self, reason: str = "") -> None:
        """差し戻す（レビュー → 作業中）"""
        self._transition_to(TaskStatus.IN_PROGRESS, f"差し戻し: {reason}")

    def update_priority(self, new_priority: Priority) -> None:
        """優先度を変更する"""
        old = self._priority
        self._priority = new_priority
        self._record_change(f"優先度変更: {old.name} → {new_priority.name}")

    def _transition_to(self, new_status: TaskStatus, message: str) -> None:
        """状態遷移のルールを強制する"""
        valid = self._VALID_TRANSITIONS.get(self._status, set())
        if new_status not in valid:
            raise ValueError(
                f"無効な状態遷移: {self._status.value} → {new_status.value}"
            )
        old_status = self._status
        self._status = new_status
        self._record_change(f"{message} ({old_status.value} → {new_status.value})")

    def _record_change(self, message: str) -> None:
        """変更履歴を記録する"""
        now = datetime.now()
        self._updated_at = now
        self._history.append((now, message))

    def get_history(self) -> list[tuple[datetime, str]]:
        """変更履歴を返す（防衛的コピー）"""
        return list(self._history)

    def __str__(self) -> str:
        overdue = " [期限切れ]" if self.is_overdue else ""
        return f"[{self._priority.name}] {self._title} ({self._status.value}){overdue}"


# 使用例
task = Task("ログイン機能の実装", priority=Priority.HIGH,
            due_date=datetime.now() + timedelta(days=7))
print(task)                    # [HIGH] ログイン機能の実装 (todo)

task.start()
print(task)                    # [HIGH] ログイン機能の実装 (in_progress)

task.submit_for_review()
print(task)                    # [HIGH] ログイン機能の実装 (in_review)

task.complete()
print(task)                    # [HIGH] ログイン機能の実装 (done)

# task.start()  # ValueError: 無効な状態遷移: done → in_progress
```

### 3.3 アイデンティティ（Identity）の重要性

アイデンティティは、同じ状態を持つ2つのオブジェクトを区別するための概念である。

```java
// Java: アイデンティティと等価性（equality）の違い

public class Money {
    private final int amount;
    private final String currency;

    public Money(int amount, String currency) {
        this.amount = amount;
        this.currency = currency;
    }

    // 値オブジェクト: 状態が同じなら等価
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;                    // アイデンティティ一致
        if (obj == null || getClass() != obj.getClass()) return false;
        Money money = (Money) obj;
        return amount == money.amount
            && Objects.equals(currency, money.currency); // 状態で比較
    }

    @Override
    public int hashCode() {
        return Objects.hash(amount, currency);
    }

    @Override
    public String toString() {
        return amount + " " + currency;
    }
}

public class Customer {
    private final String id;   // アイデンティティ: IDで区別
    private String name;
    private String email;

    public Customer(String id, String name, String email) {
        this.id = id;
        this.name = name;
        this.email = email;
    }

    // エンティティ: IDが同じなら同一
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        Customer customer = (Customer) obj;
        return Objects.equals(id, customer.id);  // IDのみで比較
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }
}

// 使用例:
// Money: 同じ金額・通貨なら「等価」
Money m1 = new Money(1000, "JPY");
Money m2 = new Money(1000, "JPY");
System.out.println(m1.equals(m2));   // true（値が同じなら等価）
System.out.println(m1 == m2);        // false（異なるオブジェクト）

// Customer: 同じIDなら「同一人物」
Customer c1 = new Customer("C001", "田中", "tanaka@example.com");
Customer c2 = new Customer("C001", "田中太郎", "t.tanaka@example.com");
System.out.println(c1.equals(c2));   // true（IDが同じなら同一）
// → 名前やメールが変わっても同一人物
```

```typescript
// TypeScript: エンティティと値オブジェクトの区別

// 値オブジェクト: 状態で等価性を判定
class Address {
  constructor(
    public readonly prefecture: string,
    public readonly city: string,
    public readonly street: string,
    public readonly zipCode: string,
  ) {}

  equals(other: Address): boolean {
    return (
      this.prefecture === other.prefecture &&
      this.city === other.city &&
      this.street === other.street &&
      this.zipCode === other.zipCode
    );
  }

  toString(): string {
    return `〒${this.zipCode} ${this.prefecture}${this.city}${this.street}`;
  }
}

// エンティティ: IDで同一性を判定
class Employee {
  private _name: string;
  private _address: Address;
  private _department: string;

  constructor(
    public readonly employeeId: string,
    name: string,
    address: Address,
    department: string,
  ) {
    this._name = name;
    this._address = address;
    this._department = department;
  }

  get name(): string { return this._name; }
  get address(): Address { return this._address; }
  get department(): string { return this._department; }

  transfer(newDepartment: string): void {
    this._department = newDepartment;
  }

  relocate(newAddress: Address): void {
    this._address = newAddress;
  }

  equals(other: Employee): boolean {
    // 社員番号が同じなら同一人物
    return this.employeeId === other.employeeId;
  }
}
```

### コード例

```typescript
// TypeScript: 銀行口座オブジェクト
class BankAccount {
  // 状態（State）
  private owner: string;
  private balance: number;
  private readonly accountNumber: string;

  constructor(owner: string, accountNumber: string, initialBalance: number = 0) {
    this.owner = owner;
    this.accountNumber = accountNumber;
    this.balance = initialBalance;
  }

  // 振る舞い（Behavior）
  deposit(amount: number): void {
    if (amount <= 0) throw new Error("入金額は正の数である必要があります");
    this.balance += amount;
  }

  withdraw(amount: number): void {
    if (amount > this.balance) throw new Error("残高不足");
    this.balance -= amount;
  }

  getBalance(): number {
    return this.balance;
  }
}

// アイデンティティ: 同じ状態でも別のオブジェクト
const account1 = new BankAccount("田中", "001", 10000);
const account2 = new BankAccount("田中", "001", 10000);
console.log(account1 === account2); // false（別のオブジェクト）
```

```python
# Python: 同じ概念
class BankAccount:
    def __init__(self, owner: str, account_number: str, initial_balance: float = 0):
        self._owner = owner
        self._account_number = account_number
        self._balance = initial_balance

    def deposit(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("入金額は正の数である必要があります")
        self._balance += amount

    def withdraw(self, amount: float) -> None:
        if amount > self._balance:
            raise ValueError("残高不足")
        self._balance -= amount

    @property
    def balance(self) -> float:
        return self._balance
```

```java
// Java: 同じ概念
public class BankAccount {
    private String owner;
    private double balance;
    private final String accountNumber;

    public BankAccount(String owner, String accountNumber, double initialBalance) {
        this.owner = owner;
        this.accountNumber = accountNumber;
        this.balance = initialBalance;
    }

    public void deposit(double amount) {
        if (amount <= 0) throw new IllegalArgumentException("入金額は正の数");
        this.balance += amount;
    }

    public void withdraw(double amount) {
        if (amount > this.balance) throw new IllegalStateException("残高不足");
        this.balance -= amount;
    }

    public double getBalance() { return balance; }
}
```

---

## 4. メッセージパッシング

```
OOPの本質はメッセージパッシング:

  手続き型的な考え方:
    result = validate(user_data)    ← 関数にデータを渡す

  OOP的な考え方:
    result = validator.validate(user_data)  ← オブジェクトにメッセージを送る

  違い:
    手続き型: 「誰が」処理するか不明
    OOP:     「validator」が責任を持って処理する

  メッセージパッシングの利点:
    1. 責任の所在が明確
    2. 実装を差し替え可能（ポリモーフィズム）
    3. テスト時にモックに差し替え可能
```

### 4.1 メッセージパッシングの実践

メッセージパッシングは単なるメソッド呼び出しではない。「オブジェクトに仕事を依頼する」という考え方である。

```typescript
// TypeScript: メッセージパッシングの実践例 - 通知システム

// 通知を送る「能力」を定義
interface NotificationSender {
  send(recipient: string, message: string): Promise<boolean>;
  readonly channelName: string;
}

// メール通知
class EmailSender implements NotificationSender {
  readonly channelName = "Email";

  async send(recipient: string, message: string): Promise<boolean> {
    console.log(`メール送信: ${recipient} → ${message}`);
    // 実際にはSMTPサーバーに接続して送信
    return true;
  }
}

// Slack通知
class SlackSender implements NotificationSender {
  readonly channelName = "Slack";

  constructor(private readonly webhookUrl: string) {}

  async send(recipient: string, message: string): Promise<boolean> {
    console.log(`Slack送信: ${recipient} → ${message}`);
    // 実際にはwebhook APIを呼び出して送信
    return true;
  }
}

// SMS通知
class SmsSender implements NotificationSender {
  readonly channelName = "SMS";

  async send(recipient: string, message: string): Promise<boolean> {
    console.log(`SMS送信: ${recipient} → ${message}`);
    // 実際にはSMS APIを呼び出して送信
    return true;
  }
}

// 通知サービス: 送信者を差し替え可能
class NotificationService {
  private senders: NotificationSender[] = [];

  addSender(sender: NotificationSender): void {
    this.senders.push(sender);
  }

  async notifyAll(recipient: string, message: string): Promise<void> {
    for (const sender of this.senders) {
      try {
        // メッセージパッシング: 各 sender に「送って」とメッセージを送る
        const success = await sender.send(recipient, message);
        if (success) {
          console.log(`${sender.channelName}: 送信成功`);
        }
      } catch (error) {
        console.error(`${sender.channelName}: 送信失敗`);
      }
    }
  }
}

// 利用例
const service = new NotificationService();
service.addSender(new EmailSender());
service.addSender(new SlackSender("https://hooks.slack.com/..."));
service.addSender(new SmsSender());

// 同じメッセージを全チャネルに送信
// 各 sender が自分の方法で処理する（ポリモーフィズム）
await service.notifyAll("user@example.com", "サーバーアラート発生");
```

### 4.2 メッセージパッシングとポリモーフィズムの関係

```python
# Python: メッセージパッシングがポリモーフィズムを実現する

from abc import ABC, abstractmethod
from typing import BinaryIO
import json
import csv
import io


class DataExporter(ABC):
    """データエクスポーターの抽象基底クラス"""

    @abstractmethod
    def export(self, data: list[dict]) -> str:
        """データをエクスポートする"""
        pass

    @abstractmethod
    def file_extension(self) -> str:
        """ファイル拡張子を返す"""
        pass


class JsonExporter(DataExporter):
    """JSON形式でエクスポート"""

    def export(self, data: list[dict]) -> str:
        return json.dumps(data, ensure_ascii=False, indent=2)

    def file_extension(self) -> str:
        return ".json"


class CsvExporter(DataExporter):
    """CSV形式でエクスポート"""

    def export(self, data: list[dict]) -> str:
        if not data:
            return ""
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
        return output.getvalue()

    def file_extension(self) -> str:
        return ".csv"


class MarkdownExporter(DataExporter):
    """Markdown表形式でエクスポート"""

    def export(self, data: list[dict]) -> str:
        if not data:
            return ""
        headers = list(data[0].keys())
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in data:
            values = [str(row.get(h, "")) for h in headers]
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    def file_extension(self) -> str:
        return ".md"


class ReportGenerator:
    """レポート生成器: エクスポーターを差し替え可能"""

    def __init__(self, exporter: DataExporter):
        self._exporter = exporter

    def generate(self, data: list[dict]) -> str:
        # メッセージパッシング: exporter に「エクスポートして」と依頼
        # どの形式で出力するかは exporter が決める
        return self._exporter.export(data)

    @property
    def output_extension(self) -> str:
        return self._exporter.file_extension()


# 使用例
data = [
    {"名前": "田中", "年齢": 25, "部署": "開発"},
    {"名前": "山田", "年齢": 30, "部署": "営業"},
]

# 同じデータ、異なるフォーマット
for exporter in [JsonExporter(), CsvExporter(), MarkdownExporter()]:
    generator = ReportGenerator(exporter)
    print(f"--- {generator.output_extension} ---")
    print(generator.generate(data))
    print()
```

### 4.3 Smalltalk スタイルのメッセージパッシング

Alan Kay が提唱した本来のメッセージパッシングは、現代の多くの言語のメソッド呼び出しとは概念的に異なる。

```
Smalltalk のメッセージパッシング:
  → オブジェクトにメッセージを送る
  → オブジェクトがメッセージを受け取り、どう処理するか自分で決める
  → メッセージに対応するメソッドがなければ doesNotUnderstand: が呼ばれる

  3 + 4
  → 3（Integerオブジェクト）に「+」というメッセージと「4」という引数を送る
  → 3 が自分で加算の方法を決める

現代の言語のメソッド呼び出し:
  → コンパイラ/インタプリタが呼び出すメソッドを解決する
  → 存在しないメソッドはコンパイルエラー（静的型付けの場合）

Ruby（Smalltalk の影響が強い）:
  → method_missing によるメッセージハンドリング
  → 動的なメソッド解決
```

```python
# Python: __getattr__ を使ったメッセージパッシング的パターン

class FluentQuery:
    """流暢なAPIを持つクエリビルダー"""

    def __init__(self):
        self._conditions: list[str] = []
        self._table: str = ""
        self._limit: int | None = None
        self._order_by: str | None = None

    def from_table(self, table: str) -> "FluentQuery":
        self._table = table
        return self

    def where(self, condition: str) -> "FluentQuery":
        self._conditions.append(condition)
        return self

    def limit(self, n: int) -> "FluentQuery":
        self._limit = n
        return self

    def order_by(self, column: str) -> "FluentQuery":
        self._order_by = column
        return self

    def build(self) -> str:
        query = f"SELECT * FROM {self._table}"
        if self._conditions:
            query += " WHERE " + " AND ".join(self._conditions)
        if self._order_by:
            query += f" ORDER BY {self._order_by}"
        if self._limit:
            query += f" LIMIT {self._limit}"
        return query


# メソッドチェーン = メッセージの連鎖
query = (
    FluentQuery()
    .from_table("users")
    .where("age >= 18")
    .where("status = 'active'")
    .order_by("created_at DESC")
    .limit(10)
    .build()
)
print(query)
# SELECT * FROM users WHERE age >= 18 AND status = 'active' ORDER BY created_at DESC LIMIT 10
```

---

## 5. OOPが解決する問題

```
OOPなし（手続き型）:
  問題1: グローバル状態の管理困難
    → 誰がどの変数を変更したか追跡不能
    → OOP: カプセル化で状態をオブジェクト内に閉じ込める

  問題2: コードの重複
    → 似た処理を何度も書く
    → OOP: 継承・コンポジションで共通化

  問題3: 変更の影響範囲が不明
    → 1箇所の変更が全体に波及
    → OOP: インターフェースで依存を制御

  問題4: 大規模コードの構造化困難
    → 1万行超えると手続き型は破綻
    → OOP: クラス・パッケージで構造化
```

### 5.1 問題と解決の具体例

```typescript
// TypeScript: OOPが解決する問題の具体例

// === 問題1: グローバル状態の管理困難 ===

// 手続き型（問題あり）
let globalUserCount = 0;
let globalTotalRevenue = 0;
// ... 100箇所から参照・変更される → カオス

// OOP（解決）
class Analytics {
  private _userCount = 0;
  private _totalRevenue = 0;

  incrementUsers(): void {
    this._userCount++;
  }

  addRevenue(amount: number): void {
    if (amount < 0) throw new Error("収益は正の数");
    this._totalRevenue += amount;
  }

  get report(): { users: number; revenue: number } {
    return { users: this._userCount, revenue: this._totalRevenue };
  }
}
// → 状態の変更は Analytics オブジェクト経由のみ
// → 不正な変更を防止できる


// === 問題2: コードの重複 ===

// 手続き型（問題あり）
function validateUserEmail(email: string): boolean {
  return /^[\w.-]+@[\w.-]+\.\w+$/.test(email);
}
function validateAdminEmail(email: string): boolean {
  return /^[\w.-]+@[\w.-]+\.\w+$/.test(email); // 同じロジック！
}

// OOP（解決）
class EmailValidator {
  private readonly pattern = /^[\w.-]+@[\w.-]+\.\w+$/;

  validate(email: string): boolean {
    return this.pattern.test(email);
  }
}
// → 1箇所にまとめ、再利用


// === 問題3: 変更の影響範囲が不明 ===

// インターフェースで契約を定義
interface PaymentProcessor {
  charge(amount: number, currency: string): Promise<PaymentResult>;
  refund(transactionId: string): Promise<RefundResult>;
}

interface PaymentResult {
  transactionId: string;
  success: boolean;
}

interface RefundResult {
  success: boolean;
  refundedAmount: number;
}

// 実装を差し替えても利用側に影響なし
class StripePaymentProcessor implements PaymentProcessor {
  async charge(amount: number, currency: string): Promise<PaymentResult> {
    // Stripe API を使った実装
    return { transactionId: "stripe_xxx", success: true };
  }

  async refund(transactionId: string): Promise<RefundResult> {
    return { success: true, refundedAmount: 1000 };
  }
}

class PayPayPaymentProcessor implements PaymentProcessor {
  async charge(amount: number, currency: string): Promise<PaymentResult> {
    // PayPay API を使った実装
    return { transactionId: "paypay_xxx", success: true };
  }

  async refund(transactionId: string): Promise<RefundResult> {
    return { success: true, refundedAmount: 1000 };
  }
}

// 利用側: PaymentProcessor だけに依存
class CheckoutService {
  constructor(private processor: PaymentProcessor) {}

  async checkout(amount: number): Promise<string> {
    const result = await this.processor.charge(amount, "JPY");
    if (!result.success) throw new Error("決済失敗");
    return result.transactionId;
  }
}
// → processor の実装を差し替えても CheckoutService は変更不要


// === 問題4: 大規模コードの構造化 ===

// モジュール構造の例
// src/
//   domain/
//     entities/User.ts, Order.ts, Product.ts
//     value-objects/Money.ts, Address.ts
//     repositories/UserRepository.ts
//   application/
//     services/OrderService.ts, UserService.ts
//   infrastructure/
//     database/PostgresUserRepository.ts
//     external/StripePaymentProcessor.ts
//   presentation/
//     controllers/OrderController.ts

// → クラスがファイルの構造化単位になる
// → 依存関係がインターフェースで明確化される
```

### 5.2 スケーラビリティの問題

OOPは特にプロジェクトが成長するときにその価値を発揮する。

```
プロジェクトの成長とパラダイムの適性:

  100行: 手続き型で十分
    → 関数をいくつか定義するだけ
    → クラスは過剰設計

  1,000行: OOPが有効になり始める
    → 状態管理が複雑になる
    → モジュール分割が必要

  10,000行: OOP無しでは管理困難
    → グローバル状態の追跡が不可能に
    → 変更の影響範囲が予測不能に
    → テストが書けなくなる

  100,000行以上: OOP + 設計パターンが必須
    → レイヤードアーキテクチャ
    → 依存性注入（DI）
    → インターフェースによる疎結合
    → テスト戦略の確立

  大規模プロジェクトでの OOP の価値:
    1. コードの構造化 → クラス/パッケージ/モジュール
    2. チーム分担 → クラスの責任境界 = チームの責任境界
    3. テスト容易性 → モック/スタブによる単体テスト
    4. 変更容易性 → インターフェースによる疎結合
    5. 知識の組織化 → ドメインモデルがビジネス知識を表現
```

---

## 6. OOPの適用領域

```
OOPが得意な領域:
  ✓ GUIアプリケーション（ウィジェット階層）
  ✓ ゲーム開発（エンティティ・コンポーネント）
  ✓ エンタープライズアプリ（ビジネスロジック）
  ✓ フレームワーク設計（拡張ポイントの提供）
  ✓ シミュレーション（現実世界のモデリング）

OOPが不得意な領域:
  ✗ データ変換パイプライン → 関数型が適切
  ✗ 数値計算・科学計算 → 手続き型/配列指向が適切
  ✗ スクリプト・グルーコード → シンプルな手続き型が適切
  ✗ 並行処理 → アクターモデル/関数型が適切

現実のプロジェクト:
  → 複数のパラダイムを組み合わせるのが最適
  → OOP + FP のマルチパラダイムが主流
```

### 6.1 GUIアプリケーションでのOOP

GUIフレームワークはOOPの代表的な適用例である。ウィジェットの階層構造と、イベント駆動の仕組みがOOPと自然に対応する。

```typescript
// TypeScript: GUIコンポーネントの階層（簡易例）

abstract class Widget {
  protected _x: number;
  protected _y: number;
  protected _width: number;
  protected _height: number;
  protected _visible: boolean = true;
  protected _parent: Widget | null = null;
  protected _children: Widget[] = [];

  constructor(x: number, y: number, width: number, height: number) {
    this._x = x;
    this._y = y;
    this._width = width;
    this._height = height;
  }

  addChild(child: Widget): void {
    child._parent = this;
    this._children.push(child);
  }

  abstract render(ctx: CanvasRenderingContext2D): void;

  renderAll(ctx: CanvasRenderingContext2D): void {
    if (!this._visible) return;
    this.render(ctx);
    for (const child of this._children) {
      child.renderAll(ctx);
    }
  }

  hide(): void { this._visible = false; }
  show(): void { this._visible = true; }
}

class Panel extends Widget {
  private _backgroundColor: string;

  constructor(
    x: number, y: number, w: number, h: number,
    backgroundColor: string = "#ffffff",
  ) {
    super(x, y, w, h);
    this._backgroundColor = backgroundColor;
  }

  render(ctx: CanvasRenderingContext2D): void {
    ctx.fillStyle = this._backgroundColor;
    ctx.fillRect(this._x, this._y, this._width, this._height);
  }
}

class Label extends Widget {
  constructor(
    x: number, y: number,
    private _text: string,
    private _fontSize: number = 14,
    private _color: string = "#000000",
  ) {
    super(x, y, 0, 0);
  }

  set text(value: string) { this._text = value; }

  render(ctx: CanvasRenderingContext2D): void {
    ctx.fillStyle = this._color;
    ctx.font = `${this._fontSize}px sans-serif`;
    ctx.fillText(this._text, this._x, this._y);
  }
}

class Button extends Widget {
  private _label: string;
  private _onClick: (() => void) | null = null;
  private _isHovered = false;

  constructor(
    x: number, y: number, w: number, h: number,
    label: string,
  ) {
    super(x, y, w, h);
    this._label = label;
  }

  set onClick(handler: () => void) {
    this._onClick = handler;
  }

  handleClick(mouseX: number, mouseY: number): void {
    if (this.containsPoint(mouseX, mouseY) && this._onClick) {
      this._onClick();
    }
  }

  private containsPoint(px: number, py: number): boolean {
    return (
      px >= this._x && px <= this._x + this._width &&
      py >= this._y && py <= this._y + this._height
    );
  }

  render(ctx: CanvasRenderingContext2D): void {
    ctx.fillStyle = this._isHovered ? "#4488ff" : "#3366cc";
    ctx.fillRect(this._x, this._y, this._width, this._height);
    ctx.fillStyle = "#ffffff";
    ctx.font = "14px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(
      this._label,
      this._x + this._width / 2,
      this._y + this._height / 2,
    );
  }
}
```

### 6.2 ゲーム開発でのOOP

```python
# Python: ゲームエンティティシステム（簡易例）

from abc import ABC, abstractmethod
import math


class Vector2D:
    """2Dベクトル"""
    def __init__(self, x: float = 0, y: float = 0):
        self.x = x
        self.y = y

    def __add__(self, other: "Vector2D") -> "Vector2D":
        return Vector2D(self.x + other.x, self.y + other.y)

    def __mul__(self, scalar: float) -> "Vector2D":
        return Vector2D(self.x * scalar, self.y * scalar)

    def magnitude(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def normalized(self) -> "Vector2D":
        mag = self.magnitude()
        if mag == 0:
            return Vector2D(0, 0)
        return Vector2D(self.x / mag, self.y / mag)

    def distance_to(self, other: "Vector2D") -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


class GameObject(ABC):
    """ゲームオブジェクトの基底クラス"""
    def __init__(self, position: Vector2D, name: str = ""):
        self.position = position
        self.name = name
        self._active = True

    @property
    def is_active(self) -> bool:
        return self._active

    def deactivate(self) -> None:
        self._active = False

    @abstractmethod
    def update(self, delta_time: float) -> None:
        """毎フレーム呼ばれる更新処理"""
        pass

    @abstractmethod
    def render(self) -> str:
        """描画用の文字列を返す"""
        pass


class Player(GameObject):
    """プレイヤーキャラクター"""
    def __init__(self, position: Vector2D, name: str = "Player"):
        super().__init__(position, name)
        self._hp = 100
        self._max_hp = 100
        self._speed = 5.0
        self._attack_power = 10
        self._level = 1
        self._experience = 0
        self._velocity = Vector2D(0, 0)

    @property
    def hp(self) -> int:
        return self._hp

    @property
    def is_alive(self) -> bool:
        return self._hp > 0

    def move(self, direction: Vector2D) -> None:
        self._velocity = direction.normalized() * self._speed

    def take_damage(self, amount: int) -> None:
        self._hp = max(0, self._hp - amount)
        if self._hp == 0:
            self.deactivate()

    def heal(self, amount: int) -> None:
        self._hp = min(self._max_hp, self._hp + amount)

    def gain_experience(self, amount: int) -> None:
        self._experience += amount
        while self._experience >= self._level * 100:
            self._experience -= self._level * 100
            self._level_up()

    def _level_up(self) -> None:
        self._level += 1
        self._max_hp += 10
        self._hp = self._max_hp
        self._attack_power += 2

    def update(self, delta_time: float) -> None:
        if not self.is_active:
            return
        self.position = self.position + self._velocity * delta_time
        self._velocity = Vector2D(0, 0)  # 入力がなければ停止

    def render(self) -> str:
        return f"[{self.name}] HP:{self._hp}/{self._max_hp} Lv:{self._level} Pos:({self.position.x:.1f},{self.position.y:.1f})"


class Enemy(GameObject):
    """敵キャラクター"""
    def __init__(self, position: Vector2D, name: str, hp: int, attack: int, speed: float):
        super().__init__(position, name)
        self._hp = hp
        self._attack = attack
        self._speed = speed
        self._target: Player | None = None

    def set_target(self, player: Player) -> None:
        self._target = player

    def take_damage(self, amount: int) -> None:
        self._hp = max(0, self._hp - amount)
        if self._hp == 0:
            self.deactivate()

    def update(self, delta_time: float) -> None:
        if not self.is_active or self._target is None:
            return

        # プレイヤーに向かって移動
        direction = Vector2D(
            self._target.position.x - self.position.x,
            self._target.position.y - self.position.y,
        )
        self.position = self.position + direction.normalized() * self._speed * delta_time

        # 攻撃範囲内ならダメージ
        if self.position.distance_to(self._target.position) < 1.0:
            self._target.take_damage(self._attack)

    def render(self) -> str:
        return f"[{self.name}] HP:{self._hp} Pos:({self.position.x:.1f},{self.position.y:.1f})"


class GameWorld:
    """ゲームワールド: 全オブジェクトを管理"""
    def __init__(self):
        self._objects: list[GameObject] = []

    def add(self, obj: GameObject) -> None:
        self._objects.append(obj)

    def update(self, delta_time: float) -> None:
        for obj in self._objects:
            if obj.is_active:
                obj.update(delta_time)
        # 非アクティブオブジェクトを除去
        self._objects = [obj for obj in self._objects if obj.is_active]

    def render(self) -> str:
        return "\n".join(obj.render() for obj in self._objects if obj.is_active)
```

### 6.3 エンタープライズアプリケーションでのOOP

```java
// Java: エンタープライズアプリケーションの例（注文管理）

// エンティティ
public class Order {
    private final String orderId;
    private final String customerId;
    private final List<OrderLine> lines;
    private OrderStatus status;
    private final LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public Order(String orderId, String customerId) {
        this.orderId = orderId;
        this.customerId = customerId;
        this.lines = new ArrayList<>();
        this.status = OrderStatus.DRAFT;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = this.createdAt;
    }

    public void addLine(Product product, int quantity) {
        if (status != OrderStatus.DRAFT) {
            throw new IllegalStateException("確定済みの注文には追加できません");
        }
        if (quantity <= 0) {
            throw new IllegalArgumentException("数量は1以上");
        }
        lines.add(new OrderLine(product, quantity));
        updatedAt = LocalDateTime.now();
    }

    public BigDecimal getTotal() {
        return lines.stream()
            .map(OrderLine::getSubtotal)
            .reduce(BigDecimal.ZERO, BigDecimal::add);
    }

    public void confirm() {
        if (lines.isEmpty()) {
            throw new IllegalStateException("空の注文は確定できません");
        }
        this.status = OrderStatus.CONFIRMED;
        this.updatedAt = LocalDateTime.now();
    }

    public void cancel() {
        if (status == OrderStatus.SHIPPED || status == OrderStatus.DELIVERED) {
            throw new IllegalStateException("出荷済みの注文はキャンセルできません");
        }
        this.status = OrderStatus.CANCELLED;
        this.updatedAt = LocalDateTime.now();
    }

    // getter省略
}

// 値オブジェクト
public class OrderLine {
    private final Product product;
    private final int quantity;

    public OrderLine(Product product, int quantity) {
        this.product = product;
        this.quantity = quantity;
    }

    public BigDecimal getSubtotal() {
        return product.getPrice().multiply(BigDecimal.valueOf(quantity));
    }
}

// 列挙型
public enum OrderStatus {
    DRAFT, CONFIRMED, SHIPPED, DELIVERED, CANCELLED
}

// サービス層
public class OrderService {
    private final OrderRepository orderRepository;
    private final InventoryService inventoryService;
    private final NotificationService notificationService;

    public OrderService(
        OrderRepository orderRepository,
        InventoryService inventoryService,
        NotificationService notificationService
    ) {
        this.orderRepository = orderRepository;
        this.inventoryService = inventoryService;
        this.notificationService = notificationService;
    }

    public void placeOrder(String orderId) {
        Order order = orderRepository.findById(orderId)
            .orElseThrow(() -> new OrderNotFoundException(orderId));

        // ビジネスルール: 在庫チェック
        for (OrderLine line : order.getLines()) {
            inventoryService.reserve(line.getProduct(), line.getQuantity());
        }

        order.confirm();
        orderRepository.save(order);

        // 通知
        notificationService.sendOrderConfirmation(order);
    }
}
```

---

## 7. 各言語のOOPスタイル

```
┌──────────────┬───────────────────────────────────────┐
│ 言語         │ OOPスタイル                            │
├──────────────┼───────────────────────────────────────┤
│ Java         │ クラスベース・純粋OOP                   │
│ C++          │ クラスベース・マルチパラダイム           │
│ Python       │ クラスベース・ダックタイピング           │
│ TypeScript   │ クラス + 構造的型付け                   │
│ Ruby         │ 純粋OOP（全てがオブジェクト）           │
│ Kotlin       │ クラスベース・データクラス               │
│ Swift        │ プロトコル指向 + クラスベース            │
│ Rust         │ トレイトベース（クラスなし）             │
│ Go           │ 構造体 + インターフェース（クラスなし）  │
│ JavaScript   │ プロトタイプベース + クラス構文          │
└──────────────┴───────────────────────────────────────┘
```

### 7.1 各言語でのOOP実装比較

同じ「図形の面積計算」を各言語で実装し、OOPスタイルの違いを見る。

```typescript
// TypeScript: 構造的型付け（ダックタイピング + 型安全）
interface HasArea {
  area(): number;
}

class TSCircle implements HasArea {
  constructor(private radius: number) {}
  area(): number { return Math.PI * this.radius ** 2; }
}

class TSRectangle implements HasArea {
  constructor(private width: number, private height: number) {}
  area(): number { return this.width * this.height; }
}

// 構造的型付け: implements を書かなくても型が一致すればOK
const customShape = {
  area(): number { return 42; }
};

function printArea(shape: HasArea): void {
  console.log(`面積: ${shape.area()}`);
}

printArea(new TSCircle(5));
printArea(new TSRectangle(3, 4));
printArea(customShape);  // OK: 構造が一致
```

```python
# Python: ダックタイピング + プロトコル（型ヒント）
from typing import Protocol
import math


class SupportsArea(Protocol):
    """面積を計算できるもの（プロトコル = 構造的型付け）"""
    def area(self) -> float: ...


class PyCircle:
    def __init__(self, radius: float):
        self._radius = radius

    def area(self) -> float:
        return math.pi * self._radius ** 2


class PyRectangle:
    def __init__(self, width: float, height: float):
        self._width = width
        self._height = height

    def area(self) -> float:
        return self._width * self._height


def print_area(shape: SupportsArea) -> None:
    """ダックタイピング: area() メソッドがあれば何でも受け付ける"""
    print(f"面積: {shape.area():.2f}")


print_area(PyCircle(5))
print_area(PyRectangle(3, 4))
```

```go
// Go: インターフェース + 構造体（クラスなし）
package main

import (
    "fmt"
    "math"
)

// インターフェース（暗黙的実装）
type Shape interface {
    Area() float64
}

// 構造体 + メソッド（クラスの代わり）
type GoCircle struct {
    Radius float64
}

func (c GoCircle) Area() float64 {
    return math.Pi * c.Radius * c.Radius
}

type GoRectangle struct {
    Width  float64
    Height float64
}

func (r GoRectangle) Area() float64 {
    return r.Width * r.Height
}

// Shape インターフェースを満たすものなら何でも受け取れる
func PrintArea(s Shape) {
    fmt.Printf("面積: %.2f\n", s.Area())
}

func main() {
    PrintArea(GoCircle{Radius: 5})
    PrintArea(GoRectangle{Width: 3, Height: 4})
}
```

```rust
// Rust: トレイト + 構造体（クラスなし、継承なし）
use std::f64::consts::PI;

trait Shape {
    fn area(&self) -> f64;

    // デフォルト実装も可能
    fn describe(&self) -> String {
        format!("面積: {:.2}", self.area())
    }
}

struct RustCircle {
    radius: f64,
}

impl Shape for RustCircle {
    fn area(&self) -> f64 {
        PI * self.radius * self.radius
    }
}

struct RustRectangle {
    width: f64,
    height: f64,
}

impl Shape for RustRectangle {
    fn area(&self) -> f64 {
        self.width * self.height
    }
}

// トレイト境界でジェネリックに
fn print_area(shape: &dyn Shape) {
    println!("{}", shape.describe());
}

fn main() {
    let circle = RustCircle { radius: 5.0 };
    let rect = RustRectangle { width: 3.0, height: 4.0 };
    print_area(&circle);
    print_area(&rect);
}
```

### 7.2 プロトタイプベースOOP（JavaScript）

JavaScriptはクラスベースではなく、プロトタイプベースのOOPを採用している。ES2015以降のclass構文はプロトタイプの糖衣構文である。

```javascript
// JavaScript: プロトタイプベースOOP

// === プロトタイプチェーン（内部メカニズム）===
const animal = {
  speak() {
    return `${this.name}が鳴く`;
  },
};

const dog = Object.create(animal); // animal をプロトタイプとする
dog.name = "ポチ";
dog.bark = function () {
  return `${this.name}: ワンワン！`;
};

console.log(dog.bark());   // ポチ: ワンワン！
console.log(dog.speak());  // ポチが鳴く（プロトタイプから継承）

// === ES2015 class 構文（糖衣構文）===
class Animal {
  constructor(name) {
    this.name = name;
  }

  speak() {
    return `${this.name}が鳴く`;
  }
}

class Dog extends Animal {
  bark() {
    return `${this.name}: ワンワン！`;
  }
}

const pochi = new Dog("ポチ");
console.log(pochi.bark());   // ポチ: ワンワン！
console.log(pochi.speak());  // ポチが鳴く

// プロトタイプチェーンの確認
console.log(pochi instanceof Dog);     // true
console.log(pochi instanceof Animal);  // true
console.log(Object.getPrototypeOf(pochi) === Dog.prototype); // true
```

---

## 8. OOPの批判と限界

OOPは万能ではない。その限界を理解しておくことで、より適切な設計判断ができる。

```
OOPに対する主要な批判:

  1. 「バナナを要求したら、バナナを持ったゴリラとジャングルがついてきた」
     — Joe Armstrong（Erlang 作者）
     → 継承による過度な依存関係
     → 1つのクラスを使うために、その親クラス、さらにその親クラス...
     → 解決策: コンポジション重視、薄い継承階層

  2. 過剰な抽象化（Over-engineering）
     → AbstractSingletonProxyFactoryBean 問題
     → 単純な処理にデザインパターンを詰め込む
     → 解決策: YAGNI（You Ain't Gonna Need It）

  3. 状態変更による複雑さ
     → 共有可変状態（Shared Mutable State）は並行処理の天敵
     → オブジェクトの状態が予測不能に変化
     → 解決策: 不変オブジェクト、関数型の要素を取り入れる

  4. 学習コストと生産性
     → 小規模プロジェクトではオーバーヘッドが大きい
     → OOP的に「正しく」書くための設計知識が必要
     → 解決策: 適切な場面で適切に使う

  5. テスト困難な密結合
     → 依存オブジェクトが多いとテストが困難
     → モック地獄（Mock Hell）
     → 解決策: 依存性注入、インターフェースによる疎結合
```

### 8.1 OOPの限界を超えるアプローチ

```typescript
// TypeScript: OOPの限界に対するモダンなアプローチ

// === コンポジション over 継承 ===

// 悪い例: 深い継承階層
// Animal → Mammal → Pet → Dog → GoldenRetriever

// 良い例: コンポジション（機能を組み合わせる）
interface Walkable {
  walk(): void;
}

interface Swimmable {
  swim(): void;
}

interface Trainable {
  train(command: string): void;
}

class DogBehavior implements Walkable, Swimmable, Trainable {
  constructor(private name: string) {}

  walk(): void {
    console.log(`${this.name}が散歩する`);
  }

  swim(): void {
    console.log(`${this.name}が泳ぐ`);
  }

  train(command: string): void {
    console.log(`${this.name}が「${command}」を覚えた`);
  }
}


// === 不変オブジェクト + ビルダーパターン ===

class ImmutableConfig {
  readonly host: string;
  readonly port: number;
  readonly database: string;
  readonly maxConnections: number;
  readonly timeout: number;

  private constructor(builder: ConfigBuilder) {
    this.host = builder.host;
    this.port = builder.port;
    this.database = builder.database;
    this.maxConnections = builder.maxConnections;
    this.timeout = builder.timeout;
  }

  static builder(): ConfigBuilder {
    return new ConfigBuilder();
  }

  // 既存の設定を元に一部変更した新しい設定を作成
  withHost(host: string): ImmutableConfig {
    return ImmutableConfig.builder()
      .setHost(host)
      .setPort(this.port)
      .setDatabase(this.database)
      .setMaxConnections(this.maxConnections)
      .setTimeout(this.timeout)
      .build();
  }
}

class ConfigBuilder {
  host = "localhost";
  port = 5432;
  database = "mydb";
  maxConnections = 10;
  timeout = 5000;

  setHost(host: string): this { this.host = host; return this; }
  setPort(port: number): this { this.port = port; return this; }
  setDatabase(db: string): this { this.database = db; return this; }
  setMaxConnections(n: number): this { this.maxConnections = n; return this; }
  setTimeout(ms: number): this { this.timeout = ms; return this; }

  build(): ImmutableConfig {
    // @ts-ignore: private constructor access for builder
    return new ImmutableConfig(this);
  }
}
```

---

## 9. OOPを学ぶためのロードマップ

```
OOP学習のロードマップ:

  Level 1: 基礎概念
    □ クラスとオブジェクトの関係
    □ コンストラクタとインスタンス化
    □ フィールドとメソッド
    □ アクセス修飾子（public, private, protected）

  Level 2: 4つの柱
    □ カプセル化（情報隠蔽、バンドリング）
    □ 継承（is-a 関係、メソッドオーバーライド）
    □ ポリモーフィズム（インターフェース、抽象クラス）
    □ 抽象化（複雑さの隠蔽、契約の定義）

  Level 3: 設計原則
    □ SOLID 原則
    □ Tell, Don't Ask
    □ Law of Demeter
    □ Composition over Inheritance

  Level 4: デザインパターン
    □ 生成パターン（Factory, Builder, Singleton）
    □ 構造パターン（Adapter, Decorator, Proxy）
    □ 振る舞いパターン（Strategy, Observer, Command）

  Level 5: アーキテクチャ
    □ レイヤードアーキテクチャ
    □ ドメイン駆動設計（DDD）
    □ クリーンアーキテクチャ
    □ 依存性注入（DI）

  Level 6: マルチパラダイム
    □ OOP + FP の融合
    □ リアクティブプログラミング
    □ アクターモデル
    □ イベントソーシング
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| OOPの本質 | データと振る舞いをオブジェクトに統合 |
| 3要素 | 状態 + 振る舞い + アイデンティティ |
| 4つの柱 | カプセル化、継承、ポリモーフィズム、抽象化 |
| メッセージ | オブジェクト間のメソッド呼び出し |
| 得意分野 | GUI、ゲーム、エンタープライズ、フレームワーク |
| 限界 | 過剰な抽象化、共有可変状態、継承の複雑さ |
| 現実 | OOP + FP のマルチパラダイムが主流 |
| 学習 | 基礎 → 4本柱 → SOLID → パターン → アーキテクチャ |

---

## 次に読むべきガイド
→ [[01-history-and-evolution.md]] — OOPの歴史と進化

---

## 参考文献
1. Kay, A. "The Early History of Smalltalk." ACM SIGPLAN Notices, 1993.
2. Gamma, E. et al. "Design Patterns." Addison-Wesley, 1994.
3. Martin, R. "Clean Code." Prentice Hall, 2008.
4. Bloch, J. "Effective Java." 3rd Ed, Addison-Wesley, 2018.
5. Armstrong, J. "Coders at Work." Apress, 2009.
6. Meyer, B. "Object-Oriented Software Construction." 2nd Ed, Prentice Hall, 1997.
