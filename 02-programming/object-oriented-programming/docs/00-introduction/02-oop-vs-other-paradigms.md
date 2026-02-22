# OOP vs 他パラダイム

> OOPは万能ではない。手続き型、関数型、リアクティブ、アクターモデルなど、各パラダイムの強みと弱みを理解し、適切に使い分けることが現代のエンジニアに求められる。

## この章で学ぶこと

- [ ] 主要パラダイムの特徴と適用領域を把握する
- [ ] OOPと関数型の根本的な設計思想の違いを理解する
- [ ] マルチパラダイムの実践的な使い分けを学ぶ
- [ ] 各パラダイムの実装を複数言語で比較する
- [ ] 現実のプロジェクトでのパラダイム選択基準を理解する

---

## 1. パラダイム比較

```
┌──────────────┬──────────────┬──────────────┬──────────────┐
│              │ 手続き型     │ OOP          │ 関数型       │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ 中心概念      │ 手順・命令   │ オブジェクト │ 関数・変換   │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ データ管理    │ グローバル変数│ カプセル化   │ イミュータブル│
├──────────────┼──────────────┼──────────────┼──────────────┤
│ コード再利用  │ 関数         │ 継承/合成    │ 高階関数     │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ 状態管理      │ 変数の変更   │ メソッドで変更│ 新しい値を返す│
├──────────────┼──────────────┼──────────────┼──────────────┤
│ 副作用        │ どこでも発生 │ メソッド内   │ 最小限に制限 │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ テスト容易性  │ △           │ ○           │ ◎           │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ 並行処理      │ 困難         │ 困難(共有状態)│ 容易(不変)   │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ 学習コスト    │ 低           │ 中           │ 高           │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

### 1.1 各パラダイムの詳細な特徴

```
手続き型プログラミング（Procedural Programming）:
  核心: 手順の記述
  代表言語: C, Pascal, BASIC, Shell Script
  特徴:
    - 上から下へ、順番に命令を実行
    - 関数でコードを構造化
    - データと処理が分離
    - グローバル変数による状態共有
  利点:
    - 学習が容易
    - 実行の流れが直感的
    - オーバーヘッドが少ない
  欠点:
    - 大規模コードで破綻しやすい
    - データの整合性が保証しにくい
    - コードの再利用が限定的

OOP（Object-Oriented Programming）:
  核心: オブジェクト間の協調
  代表言語: Java, C#, Python, TypeScript, Kotlin
  特徴:
    - データと振る舞いの統合
    - カプセル化による情報隠蔽
    - 継承による再利用
    - ポリモーフィズムによる柔軟性
  利点:
    - 大規模開発に強い
    - チーム開発でのモジュール分割が明確
    - 現実世界のモデリングが自然
  欠点:
    - 過剰な抽象化のリスク
    - 共有可変状態による並行処理の困難
    - ボイラープレートが多くなりがち

関数型プログラミング（Functional Programming）:
  核心: データ変換のパイプライン
  代表言語: Haskell, Erlang, Clojure, Elm, F#
  特徴:
    - 不変データ（immutable data）
    - 純粋関数（同じ入力 → 常に同じ出力）
    - 高階関数（関数を引数や戻り値にできる）
    - 参照透過性（副作用なし）
  利点:
    - テストが容易（純粋関数）
    - 並行処理が安全（共有状態なし）
    - 推論しやすいコード
  欠点:
    - 学習コストが高い
    - 状態管理が本質的な領域では不自然
    - パフォーマンスチューニングが難しい場合がある

リアクティブプログラミング（Reactive Programming）:
  核心: データの流れと変化の伝播
  代表: RxJS, Reactor, Akka Streams
  特徴:
    - 非同期データストリーム
    - イベント駆動
    - バックプレッシャー
  利点:
    - 非同期処理の宣言的な記述
    - UIイベント処理に適する
  欠点:
    - デバッグが困難
    - 学習コストが高い

アクターモデル（Actor Model）:
  核心: 独立した計算単位（アクター）間のメッセージパッシング
  代表: Erlang/OTP, Akka, Orleans
  特徴:
    - 各アクターが独自の状態を持つ
    - メッセージの非同期送受信
    - 故障の隔離（Let it crash）
  利点:
    - 分散システムに適する
    - スケーラビリティが高い
    - 耐障害性に優れる
  欠点:
    - メッセージ順序の保証が限定的
    - デバッグが困難
```

---

## 2. 同じ問題を各パラダイムで解く

### 問題: ユーザーリストから成人のメールアドレスを取得

```python
# === 手続き型 ===
users = [
    {"name": "田中", "age": 25, "email": "tanaka@example.com"},
    {"name": "山田", "age": 17, "email": "yamada@example.com"},
    {"name": "佐藤", "age": 30, "email": "sato@example.com"},
]

adult_emails = []
for user in users:
    if user["age"] >= 18:
        adult_emails.append(user["email"])
# → 手順を逐次的に記述
```

```python
# === OOP ===
class User:
    def __init__(self, name: str, age: int, email: str):
        self.name = name
        self.age = age
        self.email = email

    def is_adult(self) -> bool:
        return self.age >= 18

class UserRepository:
    def __init__(self, users: list[User]):
        self._users = users

    def get_adult_emails(self) -> list[str]:
        return [u.email for u in self._users if u.is_adult()]

# → 責任をオブジェクトに委譲
repo = UserRepository([
    User("田中", 25, "tanaka@example.com"),
    User("山田", 17, "yamada@example.com"),
    User("佐藤", 30, "sato@example.com"),
])
adult_emails = repo.get_adult_emails()
```

```python
# === 関数型 ===
from dataclasses import dataclass
from typing import Callable

@dataclass(frozen=True)  # イミュータブル
class User:
    name: str
    age: int
    email: str

def is_adult(user: User) -> bool:
    return user.age >= 18

def get_email(user: User) -> str:
    return user.email

users = [
    User("田中", 25, "tanaka@example.com"),
    User("山田", 17, "yamada@example.com"),
    User("佐藤", 30, "sato@example.com"),
]

adult_emails = list(map(get_email, filter(is_adult, users)))
# → データ変換のパイプライン
```

### 問題2: ECサイトの注文処理

より複雑な問題で各パラダイムの特性を比較する。

```typescript
// === 手続き型アプローチ ===

// データは分離された配列/辞書で管理
interface OrderData {
  orderId: string;
  items: { productId: string; quantity: number; price: number }[];
  status: string;
  discount: number;
}

// 関数群
function calculateSubtotal(order: OrderData): number {
  let total = 0;
  for (const item of order.items) {
    total += item.price * item.quantity;
  }
  return total;
}

function applyDiscount(order: OrderData): number {
  const subtotal = calculateSubtotal(order);
  return subtotal * (1 - order.discount);
}

function calculateTax(amount: number, rate: number = 0.1): number {
  return Math.floor(amount * rate);
}

function processOrder(order: OrderData): {
  subtotal: number;
  discount: number;
  tax: number;
  total: number;
} {
  const subtotal = calculateSubtotal(order);
  const afterDiscount = applyDiscount(order);
  const tax = calculateTax(afterDiscount);
  return {
    subtotal,
    discount: subtotal - afterDiscount,
    tax,
    total: afterDiscount + tax,
  };
}
// 問題: 関数間でデータが分散、状態管理が困難
```

```typescript
// === OOPアプローチ ===

class Money {
  constructor(private readonly _amount: number) {
    if (_amount < 0) throw new Error("金額は0以上");
  }

  get amount(): number { return this._amount; }

  add(other: Money): Money {
    return new Money(this._amount + other._amount);
  }

  subtract(other: Money): Money {
    return new Money(this._amount - other._amount);
  }

  multiply(factor: number): Money {
    return new Money(Math.floor(this._amount * factor));
  }

  toString(): string {
    return `¥${this._amount.toLocaleString()}`;
  }
}

class OrderItem {
  constructor(
    public readonly productId: string,
    public readonly productName: string,
    public readonly unitPrice: Money,
    public readonly quantity: number,
  ) {
    if (quantity <= 0) throw new Error("数量は1以上");
  }

  get subtotal(): Money {
    return this.unitPrice.multiply(this.quantity);
  }
}

type OrderStatus = "draft" | "confirmed" | "paid" | "shipped" | "delivered" | "cancelled";

class Order {
  private _items: OrderItem[] = [];
  private _status: OrderStatus = "draft";
  private _discountRate: number = 0;

  constructor(public readonly orderId: string) {}

  addItem(item: OrderItem): void {
    if (this._status !== "draft") {
      throw new Error("確定済みの注文には商品を追加できません");
    }
    this._items.push(item);
  }

  applyDiscount(rate: number): void {
    if (rate < 0 || rate > 1) throw new Error("割引率は0〜1の範囲");
    this._discountRate = rate;
  }

  get subtotal(): Money {
    return this._items.reduce(
      (sum, item) => sum.add(item.subtotal),
      new Money(0),
    );
  }

  get discountAmount(): Money {
    return this.subtotal.multiply(this._discountRate);
  }

  get afterDiscount(): Money {
    return this.subtotal.subtract(this.discountAmount);
  }

  get tax(): Money {
    return this.afterDiscount.multiply(0.1);
  }

  get total(): Money {
    return this.afterDiscount.add(this.tax);
  }

  confirm(): void {
    if (this._items.length === 0) throw new Error("空の注文は確定不可");
    this._status = "confirmed";
  }

  get status(): OrderStatus {
    return this._status;
  }

  get itemCount(): number {
    return this._items.length;
  }

  display(): string {
    const lines = [`注文 ${this.orderId} (${this._status})`];
    for (const item of this._items) {
      lines.push(`  ${item.productName} x${item.quantity} = ${item.subtotal}`);
    }
    lines.push(`  小計: ${this.subtotal}`);
    if (this._discountRate > 0) {
      lines.push(`  割引: -${this.discountAmount} (${this._discountRate * 100}%)`);
    }
    lines.push(`  税: ${this.tax}`);
    lines.push(`  合計: ${this.total}`);
    return lines.join("\n");
  }
}
// 利点: ビジネスルールがオブジェクトに内包される
```

```typescript
// === 関数型アプローチ ===

// すべて不変データ型
type FPOrderItem = Readonly<{
  productId: string;
  productName: string;
  unitPrice: number;
  quantity: number;
}>;

type FPOrder = Readonly<{
  orderId: string;
  items: ReadonlyArray<FPOrderItem>;
  status: OrderStatus;
  discountRate: number;
}>;

type OrderSummary = Readonly<{
  subtotal: number;
  discount: number;
  tax: number;
  total: number;
}>;

// 純粋関数群（副作用なし、同じ入力 → 同じ出力）
const itemSubtotal = (item: FPOrderItem): number =>
  item.unitPrice * item.quantity;

const orderSubtotal = (order: FPOrder): number =>
  order.items.reduce((sum, item) => sum + itemSubtotal(item), 0);

const calculateDiscount = (subtotal: number, rate: number): number =>
  Math.floor(subtotal * rate);

const calculateTax = (amount: number, rate: number = 0.1): number =>
  Math.floor(amount * rate);

const summarize = (order: FPOrder): OrderSummary => {
  const subtotal = orderSubtotal(order);
  const discount = calculateDiscount(subtotal, order.discountRate);
  const afterDiscount = subtotal - discount;
  const tax = calculateTax(afterDiscount);
  return {
    subtotal,
    discount,
    tax,
    total: afterDiscount + tax,
  };
};

// 注文の変更は新しいオブジェクトを返す
const addItem = (order: FPOrder, item: FPOrderItem): FPOrder => ({
  ...order,
  items: [...order.items, item],
});

const applyDiscountFP = (order: FPOrder, rate: number): FPOrder => ({
  ...order,
  discountRate: rate,
});

const confirmOrder = (order: FPOrder): FPOrder => ({
  ...order,
  status: "confirmed" as OrderStatus,
});

// パイプライン: 関数の合成
const pipe = <T>(...fns: ((arg: T) => T)[]) =>
  (initial: T): T => fns.reduce((acc, fn) => fn(acc), initial);

// 使用例
const baseOrder: FPOrder = {
  orderId: "ORD001",
  items: [],
  status: "draft",
  discountRate: 0,
};

const processedOrder = pipe<FPOrder>(
  (o) => addItem(o, { productId: "P1", productName: "本", unitPrice: 1500, quantity: 2 }),
  (o) => addItem(o, { productId: "P2", productName: "ペン", unitPrice: 200, quantity: 5 }),
  (o) => applyDiscountFP(o, 0.1),
  confirmOrder,
)(baseOrder);

const summary = summarize(processedOrder);
// baseOrder は変わらない（不変）
// processedOrder は新しいオブジェクト
```

---

## 3. OOP vs 関数型: 根本的な違い

```
Expression Problem（表現問題）:

  OOP: 新しい「型」の追加が容易、新しい「操作」の追加が困難
  FP:  新しい「操作」の追加が容易、新しい「型」の追加が困難

  例: 図形の描画

  OOP:
    Shape（抽象クラス）
    ├── Circle.draw()      ← 新しい型（Triangle）を追加するのは簡単
    ├── Rectangle.draw()      各クラスにdraw()を実装するだけ
    └── Triangle.draw()    ← でも新しい操作（area()）を追加すると
                              全クラスを修正する必要がある

  FP:
    draw(shape) = match shape with    ← 新しい操作（area()）を追加するのは簡単
    | Circle r -> ...                    新しい関数を定義するだけ
    | Rectangle w h -> ...            ← でも新しい型（Triangle）を追加すると
                                         全関数を修正する必要がある

  → どちらが良いかは「何が頻繁に変わるか」による
  → 型が増える → OOP
  → 操作が増える → FP
```

### 3.1 Expression Problemの具体例

```typescript
// TypeScript: OOPでの Expression Problem

// === OOPアプローチ: 新しい型の追加が容易 ===

interface Shape {
  area(): number;
  perimeter(): number;
  draw(): string;
}

class Circle implements Shape {
  constructor(private radius: number) {}
  area(): number { return Math.PI * this.radius ** 2; }
  perimeter(): number { return 2 * Math.PI * this.radius; }
  draw(): string { return `○ (r=${this.radius})`; }
}

class Rectangle implements Shape {
  constructor(private width: number, private height: number) {}
  area(): number { return this.width * this.height; }
  perimeter(): number { return 2 * (this.width + this.height); }
  draw(): string { return `□ (${this.width}x${this.height})`; }
}

// 新しい型（Triangle）の追加: 簡単！既存コードの変更なし
class Triangle implements Shape {
  constructor(
    private base: number,
    private height: number,
    private sideA: number,
    private sideB: number,
  ) {}
  area(): number { return this.base * this.height / 2; }
  perimeter(): number { return this.base + this.sideA + this.sideB; }
  draw(): string { return `△ (base=${this.base})`; }
}

// しかし、新しい操作（例: serialize）を追加するには
// 全クラスを修正する必要がある → 困難
// interface Shape に serialize(): string を追加
// → Circle, Rectangle, Triangle 全てに実装を追加


// === 関数型アプローチ: 新しい操作の追加が容易 ===

type FPShape =
  | { type: "circle"; radius: number }
  | { type: "rectangle"; width: number; height: number }
  | { type: "triangle"; base: number; height: number; sideA: number; sideB: number };

// 操作1
function fpArea(shape: FPShape): number {
  switch (shape.type) {
    case "circle": return Math.PI * shape.radius ** 2;
    case "rectangle": return shape.width * shape.height;
    case "triangle": return shape.base * shape.height / 2;
  }
}

// 操作2
function fpPerimeter(shape: FPShape): number {
  switch (shape.type) {
    case "circle": return 2 * Math.PI * shape.radius;
    case "rectangle": return 2 * (shape.width + shape.height);
    case "triangle": return shape.base + shape.sideA + shape.sideB;
  }
}

// 新しい操作（serialize）の追加: 簡単！既存コードの変更なし
function fpSerialize(shape: FPShape): string {
  return JSON.stringify(shape);
}

// 新しい操作（SVGに変換）の追加: 簡単！
function fpToSvg(shape: FPShape): string {
  switch (shape.type) {
    case "circle":
      return `<circle r="${shape.radius}" />`;
    case "rectangle":
      return `<rect width="${shape.width}" height="${shape.height}" />`;
    case "triangle":
      return `<polygon points="0,${shape.height} ${shape.base},${shape.height} ${shape.base/2},0" />`;
  }
}

// しかし、新しい型（pentagon）を追加するには
// 全関数を修正する必要がある → 困難
```

### 3.2 状態管理の違い

```
OOP: 可変状態をカプセル化
  account.deposit(1000)   ← オブジェクトの内部状態が変わる
  account.withdraw(500)   ← 同じオブジェクトが変化し続ける

FP: 不変データを変換
  newAccount = deposit(account, 1000)   ← 新しいオブジェクトを返す
  finalAccount = withdraw(newAccount, 500) ← 元のaccountは変わらない

  FPの利点:
    - 並行処理が安全（共有状態がない）
    - タイムトラベルデバッグが可能
    - テストが容易（同じ入力 → 同じ出力）

  OOPの利点:
    - 直感的（現実世界のモデルに近い）
    - GUIやゲームなど、状態変化が本質的な領域に適合
    - メモリ効率が良い（オブジェクトを更新するだけ）
```

```typescript
// TypeScript: 状態管理の違い - 銀行口座の例

// === OOPスタイル: 可変オブジェクト ===
class MutableBankAccount {
  private _balance: number;
  private _transactions: { amount: number; type: string; date: Date }[] = [];

  constructor(
    public readonly accountNumber: string,
    initialBalance: number = 0,
  ) {
    this._balance = initialBalance;
  }

  deposit(amount: number): void {
    if (amount <= 0) throw new Error("入金額は正の数");
    this._balance += amount;
    this._transactions.push({
      amount,
      type: "deposit",
      date: new Date(),
    });
  }

  withdraw(amount: number): void {
    if (amount > this._balance) throw new Error("残高不足");
    this._balance -= amount;
    this._transactions.push({
      amount: -amount,
      type: "withdrawal",
      date: new Date(),
    });
  }

  get balance(): number {
    return this._balance;
  }

  get transactionCount(): number {
    return this._transactions.length;
  }
}

// 使用: オブジェクトの状態が変わる
const account = new MutableBankAccount("001", 10000);
account.deposit(5000);   // balance: 15000（内部が変わる）
account.withdraw(3000);  // balance: 12000（内部が変わる）


// === 関数型スタイル: 不変データ ===
type ImmutableAccount = Readonly<{
  accountNumber: string;
  balance: number;
  transactions: ReadonlyArray<{
    amount: number;
    type: string;
    timestamp: number;
  }>;
}>;

const createAccount = (accountNumber: string, balance: number = 0): ImmutableAccount => ({
  accountNumber,
  balance,
  transactions: [],
});

const deposit = (account: ImmutableAccount, amount: number): ImmutableAccount => {
  if (amount <= 0) throw new Error("入金額は正の数");
  return {
    ...account,
    balance: account.balance + amount,
    transactions: [
      ...account.transactions,
      { amount, type: "deposit", timestamp: Date.now() },
    ],
  };
};

const withdraw = (account: ImmutableAccount, amount: number): ImmutableAccount => {
  if (amount > account.balance) throw new Error("残高不足");
  return {
    ...account,
    balance: account.balance - amount,
    transactions: [
      ...account.transactions,
      { amount: -amount, type: "withdrawal", timestamp: Date.now() },
    ],
  };
};

// 使用: 各操作が新しい状態を返す
const acc0 = createAccount("001", 10000);
const acc1 = deposit(acc0, 5000);   // 新しいオブジェクト（acc0は不変）
const acc2 = withdraw(acc1, 3000);  // 新しいオブジェクト（acc1は不変）

// タイムトラベル: 全ての状態が保存されている
console.log(acc0.balance); // 10000（元の状態）
console.log(acc1.balance); // 15000（入金後）
console.log(acc2.balance); // 12000（出金後）
```

### 3.3 副作用の管理

```python
# Python: 副作用の管理の違い

# === 手続き型: 副作用がどこでも発生 ===

total_orders = 0  # グローバル状態
log_file = None   # グローバルリソース

def process_order_procedural(order_data: dict) -> dict:
    global total_orders
    total_orders += 1  # 副作用: グローバル状態の変更

    # 副作用: ファイルI/O
    with open("orders.log", "a") as f:
        f.write(f"Order: {order_data}\n")

    # 副作用: 外部API呼び出し
    # send_email(order_data["email"], "注文確認")

    result = {
        "order_id": f"ORD-{total_orders}",
        "total": sum(item["price"] * item["qty"] for item in order_data["items"]),
    }
    return result

# 問題: 副作用が散在、テストが困難


# === 関数型: 副作用を分離 ===

from dataclasses import dataclass
from typing import Callable, TypeVar

T = TypeVar("T")
E = TypeVar("E")

@dataclass(frozen=True)
class Result:
    """成功/失敗を表現する型"""
    success: bool
    value: object = None
    error: str = ""

    @staticmethod
    def ok(value: object) -> "Result":
        return Result(success=True, value=value)

    @staticmethod
    def fail(error: str) -> "Result":
        return Result(success=False, error=error)


@dataclass(frozen=True)
class OrderInput:
    customer_email: str
    items: tuple  # 不変のタプル


@dataclass(frozen=True)
class OrderResult:
    order_id: str
    subtotal: int
    tax: int
    total: int


# 純粋関数: 副作用なし、テスト容易
def calculate_order(order_input: OrderInput, order_number: int) -> OrderResult:
    subtotal = sum(item["price"] * item["qty"] for item in order_input.items)
    tax = int(subtotal * 0.1)
    return OrderResult(
        order_id=f"ORD-{order_number:06d}",
        subtotal=subtotal,
        tax=tax,
        total=subtotal + tax,
    )


def validate_order(order_input: OrderInput) -> Result:
    if not order_input.items:
        return Result.fail("商品が空です")
    if "@" not in order_input.customer_email:
        return Result.fail("メールアドレスが無効です")
    return Result.ok(order_input)


# 副作用を持つ関数は「境界」に配置
class OrderProcessor:
    """副作用の境界: I/O、DB、外部APIをここに集約"""

    def __init__(self, db, mailer, logger):
        self._db = db
        self._mailer = mailer
        self._logger = logger

    def process(self, order_input: OrderInput) -> Result:
        # 1. バリデーション（純粋関数）
        validation = validate_order(order_input)
        if not validation.success:
            return validation

        # 2. 注文番号取得（副作用: DB）
        order_number = self._db.next_order_number()

        # 3. 計算（純粋関数）
        result = calculate_order(order_input, order_number)

        # 4. 保存（副作用: DB）
        self._db.save_order(result)

        # 5. 通知（副作用: メール）
        self._mailer.send(order_input.customer_email, f"注文確認: {result.order_id}")

        # 6. ログ（副作用: ファイル）
        self._logger.info(f"注文処理完了: {result.order_id}")

        return Result.ok(result)
```

---

## 4. マルチパラダイムの実践

```
現代のベストプラクティス:

  「ドメインモデル」→ OOP
    ビジネスエンティティの表現にクラスを使う
    例: User, Order, Product

  「データ変換」→ FP
    データのフィルタリング・変換に関数を使う
    例: map, filter, reduce

  「副作用の管理」→ FP
    I/O、DB操作を関数の境界に押しやる

  「状態を持つUI」→ OOP + リアクティブ
    コンポーネントの状態管理

実践例（TypeScript）:
```

```typescript
// ドメインモデル: OOP
class Order {
  constructor(
    public readonly id: string,
    public readonly items: OrderItem[],
    public readonly status: OrderStatus,
    public readonly createdAt: Date,
  ) {}

  get totalPrice(): number {
    // FP: データ変換
    return this.items
      .map(item => item.price * item.quantity)
      .reduce((sum, price) => sum + price, 0);
  }

  canCancel(): boolean {
    return this.status === "pending" || this.status === "confirmed";
  }
}

// データ変換: FP
const getRecentHighValueOrders = (orders: Order[]): Order[] =>
  orders
    .filter(order => order.totalPrice > 10000)
    .filter(order => order.createdAt > thirtyDaysAgo())
    .sort((a, b) => b.totalPrice - a.totalPrice);

// 副作用の管理: 関数の境界に分離
async function processOrders(repo: OrderRepository): Promise<void> {
  const orders = await repo.findAll();          // 副作用（DB）
  const highValue = getRecentHighValueOrders(orders); // 純粋関数
  await notifyAdmins(highValue);                // 副作用（通知）
}
```

### 4.1 レイヤーごとのパラダイム選択

現実のアプリケーションでは、レイヤーごとに最適なパラダイムが異なる。

```typescript
// TypeScript: レイヤーごとのパラダイム選択

// === ドメイン層: OOP（エンティティ + 値オブジェクト） ===

class Email {
  private readonly _value: string;

  constructor(value: string) {
    if (!/^[\w.-]+@[\w.-]+\.\w+$/.test(value)) {
      throw new Error(`無効なメールアドレス: ${value}`);
    }
    this._value = value.toLowerCase();
  }

  get value(): string { return this._value; }
  get domain(): string { return this._value.split("@")[1]; }

  equals(other: Email): boolean {
    return this._value === other._value;
  }

  toString(): string { return this._value; }
}

class Customer {
  private _name: string;
  private _email: Email;
  private _memberSince: Date;
  private _totalPurchases: number = 0;

  constructor(
    public readonly customerId: string,
    name: string,
    email: Email,
  ) {
    this._name = name;
    this._email = email;
    this._memberSince = new Date();
  }

  get name(): string { return this._name; }
  get email(): Email { return this._email; }
  get memberSince(): Date { return this._memberSince; }
  get totalPurchases(): number { return this._totalPurchases; }

  get membershipTier(): "bronze" | "silver" | "gold" | "platinum" {
    if (this._totalPurchases >= 1000000) return "platinum";
    if (this._totalPurchases >= 500000) return "gold";
    if (this._totalPurchases >= 100000) return "silver";
    return "bronze";
  }

  get discountRate(): number {
    switch (this.membershipTier) {
      case "platinum": return 0.15;
      case "gold": return 0.10;
      case "silver": return 0.05;
      case "bronze": return 0;
    }
  }

  recordPurchase(amount: number): void {
    this._totalPurchases += amount;
  }

  updateEmail(newEmail: Email): void {
    this._email = newEmail;
  }
}


// === アプリケーション層: FP（ユースケースのオーケストレーション） ===

// 純粋関数: ビジネスルールの計算
const calculateOrderTotal = (
  items: { price: number; quantity: number }[],
  discountRate: number,
): { subtotal: number; discount: number; tax: number; total: number } => {
  const subtotal = items.reduce(
    (sum, item) => sum + item.price * item.quantity,
    0,
  );
  const discount = Math.floor(subtotal * discountRate);
  const afterDiscount = subtotal - discount;
  const tax = Math.floor(afterDiscount * 0.1);
  return {
    subtotal,
    discount,
    tax,
    total: afterDiscount + tax,
  };
};

// データ変換パイプライン
const getEligibleCustomers = (
  customers: Customer[],
  minTier: string,
): Customer[] => {
  const tierOrder = ["bronze", "silver", "gold", "platinum"];
  const minIndex = tierOrder.indexOf(minTier);
  return customers
    .filter(c => tierOrder.indexOf(c.membershipTier) >= minIndex)
    .sort((a, b) => b.totalPurchases - a.totalPurchases);
};

// キャンペーンメールの対象者を抽出
const getCampaignTargets = (
  customers: Customer[],
  campaign: { minTier: string; minPurchases: number },
): { email: string; name: string; tier: string }[] =>
  customers
    .filter(c => c.totalPurchases >= campaign.minPurchases)
    .filter(c => {
      const tiers = ["bronze", "silver", "gold", "platinum"];
      return tiers.indexOf(c.membershipTier) >= tiers.indexOf(campaign.minTier);
    })
    .map(c => ({
      email: c.email.value,
      name: c.name,
      tier: c.membershipTier,
    }));


// === インフラ層: 手続き型（副作用を明示的に管理） ===

interface CustomerRepository {
  findById(id: string): Promise<Customer | null>;
  findAll(): Promise<Customer[]>;
  save(customer: Customer): Promise<void>;
}

interface EmailService {
  send(to: string, subject: string, body: string): Promise<boolean>;
}

class CampaignService {
  constructor(
    private readonly customerRepo: CustomerRepository,
    private readonly emailService: EmailService,
  ) {}

  async runCampaign(campaign: {
    minTier: string;
    minPurchases: number;
    subject: string;
    bodyTemplate: string;
  }): Promise<{ sent: number; failed: number }> {
    // 副作用: DB からデータ取得
    const customers = await this.customerRepo.findAll();

    // 純粋関数: 対象者の抽出
    const targets = getCampaignTargets(customers, campaign);

    // 副作用: メール送信
    let sent = 0;
    let failed = 0;
    for (const target of targets) {
      const body = campaign.bodyTemplate
        .replace("{{name}}", target.name)
        .replace("{{tier}}", target.tier);

      const success = await this.emailService.send(
        target.email,
        campaign.subject,
        body,
      );

      if (success) sent++;
      else failed++;
    }

    return { sent, failed };
  }
}
```

### 4.2 リアクティブプログラミングとの融合

```typescript
// TypeScript: リアクティブプログラミングの概念

// Observable パターン（簡易実装）
type Observer<T> = (value: T) => void;

class Observable<T> {
  private observers: Observer<T>[] = [];

  subscribe(observer: Observer<T>): () => void {
    this.observers.push(observer);
    // unsubscribe 関数を返す
    return () => {
      this.observers = this.observers.filter(o => o !== observer);
    };
  }

  protected emit(value: T): void {
    for (const observer of this.observers) {
      observer(value);
    }
  }

  // FP的な変換メソッド
  map<U>(transform: (value: T) => U): Observable<U> {
    const result = new Observable<U>();
    this.subscribe(value => {
      result.emit(transform(value));
    });
    return result;
  }

  filter(predicate: (value: T) => boolean): Observable<T> {
    const result = new Observable<T>();
    this.subscribe(value => {
      if (predicate(value)) {
        result.emit(value);
      }
    });
    return result;
  }

  debounce(ms: number): Observable<T> {
    const result = new Observable<T>();
    let timeout: ReturnType<typeof setTimeout> | null = null;
    this.subscribe(value => {
      if (timeout) clearTimeout(timeout);
      timeout = setTimeout(() => result.emit(value), ms);
    });
    return result;
  }
}

// OOP: 状態を持つイベントソース
class SearchInput extends Observable<string> {
  private _value = "";

  get value(): string { return this._value; }

  setValue(newValue: string): void {
    this._value = newValue;
    this.emit(newValue);
  }
}

// FP + リアクティブ: データ変換パイプライン
const searchInput = new SearchInput();

const searchResults = searchInput
  .debounce(300)                          // 300ms デバウンス
  .filter(query => query.length >= 2)     // 2文字以上
  .map(query => query.toLowerCase().trim()) // 正規化
  .map(query => `検索: "${query}"`);      // 表示用に変換

searchResults.subscribe(result => {
  console.log(result);
});

// 使用例
searchInput.setValue("T");        // 無視（2文字未満）
searchInput.setValue("Ty");       // 300ms後に "検索: "ty""
searchInput.setValue("TypeScript"); // 300ms後に "検索: "typescript""
```

### 4.3 アクターモデルとの比較

```typescript
// TypeScript: アクターモデルの簡易実装

type Message = {
  type: string;
  payload: unknown;
  replyTo?: Actor<unknown>;
};

class Actor<T extends Message> {
  private mailbox: T[] = [];
  private processing = false;

  constructor(
    public readonly name: string,
    private readonly handler: (message: T, self: Actor<T>) => void,
  ) {}

  send(message: T): void {
    this.mailbox.push(message);
    this.processNext();
  }

  private processNext(): void {
    if (this.processing || this.mailbox.length === 0) return;
    this.processing = true;

    const message = this.mailbox.shift()!;
    try {
      this.handler(message, this);
    } catch (error) {
      console.error(`Actor ${this.name}: エラー`, error);
      // Let it crash: エラーを隔離
    } finally {
      this.processing = false;
      if (this.mailbox.length > 0) {
        // 次のメッセージを非同期に処理
        setTimeout(() => this.processNext(), 0);
      }
    }
  }
}

// アクターモデルでの銀行口座
type BankMessage =
  | { type: "deposit"; payload: { amount: number }; replyTo?: Actor<any> }
  | { type: "withdraw"; payload: { amount: number }; replyTo?: Actor<any> }
  | { type: "getBalance"; payload: {}; replyTo: Actor<any> }
  | { type: "balanceResponse"; payload: { balance: number } };

function createBankAccountActor(
  accountId: string,
  initialBalance: number,
): Actor<BankMessage> {
  let balance = initialBalance;

  return new Actor<BankMessage>(`account-${accountId}`, (message, self) => {
    switch (message.type) {
      case "deposit":
        balance += (message.payload as { amount: number }).amount;
        console.log(`[${accountId}] 入金: ${(message.payload as any).amount} → 残高: ${balance}`);
        break;
      case "withdraw":
        const amount = (message.payload as { amount: number }).amount;
        if (amount > balance) {
          console.log(`[${accountId}] 残高不足`);
        } else {
          balance -= amount;
          console.log(`[${accountId}] 出金: ${amount} → 残高: ${balance}`);
        }
        break;
      case "getBalance":
        if (message.replyTo) {
          message.replyTo.send({
            type: "balanceResponse",
            payload: { balance },
          });
        }
        break;
    }
  });
}

// 使用例
const account = createBankAccountActor("001", 10000);
account.send({ type: "deposit", payload: { amount: 5000 } });
account.send({ type: "withdraw", payload: { amount: 3000 } });

// アクターモデルの利点:
// - 各アクターが独立した状態を持つ（共有状態なし）
// - メッセージは非同期で処理される
// - エラーが他のアクターに影響しない
// - 分散システムに自然に拡張できる
```

---

## 5. 選択指針

```
OOPを使うべき場面:
  ✓ ビジネスドメインのモデリング（エンティティが多い）
  ✓ GUIフレームワーク（ウィジェット階層）
  ✓ ゲームのエンティティシステム
  ✓ フレームワーク/ライブラリの公開API設計
  ✓ チーム開発（構造の共通理解が重要）

関数型を使うべき場面:
  ✓ データ処理パイプライン
  ✓ 並行・分散処理
  ✓ 数学的・科学的計算
  ✓ コンパイラ・パーサー
  ✓ 状態を持たない変換ロジック

手続き型を使うべき場面:
  ✓ シンプルなスクリプト
  ✓ シェルスクリプト的な処理
  ✓ プロトタイプ・使い捨てコード
  ✓ ハードウェア制御

マルチパラダイム（推奨）:
  → ドメインモデル = OOP
  → データ変換 = FP
  → 設定・スクリプト = 手続き型
  → 並行処理 = アクターモデル / CSP
```

### 5.1 判断フローチャート

```
パラダイム選択のフローチャート:

  Q1: プロジェクトの規模は？
    → 100行未満 → 手続き型
    → 100-1000行 → OOP or FP
    → 1000行以上 → Q2へ

  Q2: 主な関心事は？
    → エンティティの管理（ユーザー、注文等）→ OOP
    → データの変換・処理 → FP
    → 並行・分散処理 → アクターモデル / FP
    → UI / イベント処理 → OOP + リアクティブ
    → システムプログラミング → 手続き型 + OOP

  Q3: 何が頻繁に変わるか？
    → 型（エンティティ）が増える → OOP
    → 操作（ビジネスルール）が増える → FP
    → 両方 → マルチパラダイム

  Q4: チームの習熟度は？
    → OOP経験豊富 → OOP + FP要素
    → FP経験豊富 → FP + OOP要素
    → 混合チーム → マルチパラダイム（明確なガイドライン付き）

  Q5: テスト戦略は？
    → 単体テスト重視 → FP（純粋関数のテストが容易）
    → 統合テスト重視 → OOP（モックによるDI）
    → 両方 → マルチパラダイム
```

### 5.2 実際のプロジェクトでの判断例

```typescript
// TypeScript: ECサイトのバックエンドでの判断例

// === ドメインモデル: OOP ===
// 理由: エンティティが多く、ビジネスルールが各エンティティに紐づく
class Product {
  constructor(
    public readonly id: string,
    private _name: string,
    private _price: number,
    private _stock: number,
    private _category: string,
  ) {}

  get name(): string { return this._name; }
  get price(): number { return this._price; }
  get stock(): number { return this._stock; }
  get category(): string { return this._category; }
  get isAvailable(): boolean { return this._stock > 0; }

  reserve(quantity: number): void {
    if (quantity > this._stock) throw new Error("在庫不足");
    this._stock -= quantity;
  }

  restock(quantity: number): void {
    this._stock += quantity;
  }

  updatePrice(newPrice: number): void {
    if (newPrice < 0) throw new Error("価格は0以上");
    this._price = newPrice;
  }
}


// === レポート/集計: FP ===
// 理由: データの変換・集計が主な処理
type SalesRecord = {
  productId: string;
  category: string;
  amount: number;
  quantity: number;
  date: Date;
};

// 純粋関数で集計ロジックを記述
const totalSales = (records: SalesRecord[]): number =>
  records.reduce((sum, r) => sum + r.amount, 0);

const salesByCategory = (records: SalesRecord[]): Map<string, number> =>
  records.reduce((map, r) => {
    const current = map.get(r.category) ?? 0;
    map.set(r.category, current + r.amount);
    return map;
  }, new Map<string, number>());

const topProducts = (records: SalesRecord[], n: number): string[] => {
  const productSales = new Map<string, number>();
  for (const r of records) {
    const current = productSales.get(r.productId) ?? 0;
    productSales.set(r.productId, current + r.amount);
  }
  return [...productSales.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, n)
    .map(([id]) => id);
};

const monthlySales = (records: SalesRecord[]): Map<string, number> =>
  records.reduce((map, r) => {
    const key = `${r.date.getFullYear()}-${String(r.date.getMonth() + 1).padStart(2, "0")}`;
    const current = map.get(key) ?? 0;
    map.set(key, current + r.amount);
    return map;
  }, new Map<string, number>());


// === バッチ処理/スクリプト: 手続き型 ===
// 理由: 手順が明確で、1回きりの処理
async function runDailyReport(
  db: Database,
  mailer: EmailService,
): Promise<void> {
  // 1. データ取得
  const records = await db.getSalesRecords(new Date());

  // 2. 集計（純粋関数を利用）
  const total = totalSales(records);
  const byCategory = salesByCategory(records);
  const topN = topProducts(records, 10);

  // 3. レポート生成
  const report = formatReport(total, byCategory, topN);

  // 4. メール送信
  await mailer.send("admin@example.com", "日次売上レポート", report);

  // 5. ログ
  console.log(`日次レポート送信完了: 売上 ${total}円`);
}
```

---

## 6. パラダイム間の対応表

```
概念の対応:

  ┌──────────────────┬────────────────┬────────────────┬─────────────────┐
  │ 概念             │ OOP            │ FP             │ 手続き型        │
  ├──────────────────┼────────────────┼────────────────┼─────────────────┤
  │ データ + 操作    │ クラス         │ モジュール     │ 構造体 + 関数   │
  │ コード再利用     │ 継承           │ 高階関数       │ 関数ライブラリ  │
  │ 多態性           │ ポリモーフィズム│ パターンマッチ │ 関数ポインタ    │
  │ 状態管理         │ カプセル化     │ 不変データ     │ グローバル変数  │
  │ エラー処理       │ 例外           │ Result/Maybe型 │ エラーコード    │
  │ 依存管理         │ DI             │ 関数の引数     │ グローバル参照  │
  │ コレクション操作 │ イテレータ     │ map/filter     │ forループ       │
  │ 非同期処理       │ Future/Promise │ IO モナド      │ コールバック    │
  └──────────────────┴────────────────┴────────────────┴─────────────────┘
```

### 6.1 エラー処理の比較

```typescript
// TypeScript: 各パラダイムでのエラー処理

// === OOP: 例外ベース ===
class InsufficientFundsError extends Error {
  constructor(
    public readonly accountId: string,
    public readonly balance: number,
    public readonly requested: number,
  ) {
    super(`残高不足: 残高 ${balance}円、要求 ${requested}円`);
    this.name = "InsufficientFundsError";
  }
}

class OOPBankService {
  withdraw(accountId: string, amount: number): void {
    const account = this.findAccount(accountId);
    if (account.balance < amount) {
      throw new InsufficientFundsError(accountId, account.balance, amount);
    }
    account.balance -= amount;
  }

  private findAccount(id: string): { balance: number } {
    return { balance: 10000 }; // 簡略化
  }
}

// 呼び出し側
try {
  const service = new OOPBankService();
  service.withdraw("001", 50000);
} catch (e) {
  if (e instanceof InsufficientFundsError) {
    console.log(`残高不足: ${e.balance}円 < ${e.requested}円`);
  }
}


// === FP: Result型ベース ===
type WithdrawError =
  | { type: "INSUFFICIENT_FUNDS"; balance: number; requested: number }
  | { type: "ACCOUNT_NOT_FOUND"; accountId: string }
  | { type: "INVALID_AMOUNT"; amount: number };

type WithdrawResult = Result<{ newBalance: number }, WithdrawError>;

type Result<T, E> =
  | { ok: true; value: T }
  | { ok: false; error: E };

function fpWithdraw(
  balance: number,
  amount: number,
): WithdrawResult {
  if (amount <= 0) {
    return { ok: false, error: { type: "INVALID_AMOUNT", amount } };
  }
  if (balance < amount) {
    return {
      ok: false,
      error: { type: "INSUFFICIENT_FUNDS", balance, requested: amount },
    };
  }
  return { ok: true, value: { newBalance: balance - amount } };
}

// 呼び出し側: 全てのケースを型安全にハンドリング
const result = fpWithdraw(10000, 50000);
if (result.ok) {
  console.log(`出金成功: 新残高 ${result.value.newBalance}円`);
} else {
  switch (result.error.type) {
    case "INSUFFICIENT_FUNDS":
      console.log(`残高不足: ${result.error.balance}円 < ${result.error.requested}円`);
      break;
    case "INVALID_AMOUNT":
      console.log(`無効な金額: ${result.error.amount}円`);
      break;
    case "ACCOUNT_NOT_FOUND":
      console.log(`口座が見つかりません: ${result.error.accountId}`);
      break;
  }
}
```

### 6.2 コレクション操作の比較

```python
# Python: コレクション操作の各パラダイム比較

from dataclasses import dataclass
from typing import Callable
from functools import reduce

@dataclass(frozen=True)
class Employee:
    name: str
    department: str
    salary: int
    years: int

employees = [
    Employee("田中", "開発", 600000, 5),
    Employee("山田", "営業", 450000, 3),
    Employee("佐藤", "開発", 750000, 8),
    Employee("鈴木", "人事", 500000, 2),
    Employee("高橋", "開発", 550000, 4),
    Employee("伊藤", "営業", 400000, 1),
    Employee("渡辺", "人事", 600000, 6),
]

# === 手続き型 ===
def get_dev_avg_salary_procedural(employees):
    total = 0
    count = 0
    for emp in employees:
        if emp.department == "開発":
            total += emp.salary
            count += 1
    return total / count if count > 0 else 0

# === OOP ===
class EmployeeAnalytics:
    def __init__(self, employees: list[Employee]):
        self._employees = employees

    def average_salary_by_department(self, department: str) -> float:
        dept_employees = [e for e in self._employees if e.department == department]
        if not dept_employees:
            return 0
        return sum(e.salary for e in dept_employees) / len(dept_employees)

    def top_earners(self, n: int) -> list[Employee]:
        return sorted(self._employees, key=lambda e: e.salary, reverse=True)[:n]

    def department_report(self) -> dict[str, dict]:
        report = {}
        for emp in self._employees:
            if emp.department not in report:
                report[emp.department] = {"count": 0, "total_salary": 0, "names": []}
            report[emp.department]["count"] += 1
            report[emp.department]["total_salary"] += emp.salary
            report[emp.department]["names"].append(emp.name)
        for dept in report:
            report[dept]["avg_salary"] = report[dept]["total_salary"] / report[dept]["count"]
        return report

analytics = EmployeeAnalytics(employees)
print(analytics.average_salary_by_department("開発"))
print(analytics.top_earners(3))

# === 関数型 ===
from itertools import groupby
from operator import attrgetter

# 純粋関数のパイプライン
def fp_avg_salary_by_dept(employees: list[Employee], dept: str) -> float:
    dept_salaries = [e.salary for e in employees if e.department == dept]
    return sum(dept_salaries) / len(dept_salaries) if dept_salaries else 0

def fp_top_earners(employees: list[Employee], n: int) -> list[Employee]:
    return sorted(employees, key=attrgetter("salary"), reverse=True)[:n]

def fp_department_summary(employees: list[Employee]) -> dict[str, dict]:
    sorted_emps = sorted(employees, key=attrgetter("department"))
    return {
        dept: {
            "count": len(group := list(emps)),
            "avg_salary": sum(e.salary for e in group) / len(group),
            "names": [e.name for e in group],
        }
        for dept, emps in groupby(sorted_emps, key=attrgetter("department"))
    }

# 関数合成
def compose(*functions):
    """関数を合成する"""
    def composed(data):
        result = data
        for fn in functions:
            result = fn(result)
        return result
    return composed

# パイプライン: 開発部門の高給取りTOP2の名前
pipeline = compose(
    lambda emps: [e for e in emps if e.department == "開発"],
    lambda emps: sorted(emps, key=lambda e: e.salary, reverse=True),
    lambda emps: emps[:2],
    lambda emps: [e.name for e in emps],
)

result = pipeline(employees)
print(result)  # ['佐藤', '田中']
```

---

## 7. パラダイムの融合: 現代のトレンド

```
現代の言語はパラダイムの壁を越えて融合している:

  TypeScript: OOP + FP + 構造的型付け
  Kotlin: OOP + FP + コルーチン
  Swift: OOP + プロトコル指向 + 値型
  Rust: トレイト + FP + 所有権
  Scala: OOP + FP の完全な融合
  Python: OOP + FP + 手続き型

  「何パラダイムで書くか」ではなく
  「この問題に最適なツールは何か」で考える時代

  ベストプラクティス:
    1. デフォルトは不変データ
    2. 純粋関数を中心に設計
    3. エンティティが必要ならOOP
    4. 副作用は境界に集約
    5. テスト容易性を常に意識
```

```typescript
// TypeScript: パラダイム融合の実例

// 関数型の値オブジェクト
type Currency = "JPY" | "USD" | "EUR";

const createMoney = (amount: number, currency: Currency = "JPY") => {
  if (amount < 0) throw new Error("金額は0以上");
  return Object.freeze({ amount, currency });
};

type MoneyType = ReturnType<typeof createMoney>;

const addMoney = (a: MoneyType, b: MoneyType): MoneyType => {
  if (a.currency !== b.currency) throw new Error("通貨不一致");
  return createMoney(a.amount + b.amount, a.currency);
};

// OOPのエンティティ（内部は不変データを活用）
class Invoice {
  private readonly _lines: ReadonlyArray<{
    description: string;
    amount: MoneyType;
  }>;

  constructor(
    public readonly invoiceId: string,
    public readonly customerId: string,
    lines: { description: string; amount: MoneyType }[],
  ) {
    this._lines = Object.freeze([...lines]);
  }

  get total(): MoneyType {
    return this._lines.reduce(
      (sum, line) => addMoney(sum, line.amount),
      createMoney(0),
    );
  }

  get lineCount(): number {
    return this._lines.length;
  }

  // FP的: 新しいInvoiceを返す（不変）
  addLine(description: string, amount: MoneyType): Invoice {
    return new Invoice(
      this.invoiceId,
      this.customerId,
      [...this._lines, { description, amount }],
    );
  }
}

// 関数型のパイプライン
const getOverdueInvoices = (
  invoices: Invoice[],
  dueDate: Date,
): Invoice[] =>
  invoices
    .filter(inv => inv.total.amount > 0)
    .sort((a, b) => b.total.amount - a.total.amount);
```

---

## まとめ

| パラダイム | 核心 | 得意分野 | 弱点 |
|-----------|------|---------|------|
| 手続き型 | 手順の記述 | シンプル・スクリプト | 大規模で破綻 |
| OOP | オブジェクト | ドメインモデル・GUI | 共有状態・並行処理 |
| 関数型 | データ変換 | 並行処理・パイプライン | GUI・状態管理 |
| リアクティブ | データの流れ | 非同期・UI | デバッグ困難 |
| アクターモデル | メッセージ | 分散・耐障害性 | 順序保証困難 |
| マルチ | 適材適所 | 現代の大規模開発 | 判断力が必要 |

---

## 次に読むべきガイド
→ [[03-class-and-object.md]] — クラスとオブジェクト

---

## 参考文献
1. Wadler, P. "The Expression Problem." 1998.
2. Martin, R. "Clean Architecture." Prentice Hall, 2017.
3. Odersky, M. "Programming in Scala." Artima, 2021.
4. Armstrong, J. "Programming Erlang." Pragmatic Bookshelf, 2013.
5. Wlaschin, S. "Domain Modeling Made Functional." Pragmatic Bookshelf, 2018.
6. Milewski, B. "Category Theory for Programmers." 2018.
