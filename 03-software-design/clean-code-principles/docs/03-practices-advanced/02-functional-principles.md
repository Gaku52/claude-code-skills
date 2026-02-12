# 関数型プログラミングの原則をクリーンコードに活かす

> 純粋関数、副作用分離、高階関数、パイプラインなど、関数型プログラミングの核心的な原則をクリーンコードの文脈で解説し、より安全で保守しやすいコードの実現を支援する。

---

## 前提知識

| トピック | 内容 | 参照先 |
|---------|------|--------|
| クリーンコードの基本原則 | 命名規則・関数設計・コメントの書き方 | [00-naming-conventions.md](../00-principles/00-naming-conventions.md) |
| SOLID原則 | 単一責任、開放閉鎖、依存性逆転 | [04-solid-principles.md](../00-principles/04-solid-principles.md) |
| テスト原則 | ユニットテストの基本・テストピラミッド | [04-testing-principles.md](../01-practices/04-testing-principles.md) |
| イミュータビリティ | 不変データ構造と値オブジェクト | [00-immutability.md](./00-immutability.md) |
| Strategyパターン | 振る舞いの差し替え | [../../design-patterns-guide/docs/02-behavioral/00-strategy.md](../../design-patterns-guide/docs/02-behavioral/00-strategy.md) |

---

## この章で学ぶこと

1. **純粋関数と参照透過性**の概念を理解し、テスト可能で予測可能な関数を設計できる
2. **副作用の分離・Functional Core / Imperative Shell** パターンでアプリケーションを構造化できる
3. **高階関数・カリー化・部分適用**を活用して、再利用性と合成可能性の高いコードを書ける
4. **型安全なパイプラインとResult/Either型**で、宣言的かつ安全なデータ変換・エラーハンドリングを実装できる
5. **関数型の原則をオブジェクト指向や日常の開発**に統合し、ハイブリッドアーキテクチャを実現できる

---

## 1. 関数型プログラミングの基礎概念

### 1.1 関数型の核心的原則

```
┌──────────────────────────────────────────────────────┐
│          関数型プログラミングの5つの柱                   │
├──────────────────────────────────────────────────────┤
│                                                      │
│  ┌──────────────┐                                    │
│  │ 1. 純粋関数   │  同じ入力 → 常に同じ出力          │
│  │              │  外部状態を変更しない               │
│  └──────────────┘                                    │
│           │                                          │
│  ┌──────────────┐                                    │
│  │ 2. 不変性    │  データは変更せずコピーを作る        │
│  │              │  状態変化を明示的に管理              │
│  └──────────────┘                                    │
│           │                                          │
│  ┌──────────────┐                                    │
│  │ 3. 高階関数   │  関数を引数に取る、関数を返す       │
│  │              │  振る舞いの抽象化                   │
│  └──────────────┘                                    │
│           │                                          │
│  ┌──────────────┐                                    │
│  │ 4. 合成      │  小さい関数を組み合わせて大きな     │
│  │              │  処理を構築する                     │
│  └──────────────┘                                    │
│           │                                          │
│  ┌──────────────┐                                    │
│  │ 5. 宣言的    │  「何をするか」を記述               │
│  │              │  「どうやるか」を隠蔽               │
│  └──────────────┘                                    │
│                                                      │
└──────────────────────────────────────────────────────┘
```

### 1.2 プログラミングパラダイムのスペクトル

```
関数型の要素をどの程度取り入れるかはスペクトルで考える:

  純粋OOP ──────────── ハイブリッド ──────────── 純粋FP
    │                      │                       │
    Java                TypeScript              Haskell
   (従来)              Kotlin, Scala             Elm
                       Python, Rust              Erlang

  現実世界のアプリケーション開発:
    │
    ├── 純粋FPを全面採用する必要はない
    ├── 「使える部分に関数型の原則を適用する」プラグマティックなアプローチ
    └── テスタビリティ・予測可能性が向上する箇所に重点的に導入
```

### 1.3 命令型 vs 宣言型（関数型）

```
命令型 (How):                    宣言型/関数型 (What):
─────────────                   ─────────────────────

result = []                      result = (
for item in items:                 items
    if item.active:                  .filter(active)
        value = transform(item)      .map(transform)
        result.append(value)         .collect()
                                   )

手順を1つずつ記述                 変換のパイプラインで記述
ループ変数、条件分岐、副作用      宣言的、合成可能、テスト容易
```

```typescript
// TypeScript: 命令型 vs 宣言型の実践的比較

interface Transaction {
  id: string;
  amount: number;
  type: "income" | "expense";
  category: string;
  date: Date;
}

// NG: 命令型 — 中間変数、ループ、条件分岐が散在
function summarizeImperative(transactions: Transaction[]): Record<string, number> {
  const result: Record<string, number> = {};
  for (let i = 0; i < transactions.length; i++) {
    const tx = transactions[i];
    if (tx.type === "expense") {
      if (result[tx.category] === undefined) {
        result[tx.category] = 0;
      }
      result[tx.category] += tx.amount;
    }
  }
  // ソートのために配列に変換
  const entries = Object.entries(result);
  entries.sort((a, b) => b[1] - a[1]);
  const sorted: Record<string, number> = {};
  for (const [key, value] of entries) {
    sorted[key] = value;
  }
  return sorted;
}

// OK: 宣言型 — パイプライン的な変換の連鎖
function summarizeDeclarative(transactions: Transaction[]): Record<string, number> {
  return Object.fromEntries(
    Object.entries(
      transactions
        .filter(tx => tx.type === "expense")
        .reduce<Record<string, number>>((acc, tx) => ({
          ...acc,
          [tx.category]: (acc[tx.category] ?? 0) + tx.amount,
        }), {})
    ).sort(([, a], [, b]) => b - a)
  );
}
```

---

## 2. 純粋関数

### 2.1 純粋関数の定義と利点

```
純粋関数の2つの条件:

  1. 同じ入力に対して常に同じ出力を返す（参照透過性）
  2. 副作用がない（外部状態を変更しない）

┌────────────┐          ┌────────────────┐
│  入力 A    │ ──────> │                │ ──────> 出力 X
└────────────┘          │   純粋関数 f    │
                        │                │
  入力 A（再度）──────> │  f(A) = X      │ ──────> 出力 X（必ず同じ）
                        │  （常に同じ）   │
                        └────────────────┘
                              │
                              │ 副作用なし:
                              │ ・グローバル変数を変更しない
                              │ ・ファイルに書き込まない
                              │ ・DBを更新しない
                              │ ・ネットワーク通信しない
                              │ ・引数を変更しない
```

```
純粋関数のメリットマトリクス:

  メリット           説明                          影響度
  ──────────────────────────────────────────────────────
  テスト容易性       入出力のみテスト、モック不要      ★★★★★
  推論容易性         関数の振る舞いが引数だけで決まる  ★★★★★
  並行安全性         共有状態がないためロック不要       ★★★★☆
  キャッシュ可能     参照透過性によりメモ化が安全       ★★★★☆
  リファクタリング   関数の抽出・合成が安全            ★★★★☆
  デバッグ容易性     再現性が保証される               ★★★★☆
```

### 2.2 純粋関数の実装例

```python
# 純粋関数 vs 不純関数

# NG: 不純関数（外部状態に依存/変更）
tax_rate = 0.10  # グローバル変数

def calculate_total_impure(price: float) -> float:
    """不純: 外部変数に依存"""
    return price * (1 + tax_rate)  # tax_rateが変わると結果が変わる

total_items = []

def add_to_cart_impure(item: dict) -> None:
    """不純: 外部リストを変更"""
    total_items.append(item)  # 副作用: 外部状態を変更

# OK: 純粋関数
def calculate_total_pure(price: float, tax_rate: float) -> float:
    """純粋: 全ての依存が引数として明示"""
    return price * (1 + tax_rate)

def add_to_cart_pure(cart: tuple, item: dict) -> tuple:
    """純粋: 新しいカートを返す（元は変更しない）"""
    return cart + (item,)

# テストが簡単
assert calculate_total_pure(1000, 0.10) == 1100.0
assert calculate_total_pure(1000, 0.08) == 1080.0
# → 何度実行しても同じ結果が保証される
```

### 2.3 不純関数を純粋にリファクタリングする4つのパターン

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

# === パターン1: 依存を引数に注入する ===

# NG: 現在時刻に依存（非決定的）
def is_business_hours_impure() -> bool:
    now = datetime.now()
    return 9 <= now.hour < 18

# OK: 時刻を引数に注入
def is_business_hours(current_hour: int) -> bool:
    return 9 <= current_hour < 18

# テスト容易
assert is_business_hours(10) == True
assert is_business_hours(20) == False


# === パターン2: 副作用を戻り値に変換する ===

# NG: ログ出力（副作用）
import logging

def process_order_impure(order_id: str, amount: int) -> int:
    if amount <= 0:
        logging.error(f"Invalid amount for order {order_id}")
        raise ValueError("Invalid amount")
    tax = int(amount * 0.1)
    logging.info(f"Order {order_id}: amount={amount}, tax={tax}")
    return amount + tax

# OK: 結果とログメッセージを分離して返す
@dataclass(frozen=True)
class ProcessResult:
    total: int
    log_messages: tuple[str, ...]

def process_order_pure(order_id: str, amount: int) -> ProcessResult:
    if amount <= 0:
        return ProcessResult(total=0, log_messages=(
            f"ERROR: Invalid amount for order {order_id}",
        ))
    tax = int(amount * 0.1)
    return ProcessResult(
        total=amount + tax,
        log_messages=(f"INFO: Order {order_id}: amount={amount}, tax={tax}",),
    )

# テスト容易: ログ出力のモック不要
result = process_order_pure("ORD-001", 1000)
assert result.total == 1100
assert "amount=1000" in result.log_messages[0]


# === パターン3: 状態変更を新しい状態の返却に変える ===

# NG: オブジェクトの内部状態を変更
class BankAccountMutable:
    def __init__(self, balance: int):
        self.balance = balance

    def withdraw(self, amount: int) -> None:
        self.balance -= amount  # 副作用: 状態変更

# OK: 新しい状態を返す
@dataclass(frozen=True)
class BankAccount:
    balance: int

    def withdraw(self, amount: int) -> "BankAccount":
        return BankAccount(balance=self.balance - amount)

account = BankAccount(balance=10000)
new_account = account.withdraw(3000)
assert account.balance == 10000      # 元は変わらない
assert new_account.balance == 7000


# === パターン4: コールバックを高階関数に置き換える ===

# NG: グローバルなイベントバスに通知
event_bus = []  # グローバル状態

def complete_task_impure(task_id: str) -> None:
    event_bus.append({"type": "task_completed", "task_id": task_id})

# OK: 発行すべきイベントを戻り値に含める
@dataclass(frozen=True)
class DomainEvent:
    event_type: str
    payload: dict

def complete_task_pure(task_id: str) -> tuple[str, list[DomainEvent]]:
    """タスク完了の結果とドメインイベントを返す"""
    return (
        "completed",
        [DomainEvent(event_type="task_completed", payload={"task_id": task_id})],
    )

status, events = complete_task_pure("TASK-42")
assert status == "completed"
assert events[0].event_type == "task_completed"
```

### 2.4 参照透過性の活用

```typescript
// 参照透過性: 関数呼び出しをその結果で置き換えても意味が変わらない

// 純粋関数: 参照透過
function add(a: number, b: number): number {
  return a + b;
}

// add(2, 3) は常に 5 なので、コード中の add(2, 3) を 5 に置換可能
const x = add(2, 3) * add(2, 3);
const y = 5 * 5;  // 完全に等価

// === 参照透過性が可能にすること ===

// 1. メモ化（キャッシュ）
function memoize<T extends (...args: any[]) => any>(fn: T): T {
  const cache = new Map<string, ReturnType<T>>();
  return ((...args: any[]) => {
    const key = JSON.stringify(args);
    if (cache.has(key)) return cache.get(key)!;
    const result = fn(...args);
    cache.set(key, result);
    return result;
  }) as T;
}

// 重い計算をメモ化で高速化
const expensiveCalc = memoize((n: number): number => {
  console.log(`Computing for ${n}...`);
  return Array.from({ length: n }, (_, i) => i + 1)
    .reduce((sum, x) => sum + x, 0);
});

expensiveCalc(1000); // Computing for 1000... → 500500
expensiveCalc(1000); // キャッシュから即返却 → 500500

// 2. 遅延評価（必要になるまで計算しない）
// 3. 並列実行（順序に依存しない）
// 4. テストの独立性（セットアップ不要）
// 5. 等式推論（数学的にコードの正しさを証明できる）
```

### 2.5 メモ化の実践的な応用

```typescript
// React におけるメモ化の活用

// useMemo: 参照透過な計算のメモ化
function ExpenseReport({ transactions }: { transactions: Transaction[] }) {
  // transactions が変わらない限り再計算されない
  const summary = useMemo(() =>
    transactions
      .filter(tx => tx.type === "expense")
      .reduce<Record<string, number>>((acc, tx) => ({
        ...acc,
        [tx.category]: (acc[tx.category] ?? 0) + tx.amount,
      }), {}),
    [transactions]
  );

  // useCallback: 関数自体のメモ化（子コンポーネントの再レンダリング防止）
  const handleCategoryClick = useCallback(
    (category: string) => {
      console.log(`Selected: ${category}`);
    },
    [] // 依存なし → 同じ関数参照が維持される
  );

  return (
    <div>
      {Object.entries(summary).map(([category, amount]) => (
        <CategoryRow
          key={category}
          name={category}
          amount={amount}
          onClick={handleCategoryClick}
        />
      ))}
    </div>
  );
}
```

```python
# Python: functools.lru_cache を使ったメモ化

from functools import lru_cache

# 再帰的フィボナッチ — メモ化なしだと指数的計算量
@lru_cache(maxsize=256)
def fibonacci(n: int) -> int:
    """O(n) に最適化されたフィボナッチ（メモ化）"""
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# メモ化が安全な理由: fibonacci は純粋関数
assert fibonacci(50) == 12586269025  # 瞬時に計算

# キャッシュの状態確認
print(fibonacci.cache_info())
# CacheInfo(hits=48, misses=51, maxsize=256, currsize=51)
```

---

## 3. 副作用の分離

### 3.1 純粋コアと不純シェル（Functional Core / Imperative Shell）

```
┌──────────────────────────────────────────────┐
│           Functional Core / Imperative Shell  │
├──────────────────────────────────────────────┤
│                                              │
│  ┌────────────────────────────────┐         │
│  │     Imperative Shell (不純)    │         │
│  │  ┌──────────────────────┐     │         │
│  │  │                      │     │         │
│  │  │   Functional Core    │     │         │
│  │  │   (純粋)             │     │         │
│  │  │                      │     │         │
│  │  │  ・ビジネスロジック   │     │         │
│  │  │  ・データ変換         │     │         │
│  │  │  ・バリデーション     │     │         │
│  │  │  ・計算               │     │         │
│  │  │                      │     │         │
│  │  └──────────────────────┘     │         │
│  │                                │         │
│  │  ・DB読み書き                  │         │
│  │  ・ファイルI/O                 │         │
│  │  ・HTTP通信                   │         │
│  │  ・ログ出力                   │         │
│  │  ・時刻取得                   │         │
│  └────────────────────────────────┘         │
│                                              │
│  純粋な中心 → テスト容易、予測可能            │
│  不純な外殻 → I/Oを集約、薄く保つ            │
│                                              │
└──────────────────────────────────────────────┘
```

```
FC/IS パターンの責務分離フロー:

  HTTP Request
      │
      ▼
  ┌─────────────────────────┐
  │ Controller (Shell)       │  リクエスト解析
  │  ├ parse request         │
  │  ├ read from DB          │  ← 副作用: DB読み取り
  │  └ call core logic ──────┼──────┐
  └─────────────────────────┘      │
                                    ▼
                            ┌───────────────────┐
                            │ Core (純粋)        │
                            │  ├ validate        │
                            │  ├ calculate       │  副作用なし
                            │  ├ transform       │  テスト容易
                            │  └ return result ──┼──────┐
                            └───────────────────┘      │
                                    ▲                   ▼
  ┌─────────────────────────┐      │      ┌─────────────────┐
  │ Repository (Shell)       │      │      │ Controller      │
  │  ├ save to DB            │ ◄────┘      │  ├ save result  │
  │  └ send notification     │             │  └ return resp  │
  └─────────────────────────┘             └─────────────────┘
```

### 3.2 実装例：注文処理システム

```python
# Functional Core: 純粋なビジネスロジック
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class OrderItem:
    product_id: str
    name: str
    price: int
    quantity: int

@dataclass(frozen=True)
class Order:
    items: tuple[OrderItem, ...]
    discount_rate: float = 0.0

# --- 純粋関数群（Functional Core）---

def calculate_subtotal(order: Order) -> int:
    """小計を計算（純粋）"""
    return sum(item.price * item.quantity for item in order.items)

def apply_discount(subtotal: int, discount_rate: float) -> int:
    """割引を適用（純粋）"""
    return int(subtotal * (1 - discount_rate))

def calculate_tax(amount: int, tax_rate: float) -> int:
    """税額を計算（純粋）"""
    return int(amount * tax_rate)

def calculate_total(order: Order, tax_rate: float) -> dict:
    """合計を計算（純粋 — 全てのロジックが関数合成）"""
    subtotal = calculate_subtotal(order)
    discounted = apply_discount(subtotal, order.discount_rate)
    tax = calculate_tax(discounted, tax_rate)
    return {
        "subtotal": subtotal,
        "discount": subtotal - discounted,
        "tax": tax,
        "total": discounted + tax,
    }

def validate_order(order: Order) -> list[str]:
    """注文バリデーション（純粋 — エラーリストを返す）"""
    errors = []
    if not order.items:
        errors.append("注文に商品がありません")
    for item in order.items:
        if item.quantity <= 0:
            errors.append(f"{item.name}: 数量は1以上必要です")
        if item.price < 0:
            errors.append(f"{item.name}: 価格が不正です")
    if not (0.0 <= order.discount_rate <= 1.0):
        errors.append("割引率は0〜1の範囲で指定してください")
    return errors

# --- Imperative Shell: I/Oと副作用 ---

class OrderService:
    """不純なシェル: I/Oを担当"""

    def __init__(self, db, payment_gateway, notifier):
        self.db = db
        self.payment = payment_gateway
        self.notifier = notifier

    def process_order(self, order: Order) -> dict:
        """注文処理（不純 — I/Oを呼ぶ）"""
        # 1. 純粋なバリデーション
        errors = validate_order(order)
        if errors:
            return {"status": "error", "errors": errors}

        # 2. 純粋な計算
        totals = calculate_total(order, tax_rate=0.10)

        # 3. 不純な処理（I/O）
        payment_result = self.payment.charge(totals["total"])
        if not payment_result.success:
            return {"status": "payment_failed"}

        order_id = self.db.save_order(order, totals)
        self.notifier.send_confirmation(order_id)

        return {"status": "success", "order_id": order_id, **totals}

# テスト: 純粋部分はモック不要で簡単にテスト可能
def test_calculate_total():
    order = Order(
        items=(
            OrderItem("p1", "商品A", 1000, 2),
            OrderItem("p2", "商品B", 500, 3),
        ),
        discount_rate=0.1,
    )
    result = calculate_total(order, tax_rate=0.10)
    assert result["subtotal"] == 3500
    assert result["discount"] == 350
    assert result["total"] == 3465  # (3500-350) * 1.10
```

### 3.3 TypeScriptでの Functional Core / Imperative Shell

```typescript
// === Functional Core ===

// 不変な型定義
interface UserRegistration {
  readonly email: string;
  readonly password: string;
  readonly name: string;
}

interface ValidatedUser {
  readonly email: string;
  readonly passwordHash: string;
  readonly name: string;
  readonly normalizedEmail: string;
}

type ValidationError = { field: string; message: string };

// 純粋関数: バリデーション
function validateRegistration(
  input: UserRegistration
): Result<UserRegistration, ValidationError[]> {
  const errors: ValidationError[] = [];

  if (!input.email.includes("@")) {
    errors.push({ field: "email", message: "有効なメールアドレスを入力してください" });
  }
  if (input.password.length < 8) {
    errors.push({ field: "password", message: "パスワードは8文字以上必要です" });
  }
  if (input.name.trim().length === 0) {
    errors.push({ field: "name", message: "名前を入力してください" });
  }

  return errors.length > 0 ? Err(errors) : Ok(input);
}

// 純粋関数: データ変換（hashPassword は純粋な暗号学的ハッシュとして扱う）
function prepareUser(
  input: UserRegistration,
  hashedPassword: string
): ValidatedUser {
  return {
    email: input.email,
    passwordHash: hashedPassword,
    name: input.name.trim(),
    normalizedEmail: input.email.toLowerCase(),
  };
}

// === Imperative Shell ===

class UserRegistrationService {
  constructor(
    private readonly userRepo: UserRepository,
    private readonly hasher: PasswordHasher,
    private readonly mailer: EmailService,
  ) {}

  async register(input: UserRegistration): Promise<Result<string, string>> {
    // 1. 純粋: バリデーション
    const validated = validateRegistration(input);
    if (!validated.ok) {
      return Err(validated.error.map(e => e.message).join(", "));
    }

    // 2. 不純: 重複チェック（DB読み取り）
    const existing = await this.userRepo.findByEmail(input.email);
    if (existing) {
      return Err("このメールアドレスは既に登録されています");
    }

    // 3. 不純: パスワードハッシュ化
    const hashed = await this.hasher.hash(input.password);

    // 4. 純粋: ユーザーデータの構築
    const user = prepareUser(input, hashed);

    // 5. 不純: DB保存 + メール送信
    const userId = await this.userRepo.save(user);
    await this.mailer.sendWelcome(user.normalizedEmail, user.name);

    return Ok(userId);
  }
}
```

---

## 4. 高階関数

### 4.1 高階関数の基本パターン

```typescript
// 高階関数: 関数を引数に取る or 関数を返す

// 1. 関数を引数に取る
function filter<T>(items: T[], predicate: (item: T) => boolean): T[] {
  const result: T[] = [];
  for (const item of items) {
    if (predicate(item)) result.push(item);
  }
  return result;
}

// 2. 関数を返す（カリー化）
function createMultiplier(factor: number): (n: number) => number {
  return (n: number) => n * factor;
}

const double = createMultiplier(2);
const triple = createMultiplier(3);
console.log(double(5));  // 10
console.log(triple(5));  // 15

// 3. 関数を引数に取り、関数を返す（関数デコレータ）
function withLogging<T extends (...args: any[]) => any>(
  fn: T,
  label: string
): T {
  return ((...args: any[]) => {
    console.log(`[${label}] 呼び出し:`, args);
    const result = fn(...args);
    console.log(`[${label}] 結果:`, result);
    return result;
  }) as T;
}

const add = (a: number, b: number) => a + b;
const loggedAdd = withLogging(add, "add");
loggedAdd(2, 3);
// [add] 呼び出し: [2, 3]
// [add] 結果: 5

// 4. 部分適用
function partial<T extends (...args: any[]) => any>(
  fn: T,
  ...presetArgs: any[]
): (...remainingArgs: any[]) => ReturnType<T> {
  return (...remainingArgs) => fn(...presetArgs, ...remainingArgs);
}

const addTen = partial(add, 10);
console.log(addTen(5));  // 15
```

### 4.2 カリー化と部分適用の違い

```typescript
// カリー化: 多引数関数を「1引数関数のチェーン」に変換する
// 部分適用: 多引数関数の一部の引数を固定して新しい関数を作る

// === カリー化 ===
function curry<A, B, C>(fn: (a: A, b: B) => C): (a: A) => (b: B) => C {
  return (a: A) => (b: B) => fn(a, b);
}

const curriedAdd = curry((a: number, b: number) => a + b);
const add5 = curriedAdd(5);   // (b: number) => number
console.log(add5(3));          // 8
console.log(add5(10));         // 15

// 汎用カリー化関数
function autoCurry(fn: Function) {
  return function curried(...args: any[]): any {
    if (args.length >= fn.length) {
      return fn(...args);
    }
    return (...moreArgs: any[]) => curried(...args, ...moreArgs);
  };
}

// 実用例: ログフォーマッタ
const formatLog = autoCurry(
  (level: string, module: string, message: string) =>
    `[${level}] [${module}] ${message}`
);

const errorLog = formatLog("ERROR");          // level を固定
const dbError = errorLog("Database");          // module も固定
console.log(dbError("Connection timeout"));    // [ERROR] [Database] Connection timeout
console.log(formatLog("INFO", "API", "Request received")); // 全引数指定も可
```

### 4.3 実用的な高階関数パターン

```python
# Python: 実用的な高階関数

from functools import wraps, reduce
from typing import TypeVar, Callable, Any
import time

T = TypeVar("T")

# 1. リトライデコレータ（関数を受け取り関数を返す）
def retry(max_attempts: int = 3, delay: float = 1.0):
    """リトライロジックを関数に付与する高階関数"""
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_error = None
            for attempt in range(max_attempts):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay * (2 ** attempt))  # 指数バックオフ
            raise last_error
        return wrapper
    return decorator

@retry(max_attempts=3, delay=0.5)
def fetch_data(url: str) -> dict:
    # 実際のHTTPリクエスト
    pass

# 2. パイプライン合成
def pipe(*functions: Callable) -> Callable:
    """複数の関数を左から右に合成する"""
    def pipeline(value):
        return reduce(lambda acc, fn: fn(acc), functions, value)
    return pipeline

# 使用例
process_text = pipe(
    str.strip,              # 前後の空白除去
    str.lower,              # 小文字化
    lambda s: s.replace(" ", "_"),  # スペースをアンダースコアに
    lambda s: s[:50],       # 50文字に制限
)

result = process_text("  Hello World Example  ")
# → "hello_world_example"

# 3. バリデーション合成
def compose_validators(*validators):
    """複数のバリデータを合成"""
    def validate(value):
        errors = []
        for validator in validators:
            error = validator(value)
            if error:
                errors.append(error)
        return errors if errors else None
    return validate

def min_length(n):
    def validator(s):
        if len(s) < n:
            return f"{n}文字以上必要です"
    return validator

def max_length(n):
    def validator(s):
        if len(s) > n:
            return f"{n}文字以下にしてください"
    return validator

def matches_pattern(pattern, message):
    import re
    def validator(s):
        if not re.match(pattern, s):
            return message
    return validator

# バリデータの合成
validate_username = compose_validators(
    min_length(3),
    max_length(20),
    matches_pattern(r"^[a-zA-Z0-9_]+$", "英数字とアンダースコアのみ"),
)

errors = validate_username("ab")  # ["3文字以上必要です"]
errors = validate_username("valid_user")  # None

# 4. タイミングデコレータ（パフォーマンス計測）
def timed(fn: Callable[..., T]) -> Callable[..., T]:
    """関数の実行時間を計測する高階関数"""
    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{fn.__name__}: {elapsed:.4f}s")
        return result
    return wrapper

@timed
def heavy_computation(n: int) -> int:
    return sum(i * i for i in range(n))

heavy_computation(1_000_000)  # heavy_computation: 0.0823s
```

---

## 5. 関数型データ変換パイプライン

### 5.1 パイプライン設計

```
データ変換パイプライン:

  入力データ ──> [変換1] ──> [変換2] ──> [変換3] ──> 出力データ

  例: ユーザーリストの処理

  users ──> filter(active) ──> map(toDTO) ──> sort(byName) ──> take(10)

  各ステップ:
  ・純粋関数（副作用なし）
  ・型安全（入力型 → 出力型が明確）
  ・テスト可能（各ステップを個別にテスト）
  ・合成可能（ステップの追加・削除が容易）
```

### 5.2 TypeScriptでの型安全パイプライン

```typescript
// 型安全なパイプライン

// パイプ関数（型推論対応）
function pipe<A>(value: A): A;
function pipe<A, B>(value: A, fn1: (a: A) => B): B;
function pipe<A, B, C>(value: A, fn1: (a: A) => B, fn2: (b: B) => C): C;
function pipe<A, B, C, D>(
  value: A, fn1: (a: A) => B, fn2: (b: B) => C, fn3: (c: C) => D
): D;
function pipe(value: any, ...fns: Function[]): any {
  return fns.reduce((acc, fn) => fn(acc), value);
}

// データ変換関数
interface User {
  id: string;
  name: string;
  age: number;
  active: boolean;
  department: string;
}

interface UserDTO {
  id: string;
  displayName: string;
  department: string;
}

const filterActive = (users: User[]): User[] =>
  users.filter(u => u.active);

const filterByDepartment = (dept: string) =>
  (users: User[]): User[] =>
    users.filter(u => u.department === dept);

const toDTO = (users: User[]): UserDTO[] =>
  users.map(u => ({
    id: u.id,
    displayName: `${u.name} (${u.age})`,
    department: u.department,
  }));

const sortByName = (users: UserDTO[]): UserDTO[] =>
  [...users].sort((a, b) => a.displayName.localeCompare(b.displayName));

const take = (n: number) =>
  <T>(items: T[]): T[] => items.slice(0, n);

// パイプラインの構築と実行
const result = pipe(
  users,
  filterActive,
  filterByDepartment("engineering"),
  toDTO,
  sortByName,
  take(10),
);
// 型安全: result の型は UserDTO[]
```

### 5.3 Python での遅延評価パイプライン

```python
# ジェネレータを活用した遅延評価パイプライン
from typing import TypeVar, Callable, Iterator, Iterable
from itertools import islice

T = TypeVar("T")
U = TypeVar("U")

class LazyPipeline:
    """遅延評価パイプライン: 最終的に消費されるまで計算しない"""

    def __init__(self, source: Iterable):
        self._source = source

    def filter(self, predicate: Callable) -> "LazyPipeline":
        """条件に合う要素だけを通す（遅延）"""
        return LazyPipeline(x for x in self._source if predicate(x))

    def map(self, transform: Callable) -> "LazyPipeline":
        """各要素を変換する（遅延）"""
        return LazyPipeline(transform(x) for x in self._source)

    def flat_map(self, transform: Callable) -> "LazyPipeline":
        """各要素を変換してフラットにする（遅延）"""
        return LazyPipeline(
            item
            for x in self._source
            for item in transform(x)
        )

    def take(self, n: int) -> "LazyPipeline":
        """先頭n件だけ取る（遅延）"""
        return LazyPipeline(islice(self._source, n))

    def collect(self) -> list:
        """パイプラインを実行してリストに収集"""
        return list(self._source)

    def reduce(self, fn: Callable, initial=None):
        """畳み込みでパイプラインを実行"""
        from functools import reduce as _reduce
        if initial is not None:
            return _reduce(fn, self._source, initial)
        return _reduce(fn, self._source)


# 使用例: 100万件のログから最新エラーを取得
import json
from dataclasses import dataclass

@dataclass(frozen=True)
class LogEntry:
    timestamp: str
    level: str
    message: str
    service: str

def parse_log_line(line: str) -> LogEntry:
    data = json.loads(line)
    return LogEntry(**data)

# 遅延評価なので、100万行あっても先頭5件見つかった時点で停止
def get_recent_errors(log_lines: Iterable[str], service: str, limit: int = 5):
    return (
        LazyPipeline(log_lines)
        .map(parse_log_line)
        .filter(lambda entry: entry.level == "ERROR")
        .filter(lambda entry: entry.service == service)
        .take(limit)
        .collect()
    )

# メモリ効率: ファイル全体をメモリに読み込まない
# with open("app.log") as f:
#     errors = get_recent_errors(f, service="payment", limit=5)
```

### 5.4 Rust の Iterator パイプライン

```rust
// Rust: ゼロコスト抽象化によるパイプライン
// コンパイル時にループに最適化されるため、手書きのforループと同等の性能

#[derive(Debug, Clone)]
struct SalesRecord {
    product: String,
    region: String,
    amount: f64,
    quantity: u32,
}

fn top_products_by_region(records: &[SalesRecord], region: &str, top_n: usize) -> Vec<(String, f64)> {
    let mut product_totals: std::collections::HashMap<&str, f64> =
        records.iter()
            .filter(|r| r.region == region)
            .fold(std::collections::HashMap::new(), |mut acc, r| {
                *acc.entry(r.product.as_str()).or_insert(0.0) += r.amount;
                acc
            });

    let mut sorted: Vec<(String, f64)> = product_totals
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();

    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    sorted.into_iter().take(top_n).collect()
}

// 使用例
// let top = top_products_by_region(&sales_data, "Tokyo", 5);
// → [("Product A", 150000.0), ("Product B", 120000.0), ...]
```

---

## 6. 関数型エラーハンドリング

### 6.1 Result/Either型

```typescript
// Result型: 例外を使わないエラーハンドリング

type Result<T, E> =
  | { ok: true; value: T }
  | { ok: false; error: E };

// ヘルパー関数
function Ok<T>(value: T): Result<T, never> {
  return { ok: true, value };
}

function Err<E>(error: E): Result<never, E> {
  return { ok: false, error };
}

// Result のメソッドチェーン
function map<T, U, E>(
  result: Result<T, E>,
  fn: (value: T) => U
): Result<U, E> {
  return result.ok ? Ok(fn(result.value)) : result;
}

function flatMap<T, U, E>(
  result: Result<T, E>,
  fn: (value: T) => Result<U, E>
): Result<U, E> {
  return result.ok ? fn(result.value) : result;
}

// mapError: エラー型の変換
function mapError<T, E, F>(
  result: Result<T, E>,
  fn: (error: E) => F
): Result<T, F> {
  return result.ok ? result : Err(fn(result.error));
}
```

### 6.2 Result型の実用的な連鎖

```typescript
// 実用例: バリデーション連鎖
type ValidationError = { field: string; message: string };

function validateAge(age: number): Result<number, ValidationError> {
  if (age < 0 || age > 150) {
    return Err({ field: "age", message: "年齢は0〜150の範囲で指定" });
  }
  return Ok(age);
}

function validateName(name: string): Result<string, ValidationError> {
  if (name.length < 1 || name.length > 50) {
    return Err({ field: "name", message: "名前は1〜50文字" });
  }
  return Ok(name);
}

function validateEmail(email: string): Result<string, ValidationError> {
  if (!email.includes("@")) {
    return Err({ field: "email", message: "有効なメールアドレスを入力" });
  }
  return Ok(email);
}

// パイプラインでのエラーハンドリング
function createUser(
  name: string, age: number, email: string
): Result<User, ValidationError> {
  const nameResult = validateName(name);
  if (!nameResult.ok) return nameResult;

  const ageResult = validateAge(age);
  if (!ageResult.ok) return ageResult;

  const emailResult = validateEmail(email);
  if (!emailResult.ok) return emailResult;

  return Ok({
    id: generateId(),
    name: nameResult.value,
    age: ageResult.value,
    email: emailResult.value,
  });
}

// 全エラーを収集するパターン（Validation Applicative）
function validateAll<T, E>(
  results: Result<T, E>[]
): Result<T[], E[]> {
  const values: T[] = [];
  const errors: E[] = [];

  for (const result of results) {
    if (result.ok) {
      values.push(result.value);
    } else {
      errors.push(result.error);
    }
  }

  return errors.length > 0 ? Err(errors) : Ok(values);
}

// 使用例: 全てのバリデーションエラーを一度に返す
function createUserCollectErrors(
  name: string, age: number, email: string
): Result<User, ValidationError[]> {
  const allResults = validateAll([
    validateName(name) as Result<any, ValidationError>,
    validateAge(age) as Result<any, ValidationError>,
    validateEmail(email) as Result<any, ValidationError>,
  ]);

  if (!allResults.ok) return allResults;

  const [validName, validAge, validEmail] = allResults.value;
  return Ok({
    id: generateId(),
    name: validName,
    age: validAge,
    email: validEmail,
  });
}
```

### 6.3 Python での Result 型

```python
# Python: Result 型による例外なしエラーハンドリング
from dataclasses import dataclass
from typing import TypeVar, Generic, Union, Callable

T = TypeVar("T")
E = TypeVar("E")
U = TypeVar("U")

@dataclass(frozen=True)
class Ok(Generic[T]):
    value: T

    def is_ok(self) -> bool:
        return True

    def map(self, fn: Callable) -> "Result":
        return Ok(fn(self.value))

    def flat_map(self, fn: Callable) -> "Result":
        return fn(self.value)

    def unwrap_or(self, default) -> T:
        return self.value

@dataclass(frozen=True)
class Err(Generic[E]):
    error: E

    def is_ok(self) -> bool:
        return False

    def map(self, fn: Callable) -> "Result":
        return self  # エラーはそのまま伝播

    def flat_map(self, fn: Callable) -> "Result":
        return self

    def unwrap_or(self, default):
        return default

Result = Union[Ok[T], Err[E]]

# 実用例: ユーザー登録のバリデーションチェーン
def validate_email(email: str) -> Result:
    if "@" not in email:
        return Err(f"無効なメールアドレス: {email}")
    return Ok(email.lower())

def validate_password(password: str) -> Result:
    if len(password) < 8:
        return Err("パスワードは8文字以上必要です")
    if not any(c.isupper() for c in password):
        return Err("大文字を1つ以上含めてください")
    return Ok(password)

def validate_age(age: int) -> Result:
    if age < 13:
        return Err("13歳未満は登録できません")
    if age > 150:
        return Err("無効な年齢です")
    return Ok(age)

# チェーン: 最初のエラーで短絡
def register_user(email: str, password: str, age: int) -> Result:
    return (
        validate_email(email)
        .flat_map(lambda valid_email:
            validate_password(password)
            .flat_map(lambda valid_password:
                validate_age(age)
                .map(lambda valid_age: {
                    "email": valid_email,
                    "password_hash": hash(valid_password),
                    "age": valid_age,
                })
            )
        )
    )

# テスト
result = register_user("test@example.com", "SecurePass1", 25)
assert result.is_ok()
assert result.value["email"] == "test@example.com"

error_result = register_user("invalid", "short", 10)
assert not error_result.is_ok()
assert error_result.error == "無効なメールアドレス: invalid"
```

### 6.4 例外 vs Result 型の使い分け

```
例外を使うべき場面:
  ├── プログラマのミス（バグ）: IndexError, NullPointerException
  ├── 回復不能なエラー: OutOfMemoryError, StackOverflowError
  └── フレームワークが期待する慣習: Django, Spring の例外ハンドリング

Result型を使うべき場面:
  ├── ビジネスロジックの失敗: バリデーションエラー、権限不足
  ├── 予測可能な失敗: ファイル未発見、ネットワークタイムアウト
  ├── 複数のエラーを収集: フォームバリデーション
  └── エラーの型を明示: 呼び出し元にエラー処理を強制

判断フローチャート:
  エラーは予測可能か？
    ├── YES → 呼び出し元で処理すべきか？
    │          ├── YES → Result型
    │          └── NO  → 例外（上位で catch）
    └── NO  → 例外（バグ、修正すべき）
```

---

## 7. 関数型 vs オブジェクト指向の使い分け

### 7.1 比較表

| 観点 | 関数型 | オブジェクト指向 |
|------|--------|----------------|
| データと振る舞い | 分離 | 統合（カプセル化） |
| 状態管理 | 不変データ + 変換 | オブジェクトの内部状態 |
| 多態性 | パターンマッチ/高階関数 | サブタイプ多態 |
| 抽象化 | 関数の合成 | クラスの継承/合成 |
| 拡張の方向 | 新しい操作の追加が容易 | 新しい型の追加が容易 |
| 得意な領域 | データ変換、パイプライン | 状態管理、UIコンポーネント |
| テスト | 入出力のみ確認 | セットアップ + モック |
| 並行処理 | 不変性により安全 | ロック・同期が必要 |

### 7.2 Expression Problem

```
Expression Problem（型と操作の拡張ジレンマ）:

  OOP: 新しい型の追加は容易、新しい操作の追加は困難
  ┌────────────┐
  │ Shape      │  新しい Shape（Triangle）を追加 → 容易
  │  ├ Circle  │  新しい操作（area, perimeter）を追加
  │  ├ Square  │  → 全 Shape クラスを修正必要
  │  └ ???     │
  └────────────┘

  FP: 新しい操作の追加は容易、新しい型の追加は困難
  ┌────────────┐
  │ area(s)    │  新しい操作（perimeter）を追加 → 容易
  │ draw(s)    │  新しい型（Triangle）を追加
  │ ???(s)     │  → 全関数を修正必要
  └────────────┘

  解決策: Visitor パターン / Type Class / Protocol
  → 詳細は [../../design-patterns-guide/docs/02-behavioral/05-visitor.md]
```

### 7.3 ハイブリッドアプローチ

```
推奨: 関数型+OOPのハイブリッド
───────────────────────────────

  ┌─ ドメインモデル: イミュータブルなデータクラス (FP)
  │
  ├─ ビジネスロジック: 純粋関数 (FP)
  │
  ├─ アプリケーション層: サービスクラス + DI (OOP)
  │
  ├─ インフラ層: リポジトリ、外部サービス (OOP)
  │
  └─ データ変換: パイプライン (FP)
```

```typescript
// ハイブリッドアーキテクチャの実装例

// ===== ドメイン層: FP (不変データ + 純粋関数) =====
interface Product {
  readonly id: string;
  readonly name: string;
  readonly price: number;
  readonly stock: number;
}

interface CartItem {
  readonly product: Product;
  readonly quantity: number;
}

interface Cart {
  readonly items: readonly CartItem[];
  readonly appliedCoupon?: string;
}

// 純粋関数: カート操作
const addToCart = (cart: Cart, product: Product, quantity: number): Cart => ({
  ...cart,
  items: [
    ...cart.items.filter(item => item.product.id !== product.id),
    {
      product,
      quantity: (cart.items.find(i => i.product.id === product.id)?.quantity ?? 0) + quantity,
    },
  ],
});

const removeFromCart = (cart: Cart, productId: string): Cart => ({
  ...cart,
  items: cart.items.filter(item => item.product.id !== productId),
});

const calculateCartTotal = (cart: Cart): number =>
  cart.items.reduce((sum, item) => sum + item.product.price * item.quantity, 0);

const applyCoupon = (cart: Cart, couponCode: string, discountRate: number): Cart => ({
  ...cart,
  appliedCoupon: couponCode,
  items: cart.items.map(item => ({
    ...item,
    product: {
      ...item.product,
      price: Math.round(item.product.price * (1 - discountRate)),
    },
  })),
});

// 純粋関数: バリデーション
const validateCart = (cart: Cart): string[] => {
  const errors: string[] = [];
  if (cart.items.length === 0) {
    errors.push("カートが空です");
  }
  for (const item of cart.items) {
    if (item.quantity > item.product.stock) {
      errors.push(`${item.product.name}: 在庫不足 (在庫: ${item.product.stock})`);
    }
  }
  return errors;
};

// ===== アプリケーション層: OOP (DI + 副作用管理) =====
class CheckoutService {
  constructor(
    private readonly productRepo: ProductRepository,
    private readonly orderRepo: OrderRepository,
    private readonly paymentGateway: PaymentGateway,
  ) {}

  async checkout(cart: Cart): Promise<Result<string, string>> {
    // 1. 純粋: バリデーション
    const errors = validateCart(cart);
    if (errors.length > 0) {
      return Err(errors.join("; "));
    }

    // 2. 純粋: 合計計算
    const total = calculateCartTotal(cart);

    // 3. 不純: 決済
    const paymentResult = await this.paymentGateway.charge(total);
    if (!paymentResult.ok) {
      return Err("決済に失敗しました");
    }

    // 4. 不純: 注文保存
    const orderId = await this.orderRepo.save(cart, total);
    return Ok(orderId);
  }
}
```

---

## 8. アンチパターン

### 8.1 アンチパターン：隠れた副作用

```python
# NG: 一見純粋に見えるが副作用がある
def process_items(items: list[dict]) -> list[dict]:
    for item in items:
        item["processed"] = True  # 引数のリストを変更！
        item["timestamp"] = datetime.now()  # 非決定的
    return items

# OK: 新しいリストを返す純粋関数
def process_items(
    items: list[dict], current_time: datetime
) -> list[dict]:
    return [
        {**item, "processed": True, "timestamp": current_time}
        for item in items
    ]
```

**問題点**: 引数を直接変更する関数は呼び出し元のデータを壊す。時刻取得のような非決定的処理は引数で注入する。

**検出方法**: (1) 引数のオブジェクトに代入する操作（`obj["key"] = ...`, `obj.attr = ...`）を探す。(2) `datetime.now()`, `random()`, `uuid4()` 等の非決定的関数の呼び出しを探す。(3) Lint ルール（`no-param-reassign`）を設定する。

### 8.2 アンチパターン：過度な抽象化

```python
# NG: 関数型スタイルの過度な適用で読めないコード
result = reduce(
    lambda acc, f: f(acc),
    [
        partial(filter, lambda x: x > 0),
        partial(map, lambda x: x ** 2),
        partial(sorted, key=lambda x: -x),
        list,
    ],
    data,
)

# OK: 読みやすさを優先した関数型スタイル
positive_numbers = [x for x in data if x > 0]
squared = [x ** 2 for x in positive_numbers]
result = sorted(squared, reverse=True)

# または名前付き関数で意図を明確に
def keep_positive(nums): return [x for x in nums if x > 0]
def square_all(nums): return [x ** 2 for x in nums]
def sort_descending(nums): return sorted(nums, reverse=True)

result = sort_descending(square_all(keep_positive(data)))
```

**問題点**: 関数型パターンを無理に適用して可読性を犠牲にしてはいけない。チームの理解レベルに合わせ、名前付き関数で意図を明確にする。

**判断基準**: (1) ネストした lambda が2段以上 → 名前付き関数に切り出す。(2) reduce の中身がすぐに理解できない → 明示的なループか内包表記に変える。(3) チームの半数以上が読めないコード → シンプルにする。

### 8.3 アンチパターン：モナドの過度な入れ子

```typescript
// NG: Promise<Result<Option<T>>> の三重入れ子
async function getUser(
  id: string
): Promise<Result<Option<User>, DatabaseError>> {
  // 利用側が3段階のアンラップを強いられる
  const result = await getUser("123");
  if (!result.ok) {
    // DatabaseError 処理
  } else if (result.value === null) {
    // ユーザー未発見
  } else {
    // やっと User にアクセス
  }
}

// OK: 適切なレベルに統合
type GetUserError =
  | { type: "not_found"; userId: string }
  | { type: "database_error"; message: string };

async function getUser(
  id: string
): Promise<Result<User, GetUserError>> {
  // 利用側は Result の ok/error チェックのみ
  const result = await getUser("123");
  if (!result.ok) {
    switch (result.error.type) {
      case "not_found":
        return showNotFound();
      case "database_error":
        return showErrorPage();
    }
  }
  // すぐに User にアクセス
}
```

**問題点**: 型の入れ子が深すぎると、型安全性のメリットよりもボイラープレートの負担が上回る。`Option` と `Error` を Union Type に統合するなど、適切な抽象度を選ぶ。

### 8.4 アンチパターン：map/filter の乱用（パフォーマンス無視）

```typescript
// NG: 同じ配列を何度も走査（O(n) が4回）
const result = users
  .filter(u => u.active)
  .map(u => ({ ...u, name: u.name.toUpperCase() }))
  .filter(u => u.age >= 18)
  .map(u => u.name);

// OK: reduce で1回の走査に統合（パフォーマンスクリティカルな場合）
const result = users.reduce<string[]>((acc, u) => {
  if (u.active && u.age >= 18) {
    acc.push(u.name.toUpperCase());
  }
  return acc;
}, []);

// 注意: 可読性とパフォーマンスのトレードオフ
// - 数百件程度 → チェーン版で十分
// - 数万件以上、ホットパス → reduce 版を検討
// - まずプロファイリングで確認してから最適化
```

**問題点**: 関数型のチェーンは宣言的で読みやすいが、大量データでは中間配列のアロケーションがボトルネックになる場合がある。遅延評価パイプライン（セクション5.3）も選択肢。

---

## 9. 演習問題

### 演習1（基礎）: 売上データ変換パイプライン

**課題**: 以下の売上データを関数型パイプラインで処理し、カテゴリ別の月間売上サマリーを生成せよ。

```python
# 入力データ
from dataclasses import dataclass
from datetime import date

@dataclass(frozen=True)
class Sale:
    product: str
    category: str
    amount: int
    sale_date: date

sales = [
    Sale("商品A", "electronics", 15000, date(2025, 3, 1)),
    Sale("商品B", "books", 2500, date(2025, 3, 5)),
    Sale("商品C", "electronics", 8000, date(2025, 3, 10)),
    Sale("商品D", "clothing", 5000, date(2025, 3, 15)),
    Sale("商品E", "electronics", 22000, date(2025, 3, 20)),
    Sale("商品F", "books", 1800, date(2025, 3, 25)),
    Sale("商品G", "clothing", 12000, date(2025, 4, 1)),
    Sale("商品H", "electronics", 9500, date(2025, 4, 5)),
]

# 要件:
# 1. 2025年3月のデータだけを抽出
# 2. カテゴリ別に合計金額を集計
# 3. 金額の降順でソート
# 4. 結果を dict[str, int] で返す
```

**期待される出力**:

```python
{"electronics": 45000, "clothing": 5000, "books": 4300}
```

**模範解答**:

```python
from functools import reduce
from collections import defaultdict

# 純粋関数として定義
def filter_by_month(sales: list[Sale], year: int, month: int) -> list[Sale]:
    return [s for s in sales if s.sale_date.year == year and s.sale_date.month == month]

def group_by_category(sales: list[Sale]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for sale in sales:
        totals[sale.category] = totals.get(sale.category, 0) + sale.amount
    return totals

def sort_by_value_desc(data: dict[str, int]) -> dict[str, int]:
    return dict(sorted(data.items(), key=lambda x: -x[1]))

# パイプライン合成
def pipe(*functions):
    def pipeline(value):
        return reduce(lambda acc, fn: fn(acc), functions, value)
    return pipeline

monthly_summary = pipe(
    lambda s: filter_by_month(s, 2025, 3),
    group_by_category,
    sort_by_value_desc,
)

result = monthly_summary(sales)
assert result == {"electronics": 45000, "clothing": 5000, "books": 4300}
print(result)
# → {'electronics': 45000, 'clothing': 5000, 'books': 4300}
```

---

### 演習2（応用）: Functional Core / Imperative Shell でユーザー登録を実装

**課題**: 以下の要件を満たすユーザー登録システムを、Functional Core / Imperative Shell パターンで実装せよ。

```
要件:
  1. メール: @を含む、255文字以下、小文字正規化
  2. パスワード: 8文字以上、英大文字・小文字・数字をそれぞれ1つ以上含む
  3. 名前: 1〜50文字、前後の空白をトリム
  4. 年齢: 13〜150歳

Functional Core:
  - 全バリデーション関数は純粋
  - バリデーション結果は Result 型で返す
  - 全てのエラーを収集して返す（最初のエラーで停止しない）

Imperative Shell:
  - DB保存（モック可）
  - ウェルカムメール送信（モック可）
```

**期待される出力**:

```python
# 正常系
result = register("Alice", "alice@example.com", "Passw0rd", 25)
# → Ok({"id": "usr-xxx", "email": "alice@example.com", "name": "Alice"})

# エラー系（全エラー収集）
result = register("", "invalid", "short", 10)
# → Err(["名前を入力してください", "無効なメールアドレスです",
#         "パスワードは8文字以上必要です", "13歳未満は登録できません"])
```

**模範解答**:

```python
from dataclasses import dataclass, field
from typing import Union
import re
import uuid

# Result型
@dataclass(frozen=True)
class Ok:
    value: object
    def is_ok(self): return True

@dataclass(frozen=True)
class Err:
    errors: list[str]
    def is_ok(self): return False

Result = Union[Ok, Err]

# === Functional Core (純粋関数) ===

def validate_name(name: str) -> list[str]:
    errors = []
    trimmed = name.strip()
    if len(trimmed) == 0:
        errors.append("名前を入力してください")
    elif len(trimmed) > 50:
        errors.append("名前は50文字以下にしてください")
    return errors

def validate_email(email: str) -> list[str]:
    errors = []
    if "@" not in email:
        errors.append("無効なメールアドレスです")
    elif len(email) > 255:
        errors.append("メールアドレスは255文字以下にしてください")
    return errors

def validate_password(password: str) -> list[str]:
    errors = []
    if len(password) < 8:
        errors.append("パスワードは8文字以上必要です")
    if not re.search(r"[A-Z]", password):
        errors.append("パスワードに英大文字を含めてください")
    if not re.search(r"[a-z]", password):
        errors.append("パスワードに英小文字を含めてください")
    if not re.search(r"\d", password):
        errors.append("パスワードに数字を含めてください")
    return errors

def validate_age(age: int) -> list[str]:
    errors = []
    if age < 13:
        errors.append("13歳未満は登録できません")
    elif age > 150:
        errors.append("無効な年齢です")
    return errors

def validate_all(name: str, email: str, password: str, age: int) -> Result:
    """全バリデーションを実行し、全エラーを収集（純粋関数）"""
    all_errors = (
        validate_name(name)
        + validate_email(email)
        + validate_password(password)
        + validate_age(age)
    )
    if all_errors:
        return Err(all_errors)
    return Ok({
        "name": name.strip(),
        "email": email.lower(),
        "age": age,
    })

# === Imperative Shell ===

class UserService:
    def __init__(self, db, mailer):
        self.db = db
        self.mailer = mailer

    def register(self, name: str, email: str, password: str, age: int) -> Result:
        # 1. 純粋: バリデーション
        validation = validate_all(name, email, password, age)
        if not validation.is_ok():
            return validation

        user_data = validation.value

        # 2. 不純: DB保存
        user_id = str(uuid.uuid4())
        self.db.save({"id": user_id, **user_data})

        # 3. 不純: メール送信
        self.mailer.send_welcome(user_data["email"], user_data["name"])

        return Ok({"id": user_id, **user_data})

# テスト: 純粋部分はモック不要
def test_validate_all_success():
    result = validate_all("Alice", "alice@example.com", "Passw0rd", 25)
    assert result.is_ok()
    assert result.value["email"] == "alice@example.com"

def test_validate_all_collects_all_errors():
    result = validate_all("", "invalid", "short", 10)
    assert not result.is_ok()
    assert len(result.errors) == 4  # 4つのエラーが全て収集される
    assert "名前を入力してください" in result.errors
    assert "無効なメールアドレスです" in result.errors
    assert "パスワードは8文字以上必要です" in result.errors
    assert "13歳未満は登録できません" in result.errors

test_validate_all_success()
test_validate_all_collects_all_errors()
print("All tests passed!")
```

---

### 演習3（発展）: Result 型で注文処理パイプラインを実装

**課題**: 以下の注文処理を Result 型のチェーンで実装し、エラー伝播をパイプラインで処理せよ。

```
処理フロー:
  1. 在庫チェック → 在庫不足なら InsufficientStock エラー
  2. 価格計算   → 合計0円以下なら InvalidTotal エラー
  3. 割引適用   → 割引後の金額を計算
  4. 税計算     → 税込み金額を返す

全ステップが Result 型を返し、エラーはチェーン全体で伝播する
```

**期待される出力**:

```python
# 正常系
result = process_order(order, inventory)
# → Ok({"subtotal": 3500, "discount": 350, "tax": 315, "total": 3465})

# 在庫不足
result = process_order(large_order, limited_inventory)
# → Err(OrderError("insufficient_stock", "商品A: 在庫不足 (要求: 100, 在庫: 5)"))
```

**模範解答**:

```python
from dataclasses import dataclass
from typing import Union

@dataclass(frozen=True)
class OrderError:
    error_type: str
    message: str

@dataclass(frozen=True)
class OkResult:
    value: dict
    def is_ok(self): return True
    def then(self, fn):
        return fn(self.value)

@dataclass(frozen=True)
class ErrResult:
    error: OrderError
    def is_ok(self): return False
    def then(self, fn):
        return self  # エラーはそのまま伝播

OrderResult = Union[OkResult, ErrResult]

@dataclass(frozen=True)
class OrderItem:
    product_id: str
    name: str
    price: int
    quantity: int

@dataclass(frozen=True)
class OrderRequest:
    items: tuple[OrderItem, ...]
    discount_rate: float = 0.0
    tax_rate: float = 0.10

# 各ステップの純粋関数（Result を返す）

def check_stock(order: OrderRequest, inventory: dict[str, int]) -> OrderResult:
    for item in order.items:
        available = inventory.get(item.product_id, 0)
        if item.quantity > available:
            return ErrResult(OrderError(
                "insufficient_stock",
                f"{item.name}: 在庫不足 (要求: {item.quantity}, 在庫: {available})"
            ))
    subtotal = sum(i.price * i.quantity for i in order.items)
    return OkResult({"items": order.items, "subtotal": subtotal,
                      "discount_rate": order.discount_rate, "tax_rate": order.tax_rate})

def validate_total(data: dict) -> OrderResult:
    if data["subtotal"] <= 0:
        return ErrResult(OrderError("invalid_total", "合計金額が0以下です"))
    return OkResult(data)

def apply_discount(data: dict) -> OrderResult:
    subtotal = data["subtotal"]
    discount = int(subtotal * data["discount_rate"])
    discounted = subtotal - discount
    return OkResult({**data, "discount": discount, "discounted": discounted})

def apply_tax(data: dict) -> OrderResult:
    discounted = data["discounted"]
    tax = int(discounted * data["tax_rate"])
    total = discounted + tax
    return OkResult({
        "subtotal": data["subtotal"],
        "discount": data["discount"],
        "tax": tax,
        "total": total,
    })

# パイプライン実行
def process_order(order: OrderRequest, inventory: dict[str, int]) -> OrderResult:
    return (
        check_stock(order, inventory)
        .then(validate_total)
        .then(apply_discount)
        .then(apply_tax)
    )

# テスト
order = OrderRequest(
    items=(
        OrderItem("p1", "商品A", 1000, 2),
        OrderItem("p2", "商品B", 500, 3),
    ),
    discount_rate=0.1,
    tax_rate=0.10,
)

inventory = {"p1": 10, "p2": 20}
result = process_order(order, inventory)
assert result.is_ok()
assert result.value["subtotal"] == 3500
assert result.value["discount"] == 350
assert result.value["tax"] == 315
assert result.value["total"] == 3465
print(f"OK: {result.value}")

# 在庫不足テスト
limited_inventory = {"p1": 1, "p2": 20}
error_result = process_order(order, limited_inventory)
assert not error_result.is_ok()
assert error_result.error.error_type == "insufficient_stock"
print(f"Error: {error_result.error.message}")

print("All tests passed!")
# 出力:
# OK: {'subtotal': 3500, 'discount': 350, 'tax': 315, 'total': 3465}
# Error: 商品A: 在庫不足 (要求: 2, 在庫: 1)
# All tests passed!
```

---

## 10. FAQ

### Q1: 関数型プログラミングはパフォーマンスが悪いのでは？

**A**: 新しいオブジェクトの生成にコストがかかるのは事実だが、現代のGCは短命オブジェクトの処理が非常に高速。構造共有や遅延評価を使えばパフォーマンスへの影響は最小限。JVMの場合、JITコンパイラがインライン化やエスケープ解析で最適化する。ボトルネックが確認された箇所のみ可変データを使うのが現実的。

具体的な数値例として、Immutable.js の Map は100万エントリで通常の Object と比較して更新が約2倍遅いが、構造共有により変更検出（===比較）は O(1) で即座に完了する。React の shouldComponentUpdate のような場面では、不変データの方がトータルで高速になる。

### Q2: React/Reduxと関数型プログラミングの関係は？

**A**: Reactは関数型の原則を多く取り入れている。(1) コンポーネントは「props → JSX」の純粋関数、(2) Reduxは「(state, action) → newState」の純粋なリデューサ、(3) useStateのイミュータブルな状態更新、(4) useMemoの参照透過性によるメモ化。フロントエンド開発者は自然に関数型の恩恵を受けている。

さらに React 18+ の Concurrent Features は参照透過性を前提としている。レンダリングが中断・再開されてもUI が一貫するのは、レンダリング関数が純粋であるため。`StrictMode` で2回レンダリングされるのも、純粋性を検証するための仕組み。

### Q3: チームに関数型の原則をどう導入するか？

**A**: 段階的な導入ロードマップを推奨する:

1. **Week 1-2**: `map/filter/reduce` の活用から始める（for文の置き換え）
2. **Week 3-4**: 純粋関数の概念を共有し、新規コードで副作用を分離する習慣をつける
3. **Month 2**: イミュータブルなデータクラスを導入（`dataclass(frozen=True)`, `readonly`）
4. **Month 3**: Lintルールで不変性を強制（`no-param-reassign`, `prefer-const`）
5. **Month 4+**: Result型やパイプラインの導入を検討

一気にHaskellスタイルを強いるのではなく、各ステップで「テストが書きやすくなった」「バグが減った」という成果を実感させることが重要。

### Q4: 純粋関数のテストは本当にモック不要なのか？

**A**: 純粋関数のテストにはモックが一切不要。入力を渡して出力を確認するだけでよい。これが Functional Core / Imperative Shell の最大のメリット。ビジネスロジックの80%以上を純粋関数で書ければ、テストスイートの大半はシンプルな入出力テストになり、テスト実行時間も大幅に短縮される。

```python
# 純粋関数: モック不要
def test_calculate_total():
    order = Order(items=(OrderItem("p1", "A", 1000, 2),), discount_rate=0.1)
    result = calculate_total(order, tax_rate=0.1)
    assert result["total"] == 1980  # 入力→出力のみ確認

# 不純なシェル: モックが必要（だが薄い）
def test_order_service(mocker):
    mock_db = mocker.Mock()
    mock_payment = mocker.Mock(return_value=PaymentResult(success=True))
    service = OrderService(db=mock_db, payment=mock_payment, notifier=mocker.Mock())
    # ...
```

### Q5: 関数型プログラミングと依存性注入（DI）はどう関係するのか？

**A**: 関数型では DI を「関数の引数」として実現する。OOP の DI コンテナを使うこともできるが、高階関数による「関数レベルの DI」も有効。

```typescript
// OOP的DI
class OrderService {
  constructor(private repo: OrderRepository) {}
  getOrder(id: string) { return this.repo.find(id); }
}

// 関数型DI: 依存を引数（高階関数）で渡す
const getOrder = (repo: OrderRepository) => (id: string) => repo.find(id);
const getOrderFromDB = getOrder(new PostgresOrderRepository());
const getOrderFromMock = getOrder(new MockOrderRepository());  // テスト用
```

両方のアプローチを使い分けるのが現実的。サービス層は OOP の DI、ビジネスロジックは引数注入が見通しがよい。

### Q6: イベントソーシングと関数型プログラミングの関係は？

**A**: イベントソーシングは本質的に関数型のパターン。現在の状態は「初期状態 + イベント列の左畳み込み（reduce/fold）」で計算される。

```
state = reduce(apply_event, events, initial_state)

イベント: [Created, ItemAdded, ItemAdded, Discounted, Confirmed]
         ↓ fold
状態:    Order(items=2, discount=10%, status=confirmed)
```

イベントは不変（過去のイベントは変更しない）、apply_event は純粋関数（同じイベント列からは常に同じ状態が再現される）。これにより完全な監査ログ、時点復元、デバッグが容易になる。詳細は [../../system-design-guide/docs/02-architecture/](../../system-design-guide/docs/02-architecture/) を参照。

---

## 11. まとめ

| カテゴリ | ポイント |
|---------|---------|
| 純粋関数 | 同じ入力→同じ出力、副作用なし。テスト・推論が容易 |
| 参照透過性 | 関数呼び出しを結果で置換可能。メモ化・並列化の基盤 |
| 副作用分離 | Functional Core / Imperative Shell で構造化 |
| 高階関数 | 振る舞いの抽象化。デコレータ、カリー化、部分適用 |
| パイプライン | データ変換を宣言的に合成。遅延評価で大量データにも対応 |
| エラー処理 | Result/Either型で例外を使わない安全なエラー伝播 |
| 不変性 | データは変更せずコピー。並行安全、変更追跡が容易 |
| ハイブリッド | FP + OOP の適材適所。ドメイン=FP、インフラ=OOP |
| 導入戦略 | map/filter→純粋関数→不変データ→パイプラインの順で段階的に |

```
導入の成熟度モデル:

  Level 0: 命令型のみ（for, while, 状態変更）
      ↓
  Level 1: map/filter/reduce の活用
      ↓
  Level 2: 純粋関数の意識的な分離
      ↓
  Level 3: Functional Core / Imperative Shell
      ↓
  Level 4: Result型 + パイプライン合成
      ↓
  Level 5: 型レベルプログラミング、Phantom Types 等
```

---

## 次に読むべきガイド

- [00-immutability.md](./00-immutability.md) — イミュータビリティの原則（不変データ構造と構造共有の詳細）
- [01-composition-over-inheritance.md](./01-composition-over-inheritance.md) — 継承より合成の原則（Strategy, Decorator との連携）
- [03-api-design.md](./03-api-design.md) — API設計（関数型エラーハンドリングのAPI適用）
- [../01-practices/04-testing-principles.md](../01-practices/04-testing-principles.md) — テスト原則（純粋関数のテスト戦略）
- [../../design-patterns-guide/docs/03-functional/](../../design-patterns-guide/docs/03-functional/) — 関数型デザインパターン（Monad, Functor）
- [../../design-patterns-guide/docs/02-behavioral/00-strategy.md](../../design-patterns-guide/docs/02-behavioral/00-strategy.md) — Strategyパターン（高階関数との対比）
- [../../system-design-guide/docs/00-fundamentals/](../../system-design-guide/docs/00-fundamentals/) — システム設計の基礎（関数型アーキテクチャの全体像）

---

## 参考文献

1. Michael Feathers, **"Functional Design"** — 関数型設計の実践ガイド
2. Eric Normand, **"Grokking Simplicity"** (Manning, 2021) — 実用的な関数型プログラミング入門
3. Gary Bernhardt, **"Functional Core, Imperative Shell"** — https://www.destroyallsoftware.com/screencasts/catalog/functional-core-imperative-shell
4. Martin Fowler, **"Collection Pipeline"** — https://martinfowler.com/articles/collection-pipeline/
5. Scott Wlaschin, **"Domain Modeling Made Functional"** (Pragmatic, 2018) — 関数型ドメイン駆動設計
6. Enrico Buonanno, **"Functional Programming in C#"** (Manning, 2nd ed., 2022) — 実践的な関数型プログラミング
7. Brian Lonsdorf, **"Professor Frisby's Mostly Adequate Guide to Functional Programming"** — https://mostly-adequate.gitbook.io/mostly-adequate-guide/ — 無料のオンライン FP ガイド
8. Rust by Example, **"Iterators"** — https://doc.rust-lang.org/rust-by-example/trait/iter.html — Rust のイテレータパイプライン
9. Haskell Wiki, **"Functional programming"** — https://wiki.haskell.org/Functional_programming — 関数型プログラミングの理論的基盤
10. Mark Seemann, **"From dependency injection to dependency rejection"** — https://blog.ploeh.dk/2017/01/27/from-dependency-injection-to-dependency-rejection/ — FP 視点の DI
