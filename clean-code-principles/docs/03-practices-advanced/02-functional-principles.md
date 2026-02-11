# 関数型プログラミングの原則をクリーンコードに活かす

> 純粋関数、副作用分離、高階関数、パイプラインなど、関数型プログラミングの核心的な原則をクリーンコードの文脈で解説し、より安全で保守しやすいコードの実現を支援する。

---

## この章で学ぶこと

1. **純粋関数と参照透過性**の概念を理解し、テスト可能で予測可能な関数を設計できる
2. **副作用の分離・高階関数・パイプライン**を活用して、読みやすく合成可能なコードを書ける
3. **関数型の原則をオブジェクト指向や日常の開発**に統合し、実用的なクリーンコードを実現できる

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

### 1.2 命令型 vs 宣言型（関数型）

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

### 2.3 参照透過性の活用

```typescript
// 参照透過性: 関数呼び出しをその結果で置き換えても意味が変わらない

// 純粋関数: 参照透過
function add(a: number, b: number): number {
  return a + b;
}

// add(2, 3) は常に 5 なので、コード中の add(2, 3) を 5 に置換可能
const x = add(2, 3) * add(2, 3);
const y = 5 * 5;  // 完全に等価

// これが可能にすること:
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

// 2. 遅延評価（必要になるまで計算しない）
// 3. 並列実行（順序に依存しない）
// 4. テストの独立性（セットアップ不要）
```

---

## 3. 副作用の分離

### 3.1 純粋コアと不純シェル

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

### 4.2 実用的な高階関数パターン

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
| 得意な領域 | データ変換、パイプライン | 状態管理、UIコンポーネント |
| テスト | 入出力のみ確認 | セットアップ + モック |

### 7.2 ハイブリッドアプローチ

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

---

## 9. FAQ

### Q1: 関数型プログラミングはパフォーマンスが悪いのでは？

**A**: 新しいオブジェクトの生成にコストがかかるのは事実だが、現代のGCは短命オブジェクトの処理が非常に高速。構造共有や遅延評価を使えばパフォーマンスへの影響は最小限。JVMの場合、JITコンパイラがインライン化やエスケープ解析で最適化する。ボトルネックが確認された箇所のみ可変データを使うのが現実的。

### Q2: React/Reduxと関数型プログラミングの関係は？

**A**: Reactは関数型の原則を多く取り入れている。(1) コンポーネントは「props → JSX」の純粋関数、(2) Reduxは「(state, action) → newState」の純粋なリデューサ、(3) useStateのイミュータブルな状態更新、(4) useMemoの参照透過性によるメモ化。フロントエンド開発者は自然に関数型の恩恵を受けている。

### Q3: チームに関数型の原則をどう導入するか？

**A**: (1) まずmap/filter/reduceの活用から始める（for文の置き換え）、(2) 純粋関数の概念を共有し、副作用を分離する習慣をつける、(3) イミュータブルなデータクラスを導入、(4) Lintルールで不変性を強制（no-param-reassign等）。一気にHaskellスタイルを強いるのではなく、段階的に導入する。

---

## 10. まとめ

| カテゴリ | ポイント |
|---------|---------|
| 純粋関数 | 同じ入力→同じ出力、副作用なし。テスト・推論が容易 |
| 副作用分離 | Functional Core / Imperative Shell で構造化 |
| 高階関数 | 振る舞いの抽象化。デコレータ、カリー化、部分適用 |
| パイプライン | データ変換を宣言的に合成。読みやすく拡張しやすい |
| エラー処理 | Result/Either型で例外を使わない安全なエラー伝播 |
| 不変性 | データは変更せずコピー。並行安全、変更追跡が容易 |
| 導入戦略 | map/filter→純粋関数→不変データ→パイプラインの順で段階的に |

---

## 次に読むべきガイド

- [00-immutability.md](./00-immutability.md) — イミュータビリティの原則
- [01-composition-over-inheritance.md](./01-composition-over-inheritance.md) — 継承より合成の原則
- リアクティブプログラミング — ストリームと関数型の融合

---

## 参考文献

1. Michael Feathers, "Functional Design" — 関数型設計の実践ガイド
2. Eric Normand, "Grokking Simplicity" — 実用的な関数型プログラミング入門
3. Gary Bernhardt, "Functional Core, Imperative Shell" — https://www.destroyallsoftware.com/screencasts/catalog/functional-core-imperative-shell
4. Martin Fowler, "Collection Pipeline" — https://martinfowler.com/articles/collection-pipeline/
5. Scott Wlaschin, "Domain Modeling Made Functional" — 関数型ドメイン駆動設計
