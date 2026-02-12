# 継承より合成の原則

> なぜ合成（コンポジション）が継承よりも柔軟で保守しやすいか。デザインパターンとの関係、言語別の実装手法、リファクタリング戦略を体系的に解説する。

---

## この章で学ぶこと

1. **継承の問題点と合成の優位性**を理論的に理解し、設計判断の根拠を説明できる
2. **Strategy、Decorator、Delegate**など合成ベースのデザインパターンを実装できる
3. **既存の継承階層を合成にリファクタリング**する手法を習得し、保守性を改善できる
4. **言語ごとの合成の実装手法**（Python Mixin、TypeScript Mixin、Rust Trait、Go Interface）を使い分けられる
5. **継承が適切な場面と不適切な場面**を正確に判断し、設計レビューで根拠を説明できる

---

## 前提知識

このガイドを理解するには、以下の知識が必要です。

| 前提知識 | 参照先 |
|---------|-------|
| オブジェクト指向の基本（クラス、継承、ポリモーフィズム） | [00-principles/02-solid.md](../00-principles/02-solid.md) |
| SOLID原則（特にLSP、OCP、DIP） | [00-principles/02-solid.md](../00-principles/02-solid.md) |
| インターフェースと抽象クラスの違い | `02-programming/` |
| デザインパターンの基本概念 | `design-patterns-guide/docs/00-creational/` |
| 依存性注入（DI）の基礎 | [01-practices/03-dependency-injection.md](../01-practices/03-dependency-injection.md) |

---

## 1. なぜ「継承より合成」なのか

### 1.1 継承の問題点

```
┌──────────────────────────────────────────────────────┐
│              継承の5つの問題点                          │
├──────────────────────────────────────────────────────┤
│                                                      │
│  1. 脆い基底クラス問題 (Fragile Base Class)           │
│     基底クラスの変更が全子クラスに波及                  │
│                                                      │
│  2. ダイヤモンド問題 (Diamond Problem)                │
│     多重継承で同名メソッドが衝突                       │
│                                                      │
│  3. 深い階層 → 理解困難                               │
│     A → B → C → D → E... どこに何がある？             │
│                                                      │
│  4. is-a 関係の強制                                   │
│     「正方形 is-a 長方形」は本当に成立するか？          │
│                                                      │
│  5. 単一継承の制限                                    │
│     1つの軸でしか分類できない                          │
│     (鳥 is-a 動物 だが、鳥 is-a 飛行体 でもある)       │
│                                                      │
└──────────────────────────────────────────────────────┘
```

### 1.2 脆い基底クラス問題の具体例

```java
// 脆い基底クラス問題の典型例（Effective Java Item 18より）

// 基底クラス: HashSetを継承して追加回数を記録するクラス
public class CountingHashSet<E> extends HashSet<E> {
    private int addCount = 0;

    @Override
    public boolean add(E e) {
        addCount++;
        return super.add(e);
    }

    @Override
    public boolean addAll(Collection<? extends E> c) {
        addCount += c.size();
        return super.addAll(c); // ← ここが問題！
    }

    public int getAddCount() {
        return addCount;
    }
}

// 使用
CountingHashSet<String> s = new CountingHashSet<>();
s.addAll(List.of("A", "B", "C"));
s.getAddCount(); // 期待: 3, 実際: 6！

// なぜか？
// HashSet.addAll() は内部で add() を呼ぶ。
// CountingHashSet.addAll() で addCount += 3 した後、
// super.addAll() が内部で add() を3回呼び、
// それぞれで addCount++ されるため、合計6になる。
//
// しかもこれはHashSetの「実装の詳細」であり、
// 仕様には記載されていない。
// 将来のJavaバージョンで内部実装が変われば、
// このコードは別のバグを生む可能性がある。
```

### 1.3 継承 vs 合成の比較

```
継承 (is-a):                     合成 (has-a):
─────────                       ─────────

  Animal                         ┌─────────┐
    │                            │ Duck    │
    ├── Bird                     │         │
    │    ├── Duck                │ swim: SwimBehavior
    │    ├── Penguin             │ fly:  FlyBehavior
    │    └── Eagle               │ sound: QuackBehavior
    │                            └─────────┘
    └── Fish
         └── FlyingFish          ← FlyingFishは？
                                     Bird? Fish? 両方？
                                     → 合成なら簡単に組合せ可能
```

| 比較軸 | 継承 | 合成 |
|--------|------|------|
| 結合度 | 高い（親子密結合） | 低い（インターフェース経由） |
| 柔軟性 | 低い（コンパイル時固定） | 高い（実行時変更可能） |
| 再利用性 | 限定的（階層に依存） | 高い（任意に組合せ） |
| テスト容易性 | 困難（親クラス依存） | 容易（モック可能） |
| 理解しやすさ | 深い階層は困難 | フラットで明快 |
| 多重分類 | 困難（単一継承の壁） | 容易（複数の振る舞いを持てる） |
| 変更の影響範囲 | 大きい（全子クラスに波及） | 小さい（変更箇所に局所化） |
| カプセル化 | 破壊される（protected公開） | 維持される（内部実装を隠蔽） |

### 1.4 リスコフの置換原則（LSP）と継承の限界

```
正方形-長方形問題: LSP違反の典型例

  Rectangle (長方形)
    │
    └── Square (正方形)

  長方形のコード:
    rect.setWidth(5)
    rect.setHeight(3)
    assert rect.area() == 15  // 期待通り

  正方形でも動くか？
    square.setWidth(5)   // → 内部でheightも5に
    square.setHeight(3)  // → 内部でwidthも3に
    assert square.area() == 15  // 失敗！ area() = 9

  問題: 正方形は長方形の「振る舞い」を継承できない
  → is-a 関係が数学的には成立しても、プログラム上は成立しない

  合成による解決:
    Shape { dimensions: Dimensions }
    Dimensions は Width×Height / Side など異なる実装
```

```python
# LSP違反の例と修正

# NG: 継承でLSP違反
class Rectangle:
    def __init__(self, width: float, height: float):
        self._width = width
        self._height = height

    @property
    def width(self) -> float:
        return self._width

    @width.setter
    def width(self, value: float) -> None:
        self._width = value

    @property
    def height(self) -> float:
        return self._height

    @height.setter
    def height(self, value: float) -> None:
        self._height = value

    def area(self) -> float:
        return self._width * self._height

class Square(Rectangle):  # LSP違反！
    def __init__(self, side: float):
        super().__init__(side, side)

    @Rectangle.width.setter
    def width(self, value: float) -> None:
        self._width = value
        self._height = value  # 幅を変えると高さも変わる

    @Rectangle.height.setter
    def height(self, value: float) -> None:
        self._width = value
        self._height = value

# OK: 合成で解決
from abc import ABC, abstractmethod
from dataclasses import dataclass

class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        pass

@dataclass(frozen=True)
class Rectangle(Shape):
    width: float
    height: float

    def area(self) -> float:
        return self.width * self.height

@dataclass(frozen=True)
class Square(Shape):
    side: float

    def area(self) -> float:
        return self.side ** 2

# Shape のリストに対して area() を安全に呼べる
shapes: list[Shape] = [Rectangle(5, 3), Square(4)]
total_area = sum(s.area() for s in shapes)  # 15 + 16 = 31
```

---

## 2. 合成ベースのデザインパターン

### 2.1 Strategy パターン

```python
# Strategy パターン: 振る舞いを外部から注入
from abc import ABC, abstractmethod
from dataclasses import dataclass

# === 戦略インターフェース ===
class SortStrategy(ABC):
    @abstractmethod
    def sort(self, data: list) -> list:
        pass

    @abstractmethod
    def name(self) -> str:
        pass

class QuickSort(SortStrategy):
    def sort(self, data: list) -> list:
        if len(data) <= 1:
            return data
        pivot = data[0]
        left = [x for x in data[1:] if x <= pivot]
        right = [x for x in data[1:] if x > pivot]
        return self.sort(left) + [pivot] + self.sort(right)

    def name(self) -> str:
        return "QuickSort"

class MergeSort(SortStrategy):
    def sort(self, data: list) -> list:
        if len(data) <= 1:
            return data
        mid = len(data) // 2
        left = self.sort(data[:mid])
        right = self.sort(data[mid:])
        return self._merge(left, right)

    def _merge(self, left, right):
        result = []
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i]); i += 1
            else:
                result.append(right[j]); j += 1
        return result + left[i:] + right[j:]

    def name(self) -> str:
        return "MergeSort"

class InsertionSort(SortStrategy):
    """小さなデータセットに最適"""
    def sort(self, data: list) -> list:
        result = list(data)
        for i in range(1, len(result)):
            key = result[i]
            j = i - 1
            while j >= 0 and result[j] > key:
                result[j + 1] = result[j]
                j -= 1
            result[j + 1] = key
        return result

    def name(self) -> str:
        return "InsertionSort"

# === 合成: 戦略を持つクラス ===
@dataclass
class DataProcessor:
    sort_strategy: SortStrategy  # has-a（合成）

    def process(self, data: list) -> list:
        print(f"  使用アルゴリズム: {self.sort_strategy.name()}")
        return self.sort_strategy.sort(data)

# === 実行時に戦略を選択 ===
def choose_strategy(data_size: int) -> SortStrategy:
    """データサイズに応じて最適な戦略を選択"""
    if data_size < 50:
        return InsertionSort()
    elif data_size < 10000:
        return QuickSort()
    else:
        return MergeSort()

data = [3, 1, 4, 1, 5, 9, 2, 6]
strategy = choose_strategy(len(data))
processor = DataProcessor(sort_strategy=strategy)
result = processor.process(data)
print(f"  結果: {result}")

# 戦略の切り替えも簡単
processor.sort_strategy = MergeSort()
result2 = processor.process(data)
```

### 2.2 Decorator パターン

```typescript
// Decorator パターン: 機能を動的に追加
// 継承ではなく合成で機能を積み重ねる

interface Logger {
  log(message: string): void;
}

// === 基本実装 ===
class ConsoleLogger implements Logger {
  log(message: string): void {
    console.log(message);
  }
}

class FileLogger implements Logger {
  constructor(private filePath: string) {}

  log(message: string): void {
    // fs.appendFileSync(this.filePath, message + '\n');
    console.log(`[File: ${this.filePath}] ${message}`);
  }
}

// === デコレータ: タイムスタンプ追加 ===
class TimestampDecorator implements Logger {
  constructor(private inner: Logger) {} // 合成

  log(message: string): void {
    const timestamp = new Date().toISOString();
    this.inner.log(`[${timestamp}] ${message}`);
  }
}

// === デコレータ: ログレベル追加 ===
class LevelDecorator implements Logger {
  constructor(
    private inner: Logger,
    private level: string = "INFO"
  ) {}

  log(message: string): void {
    this.inner.log(`[${this.level}] ${message}`);
  }
}

// === デコレータ: JSON形式変換 ===
class JsonDecorator implements Logger {
  constructor(private inner: Logger) {}

  log(message: string): void {
    const json = JSON.stringify({
      message,
      timestamp: new Date().toISOString(),
    });
    this.inner.log(json);
  }
}

// === デコレータ: フィルタリング ===
class FilterDecorator implements Logger {
  constructor(
    private inner: Logger,
    private minLevel: string
  ) {}

  private levelOrder: Record<string, number> = {
    DEBUG: 0,
    INFO: 1,
    WARN: 2,
    ERROR: 3,
  };

  log(message: string): void {
    // ログレベルのフィルタリングロジック
    this.inner.log(message);
  }
}

// === 使用: デコレータを自由に組み合わせ ===

// 組み合わせ1: タイムスタンプ + レベル付きコンソールログ
const logger1 = new TimestampDecorator(
  new LevelDecorator(new ConsoleLogger(), "DEBUG")
);
logger1.log("テストメッセージ");
// 出力: [2026-01-15T10:30:00.000Z] [DEBUG] テストメッセージ

// 組み合わせ2: JSONフォーマットのファイルログ
const logger2 = new JsonDecorator(new FileLogger("/var/log/app.log"));
logger2.log("エラー発生");

// 組み合わせ3: 本番環境用（タイムスタンプ + JSON）
const prodLogger = new TimestampDecorator(
  new JsonDecorator(new ConsoleLogger())
);

// ★ 重要: 継承でこれを実現しようとすると
// ConsoleLogger, FileLogger × Timestamp, Level, Json, Filter
// = 2 × 4 = 8 クラスが必要（さらに組み合わせで爆発）
// 合成なら必要なデコレータを組み合わせるだけ
```

### 2.3 委譲（Delegation）パターン

```java
// 委譲パターン: 継承の代わりに内部オブジェクトに処理を委譲

// === NG: 継承でStackを実装 ===
// public class Stack<T> extends ArrayList<T> {
//     // ArrayListの全メソッドが公開されてしまう
//     // add(index, element)でスタックの規約が破れる
//     // sort() や set() など不要なメソッドも使える
// }

// === OK: 合成+委譲でStackを実装 ===
public class Stack<T> {
    private final List<T> elements = new ArrayList<>(); // has-a

    public void push(T item) {
        Objects.requireNonNull(item, "null は push できません");
        elements.add(item);  // 委譲
    }

    public T pop() {
        if (elements.isEmpty()) {
            throw new EmptyStackException();
        }
        return elements.remove(elements.size() - 1);  // 委譲
    }

    public T peek() {
        if (elements.isEmpty()) {
            throw new EmptyStackException();
        }
        return elements.get(elements.size() - 1);  // 委譲
    }

    public boolean isEmpty() {
        return elements.isEmpty();  // 委譲
    }

    public int size() {
        return elements.size();  // 委譲
    }

    // ArrayListのadd(index)やsort()は公開されない
    // → スタックの不変条件が保証される
}

// === 実用例: ドメインオブジェクトのコレクション ===
public class OrderList {
    private final List<Order> orders = new ArrayList<>();

    public void place(Order order) {
        if (!order.isValid()) {
            throw new IllegalArgumentException("不正な注文");
        }
        orders.add(order);
    }

    public List<Order> findByStatus(OrderStatus status) {
        return orders.stream()
            .filter(o -> o.getStatus() == status)
            .toList(); // 不変リストを返す
    }

    public int totalAmount() {
        return orders.stream()
            .mapToInt(Order::getAmount)
            .sum();
    }

    // List の remove(), set(), sort() は公開しない
    // → ビジネスルールに沿った操作のみ許可
}
```

### 2.4 合成による依存性注入（DI）

```python
# 合成 + DI: テスト容易で柔軟な設計
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

# === ポートの定義（インターフェース） ===
class UserRepository(Protocol):
    def find_by_id(self, user_id: str) -> dict | None: ...
    def save(self, user: dict) -> None: ...

class EmailSender(Protocol):
    def send(self, to: str, subject: str, body: str) -> bool: ...

class Logger(Protocol):
    def info(self, message: str) -> None: ...
    def error(self, message: str) -> None: ...

# === アダプタの実装 ===
class PostgresUserRepository:
    def __init__(self, connection_string: str):
        self.conn_str = connection_string

    def find_by_id(self, user_id: str) -> dict | None:
        # 実際のDB操作
        return {"id": user_id, "name": "テスト", "email": "test@example.com"}

    def save(self, user: dict) -> None:
        # 実際のDB操作
        pass

class SmtpEmailSender:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

    def send(self, to: str, subject: str, body: str) -> bool:
        # 実際のメール送信
        return True

class ConsoleLogger:
    def info(self, message: str) -> None:
        print(f"[INFO] {message}")

    def error(self, message: str) -> None:
        print(f"[ERROR] {message}")

# === サービス: 合成で依存を注入 ===
@dataclass
class UserService:
    """ユーザーサービス（合成で依存を持つ）"""
    repository: UserRepository    # has-a
    email_sender: EmailSender     # has-a
    logger: Logger                # has-a

    def register(self, name: str, email: str) -> dict:
        self.logger.info(f"ユーザー登録開始: {name}")

        user = {"id": f"user-{hash(email)}", "name": name, "email": email}
        self.repository.save(user)

        self.email_sender.send(
            to=email,
            subject="登録完了",
            body=f"{name}さん、登録ありがとうございます。"
        )

        self.logger.info(f"ユーザー登録完了: {user['id']}")
        return user

# === 本番環境 ===
prod_service = UserService(
    repository=PostgresUserRepository("postgresql://localhost/mydb"),
    email_sender=SmtpEmailSender("smtp.example.com", 587),
    logger=ConsoleLogger(),
)

# === テスト用: モックを注入 ===
class MockUserRepository:
    def __init__(self):
        self.saved_users = []

    def find_by_id(self, user_id: str) -> dict | None:
        return next((u for u in self.saved_users if u["id"] == user_id), None)

    def save(self, user: dict) -> None:
        self.saved_users.append(user)

class MockEmailSender:
    def __init__(self):
        self.sent_emails = []

    def send(self, to: str, subject: str, body: str) -> bool:
        self.sent_emails.append({"to": to, "subject": subject, "body": body})
        return True

class NullLogger:
    def info(self, message: str) -> None: pass
    def error(self, message: str) -> None: pass

# テスト
mock_repo = MockUserRepository()
mock_email = MockEmailSender()
test_service = UserService(
    repository=mock_repo,
    email_sender=mock_email,
    logger=NullLogger(),
)

user = test_service.register("田中", "tanaka@example.com")
assert len(mock_repo.saved_users) == 1
assert len(mock_email.sent_emails) == 1
assert mock_email.sent_emails[0]["to"] == "tanaka@example.com"
```

---

## 3. 合成パターン一覧と比較

### 3.1 パターン比較表

| パターン | 目的 | 構造 | 使用場面 |
|---------|------|------|---------|
| Strategy | 振る舞いの切替 | コンテキスト has-a 戦略 | アルゴリズム選択 |
| Decorator | 機能の動的追加 | デコレータ has-a コンポーネント | ミドルウェア、ログ |
| Delegate | 実装の委譲 | ラッパー has-a 被委譲 | APIの制限公開 |
| Observer | イベント通知 | サブジェクト has-a オブザーバ群 | イベント駆動 |
| Composite | ツリー構造 | ノード has-a 子ノード群 | UI、ファイルシステム |
| State | 状態遷移 | コンテキスト has-a 状態 | ステートマシン |
| Bridge | 抽象と実装の分離 | 抽象 has-a 実装 | クロスプラットフォーム |
| Adapter | インターフェース変換 | アダプタ has-a 被適応者 | レガシー統合 |

### 3.2 合成の実現手法比較

| 手法 | 言語 | 特徴 | 適用場面 |
|------|------|------|---------|
| インターフェース + DI | Java, TypeScript | 最も型安全 | 大規模システム |
| Protocol | Python | ダックタイピング | Python全般 |
| Trait | Rust, Scala | ゼロコスト抽象 | パフォーマンス重視 |
| Mixin | TypeScript, Python | 手軽に機能追加 | 小規模な機能追加 |
| Higher-Order Function | 全言語 | 最も軽量 | 関数レベルの合成 |
| Interface embedding | Go | 暗黙のインターフェース | Go全般 |

### 3.3 Mixin/Trait パターン（多重合成）

```typescript
// TypeScript: Mixinで合成的に機能を追加

type Constructor<T = {}> = new (...args: any[]) => T;

// === Mixin: シリアライズ機能 ===
function Serializable<TBase extends Constructor>(Base: TBase) {
  return class extends Base {
    serialize(): string {
      return JSON.stringify(this);
    }

    static deserialize<T>(json: string): T {
      return JSON.parse(json) as T;
    }
  };
}

// === Mixin: バリデーション機能 ===
function Validatable<TBase extends Constructor>(Base: TBase) {
  return class extends Base {
    private validationErrors: string[] = [];

    validate(): boolean {
      this.validationErrors = [];
      return this.validationErrors.length === 0;
    }

    getErrors(): string[] {
      return [...this.validationErrors];
    }

    protected addError(error: string): void {
      this.validationErrors.push(error);
    }
  };
}

// === Mixin: 監査ログ機能 ===
function Auditable<TBase extends Constructor>(Base: TBase) {
  return class extends Base {
    createdAt: Date = new Date();
    updatedAt: Date = new Date();
    createdBy: string = "";
    updatedBy: string = "";

    markCreated(user: string): void {
      this.createdBy = user;
      this.createdAt = new Date();
    }

    markUpdated(user: string): void {
      this.updatedBy = user;
      this.updatedAt = new Date();
    }
  };
}

// === Mixin: イベント発行機能 ===
function EventEmittable<TBase extends Constructor>(Base: TBase) {
  return class extends Base {
    private listeners: Map<string, Function[]> = new Map();

    on(event: string, handler: Function): void {
      const handlers = this.listeners.get(event) || [];
      this.listeners.set(event, [...handlers, handler]);
    }

    emit(event: string, ...args: any[]): void {
      const handlers = this.listeners.get(event) || [];
      handlers.forEach((h) => h(...args));
    }
  };
}

// === 基底クラス ===
class BaseEntity {
  constructor(public id: string) {}
}

// === 合成: 必要な機能を組み合わせる ===
class User extends Auditable(
  Validatable(Serializable(EventEmittable(BaseEntity)))
) {
  constructor(id: string, public name: string, public email: string) {
    super(id);
  }
}

const user = new User("1", "田中", "tanaka@example.com");
user.markCreated("admin");
user.on("updated", () => console.log("ユーザーが更新されました"));
const json = user.serialize();
console.log(user.validate());
```

### 3.4 Rust Trait による合成

```rust
// Rust: Traitによるゼロコスト合成

// Traitの定義（インターフェース相当）
trait Drawable {
    fn draw(&self);
    fn bounding_box(&self) -> (f64, f64, f64, f64);
}

trait Resizable {
    fn resize(&mut self, factor: f64);
}

trait Serializable {
    fn serialize(&self) -> String;
    fn deserialize(data: &str) -> Self where Self: Sized;
}

// 構造体がTraitを実装（継承ではなく合成的）
struct Circle {
    x: f64,
    y: f64,
    radius: f64,
}

impl Drawable for Circle {
    fn draw(&self) {
        println!("Drawing circle at ({}, {}) r={}", self.x, self.y, self.radius);
    }

    fn bounding_box(&self) -> (f64, f64, f64, f64) {
        (
            self.x - self.radius,
            self.y - self.radius,
            self.x + self.radius,
            self.y + self.radius,
        )
    }
}

impl Resizable for Circle {
    fn resize(&mut self, factor: f64) {
        self.radius *= factor;
    }
}

impl Serializable for Circle {
    fn serialize(&self) -> String {
        format!("circle:{},{},{}", self.x, self.y, self.radius)
    }

    fn deserialize(data: &str) -> Self {
        let parts: Vec<f64> = data.strip_prefix("circle:")
            .unwrap()
            .split(',')
            .map(|s| s.parse().unwrap())
            .collect();
        Circle { x: parts[0], y: parts[1], radius: parts[2] }
    }
}

// Trait境界で複数のTraitを要求（合成的な型制約）
fn save_drawable<T: Drawable + Serializable>(item: &T) {
    item.draw();
    let data = item.serialize();
    println!("Saved: {}", data);
}

// Trait Object で動的ディスパッチ（多態性）
fn draw_all(items: &[Box<dyn Drawable>]) {
    for item in items {
        item.draw();
    }
}
```

### 3.5 Go Interface Embedding（合成的インターフェース）

```go
// Go: 暗黙的インターフェースと構造体埋め込みによる合成

// 小さなインターフェースを定義
type Reader interface {
    Read(p []byte) (n int, err error)
}

type Writer interface {
    Write(p []byte) (n int, err error)
}

type Closer interface {
    Close() error
}

// インターフェースの合成（埋め込み）
type ReadWriter interface {
    Reader
    Writer
}

type ReadWriteCloser interface {
    Reader
    Writer
    Closer
}

// 構造体の合成（埋め込み）
type Logger struct {
    prefix string
}

func (l *Logger) Log(msg string) {
    fmt.Printf("[%s] %s\n", l.prefix, msg)
}

type Validator struct{}

func (v *Validator) Validate(data string) error {
    if data == "" {
        return fmt.Errorf("data is empty")
    }
    return nil
}

// 合成: 埋め込みで機能を組み合わせる
type UserService struct {
    Logger     // 埋め込み: UserService.Log() が使える
    Validator  // 埋め込み: UserService.Validate() が使える
    repo UserRepository
}

func NewUserService(repo UserRepository) *UserService {
    return &UserService{
        Logger:    Logger{prefix: "UserService"},
        Validator: Validator{},
        repo:      repo,
    }
}

func (s *UserService) Create(name string) error {
    if err := s.Validate(name); err != nil {
        s.Log("Validation failed: " + err.Error())
        return err
    }
    s.Log("Creating user: " + name)
    return s.repo.Save(name)
}
```

---

## 4. 継承から合成へのリファクタリング

### 4.1 リファクタリング手順

```
Step 1: 継承階層を分析
──────────────────────
  Base
   ├── ChildA (override: methodX, methodY)
   ├── ChildB (override: methodX, methodZ)
   └── ChildC (override: methodY, methodZ)

Step 2: 振る舞いの軸を特定
──────────────────────────
  methodX の振る舞い → BehaviorX インターフェース
  methodY の振る舞い → BehaviorY インターフェース
  methodZ の振る舞い → BehaviorZ インターフェース

Step 3: インターフェースと実装を抽出
──────────────────────────────────
  interface BehaviorX { doX(); }
  class BehaviorXImpl1 implements BehaviorX { ... }
  class BehaviorXImpl2 implements BehaviorX { ... }

Step 4: 合成に書き換え
─────────────────────
  class Entity:
      behaviorX: BehaviorX
      behaviorY: BehaviorY
      behaviorZ: BehaviorZ

  entityA = Entity(BehaviorXImpl1, BehaviorYImpl1, default)
  entityB = Entity(BehaviorXImpl1, default, BehaviorZImpl1)
  entityC = Entity(default, BehaviorYImpl1, BehaviorZImpl1)

Step 5: ファクトリで構成を管理
──────────────────────────────
  EntityFactory.createTypeA() → 適切な組み合わせを返す
```

### 4.2 具体例：通知システムのリファクタリング

```python
# Before: 継承ベースの通知システム（問題あり）

# class Notification:
#     def send(self, message): ...
#     def format(self, message): return message
#
# class EmailNotification(Notification):
#     def send(self, message): ...    # メール送信
#     def format(self, message): ...  # HTML形式
#
# class SlackNotification(Notification):
#     def send(self, message): ...    # Slack送信
#     def format(self, message): ...  # Markdown形式
#
# class UrgentEmailNotification(EmailNotification):
#     def format(self, message): ...  # HTML + 赤文字
#
# class UrgentSlackNotification(SlackNotification):
#     def format(self, message): ...  # Markdown + :alert:
#
# → 配送方法 × フォーマット × 緊急度で組み合わせ爆発！
# → 3配送 × 3フォーマット × 3緊急度 = 27クラスが必要

# After: 合成ベースのリファクタリング
from abc import ABC, abstractmethod
from dataclasses import dataclass

# === 配送戦略 ===
class DeliveryChannel(ABC):
    @abstractmethod
    def deliver(self, formatted_message: str, recipient: str) -> bool:
        pass

    @abstractmethod
    def channel_name(self) -> str:
        pass

class EmailChannel(DeliveryChannel):
    def deliver(self, formatted_message: str, recipient: str) -> bool:
        print(f"Email to {recipient}: {formatted_message}")
        return True

    def channel_name(self) -> str:
        return "Email"

class SlackChannel(DeliveryChannel):
    def __init__(self, webhook_url: str = ""):
        self.webhook_url = webhook_url

    def deliver(self, formatted_message: str, recipient: str) -> bool:
        print(f"Slack to {recipient}: {formatted_message}")
        return True

    def channel_name(self) -> str:
        return "Slack"

class SMSChannel(DeliveryChannel):
    def deliver(self, formatted_message: str, recipient: str) -> bool:
        print(f"SMS to {recipient}: {formatted_message}")
        return True

    def channel_name(self) -> str:
        return "SMS"

# === フォーマット戦略 ===
class Formatter(ABC):
    @abstractmethod
    def format(self, message: str) -> str:
        pass

class PlainFormatter(Formatter):
    def format(self, message: str) -> str:
        return message

class HTMLFormatter(Formatter):
    def format(self, message: str) -> str:
        return f"<html><body><p>{message}</p></body></html>"

class MarkdownFormatter(Formatter):
    def format(self, message: str) -> str:
        return f"**{message}**"

# === 緊急度デコレータ（合成） ===
class UrgencyDecorator(Formatter):
    def __init__(self, inner: Formatter, prefix: str = "[緊急]"):
        self.inner = inner
        self.prefix = prefix

    def format(self, message: str) -> str:
        return self.inner.format(f"{self.prefix} {message}")

# === 合成: 自由に組み合わせ可能 ===
@dataclass
class NotificationService:
    channel: DeliveryChannel
    formatter: Formatter

    def send(self, message: str, recipient: str) -> bool:
        formatted = self.formatter.format(message)
        return self.channel.deliver(formatted, recipient)

# 使用例: 3コンポーネント × 3組み合わせ = 合計6クラスで全組み合わせ対応
normal_email = NotificationService(EmailChannel(), HTMLFormatter())
urgent_slack = NotificationService(
    SlackChannel(),
    UrgencyDecorator(MarkdownFormatter())
)
urgent_sms = NotificationService(
    SMSChannel(),
    UrgencyDecorator(PlainFormatter(), prefix="[至急]")
)

normal_email.send("会議のお知らせ", "tanaka@example.com")
urgent_slack.send("サーバーダウン", "#alerts")
urgent_sms.send("緊急メンテナンス", "+81-90-XXXX-XXXX")
```

### 4.3 段階的リファクタリングのガイドライン

```
リファクタリング判断フロー:

  継承階層を発見
       │
       ▼
  Q1: 階層は3段以内か？
       │
    Yes │ No
       │  └──→ 合成にリファクタリング（優先度: 高）
       ▼
  Q2: 組み合わせ爆発が起きていないか？
       │
    No  │ Yes
       │  └──→ Bridge/Strategyパターンに分解（優先度: 高）
       ▼
  Q3: 基底クラスの変更で子クラスが壊れたことがあるか？
       │
    No  │ Yes
       │  └──→ 委譲パターンに書き換え（優先度: 中）
       ▼
  Q4: LSPを満たしているか？
       │
    Yes │ No
       │  └──→ インターフェース + 合成に変更（優先度: 中）
       ▼
  現状維持（継承が適切な場合）
```

---

## 5. 継承が適切な場合

### 5.1 継承を使うべき場面

```
継承が適切なケース:
──────────────────

1. 真の is-a 関係（リスコフの置換原則を満たす）
   - IOException is-a Exception ← OK
   - ArrayList is-a List ← OK
   - HttpServletRequest is-a ServletRequest ← OK

2. フレームワークが継承を要求する場合
   - Android Activity / Fragment
   - Django View / ModelAdmin
   - JUnit TestCase（レガシー）

3. テンプレートメソッドパターン
   - アルゴリズムの骨格は固定、詳細をオーバーライド
   - AbstractList の get(index) / size() のみオーバーライド

4. 共通の状態と振る舞いが密接に関連する場合
   - GUI ウィジェット階層（Button is-a Widget）

判断基準:
  □ 子クラスは親クラスの完全な代替になるか？ (LSP)
  □ 継承階層は3段以下か？
  □ 将来の拡張で組み合わせ爆発が起きないか？
  □ 親クラスの変更が子クラスを壊さないか？
  □ 子クラスが親クラスの全メソッドを意味的に持つべきか？

  1つでも No → 合成を検討
```

### 5.2 テンプレートメソッドパターン（適切な継承の例）

```python
# テンプレートメソッドパターン: 継承が適切な場面

from abc import ABC, abstractmethod

class DataExporter(ABC):
    """データエクスポートの骨格（テンプレートメソッド）"""

    def export(self, data: list[dict]) -> str:
        """エクスポートの手順は固定（テンプレート）"""
        # Step 1: ヘッダーを書く
        result = self.write_header()
        # Step 2: 各レコードを書く
        for record in data:
            result += self.write_record(record)
        # Step 3: フッターを書く
        result += self.write_footer()
        return result

    @abstractmethod
    def write_header(self) -> str:
        """サブクラスでフォーマット固有のヘッダーを実装"""
        pass

    @abstractmethod
    def write_record(self, record: dict) -> str:
        """サブクラスでフォーマット固有のレコード出力を実装"""
        pass

    @abstractmethod
    def write_footer(self) -> str:
        """サブクラスでフォーマット固有のフッターを実装"""
        pass

class CsvExporter(DataExporter):
    def write_header(self) -> str:
        return "id,name,email\n"

    def write_record(self, record: dict) -> str:
        return f"{record['id']},{record['name']},{record['email']}\n"

    def write_footer(self) -> str:
        return ""

class JsonExporter(DataExporter):
    def __init__(self):
        self._first = True

    def write_header(self) -> str:
        self._first = True
        return "[\n"

    def write_record(self, record: dict) -> str:
        import json
        prefix = "  " if self._first else ",\n  "
        self._first = False
        return prefix + json.dumps(record, ensure_ascii=False)

    def write_footer(self) -> str:
        return "\n]"

# 使用
data = [
    {"id": "1", "name": "田中", "email": "tanaka@example.com"},
    {"id": "2", "name": "鈴木", "email": "suzuki@example.com"},
]

csv_result = CsvExporter().export(data)
json_result = JsonExporter().export(data)
print(csv_result)
print(json_result)
```

---

## 6. アンチパターン

### 6.1 アンチパターン：ゴッド継承階層

```
NG: 過度に深い継承階層
  Component
   └── VisualComponent
        └── InteractiveComponent
             └── FormComponent
                  └── InputComponent
                       └── TextInputComponent
                            └── SearchInputComponent
                                 └── AutoCompleteSearchInput
                                      └── ... (まだ続く)

問題:
- 1つの変更が8段以上に波及
- 「あの機能はどの階層にあるか」が不明
- テスト時にモックすべき階層が多すぎる
- 新しい種類の入力を追加するのに全階層を理解する必要がある

OK: フラットな合成構造
  SearchInput:
    - renderer: AutoCompleteRenderer  // 見た目
    - validator: SearchValidator      // 検証
    - formatter: SearchFormatter      // フォーマット
    - behavior: SearchBehavior        // 振る舞い
    - accessiblity: AriaProps         // アクセシビリティ
```

### 6.2 アンチパターン：継承による横断的関心事の実装

```java
// NG: ログ機能を継承で追加
class LoggableService extends BaseService {
    @Override
    public void execute() {
        log("開始");
        super.execute();
        log("終了");
    }
}
// → 全サービスがLoggableServiceを継承？
// → キャッシュも追加したい → CachingLoggableService？
// → 認証も → AuthCachingLoggableService？ → 組み合わせ爆発

// OK: デコレータ/アスペクトで合成的に追加
class LoggingDecorator implements Service {
    private final Service inner;
    private final Logger logger;

    public LoggingDecorator(Service inner, Logger logger) {
        this.inner = inner;
        this.logger = logger;
    }

    @Override
    public void execute() {
        logger.info("開始: " + inner.getClass().getSimpleName());
        try {
            inner.execute();
            logger.info("完了");
        } catch (Exception e) {
            logger.error("失敗: " + e.getMessage());
            throw e;
        }
    }
}

class CachingDecorator implements Service {
    private final Service inner;
    private final Cache cache;

    public CachingDecorator(Service inner, Cache cache) {
        this.inner = inner;
        this.cache = cache;
    }

    @Override
    public void execute() {
        String key = inner.getClass().getSimpleName();
        if (!cache.has(key)) {
            inner.execute();
            cache.set(key, true);
        }
    }
}

class AuthDecorator implements Service {
    private final Service inner;
    private final AuthService auth;

    public AuthDecorator(Service inner, AuthService auth) {
        this.inner = inner;
        this.auth = auth;
    }

    @Override
    public void execute() {
        if (!auth.isAuthenticated()) {
            throw new UnauthorizedException();
        }
        inner.execute();
    }
}

// 自由に組み合わせ
Service service = new AuthDecorator(
    new LoggingDecorator(
        new CachingDecorator(
            new ActualService(),
            cache
        ),
        logger
    ),
    authService
);
```

### 6.3 アンチパターン：不要な抽象基底クラス

```python
# NG: 1つしか子クラスがないのに抽象クラスを作る
class AbstractUserRepository(ABC):
    @abstractmethod
    def find_by_id(self, user_id: str) -> User: ...
    @abstractmethod
    def save(self, user: User) -> None: ...

class PostgresUserRepository(AbstractUserRepository):
    # 唯一の実装
    def find_by_id(self, user_id: str) -> User: ...
    def save(self, user: User) -> None: ...

# → YAGNI（You Ain't Gonna Need It）違反
# → 2つ目の実装が必要になった時に抽象化すればよい

# OK: まずは具体クラスで十分
class UserRepository:
    def find_by_id(self, user_id: str) -> User: ...
    def save(self, user: User) -> None: ...

# Python では Protocol で後からインターフェースを定義できる
# class UserRepositoryProtocol(Protocol):
#     def find_by_id(self, user_id: str) -> User: ...
#     def save(self, user: User) -> None: ...
# → 既存のクラスを変更せずにインターフェースを満たせる
```

---

## 7. 実践演習

### 演習1（基礎）: 決済システムの合成設計

**課題**: 以下の要件を満たす決済システムをPythonで合成ベースに設計してください。

```python
# 要件:
# 1. 決済手段: CreditCard, BankTransfer, PayPay の3種
# 2. 通知手段: Email, SMS の2種
# 3. ログ: Console, File の2種
# 4. PaymentService は決済手段・通知手段・ログを合成で受け取る
# 5. process_payment(amount, user) メソッドを実装
# 6. 全ての組み合わせをクラス爆発なしに実現する
```

**期待される実装**:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

# === 決済手段インターフェース ===
class PaymentMethod(ABC):
    @abstractmethod
    def charge(self, amount: int, user_id: str) -> bool:
        pass

    @abstractmethod
    def name(self) -> str:
        pass

class CreditCard(PaymentMethod):
    def charge(self, amount: int, user_id: str) -> bool:
        print(f"  クレジットカード決済: {amount}円 (user: {user_id})")
        return True

    def name(self) -> str:
        return "CreditCard"

class BankTransfer(PaymentMethod):
    def charge(self, amount: int, user_id: str) -> bool:
        print(f"  銀行振込: {amount}円 (user: {user_id})")
        return True

    def name(self) -> str:
        return "BankTransfer"

class PayPay(PaymentMethod):
    def charge(self, amount: int, user_id: str) -> bool:
        print(f"  PayPay決済: {amount}円 (user: {user_id})")
        return True

    def name(self) -> str:
        return "PayPay"

# === 通知インターフェース ===
class Notifier(ABC):
    @abstractmethod
    def notify(self, user_id: str, message: str) -> None:
        pass

class EmailNotifier(Notifier):
    def notify(self, user_id: str, message: str) -> None:
        print(f"  Email通知 -> {user_id}: {message}")

class SMSNotifier(Notifier):
    def notify(self, user_id: str, message: str) -> None:
        print(f"  SMS通知 -> {user_id}: {message}")

# === ログインターフェース ===
class PaymentLogger(ABC):
    @abstractmethod
    def log(self, message: str) -> None:
        pass

class ConsolePaymentLogger(PaymentLogger):
    def log(self, message: str) -> None:
        print(f"  [LOG] {message}")

class FilePaymentLogger(PaymentLogger):
    def __init__(self, path: str = "payment.log"):
        self.path = path

    def log(self, message: str) -> None:
        print(f"  [FILE:{self.path}] {message}")

# === 合成: PaymentService ===
@dataclass
class PaymentService:
    payment_method: PaymentMethod
    notifier: Notifier
    logger: PaymentLogger

    def process_payment(self, amount: int, user_id: str) -> bool:
        self.logger.log(
            f"決済開始: {self.payment_method.name()} {amount}円 user={user_id}"
        )
        success = self.payment_method.charge(amount, user_id)
        if success:
            self.notifier.notify(user_id, f"決済完了: {amount}円")
            self.logger.log("決済成功")
        else:
            self.logger.log("決済失敗")
        return success

# === テスト ===
# 組み合わせ1: クレカ + Email + コンソールログ
service1 = PaymentService(CreditCard(), EmailNotifier(), ConsolePaymentLogger())
service1.process_payment(5000, "user-001")

print()

# 組み合わせ2: PayPay + SMS + ファイルログ
service2 = PaymentService(PayPay(), SMSNotifier(), FilePaymentLogger())
service2.process_payment(1200, "user-002")

# 3手段 × 2通知 × 2ログ = 12通りの組み合わせが6クラスで実現
print("\n全テスト通過！")
```

**期待される出力**:
```
  [LOG] 決済開始: CreditCard 5000円 user=user-001
  クレジットカード決済: 5000円 (user: user-001)
  Email通知 -> user-001: 決済完了: 5000円
  [LOG] 決済成功

  [FILE:payment.log] 決済開始: PayPay 1200円 user=user-002
  PayPay決済: 1200円 (user: user-002)
  SMS通知 -> user-002: 決済完了: 1200円
  [FILE:payment.log] 決済成功

全テスト通過！
```

---

### 演習2（応用）: ミドルウェアパイプラインの合成

**課題**: TypeScriptでHTTPミドルウェアをデコレータパターンで合成するシステムを実装してください。

```typescript
// 要件:
// 1. Handler インターフェース: handle(request) => response
// 2. LoggingMiddleware: リクエスト/レスポンスをログ出力
// 3. AuthMiddleware: Authorizationヘッダーを検証
// 4. RateLimitMiddleware: レート制限チェック
// 5. ミドルウェアを任意の順序で組み合わせ可能
```

**期待される実装**:

```typescript
interface Request {
  readonly method: string;
  readonly path: string;
  readonly headers: Readonly<Record<string, string>>;
  readonly body?: string;
}

interface Response {
  readonly status: number;
  readonly body: string;
}

interface Handler {
  handle(request: Request): Response;
}

// === 実際のハンドラ ===
class UserHandler implements Handler {
  handle(request: Request): Response {
    return { status: 200, body: JSON.stringify({ user: "田中" }) };
  }
}

// === ミドルウェア（デコレータ） ===
class LoggingMiddleware implements Handler {
  constructor(private inner: Handler) {}

  handle(request: Request): Response {
    console.log(`→ ${request.method} ${request.path}`);
    const response = this.inner.handle(request);
    console.log(`← ${response.status}`);
    return response;
  }
}

class AuthMiddleware implements Handler {
  constructor(
    private inner: Handler,
    private validTokens: Set<string>
  ) {}

  handle(request: Request): Response {
    const token = request.headers["authorization"];
    if (!token || !this.validTokens.has(token)) {
      return { status: 401, body: "Unauthorized" };
    }
    return this.inner.handle(request);
  }
}

class RateLimitMiddleware implements Handler {
  private requestCounts = new Map<string, number>();

  constructor(
    private inner: Handler,
    private maxRequests: number = 10
  ) {}

  handle(request: Request): Response {
    const clientIp = request.headers["x-forwarded-for"] || "unknown";
    const count = this.requestCounts.get(clientIp) || 0;

    if (count >= this.maxRequests) {
      return { status: 429, body: "Too Many Requests" };
    }

    this.requestCounts.set(clientIp, count + 1);
    return this.inner.handle(request);
  }
}

// === パイプライン構築 ===
const validTokens = new Set(["Bearer token123"]);

const pipeline = new LoggingMiddleware(
  new AuthMiddleware(
    new RateLimitMiddleware(new UserHandler(), 100),
    validTokens
  )
);

// テスト1: 正常リクエスト
const response1 = pipeline.handle({
  method: "GET",
  path: "/users/1",
  headers: { authorization: "Bearer token123", "x-forwarded-for": "1.2.3.4" },
});
console.log("Response:", response1);

// テスト2: 認証失敗
const response2 = pipeline.handle({
  method: "GET",
  path: "/users/1",
  headers: { "x-forwarded-for": "1.2.3.4" },
});
console.log("Response:", response2);
```

**期待される出力**:
```
→ GET /users/1
← 200
Response: { status: 200, body: '{"user":"田中"}' }
→ GET /users/1
← 401
Response: { status: 401, body: 'Unauthorized' }
```

---

### 演習3（発展）: 継承から合成へのリファクタリング

**課題**: 以下の継承ベースのゲームキャラクターシステムを合成ベースにリファクタリングしてください。

```python
# Before: 継承ベース（問題あり）
# class Character:
#     def attack(self): return 10
#     def defend(self): return 5
#     def move(self): return "walk"
#
# class Warrior(Character):
#     def attack(self): return 20
#     def defend(self): return 15
#
# class Mage(Character):
#     def attack(self): return 25  # 魔法攻撃
#     def move(self): return "teleport"
#
# class FlyingWarrior(Warrior):  # ← 問題！飛行 + 戦士
#     def move(self): return "fly"
#
# class FlyingMage(Mage):  # ← 飛行 + 魔法使い
#     def move(self): return "fly"
#
# → 飛行 × 戦闘スタイル × 移動方法 で組み合わせ爆発

# 要件:
# 1. AttackStrategy, DefenseStrategy, MovementStrategy を定義
# 2. Character は3つの戦略を合成で持つ
# 3. 実行時に戦略を変更可能（パワーアップアイテム等）
# 4. 新しい組み合わせをクラス追加なしに作成可能
```

**期待される実装**:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

# === 攻撃戦略 ===
class AttackStrategy(ABC):
    @abstractmethod
    def attack(self) -> tuple[int, str]:
        """(ダメージ, 攻撃名)"""
        pass

class SwordAttack(AttackStrategy):
    def attack(self) -> tuple[int, str]:
        return (20, "剣で斬る")

class MagicAttack(AttackStrategy):
    def attack(self) -> tuple[int, str]:
        return (25, "ファイアボール")

class BowAttack(AttackStrategy):
    def attack(self) -> tuple[int, str]:
        return (15, "矢を射る")

class DualWield(AttackStrategy):
    """合成の合成: 二刀流"""
    def __init__(self, main: AttackStrategy, sub: AttackStrategy):
        self.main = main
        self.sub = sub

    def attack(self) -> tuple[int, str]:
        main_dmg, main_name = self.main.attack()
        sub_dmg, sub_name = self.sub.attack()
        return (main_dmg + sub_dmg // 2, f"{main_name} + {sub_name}")

# === 防御戦略 ===
class DefenseStrategy(ABC):
    @abstractmethod
    def defend(self) -> tuple[int, str]:
        """(防御力, 防御名)"""
        pass

class ShieldDefense(DefenseStrategy):
    def defend(self) -> tuple[int, str]:
        return (15, "盾で防御")

class MagicBarrier(DefenseStrategy):
    def defend(self) -> tuple[int, str]:
        return (20, "魔法障壁")

class DodgeDefense(DefenseStrategy):
    def defend(self) -> tuple[int, str]:
        return (10, "回避")

# === 移動戦略 ===
class MovementStrategy(ABC):
    @abstractmethod
    def move(self) -> tuple[int, str]:
        """(速度, 移動名)"""
        pass

class WalkMovement(MovementStrategy):
    def move(self) -> tuple[int, str]:
        return (5, "歩く")

class FlyMovement(MovementStrategy):
    def move(self) -> tuple[int, str]:
        return (15, "飛行")

class TeleportMovement(MovementStrategy):
    def move(self) -> tuple[int, str]:
        return (100, "テレポート")

# === キャラクター: 合成で構成 ===
@dataclass
class Character:
    name: str
    attack_strategy: AttackStrategy
    defense_strategy: DefenseStrategy
    movement_strategy: MovementStrategy
    hp: int = 100

    def perform_attack(self) -> str:
        damage, attack_name = self.attack_strategy.attack()
        return f"{self.name}: {attack_name} (ダメージ: {damage})"

    def perform_defense(self) -> str:
        defense, defense_name = self.defense_strategy.defend()
        return f"{self.name}: {defense_name} (防御力: {defense})"

    def perform_move(self) -> str:
        speed, move_name = self.movement_strategy.move()
        return f"{self.name}: {move_name} (速度: {speed})"

    def equip_attack(self, strategy: AttackStrategy) -> None:
        """実行時に攻撃戦略を変更"""
        self.attack_strategy = strategy

    def equip_movement(self, strategy: MovementStrategy) -> None:
        """実行時に移動戦略を変更"""
        self.movement_strategy = strategy

# === ファクトリ: 典型的な構成を提供 ===
class CharacterFactory:
    @staticmethod
    def create_warrior(name: str) -> Character:
        return Character(name, SwordAttack(), ShieldDefense(), WalkMovement())

    @staticmethod
    def create_mage(name: str) -> Character:
        return Character(name, MagicAttack(), MagicBarrier(), TeleportMovement())

    @staticmethod
    def create_ranger(name: str) -> Character:
        return Character(name, BowAttack(), DodgeDefense(), WalkMovement())

    @staticmethod
    def create_flying_warrior(name: str) -> Character:
        return Character(name, SwordAttack(), ShieldDefense(), FlyMovement())

# === テスト ===
warrior = CharacterFactory.create_warrior("アーサー")
mage = CharacterFactory.create_mage("マーリン")
ranger = CharacterFactory.create_ranger("レゴラス")

print(warrior.perform_attack())   # アーサー: 剣で斬る (ダメージ: 20)
print(mage.perform_attack())      # マーリン: ファイアボール (ダメージ: 25)
print(ranger.perform_move())      # レゴラス: 歩く (速度: 5)

# パワーアップ: 飛行能力を獲得！
warrior.equip_movement(FlyMovement())
print(warrior.perform_move())     # アーサー: 飛行 (速度: 15)

# 二刀流に変更
warrior.equip_attack(DualWield(SwordAttack(), BowAttack()))
print(warrior.perform_attack())   # アーサー: 剣で斬る + 矢を射る (ダメージ: 27)

# カスタムキャラ: 魔法剣士（クラス追加なし！）
magic_knight = Character(
    "セシル", MagicAttack(), ShieldDefense(), WalkMovement()
)
print(magic_knight.perform_attack())  # セシル: ファイアボール (ダメージ: 25)
print(magic_knight.perform_defense()) # セシル: 盾で防御 (防御力: 15)

print("\n全テスト通過！")
```

**期待される出力**:
```
アーサー: 剣で斬る (ダメージ: 20)
マーリン: ファイアボール (ダメージ: 25)
レゴラス: 歩く (速度: 5)
アーサー: 飛行 (速度: 15)
アーサー: 剣で斬る + 矢を射る (ダメージ: 27)
セシル: ファイアボール (ダメージ: 25)
セシル: 盾で防御 (防御力: 15)

全テスト通過！
```

---

## 8. FAQ

### Q1: 合成だとコード量が増えないか？

**A**: 短期的にはインターフェース定義やファクトリが増えるため、コード量は若干増える。しかし長期的には、機能追加時に既存コードの変更なしに新しい組み合わせが作れるため、総コード量は減少する。また、各コンポーネントが小さく独立しているため、理解・テスト・変更が容易になり、開発速度は向上する。継承で27クラス必要なところを合成なら9クラスで実現できるケースもある。

### Q2: 合成と依存性注入（DI）の関係は？

**A**: 合成と依存性注入は密接に関連する。合成は「何を組み合わせるか」の設計、DIは「どのように組み合わせを構成するか」の実装手法。DIコンテナ（Spring、Dagger等）は合成パターンの構成を自動化する。つまりDIは合成を実現するための強力なツール。合成を前提とした設計にDIを導入すると、テスト時のモック差し替えが容易になり、本番環境とテスト環境で異なるコンポーネントを注入できる。

### Q3: TypeScript/JavaScriptではMixinとクラス継承のどちらを使うべきか？

**A**: TypeScriptではインターフェース + 合成が推奨される。Mixinはクラス継承の制約を回避する手法だが、型安全性が弱まる場合がある。小規模な機能追加にはMixinが便利だが、大規模なシステムではインターフェースベースの合成+DIが保守しやすい。Reactのカスタムフック（useXxx）は関数レベルの合成パターンであり、クラスベースの合成より軽量で推奨される。

### Q4: 合成を使うとパフォーマンスが低下しないか？

**A**: メソッド呼び出しが1段増える（委譲のコスト）ため、理論上はわずかなオーバーヘッドがある。しかしJITコンパイラによるインライン化やメソッドディスパッチの最適化により、実測で差が出ることはほぼない。Rustのtrait/genericsはゼロコスト抽象を実現し、コンパイル時に静的ディスパッチに変換される。パフォーマンスの懸念よりも、設計の柔軟性・保守性の向上の方がはるかに価値が大きい。

### Q5: 既存の大規模な継承階層をどうリファクタリングすべきか？

**A**: 一度にすべてをリファクタリングしようとしないこと。(1) まず新しいコードから合成パターンを適用する。(2) バグ修正や機能追加のタイミングで周辺コードを段階的に書き換える。(3) リファクタリング対象は「変更頻度が高い」「テストが書きにくい」「組み合わせ爆発が起きている」箇所から優先する。Martin Fowlerの「Strangler Fig Pattern」（絞め殺しの木パターン）を参考に、古い継承を新しい合成で徐々に置き換えていく。

---

## 9. まとめ

| カテゴリ | ポイント |
|---------|---------|
| 基本原則 | デフォルトは合成、継承はLSPを満たす場合のみ |
| Strategy | 振る舞いの実行時切替が必要な場合 |
| Decorator | 機能の動的追加・組み合わせに最適 |
| Delegate | 内部実装を隠蔽しつつ機能を公開 |
| Mixin/Trait | 横断的機能の付与（言語サポートがある場合） |
| DI | 合成パターンの構成を自動化するツール |
| リファクタリング | 振る舞いの軸を特定→インターフェース抽出→合成に置換 |
| 継承の適用 | is-a関係が真に成立する場合、3段以下に制限 |
| 組み合わせ爆発 | N×M問題はBridge/Strategyで分解 |

| 判断基準 | 継承を選ぶ | 合成を選ぶ |
|---------|----------|----------|
| 関係 | 真のis-a | has-a / can-do |
| 拡張方向 | 1軸のみ | 複数軸の組み合わせ |
| 変更頻度 | 低い（安定した階層） | 高い（要件変更が多い） |
| テスト | 統合テスト中心 | ユニットテスト中心 |
| 実行時変更 | 不要 | 必要 |
| 階層の深さ | 3段以下 | フラット |

---

## 次に読むべきガイド

- [00-immutability.md](./00-immutability.md) -- イミュータビリティの原則（合成と不変データの組み合わせ）
- [02-functional-principles.md](./02-functional-principles.md) -- 関数型プログラミングの原則（関数の合成）
- [03-api-design.md](./03-api-design.md) -- API設計（合成的なミドルウェア設計）
- [00-principles/02-solid.md](../00-principles/02-solid.md) -- SOLID原則（LSP、OCP、DIPと合成の関係）
- [02-refactoring/01-refactoring-catalog.md](../02-refactoring/01-refactoring-catalog.md) -- リファクタリングカタログ
- `design-patterns-guide/docs/01-structural/` -- 構造パターン（Decorator、Bridge、Composite等）
- `design-patterns-guide/docs/02-behavioral/` -- 振る舞いパターン（Strategy、State、Observer等）

---

## 参考文献

1. Gamma et al., "Design Patterns: Elements of Reusable Object-Oriented Software" -- GoF パターンの原典
2. Joshua Bloch, "Effective Java" 第3版 -- Item 18: Favor composition over inheritance
3. Robert C. Martin, "Agile Software Development: Principles, Patterns, and Practices" -- OCP, LSP
4. Sandi Metz, "Practical Object-Oriented Design in Ruby" -- Composition chapter
5. Martin Fowler, "Refactoring" 第2版 -- Replace Inheritance with Delegation
6. Head First Design Patterns, 2nd Edition -- Strategy, Decorator パターンの詳解
7. Go Proverbs -- "Don't just check errors, handle them gracefully" / Composition over inheritance in Go
8. Rust Book, "Traits: Defining Shared Behavior" -- https://doc.rust-lang.org/book/ch10-02-traits.html
9. Sam Newman, "Building Microservices" -- サービスの合成とオーケストレーション
10. Michael Feathers, "Working Effectively with Legacy Code" -- レガシーコードの段階的リファクタリング
