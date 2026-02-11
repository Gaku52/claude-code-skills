# 継承より合成の原則

> なぜ合成（コンポジション）が継承よりも柔軟で保守しやすいか。デザインパターンとの関係、言語別の実装手法、リファクタリング戦略を体系的に解説する。

---

## この章で学ぶこと

1. **継承の問題点と合成の優位性**を理論的に理解し、設計判断の根拠を説明できる
2. **Strategy、Decorator、Delegate**など合成ベースのデザインパターンを実装できる
3. **既存の継承階層を合成にリファクタリング**する手法を習得し、保守性を改善できる

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

### 1.2 継承 vs 合成の比較

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

---

## 2. 合成ベースのデザインパターン

### 2.1 Strategy パターン

```python
# Strategy パターン: 振る舞いを外部から注入
from abc import ABC, abstractmethod
from dataclasses import dataclass

# 戦略インターフェース
class SortStrategy(ABC):
    @abstractmethod
    def sort(self, data: list) -> list:
        pass

class QuickSort(SortStrategy):
    def sort(self, data: list) -> list:
        if len(data) <= 1:
            return data
        pivot = data[0]
        left = [x for x in data[1:] if x <= pivot]
        right = [x for x in data[1:] if x > pivot]
        return self.sort(left) + [pivot] + self.sort(right)

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

# 合成: 戦略を持つクラス
@dataclass
class DataProcessor:
    sort_strategy: SortStrategy  # has-a（合成）

    def process(self, data: list) -> list:
        # ソート戦略は実行時に差し替え可能
        return self.sort_strategy.sort(data)

# 使用: 戦略を自由に切り替え
processor = DataProcessor(sort_strategy=QuickSort())
result1 = processor.process([3, 1, 4, 1, 5])

processor.sort_strategy = MergeSort()  # 実行時に切り替え
result2 = processor.process([3, 1, 4, 1, 5])
```

### 2.2 Decorator パターン

```typescript
// Decorator パターン: 機能を動的に追加
// 継承ではなく合成で機能を積み重ねる

interface Logger {
  log(message: string): void;
}

// 基本実装
class ConsoleLogger implements Logger {
  log(message: string): void {
    console.log(message);
  }
}

// デコレータ: タイムスタンプ追加
class TimestampDecorator implements Logger {
  constructor(private inner: Logger) {} // 合成

  log(message: string): void {
    const timestamp = new Date().toISOString();
    this.inner.log(`[${timestamp}] ${message}`);
  }
}

// デコレータ: ログレベル追加
class LevelDecorator implements Logger {
  constructor(
    private inner: Logger,
    private level: string = "INFO"
  ) {}

  log(message: string): void {
    this.inner.log(`[${this.level}] ${message}`);
  }
}

// デコレータ: JSON形式変換
class JsonDecorator implements Logger {
  constructor(private inner: Logger) {}

  log(message: string): void {
    const json = JSON.stringify({ message, timestamp: Date.now() });
    this.inner.log(json);
  }
}

// 使用: デコレータを自由に組み合わせ
const logger = new TimestampDecorator(
  new LevelDecorator(
    new ConsoleLogger(),
    "DEBUG"
  )
);
logger.log("テストメッセージ");
// 出力: [2026-01-15T10:30:00.000Z] [DEBUG] テストメッセージ

// 別の組み合わせ
const jsonLogger = new JsonDecorator(new ConsoleLogger());
jsonLogger.log("テスト");
// 出力: {"message":"テスト","timestamp":1737000000000}
```

### 2.3 委譲（Delegation）パターン

```java
// 委譲パターン: 継承の代わりに内部オブジェクトに処理を委譲

// NG: 継承でStackを実装
// public class Stack<T> extends ArrayList<T> {
//     // ArrayListの全メソッドが公開されてしまう
//     // add(index, element)でスタックの規約が破れる
// }

// OK: 合成+委譲でStackを実装
public class Stack<T> {
    private final List<T> elements = new ArrayList<>(); // has-a

    public void push(T item) {
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
```

---

## 3. 合成を活用した設計パターン一覧

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

### 3.2 Mixin/Trait パターン（多重合成）

```typescript
// TypeScript: Mixinで合成的に機能を追加

type Constructor<T = {}> = new (...args: any[]) => T;

// Mixin: シリアライズ機能
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

// Mixin: バリデーション機能
function Validatable<TBase extends Constructor>(Base: TBase) {
  return class extends Base {
    private validationErrors: string[] = [];

    validate(): boolean {
      this.validationErrors = [];
      // 各フィールドの検証ロジック
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

// Mixin: 監査ログ機能
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

// 基底クラス
class BaseEntity {
  constructor(public id: string) {}
}

// 合成: 必要な機能を組み合わせる
class User extends Auditable(Validatable(Serializable(BaseEntity))) {
  constructor(id: string, public name: string, public email: string) {
    super(id);
  }
}

const user = new User("1", "田中", "tanaka@example.com");
user.markCreated("admin");
const json = user.serialize();
console.log(user.validate());
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

Step 3: 合成に書き換え
─────────────────────
  class Entity:
      behaviorX: BehaviorX
      behaviorY: BehaviorY
      behaviorZ: BehaviorZ

  entityA = Entity(BehaviorXImpl1, BehaviorYImpl1, default)
  entityB = Entity(BehaviorXImpl1, default, BehaviorZImpl1)
  entityC = Entity(default, BehaviorYImpl1, BehaviorZImpl1)
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

# After: 合成ベースのリファクタリング
from abc import ABC, abstractmethod
from dataclasses import dataclass

# 配送戦略
class DeliveryChannel(ABC):
    @abstractmethod
    def deliver(self, formatted_message: str, recipient: str) -> bool:
        pass

class EmailChannel(DeliveryChannel):
    def deliver(self, formatted_message: str, recipient: str) -> bool:
        print(f"Email to {recipient}: {formatted_message}")
        return True

class SlackChannel(DeliveryChannel):
    def deliver(self, formatted_message: str, recipient: str) -> bool:
        print(f"Slack to {recipient}: {formatted_message}")
        return True

class SMSChannel(DeliveryChannel):
    def deliver(self, formatted_message: str, recipient: str) -> bool:
        print(f"SMS to {recipient}: {formatted_message}")
        return True

# フォーマット戦略
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

# 緊急度デコレータ
class UrgencyDecorator(Formatter):
    def __init__(self, inner: Formatter, prefix: str = "[緊急]"):
        self.inner = inner
        self.prefix = prefix

    def format(self, message: str) -> str:
        return self.inner.format(f"{self.prefix} {message}")

# 合成: 自由に組み合わせ可能
@dataclass
class NotificationService:
    channel: DeliveryChannel
    formatter: Formatter

    def send(self, message: str, recipient: str) -> bool:
        formatted = self.formatter.format(message)
        return self.channel.deliver(formatted, recipient)

# 使用例: どんな組み合わせも可能
normal_email = NotificationService(EmailChannel(), HTMLFormatter())
urgent_slack = NotificationService(
    SlackChannel(),
    UrgencyDecorator(MarkdownFormatter())
)
urgent_sms = NotificationService(
    SMSChannel(),
    UrgencyDecorator(PlainFormatter(), prefix="[至急]")
)
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

2. フレームワークが継承を要求する場合
   - Android Activity / Fragment
   - Django View

3. テンプレートメソッドパターン
   - アルゴリズムの骨格は固定、詳細をオーバーライド

判断基準:
  □ 子クラスは親クラスの完全な代替になるか？ (LSP)
  □ 継承階層は3段以下か？
  □ 将来の拡張で組み合わせ爆発が起きないか？
  □ 親クラスの変更が子クラスを壊さないか？

  1つでも No → 合成を検討
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

OK: フラットな合成構造
  SearchInput:
    - renderer: AutoCompleteRenderer
    - validator: SearchValidator
    - formatter: SearchFormatter
    - behavior: SearchBehavior
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
// → 全サービスがLoggableServiceを継承？他の横断的関心事は？

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
        logger.info("開始");
        inner.execute();
        logger.info("終了");
    }
}

// 自由に組み合わせ
Service service = new LoggingDecorator(
    new MetricsDecorator(
        new RetryDecorator(
            new ActualService(),
            3
        )
    ),
    logger
);
```

---

## 7. FAQ

### Q1: 合成だとコード量が増えないか？

**A**: 短期的にはインターフェース定義やファクトリが増えるため、コード量は若干増える。しかし長期的には、機能追加時に既存コードの変更なしに新しい組み合わせが作れるため、総コード量は減少する。また、各コンポーネントが小さく独立しているため、理解・テスト・変更が容易になり、開発速度は向上する。

### Q2: 合成と依存性注入（DI）の関係は？

**A**: 合成と依存性注入は密接に関連する。合成は「何を組み合わせるか」の設計、DIは「どのように組み合わせを構成するか」の実装手法。DIコンテナ（Spring、Dagger等）は合成パターンの構成を自動化する。つまりDIは合成を実現するための強力なツール。

### Q3: TypeScript/JavaScriptではMixinとクラス継承のどちらを使うべきか？

**A**: TypeScriptではインターフェース + 合成が推奨される。Mixinはクラス継承の制約を回避する手法だが、型安全性が弱まる場合がある。小規模な機能追加にはMixinが便利だが、大規模なシステムではインターフェースベースの合成+DIが保守しやすい。

---

## 8. まとめ

| カテゴリ | ポイント |
|---------|---------|
| 基本原則 | デフォルトは合成、継承はLSPを満たす場合のみ |
| Strategy | 振る舞いの実行時切替が必要な場合 |
| Decorator | 機能の動的追加・組み合わせに最適 |
| Delegate | 内部実装を隠蔽しつつ機能を公開 |
| Mixin | 横断的機能の付与（TypeScript/Pythonで有効） |
| リファクタリング | 振る舞いの軸を特定→インターフェース抽出→合成に置換 |
| 継承の適用 | is-a関係が真に成立する場合、3段以下に制限 |

---

## 次に読むべきガイド

- [00-immutability.md](./00-immutability.md) — イミュータビリティの原則
- [02-functional-principles.md](./02-functional-principles.md) — 関数型プログラミングの原則
- デザインパターン詳解 — GoFパターンの深掘り

---

## 参考文献

1. Gamma et al., "Design Patterns: Elements of Reusable Object-Oriented Software" — GoF パターンの原典
2. Joshua Bloch, "Effective Java" 第3版 — Item 18: Favor composition over inheritance
3. Robert C. Martin, "Agile Software Development: Principles, Patterns, and Practices" — OCP, LSP
4. Sandi Metz, "Practical Object-Oriented Design in Ruby" — Composition chapter
5. Martin Fowler, "Refactoring" 第2版 — Replace Inheritance with Delegation
