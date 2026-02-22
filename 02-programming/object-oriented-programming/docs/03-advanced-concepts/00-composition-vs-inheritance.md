# コンポジション vs 継承

> 「継承よりコンポジションを優先せよ」はGoF以来の鉄則。しかし盲目的にコンポジションを使うのではなく、両者のトレードオフを理解して使い分けることが重要。

## この章で学ぶこと

- [ ] コンポジションと継承の本質的な違いを理解する
- [ ] 「継承よりコンポジション」の理由を把握する
- [ ] 実践的な判断基準を学ぶ
- [ ] デザインパターンにおけるコンポジションの活用を習得する
- [ ] 継承が適切な場面とその設計指針を理解する

---

## 1. 本質的な違い

```
継承（Inheritance）: is-a 関係
  → Dog is-a Animal（犬は動物である）
  → 親のすべてを引き継ぐ（強い結合）
  → コンパイル時に関係が固定

コンポジション（Composition）: has-a 関係
  → Car has-a Engine（車はエンジンを持つ）
  → 必要な部品を組み合わせる（弱い結合）
  → 実行時に部品を差し替え可能

  継承:
    ┌─────────┐
    │ Animal  │
    └────┬────┘
         │ is-a
    ┌────┴────┐
    │   Dog   │
    └─────────┘

  コンポジション:
    ┌─────────┐     ┌────────┐
    │   Car   │────→│ Engine │  has-a
    │         │────→│ Wheels │  has-a
    │         │────→│ GPS    │  has-a
    └─────────┘     └────────┘

委譲（Delegation）:
  → コンポジションの一形態
  → 内部のオブジェクトにメソッド呼び出しを転送
  → 「自分で処理する」のではなく「持っているものに頼む」

集約（Aggregation）:
  → コンポジションの弱い形態
  → 「部品」が独立して存在できる
  → Car has-a Driver（ドライバーは車がなくても存在する）

  コンポジション: 部品はオーナーと共に生死する
  集約: 部品はオーナーとは独立して存在する
```

### 1.1 継承の構造的問題

```
継承のメカニズム:

  class Dog extends Animal {
    // Dog は Animal の全てを自動的に引き継ぐ
    // 1. public メソッド → そのまま公開
    // 2. protected メソッド → アクセス可能
    // 3. private メソッド → アクセス不可だが存在する
    // 4. フィールド → すべて引き継ぐ
  }

問題点:

  1. カプセル化の破壊:
     → protected フィールドにアクセスできてしまう
     → 親クラスの内部実装に依存してしまう
     → 親クラスのリファクタリングが困難に

  2. 脆い基底クラス問題（Fragile Base Class Problem）:
     → 親クラスの変更が子クラスを予期せず壊す
     → 子クラスが親の実装詳細に依存しているため

  3. 密結合:
     → 親クラスのインターフェースすべてを強制的に継承
     → 不要なメソッドもすべて公開される

  4. 単一継承の制約（Java, C#, TypeScript）:
     → 1つの親クラスしか持てない
     → 複数の振る舞いを組み合わせられない
```

### 1.2 脆い基底クラス問題の具体例

```java
// ❌ 脆い基底クラス問題の実例

// Java の HashSet を継承した「カウント付きセット」
public class CountingSet<E> extends HashSet<E> {
    private int addCount = 0;

    @Override
    public boolean add(E e) {
        addCount++;
        return super.add(e);
    }

    @Override
    public boolean addAll(Collection<? extends E> c) {
        addCount += c.size();
        return super.addAll(c);
    }

    public int getAddCount() {
        return addCount;
    }
}

// 使ってみる
CountingSet<String> s = new CountingSet<>();
s.addAll(Arrays.asList("A", "B", "C"));
System.out.println(s.getAddCount()); // 期待: 3, 実際: 6 !!!

// なぜ 6 になるのか？
// HashSet.addAll() の内部で add() を呼んでいる!
// addAll() で +3、add() x 3 で +3 = 合計 6
// → 親クラスの実装詳細に依存してしまった

// ✅ コンポジションで解決
public class CountingSet<E> implements Set<E> {
    private final Set<E> delegate = new HashSet<>();
    private int addCount = 0;

    @Override
    public boolean add(E e) {
        addCount++;
        return delegate.add(e);  // 委譲
    }

    @Override
    public boolean addAll(Collection<? extends E> c) {
        addCount += c.size();
        return delegate.addAll(c);  // 委譲（内部で add() を呼んでも影響なし）
    }

    @Override
    public int size() { return delegate.size(); }

    @Override
    public boolean contains(Object o) { return delegate.contains(o); }

    // ... 他の Set メソッドもすべて delegate に委譲

    public int getAddCount() {
        return addCount;
    }
}

CountingSet<String> s = new CountingSet<>();
s.addAll(Arrays.asList("A", "B", "C"));
System.out.println(s.getAddCount()); // ✅ 3（正しい！）
```

---

## 2. なぜ「継承よりコンポジション」か

```
継承の問題:
  1. 強い結合: 親の変更が全子クラスに波及
  2. カプセル化の破壊: 子が親の実装詳細に依存
  3. 柔軟性の欠如: 実行時に振る舞いを変更できない
  4. 爆発的な組み合わせ:

  例: ゲームキャラクター
  継承で設計すると:
    Character
    ├── Warrior
    │   ├── FireWarrior
    │   ├── IceWarrior
    │   └── FlyingWarrior
    ├── Mage
    │   ├── FireMage
    │   ├── IceMage
    │   └── FlyingMage
    └── Archer
        ├── FireArcher
        ├── IceArcher
        └── FlyingArcher
    → 3属性 × 3職業 = 9クラス
    → 新属性追加で +3クラス、新職業追加で +3クラス

  コンポジションで設計すると:
    Character
    ├── has-a: AttackStyle（Warrior, Mage, Archer）
    ├── has-a: Element（Fire, Ice, Lightning）
    └── has-a: Movement（Walk, Fly, Teleport）
    → 3 + 3 + 3 = 9コンポーネント
    → 新属性追加で +1コンポーネント
```

### 2.1 コンポジションによるリファクタリング

```typescript
// ❌ 継承: クラスの爆発
class Animal {
  eat(): void { console.log("食べる"); }
}
class FlyingAnimal extends Animal {
  fly(): void { console.log("飛ぶ"); }
}
class SwimmingAnimal extends Animal {
  swim(): void { console.log("泳ぐ"); }
}
class FlyingSwimmingAnimal extends ??? {
  // 多重継承できない！
}

// ✅ コンポジション: 柔軟な組み合わせ
interface MovementAbility {
  move(): string;
}

class Flying implements MovementAbility {
  move(): string { return "空を飛ぶ"; }
}

class Swimming implements MovementAbility {
  move(): string { return "水中を泳ぐ"; }
}

class Walking implements MovementAbility {
  move(): string { return "地上を歩く"; }
}

class Animal {
  private abilities: MovementAbility[] = [];

  addAbility(ability: MovementAbility): void {
    this.abilities.push(ability);
  }

  moveAll(): string[] {
    return this.abilities.map(a => a.move());
  }
}

// カモ: 飛べる + 泳げる + 歩ける
const duck = new Animal();
duck.addAbility(new Flying());
duck.addAbility(new Swimming());
duck.addAbility(new Walking());
console.log(duck.moveAll()); // ["空を飛ぶ", "水中を泳ぐ", "地上を歩く"]
```

### 2.2 ゲームキャラクターのコンポジション設計

```typescript
// ECS（Entity-Component-System）的なコンポジション設計

// === コンポーネント（振る舞いの部品）===
interface AttackBehavior {
  attack(target: string): string;
  getRange(): number;
}

interface DefenseBehavior {
  defend(): string;
  getArmor(): number;
}

interface MovementBehavior {
  move(direction: string): string;
  getSpeed(): number;
}

interface ElementalPower {
  element(): string;
  specialAttack(target: string): string;
}

// === 具体的なコンポーネント ===
class SwordAttack implements AttackBehavior {
  attack(target: string): string {
    return `剣で${target}を斬りつけた！`;
  }
  getRange(): number { return 1; }
}

class BowAttack implements AttackBehavior {
  attack(target: string): string {
    return `弓矢で${target}を射た！`;
  }
  getRange(): number { return 10; }
}

class MagicAttack implements AttackBehavior {
  attack(target: string): string {
    return `魔法で${target}を攻撃した！`;
  }
  getRange(): number { return 5; }
}

class ShieldDefense implements DefenseBehavior {
  defend(): string { return "盾で防御！"; }
  getArmor(): number { return 50; }
}

class DodgeDefense implements DefenseBehavior {
  defend(): string { return "素早く回避！"; }
  getArmor(): number { return 10; }
}

class MagicBarrier implements DefenseBehavior {
  defend(): string { return "魔法障壁を展開！"; }
  getArmor(): number { return 30; }
}

class WalkMovement implements MovementBehavior {
  move(direction: string): string { return `${direction}に歩いた`; }
  getSpeed(): number { return 3; }
}

class FlyMovement implements MovementBehavior {
  move(direction: string): string { return `${direction}に飛んだ`; }
  getSpeed(): number { return 8; }
}

class TeleportMovement implements MovementBehavior {
  move(direction: string): string { return `${direction}にテレポートした`; }
  getSpeed(): number { return 100; }
}

class FirePower implements ElementalPower {
  element(): string { return "炎"; }
  specialAttack(target: string): string {
    return `${target}を炎で焼き尽くした！`;
  }
}

class IcePower implements ElementalPower {
  element(): string { return "氷"; }
  specialAttack(target: string): string {
    return `${target}を氷で凍らせた！`;
  }
}

class LightningPower implements ElementalPower {
  element(): string { return "雷"; }
  specialAttack(target: string): string {
    return `${target}に雷を落とした！`;
  }
}

// === キャラクター（コンポジション）===
class Character {
  constructor(
    public name: string,
    private attackBehavior: AttackBehavior,
    private defenseBehavior: DefenseBehavior,
    private movementBehavior: MovementBehavior,
    private elementalPower?: ElementalPower,
  ) {}

  performAttack(target: string): string {
    return this.attackBehavior.attack(target);
  }

  performDefense(): string {
    return this.defenseBehavior.defend();
  }

  performMove(direction: string): string {
    return this.movementBehavior.move(direction);
  }

  performSpecial(target: string): string {
    if (!this.elementalPower) {
      return "特殊能力を持っていない";
    }
    return this.elementalPower.specialAttack(target);
  }

  // 実行時に振る舞いを変更可能！
  setAttackBehavior(attack: AttackBehavior): void {
    this.attackBehavior = attack;
  }

  setElementalPower(power: ElementalPower): void {
    this.elementalPower = power;
  }

  describe(): string {
    const parts = [
      `[${this.name}]`,
      `攻撃: ${this.attackBehavior.constructor.name}`,
      `防御: ${this.defenseBehavior.constructor.name}`,
      `移動: ${this.movementBehavior.constructor.name}`,
    ];
    if (this.elementalPower) {
      parts.push(`属性: ${this.elementalPower.element()}`);
    }
    return parts.join(" / ");
  }
}

// === 使用例 ===
// 炎の剣士: 剣 + 盾 + 歩行 + 炎
const fireWarrior = new Character(
  "炎の剣士",
  new SwordAttack(),
  new ShieldDefense(),
  new WalkMovement(),
  new FirePower(),
);

// 氷の魔法使い: 魔法 + 魔法障壁 + テレポート + 氷
const iceMage = new Character(
  "氷の魔法使い",
  new MagicAttack(),
  new MagicBarrier(),
  new TeleportMovement(),
  new IcePower(),
);

// 雷の弓使い: 弓 + 回避 + 飛行 + 雷
const lightningArcher = new Character(
  "雷の弓使い",
  new BowAttack(),
  new DodgeDefense(),
  new FlyMovement(),
  new LightningPower(),
);

// ゲーム中に装備を変更！
fireWarrior.setAttackBehavior(new BowAttack()); // 弓に持ち替え
fireWarrior.setElementalPower(new IcePower());   // 属性変更

console.log(fireWarrior.performAttack("ドラゴン"));  // "弓矢でドラゴンを射た！"
console.log(fireWarrior.performSpecial("ドラゴン")); // "ドラゴンを氷で凍らせた！"
```

### 2.3 Python でのコンポジション

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol, Optional


# === コンポーネントの定義 ===
class Renderer(Protocol):
    """レンダリング戦略"""
    def render(self, data: dict) -> str: ...


class Validator(Protocol):
    """バリデーション戦略"""
    def validate(self, data: dict) -> list[str]: ...


class Serializer(Protocol):
    """シリアライズ戦略"""
    def serialize(self, data: dict) -> str: ...
    def deserialize(self, raw: str) -> dict: ...


class Logger(Protocol):
    """ログ出力戦略"""
    def info(self, message: str) -> None: ...
    def error(self, message: str) -> None: ...


# === コンポーネントの実装 ===
class HtmlRenderer:
    def render(self, data: dict) -> str:
        rows = "".join(
            f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in data.items()
        )
        return f"<table>{rows}</table>"


class MarkdownRenderer:
    def render(self, data: dict) -> str:
        header = "| Key | Value |\n|-----|-------|\n"
        rows = "\n".join(f"| {k} | {v} |" for k, v in data.items())
        return header + rows


class JsonRenderer:
    def render(self, data: dict) -> str:
        import json
        return json.dumps(data, indent=2, ensure_ascii=False)


class StrictValidator:
    """厳密なバリデーション"""
    def __init__(self, required_fields: list[str]):
        self.required_fields = required_fields

    def validate(self, data: dict) -> list[str]:
        errors = []
        for field_name in self.required_fields:
            if field_name not in data or not data[field_name]:
                errors.append(f"'{field_name}' は必須です")
        return errors


class LenientValidator:
    """緩いバリデーション（警告のみ）"""
    def validate(self, data: dict) -> list[str]:
        return []  # 常に OK


class JsonSerializer:
    def serialize(self, data: dict) -> str:
        import json
        return json.dumps(data, ensure_ascii=False)

    def deserialize(self, raw: str) -> dict:
        import json
        return json.loads(raw)


class ConsoleLogger:
    def info(self, message: str) -> None:
        print(f"[INFO] {message}")

    def error(self, message: str) -> None:
        print(f"[ERROR] {message}")


class NullLogger:
    """何も出力しないロガー（テスト用）"""
    def info(self, message: str) -> None:
        pass

    def error(self, message: str) -> None:
        pass


# === コンポジションで組み立てる ===
@dataclass
class ReportGenerator:
    """レポート生成器: コンポーネントを組み合わせて構築"""
    renderer: Renderer
    validator: Validator
    serializer: Serializer
    logger: Logger

    def generate(self, data: dict) -> str:
        self.logger.info(f"レポート生成開始: {len(data)} 項目")

        # バリデーション
        errors = self.validator.validate(data)
        if errors:
            for error in errors:
                self.logger.error(f"バリデーションエラー: {error}")
            raise ValueError(f"バリデーション失敗: {errors}")

        # レンダリング
        rendered = self.renderer.render(data)
        self.logger.info(f"レンダリング完了: {len(rendered)} 文字")

        return rendered

    def save(self, data: dict, filepath: str) -> None:
        serialized = self.serializer.serialize(data)
        with open(filepath, "w") as f:
            f.write(serialized)
        self.logger.info(f"保存完了: {filepath}")

    def load(self, filepath: str) -> dict:
        with open(filepath) as f:
            raw = f.read()
        return self.serializer.deserialize(raw)


# === 使用例: 異なる用途で異なるコンポーネントを組み合わせ ===

# 本番環境: HTML + 厳密バリデーション + JSON + コンソールログ
production_report = ReportGenerator(
    renderer=HtmlRenderer(),
    validator=StrictValidator(["title", "author", "date"]),
    serializer=JsonSerializer(),
    logger=ConsoleLogger(),
)

# 開発環境: Markdown + 緩いバリデーション + JSON + 無出力ログ
dev_report = ReportGenerator(
    renderer=MarkdownRenderer(),
    validator=LenientValidator(),
    serializer=JsonSerializer(),
    logger=NullLogger(),
)

# テスト環境: JSON + 緩いバリデーション + JSON + 無出力ログ
test_report = ReportGenerator(
    renderer=JsonRenderer(),
    validator=LenientValidator(),
    serializer=JsonSerializer(),
    logger=NullLogger(),
)

# 同じ ReportGenerator クラスだが、振る舞いが全く異なる
data = {"title": "月次報告", "author": "田中", "date": "2026-01-01"}
print(production_report.generate(data))  # HTML形式
print(dev_report.generate(data))         # Markdown形式
print(test_report.generate(data))        # JSON形式
```

---

## 3. Strategy パターンとの関係

```typescript
// コンポジション + Strategy = 実行時に振る舞いを変更

interface SortStrategy {
  sort<T>(data: T[], compareFn: (a: T, b: T) => number): T[];
}

class QuickSort implements SortStrategy {
  sort<T>(data: T[], compareFn: (a: T, b: T) => number): T[] {
    // クイックソートの実装
    return [...data].sort(compareFn);
  }
}

class MergeSort implements SortStrategy {
  sort<T>(data: T[], compareFn: (a: T, b: T) => number): T[] {
    // マージソートの実装
    return [...data].sort(compareFn);
  }
}

class DataProcessor {
  // コンポジション: 戦略を外部から注入
  constructor(private sortStrategy: SortStrategy) {}

  // 実行時に戦略を変更可能
  setSortStrategy(strategy: SortStrategy): void {
    this.sortStrategy = strategy;
  }

  process(data: number[]): number[] {
    return this.sortStrategy.sort(data, (a, b) => a - b);
  }
}

const processor = new DataProcessor(new QuickSort());
processor.process([3, 1, 4, 1, 5]);
// データ量が増えたら戦略を変更
processor.setSortStrategy(new MergeSort());
```

### 3.1 デザインパターンとコンポジション

```
コンポジションを活用するデザインパターン:

  Strategy パターン:
    → 振る舞いを交換可能にする
    → 例: ソートアルゴリズム、認証戦略

  Decorator パターン:
    → 既存のオブジェクトに機能を追加
    → 例: ストリーム処理、ミドルウェア

  Observer パターン:
    → イベントの通知
    → 例: UIイベント、Pub/Sub

  Composite パターン:
    → ツリー構造の表現
    → 例: UIコンポーネント、ファイルシステム

  Bridge パターン:
    → 抽象と実装を分離
    → 例: プラットフォーム別の描画

  State パターン:
    → 状態に応じて振る舞いを変更
    → 例: ワークフロー、TCP接続

  Chain of Responsibility パターン:
    → 処理の連鎖
    → 例: ミドルウェアチェーン、バリデーション
```

### 3.2 Decorator パターン（コンポジションの応用）

```typescript
// Decorator パターン: 継承ではなくコンポジションで機能拡張

interface Logger {
  log(message: string): void;
}

class ConsoleLogger implements Logger {
  log(message: string): void {
    console.log(message);
  }
}

// デコレーター: 基本のロガーを包んで機能追加
class TimestampLogger implements Logger {
  constructor(private inner: Logger) {}

  log(message: string): void {
    const timestamp = new Date().toISOString();
    this.inner.log(`[${timestamp}] ${message}`);
  }
}

class PrefixLogger implements Logger {
  constructor(private inner: Logger, private prefix: string) {}

  log(message: string): void {
    this.inner.log(`${this.prefix} ${message}`);
  }
}

class JsonLogger implements Logger {
  constructor(private inner: Logger) {}

  log(message: string): void {
    this.inner.log(JSON.stringify({
      message,
      timestamp: new Date().toISOString(),
      level: "info",
    }));
  }
}

class FilterLogger implements Logger {
  constructor(private inner: Logger, private minLevel: string) {}

  log(message: string): void {
    // フィルタリングロジック
    if (this.shouldLog(message)) {
      this.inner.log(message);
    }
  }

  private shouldLog(message: string): boolean {
    // 簡易的なフィルタリング
    return !message.startsWith("[DEBUG]");
  }
}

// デコレーターを組み合わせて使う
const logger = new TimestampLogger(
  new PrefixLogger(
    new FilterLogger(
      new ConsoleLogger(),
      "info"
    ),
    "[MyApp]"
  )
);

logger.log("Hello, World!");
// → [2026-01-15T10:30:00.000Z] [MyApp] Hello, World!

// 継承で同じことをやろうとすると:
// TimestampConsoleLogger, TimestampFileLogger,
// PrefixConsoleLogger, PrefixFileLogger,
// TimestampPrefixConsoleLogger, TimestampPrefixFileLogger...
// → クラスの爆発！
```

```python
# Python: デコレーター（関数）を使ったコンポジション
from functools import wraps
from typing import Callable, Any
import time
import logging


def with_logging(func: Callable) -> Callable:
    """ログ出力を追加するデコレーター"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        logging.info(f"{func.__name__} returned {result}")
        return result
    return wrapper


def with_timing(func: Callable) -> Callable:
    """実行時間計測を追加するデコレーター"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper


def with_retry(max_retries: int = 3, delay: float = 1.0):
    """リトライを追加するデコレーター"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    print(f"Attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator


def with_cache(func: Callable) -> Callable:
    """結果をキャッシュするデコレーター"""
    cache = {}
    @wraps(func)
    def wrapper(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return wrapper


# デコレーターを組み合わせ（コンポジション）
@with_logging
@with_timing
@with_retry(max_retries=3)
@with_cache
def fetch_data(url: str) -> dict:
    """外部APIからデータを取得"""
    import requests
    response = requests.get(url)
    return response.json()

# 実行すると:
# 1. ログ出力（with_logging）
# 2. 時間計測開始（with_timing）
# 3. リトライ処理（with_retry）
# 4. キャッシュ確認（with_cache）
# 5. 実際の関数実行
```

### 3.3 State パターン（コンポジションで状態管理）

```typescript
// State パターン: 状態オブジェクトをコンポジションで持つ

interface OrderState {
  name: string;
  canConfirm(): boolean;
  canShip(): boolean;
  canCancel(): boolean;
  canDeliver(): boolean;
  confirm(order: Order): void;
  ship(order: Order): void;
  cancel(order: Order): void;
  deliver(order: Order): void;
}

class PendingState implements OrderState {
  name = "pending";
  canConfirm() { return true; }
  canShip() { return false; }
  canCancel() { return true; }
  canDeliver() { return false; }

  confirm(order: Order): void {
    console.log("注文を確認しました");
    order.setState(new ConfirmedState());
  }
  ship(order: Order): void {
    throw new Error("未確認の注文は出荷できません");
  }
  cancel(order: Order): void {
    console.log("注文をキャンセルしました");
    order.setState(new CancelledState());
  }
  deliver(order: Order): void {
    throw new Error("未確認の注文は配達できません");
  }
}

class ConfirmedState implements OrderState {
  name = "confirmed";
  canConfirm() { return false; }
  canShip() { return true; }
  canCancel() { return true; }
  canDeliver() { return false; }

  confirm(order: Order): void {
    throw new Error("既に確認済みです");
  }
  ship(order: Order): void {
    console.log("注文を出荷しました");
    order.setState(new ShippedState());
  }
  cancel(order: Order): void {
    console.log("確認済み注文をキャンセルしました（返金処理開始）");
    order.setState(new CancelledState());
  }
  deliver(order: Order): void {
    throw new Error("出荷前に配達はできません");
  }
}

class ShippedState implements OrderState {
  name = "shipped";
  canConfirm() { return false; }
  canShip() { return false; }
  canCancel() { return false; }
  canDeliver() { return true; }

  confirm(order: Order): void { throw new Error("出荷済み"); }
  ship(order: Order): void { throw new Error("既に出荷済み"); }
  cancel(order: Order): void { throw new Error("出荷済みの注文はキャンセルできません"); }
  deliver(order: Order): void {
    console.log("注文を配達しました");
    order.setState(new DeliveredState());
  }
}

class DeliveredState implements OrderState {
  name = "delivered";
  canConfirm() { return false; }
  canShip() { return false; }
  canCancel() { return false; }
  canDeliver() { return false; }

  confirm() { throw new Error("配達済み"); }
  ship() { throw new Error("配達済み"); }
  cancel() { throw new Error("配達済みの注文はキャンセルできません"); }
  deliver() { throw new Error("既に配達済み"); }
}

class CancelledState implements OrderState {
  name = "cancelled";
  canConfirm() { return false; }
  canShip() { return false; }
  canCancel() { return false; }
  canDeliver() { return false; }

  confirm() { throw new Error("キャンセル済み"); }
  ship() { throw new Error("キャンセル済み"); }
  cancel() { throw new Error("既にキャンセル済み"); }
  deliver() { throw new Error("キャンセル済み"); }
}

class Order {
  private state: OrderState = new PendingState(); // コンポジション

  setState(state: OrderState): void {
    console.log(`状態変更: ${this.state.name} → ${state.name}`);
    this.state = state;
  }

  confirm(): void { this.state.confirm(this); }
  ship(): void { this.state.ship(this); }
  cancel(): void { this.state.cancel(this); }
  deliver(): void { this.state.deliver(this); }

  getStatus(): string { return this.state.name; }
}

// 使用例
const order = new Order();
console.log(order.getStatus());  // "pending"
order.confirm();                  // 状態変更: pending → confirmed
order.ship();                     // 状態変更: confirmed → shipped
order.deliver();                  // 状態変更: shipped → delivered
// order.cancel();                // Error: 配達済みの注文はキャンセルできません
```

---

## 4. 継承が適切な場面

```
継承を使うべき場面:
  ✓ 明確な is-a 関係（ListはCollectionである）
  ✓ フレームワークの拡張ポイント（AbstractController）
  ✓ テンプレートメソッドパターン
  ✓ 子クラスが親の全メソッドを意味的に満たす
  ✓ 型階層が安定している（頻繁に変わらない）

コンポジションを使うべき場面:
  ✓ has-a 関係（CarはEngineを持つ）
  ✓ 振る舞いの組み合わせが必要
  ✓ 実行時に振る舞いを変更したい
  ✓ 複数の「機能」を組み合わせたい
  ✓ テスト時にモックに差し替えたい

迷ったとき:
  → コンポジションを選ぶ（より安全）
  → 「このクラスは本当に親の "一種" か？」を自問
  → 「コード再利用のためだけの継承」は避ける
```

### 4.1 テンプレートメソッドパターン（継承の適切な使用例）

```python
from abc import ABC, abstractmethod
from typing import Any


class ETLPipeline(ABC):
    """ETL（Extract-Transform-Load）パイプラインの基底クラス

    テンプレートメソッドパターン: アルゴリズムの骨格を定義し、
    具体的なステップをサブクラスに委ねる。
    """

    def run(self) -> dict:
        """テンプレートメソッド: ETLの全体フロー（final）"""
        self._log("パイプライン開始")

        # 1. データ抽出
        raw_data = self.extract()
        self._log(f"抽出完了: {len(raw_data)} レコード")

        # 2. データ変換
        transformed = self.transform(raw_data)
        self._log(f"変換完了: {len(transformed)} レコード")

        # 3. バリデーション（オプショナルフック）
        valid_data = self.validate(transformed)
        self._log(f"バリデーション完了: {len(valid_data)} レコード")

        # 4. データロード
        result = self.load(valid_data)
        self._log(f"ロード完了")

        # 5. 後処理（オプショナルフック）
        self.after_load(result)

        return result

    @abstractmethod
    def extract(self) -> list[dict]:
        """データを抽出する（サブクラスで実装）"""
        ...

    @abstractmethod
    def transform(self, data: list[dict]) -> list[dict]:
        """データを変換する（サブクラスで実装）"""
        ...

    @abstractmethod
    def load(self, data: list[dict]) -> dict:
        """データをロードする（サブクラスで実装）"""
        ...

    def validate(self, data: list[dict]) -> list[dict]:
        """バリデーション（デフォルト: すべて通過）"""
        return data

    def after_load(self, result: dict) -> None:
        """後処理（デフォルト: 何もしない）"""
        pass

    def _log(self, message: str) -> None:
        print(f"[{self.__class__.__name__}] {message}")


# サブクラス: CSV → PostgreSQL のETL
class CsvToPostgresETL(ETLPipeline):
    def __init__(self, csv_path: str, db_connection):
        self.csv_path = csv_path
        self.db = db_connection

    def extract(self) -> list[dict]:
        import csv
        with open(self.csv_path) as f:
            reader = csv.DictReader(f)
            return list(reader)

    def transform(self, data: list[dict]) -> list[dict]:
        # 型変換やクレンジング
        for row in data:
            row["price"] = float(row.get("price", 0))
            row["name"] = row.get("name", "").strip()
        return data

    def validate(self, data: list[dict]) -> list[dict]:
        # 価格が正の値のデータのみ
        return [row for row in data if row["price"] > 0]

    def load(self, data: list[dict]) -> dict:
        # PostgreSQLにINSERT
        count = 0
        for row in data:
            self.db.execute(
                "INSERT INTO products (name, price) VALUES (%s, %s)",
                (row["name"], row["price"]),
            )
            count += 1
        return {"inserted": count}


# サブクラス: API → Elasticsearch のETL
class ApiToElasticsearchETL(ETLPipeline):
    def __init__(self, api_url: str, es_client):
        self.api_url = api_url
        self.es = es_client

    def extract(self) -> list[dict]:
        import requests
        response = requests.get(self.api_url)
        return response.json()["results"]

    def transform(self, data: list[dict]) -> list[dict]:
        # Elasticsearch用にドキュメント変換
        return [
            {
                "_index": "products",
                "_id": item["id"],
                "_source": {
                    "name": item["name"],
                    "price": item["price"],
                    "category": item.get("category", "uncategorized"),
                },
            }
            for item in data
        ]

    def load(self, data: list[dict]) -> dict:
        # Elasticsearchにバルクインサート
        from elasticsearch.helpers import bulk
        success, errors = bulk(self.es, data)
        return {"success": success, "errors": len(errors)}

    def after_load(self, result: dict) -> None:
        # インデックスのリフレッシュ
        self.es.indices.refresh(index="products")
```

### 4.2 フレームワーク拡張（継承の適切な使用例）

```typescript
// フレームワークが提供する基底クラスの拡張
// → これは継承が適切な場面

// React の クラスコンポーネント（歴史的な例）
abstract class Component<P, S> {
  constructor(public props: P) {}
  abstract render(): VNode;

  setState(newState: Partial<S>): void {
    // フレームワーク内部の処理
  }

  componentDidMount(): void {}
  componentWillUnmount(): void {}
  shouldComponentUpdate(nextProps: P, nextState: S): boolean {
    return true;
  }
}

// フレームワークの拡張ポイントとして継承
class UserProfile extends Component<UserProps, UserState> {
  componentDidMount(): void {
    this.fetchUser(this.props.userId);
  }

  render(): VNode {
    // UIの描画
  }
}

// Express のミドルウェア基底クラス（仮想例）
abstract class Middleware {
  abstract handle(req: Request, res: Response, next: NextFunction): void;

  protected sendError(res: Response, status: number, message: string): void {
    res.status(status).json({ error: message });
  }
}

class AuthMiddleware extends Middleware {
  handle(req: Request, res: Response, next: NextFunction): void {
    const token = req.headers.authorization;
    if (!token) {
      this.sendError(res, 401, "認証が必要です");
      return;
    }
    // トークン検証...
    next();
  }
}

class RateLimitMiddleware extends Middleware {
  private requests = new Map<string, number[]>();

  handle(req: Request, res: Response, next: NextFunction): void {
    const ip = req.ip;
    const now = Date.now();
    const windowMs = 60000; // 1分

    const reqs = this.requests.get(ip) ?? [];
    const recent = reqs.filter(t => now - t < windowMs);

    if (recent.length >= 100) {
      this.sendError(res, 429, "リクエスト制限を超えました");
      return;
    }

    recent.push(now);
    this.requests.set(ip, recent);
    next();
  }
}
```

---

## 5. コンポジション vs 継承の判断フローチャート

```
判断フローチャート:

  Q1: 「B は A の一種か？」（is-a 関係か？）
  │
  ├── No → コンポジション
  │
  └── Yes
      │
      Q2: 「B は A の全メソッドを正しく実装できるか？」
      │
      ├── No → コンポジション（+ ISPでインターフェース分離）
      │
      └── Yes
          │
          Q3: 「A の実装詳細に B が依存する必要があるか？」
          │
          ├── Yes → 継承（ただし protected の使用を最小限に）
          │
          └── No
              │
              Q4: 「B の振る舞いは実行時に変更する必要があるか？」
              │
              ├── Yes → コンポジション（Strategy パターン）
              │
              └── No
                  │
                  Q5: 「型階層は安定しているか？」
                  │
                  ├── Yes → 継承で OK
                  │
                  └── No → コンポジション（将来の変更に備える）

具体的な判断例:

  ArrayList extends AbstractList → ✅ 継承（is-a + 安定した型階層）
  Stack extends Vector → ❌ 継承（Stack is-a Vector ではない）
  CountingSet extends HashSet → ❌ 継承（脆い基底クラス問題）
  Button extends Component → ✅ 継承（フレームワーク拡張）
  Car has-a Engine → ✅ コンポジション（has-a 関係）
  Logger has-a Formatter → ✅ コンポジション（実行時変更）
```

### 5.1 実務でよくあるケースの判断

```typescript
// ケース1: ログ出力のカスタマイズ
// ❌ 継承
class FileLogger extends ConsoleLogger { ... }
class JsonLogger extends FileLogger { ... }
// → ロガーは is-a 関係ではなく、出力先の違い

// ✅ コンポジション
class Logger {
  constructor(
    private transport: LogTransport,  // 出力先
    private formatter: LogFormatter,  // フォーマット
    private filter: LogFilter,        // フィルタ
  ) {}
}

// ケース2: HTTPクライアントの認証
// ❌ 継承
class AuthenticatedHttpClient extends HttpClient { ... }
class OAuthHttpClient extends AuthenticatedHttpClient { ... }

// ✅ コンポジション
class HttpClient {
  constructor(private auth: AuthStrategy) {}
  // BasicAuth, BearerToken, OAuth, NoAuth を差し替え可能
}

// ケース3: バリデーションロジック
// ❌ 継承
class EmailValidator extends StringValidator { ... }
class StrongPasswordValidator extends PasswordValidator { ... }

// ✅ コンポジション
class CompositeValidator implements Validator {
  constructor(private validators: Validator[]) {}
  validate(value: string): ValidationResult {
    const errors = this.validators
      .map(v => v.validate(value))
      .filter(r => !r.isValid);
    return errors.length === 0
      ? { isValid: true }
      : { isValid: false, errors: errors.flatMap(r => r.errors) };
  }
}

// バリデーションルールを組み合わせ
const passwordValidator = new CompositeValidator([
  new MinLengthValidator(8),
  new MaxLengthValidator(100),
  new ContainsUppercaseValidator(),
  new ContainsLowercaseValidator(),
  new ContainsDigitValidator(),
  new ContainsSpecialCharValidator(),
]);

// ケース4: データリポジトリ
// ❌ 継承
class CachedUserRepository extends PostgresUserRepository { ... }
// → キャッシュは永続化戦略ではない

// ✅ コンポジション（Decorator パターン）
class CachedRepository<T> implements Repository<T> {
  constructor(
    private inner: Repository<T>,
    private cache: CacheStore,
  ) {}

  async findById(id: string): Promise<T | null> {
    const cached = await this.cache.get(id);
    if (cached) return cached;
    const result = await this.inner.findById(id);
    if (result) await this.cache.set(id, result);
    return result;
  }
}

const userRepo = new CachedRepository(
  new PostgresUserRepository(db),
  new RedisCache(redis),
);
```

---

## 6. 言語ごとのコンポジション支援機能

```
各言語のコンポジション支援:

  Rust:
    → トレイト + impl → 明示的なコンポジション
    → 継承なし（設計上の意思決定）
    → derive マクロ → 自動実装

  Go:
    → 埋め込み（embedding）→ 委譲の糖衣構文
    → インターフェースは暗黙的
    → 継承なし（設計上の意思決定）

  Kotlin:
    → by キーワード → 委譲の糖衣構文
    → data class → 値オブジェクトの自動生成

  Swift:
    → protocol extension → プロトコルにデフォルト実装
    → protocol composition → プロトコルの合成

  TypeScript:
    → ミックスイン → クラス式による合成
    → インターセクション型 → 型レベルの合成
```

```go
// Go: 埋め込み（Embedding）によるコンポジション
type Logger struct{}

func (l *Logger) Log(msg string) {
    fmt.Printf("[LOG] %s\n", msg)
}

type Metrics struct{}

func (m *Metrics) RecordLatency(duration time.Duration) {
    fmt.Printf("[METRICS] latency: %v\n", duration)
}

// 埋め込みによるコンポジション（委譲の糖衣構文）
type Service struct {
    Logger   // 埋め込み: Service.Log() が使える
    Metrics  // 埋め込み: Service.RecordLatency() が使える
    db *sql.DB
}

func (s *Service) GetUser(id string) (*User, error) {
    start := time.Now()
    s.Log(fmt.Sprintf("Getting user: %s", id))  // Logger のメソッド

    var user User
    err := s.db.QueryRow("SELECT * FROM users WHERE id = $1", id).
        Scan(&user.ID, &user.Name)

    s.RecordLatency(time.Since(start))  // Metrics のメソッド
    return &user, err
}
```

```kotlin
// Kotlin: by キーワードによる委譲
interface Printer {
    fun print(message: String)
}

class ConsolePrinter : Printer {
    override fun print(message: String) {
        println(message)
    }
}

// by キーワードで委譲: printer に処理を委譲
class TimestampPrinter(private val printer: Printer) : Printer by printer {
    // print() は自動的に printer に委譲される

    // 必要に応じてオーバーライド
    override fun print(message: String) {
        val timestamp = java.time.LocalDateTime.now()
        printer.print("[$timestamp] $message")
    }
}

// 複数のインターフェースの委譲
interface Logger {
    fun log(message: String)
}

interface Cache {
    fun get(key: String): String?
    fun set(key: String, value: String)
}

class MyService(
    logger: Logger,
    cache: Cache,
) : Logger by logger, Cache by cache {
    // Logger と Cache の全メソッドが自動委譲
    // このクラスでは追加のビジネスロジックのみ定義

    fun processRequest(key: String): String {
        log("Processing request for key: $key")
        val cached = get(key)
        if (cached != null) {
            log("Cache hit for key: $key")
            return cached
        }
        val result = "computed_result"
        set(key, result)
        return result
    }
}
```

```rust
// Rust: トレイトによるコンポジション（継承なし）
trait Drawable {
    fn draw(&self);
}

trait Clickable {
    fn on_click(&mut self);
}

trait Resizable {
    fn resize(&mut self, width: u32, height: u32);
}

// 複数のトレイトを実装（コンポジション的）
struct Button {
    label: String,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    click_count: u32,
}

impl Drawable for Button {
    fn draw(&self) {
        println!("Drawing button '{}' at ({}, {})", self.label, self.x, self.y);
    }
}

impl Clickable for Button {
    fn on_click(&mut self) {
        self.click_count += 1;
        println!("Button '{}' clicked! (count: {})", self.label, self.click_count);
    }
}

impl Resizable for Button {
    fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }
}

// トレイトオブジェクトで動的ディスパッチ
fn draw_all(items: &[&dyn Drawable]) {
    for item in items {
        item.draw();
    }
}

// トレイト境界でジェネリクスの制約
fn interactive<T: Drawable + Clickable + Resizable>(widget: &mut T) {
    widget.draw();
    widget.on_click();
    widget.resize(200, 100);
    widget.draw();
}
```

---

## まとめ

| 観点 | 継承 | コンポジション |
|------|------|---------------|
| 関係 | is-a | has-a |
| 結合度 | 強い | 弱い |
| 柔軟性 | 低い | 高い |
| 実行時変更 | 不可 | 可能 |
| 推奨度 | 限定的 | 優先 |
| テスト容易性 | 低い | 高い |
| 再利用性 | 型階層に依存 | 独立して再利用可能 |

```
実践的な指針:

  1. デフォルトはコンポジション
     → 迷ったらコンポジションを選ぶ
     → 後から継承に変更するより、後からコンポジションに変更する方が困難

  2. 継承を使う条件:
     → 明確な is-a 関係がある
     → 親クラスの全メソッドがサブクラスで意味を持つ（LSP準拠）
     → 型階層が安定している
     → フレームワークが要求している

  3. 「コード再利用のための継承」は避ける
     → 共通コードが欲しいだけならユーティリティクラスやヘルパー関数
     → 振る舞いの再利用ならトレイト/ミックスイン

  4. 継承の深さは2〜3レベルまで
     → 深い継承ツリーは理解が困難
     → 「A → B → C → D → E」は危険信号

  5. 継承よりインターフェース
     → 型の互換性が必要ならインターフェースで十分
     → 実装の共有はコンポジション + 委譲で
```

---

## 次に読むべきガイド
→ [[01-interfaces-and-traits.md]] — インターフェースとトレイト

---

## 参考文献
1. Gamma, E. et al. "Design Patterns." Addison-Wesley, 1994. (Favor composition over inheritance)
2. Bloch, J. "Effective Java." Item 18: Favor composition over inheritance. 3rd Edition, 2018.
3. Martin, R. "Clean Architecture." Prentice Hall, 2017.
4. Sandi Metz. "Practical Object-Oriented Design in Ruby." 2nd Edition, 2018.
5. The Go Programming Language Specification. "Embedding." golang.org.
6. The Rust Programming Language. "Traits." doc.rust-lang.org.
