# Strategy パターン

> アルゴリズムのファミリーを定義し、それぞれを **カプセル化** して交換可能にする振る舞いパターン。実行時にアルゴリズムを切り替えられる。

---

## この章で学ぶこと

1. Strategy パターンの構造と、条件分岐の排除による OCP 準拠の設計
2. DI（依存性注入）との関係と、関数型アプローチでの実現方法
3. Strategy の過剰適用を避ける判断基準

---

## 1. Strategy の構造

```
+-------------+       +-------------------+
|   Context   |------>|   Strategy        |
+-------------+       |   (interface)     |
| - strategy  |       +-------------------+
| + execute() |       | + execute(data)   |
+-------------+       +-------------------+
                              ^
                       _______|_______
                      |               |
               +------------+  +------------+
               | StrategyA  |  | StrategyB  |
               +------------+  +------------+
               | + execute() |  | + execute() |
               +------------+  +------------+
```

---

## 2. コード例

### コード例 1: 料金計算 Strategy

```typescript
interface PricingStrategy {
  calculate(basePrice: number): number;
}

class RegularPricing implements PricingStrategy {
  calculate(basePrice: number): number {
    return basePrice;
  }
}

class PremiumPricing implements PricingStrategy {
  calculate(basePrice: number): number {
    return basePrice * 0.9; // 10%割引
  }
}

class StudentPricing implements PricingStrategy {
  calculate(basePrice: number): number {
    return basePrice * 0.7; // 30%割引
  }
}

class ShoppingCart {
  constructor(private pricingStrategy: PricingStrategy) {}

  setPricingStrategy(strategy: PricingStrategy): void {
    this.pricingStrategy = strategy;
  }

  checkout(items: { price: number }[]): number {
    const base = items.reduce((sum, i) => sum + i.price, 0);
    return this.pricingStrategy.calculate(base);
  }
}

// 使用: 実行時に戦略を切り替え
const cart = new ShoppingCart(new RegularPricing());
console.log(cart.checkout([{ price: 1000 }])); // 1000

cart.setPricingStrategy(new StudentPricing());
console.log(cart.checkout([{ price: 1000 }])); // 700
```

### コード例 2: 関数型 Strategy

```typescript
// クラスを使わず関数で Strategy を実現
type SortStrategy<T> = (a: T, b: T) => number;

const byName: SortStrategy<User> = (a, b) => a.name.localeCompare(b.name);
const byAge: SortStrategy<User> = (a, b) => a.age - b.age;
const byCreatedDesc: SortStrategy<User> = (a, b) =>
  b.createdAt.getTime() - a.createdAt.getTime();

function sortUsers(users: User[], strategy: SortStrategy<User>): User[] {
  return [...users].sort(strategy);
}

// 使用
const sorted = sortUsers(users, byAge);
```

### コード例 3: バリデーション Strategy

```typescript
interface ValidationStrategy {
  validate(value: string): { valid: boolean; message?: string };
}

class EmailValidation implements ValidationStrategy {
  validate(value: string) {
    const valid = /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value);
    return { valid, message: valid ? undefined : "Invalid email format" };
  }
}

class PasswordValidation implements ValidationStrategy {
  validate(value: string) {
    if (value.length < 8) return { valid: false, message: "Min 8 chars" };
    if (!/[A-Z]/.test(value)) return { valid: false, message: "Need uppercase" };
    if (!/[0-9]/.test(value)) return { valid: false, message: "Need digit" };
    return { valid: true };
  }
}

class FormField {
  constructor(
    private name: string,
    private strategy: ValidationStrategy
  ) {}

  validate(value: string) {
    return this.strategy.validate(value);
  }
}
```

### コード例 4: Python — Strategy with Protocol

```python
from typing import Protocol

class CompressionStrategy(Protocol):
    def compress(self, data: bytes) -> bytes: ...
    def decompress(self, data: bytes) -> bytes: ...

class GzipCompression:
    def compress(self, data: bytes) -> bytes:
        import gzip
        return gzip.compress(data)

    def decompress(self, data: bytes) -> bytes:
        import gzip
        return gzip.decompress(data)

class ZstdCompression:
    def compress(self, data: bytes) -> bytes:
        import zstandard
        return zstandard.ZstdCompressor().compress(data)

    def decompress(self, data: bytes) -> bytes:
        import zstandard
        return zstandard.ZstdDecompressor().decompress(data)

class FileProcessor:
    def __init__(self, compression: CompressionStrategy):
        self._compression = compression

    def save(self, data: bytes, path: str) -> None:
        compressed = self._compression.compress(data)
        with open(path, "wb") as f:
            f.write(compressed)

processor = FileProcessor(GzipCompression())
processor.save(b"Hello World", "data.gz")
```

### コード例 5: Strategy の動的選択

```typescript
// Registry + Strategy
class StrategyRegistry<T> {
  private strategies = new Map<string, T>();

  register(name: string, strategy: T): void {
    this.strategies.set(name, strategy);
  }

  get(name: string): T {
    const s = this.strategies.get(name);
    if (!s) throw new Error(`Strategy "${name}" not found`);
    return s;
  }
}

const pricingRegistry = new StrategyRegistry<PricingStrategy>();
pricingRegistry.register("regular", new RegularPricing());
pricingRegistry.register("premium", new PremiumPricing());
pricingRegistry.register("student", new StudentPricing());

// 設定や API パラメータから動的に選択
const strategy = pricingRegistry.get(user.membershipType);
const total = strategy.calculate(basePrice);
```

---

## 3. if/else の排除

```
BEFORE (条件分岐の肥大化):
function calculate(type, price) {
  if (type === "regular") return price;
  else if (type === "premium") return price * 0.9;
  else if (type === "student") return price * 0.7;
  else if (type === "senior") return price * 0.8;
  // ... 追加のたびに変更 → OCP 違反
}

AFTER (Strategy パターン):
strategies.get(type).calculate(price);
  → 新しい型は register するだけ → OCP 準拠
```

---

## 4. 比較表

### 比較表 1: Strategy vs State vs Command

| 観点 | Strategy | State | Command |
|------|:---:|:---:|:---:|
| 目的 | アルゴリズム交換 | 状態依存の振る舞い | 操作のカプセル化 |
| 交換タイミング | クライアントが決定 | 内部状態で自動遷移 | キュー/履歴に保存 |
| 典型的な数 | 少数 | 有限個の状態 | 多数のコマンド |
| Undo | なし | なし | あり |

### 比較表 2: クラス Strategy vs 関数 Strategy

| 観点 | クラスベース | 関数ベース |
|------|:---:|:---:|
| 状態保持 | 可能 | クロージャで可能 |
| テスト | インスタンス化 | 直接呼び出し |
| コード量 | 多い | 少ない |
| 型安全性 | 高い | 中 |
| 適用場面 | 複雑な戦略 | 単純な戦略 |

---

## 5. アンチパターン

### アンチパターン 1: 戦略が2つしかないのに Strategy パターン

```typescript
// BAD: 過剰設計
interface GreetingStrategy {
  greet(name: string): string;
}
class FormalGreeting implements GreetingStrategy {
  greet(name: string) { return `Dear ${name}`; }
}
class CasualGreeting implements GreetingStrategy {
  greet(name: string) { return `Hi ${name}`; }
}
// ↑ 三項演算子で十分
const greet = (name: string, formal: boolean) =>
  formal ? `Dear ${name}` : `Hi ${name}`;
```

**改善**: バリエーションが3つ以上、または将来の追加が見込まれる場合にのみ導入する。

### アンチパターン 2: Context が Strategy の内部を知っている

```typescript
// BAD: Context が Strategy の型をチェック
class Context {
  execute(): void {
    if (this.strategy instanceof StrategyA) {
      // StrategyA 固有の前処理
    }
    this.strategy.execute();
  }
}
```

**改善**: Context は Strategy インタフェースのみに依存し、具象型を知らない設計にする。

---

## 6. FAQ

### Q1: Strategy と DI は同じですか？

DI は依存の注入メカニズム、Strategy はアルゴリズム交換のパターンです。DI は Strategy を実現する手段として使えますが、Strategy は DI なしでも実装可能です。

### Q2: JavaScript では関数を渡すだけで Strategy は実現できますか？

はい。コールバック関数は Strategy パターンの軽量な実装です。`Array.sort(compareFn)` が典型例です。複雑な状態を持つ戦略にはクラスが適しています。

### Q3: Strategy と Template Method の違いは？

Strategy は委譲（has-a）でアルゴリズム全体を交換します。Template Method は継承（is-a）でアルゴリズムの一部をオーバーライドします。Strategy の方が柔軟性が高く推奨されます。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| 目的 | アルゴリズムをカプセル化して交換可能に |
| OCP | 新しい戦略を追加しても既存コード変更不要 |
| 関数型 | 関数を渡すだけでも実現可能 |
| 判断基準 | 3+ バリエーション or 将来の拡張が見込まれる |
| 注意 | Context は Strategy の具象型を知らない |

---

## 次に読むべきガイド

- [Command パターン](./02-command.md) — 操作のカプセル化
- [State パターン](./03-state.md) — 状態遷移
- [SOLID 原則](../../../clean-code-principles/docs/00-principles/01-solid.md) — OCP の詳細

---

## 参考文献

1. Gamma, E. et al. (1994). *Design Patterns: Elements of Reusable Object-Oriented Software*. Addison-Wesley.
2. Freeman, E. et al. (2004). *Head First Design Patterns*. O'Reilly Media.
3. Refactoring.Guru — Strategy. https://refactoring.guru/design-patterns/strategy
