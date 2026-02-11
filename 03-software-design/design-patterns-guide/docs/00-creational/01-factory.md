# Factory Method / Abstract Factory パターン

> オブジェクト生成をサブクラスや専用ファクトリに **委譲** し、生成ロジックを利用側から分離する生成パターン。

---

## この章で学ぶこと

1. Factory Method と Abstract Factory の違いと使い分け
2. 生成ロジックの抽象化による柔軟性とテスト容易性の向上
3. 実務での適用場面と過剰設計を避けるための判断基準

---

## 1. Factory Method の構造

```
+----------------+         +----------------+
|   Creator      |         |   Product      |
|----------------|         |  (interface)   |
| + factoryMethod|-------->+----------------+
| + operation()  |         | + use()        |
+----------------+         +----------------+
        ^                          ^
        |                          |
+----------------+         +----------------+
| ConcreteCreator|         | ConcreteProduct|
|----------------|         +----------------+
| + factoryMethod|-------->| + use()        |
+----------------+         +----------------+
```

---

## 2. Abstract Factory の構造

```
+-------------------------+       +----------+  +----------+
| AbstractFactory         |       | ProductA |  | ProductB |
|-------------------------|       |(interface)|  |(interface)|
| + createProductA()      |       +----------+  +----------+
| + createProductB()      |            ^              ^
+-------------------------+            |              |
        ^           ^           +----------+  +----------+
        |           |           |ConcreteA1|  |ConcreteB1|
+------------+ +------------+  +----------+  +----------+
|Factory1    | |Factory2    |
|createA()->A1 |createA()->A2
|createB()->B1 |createB()->B2
+------------+ +------------+
```

---

## 3. コード例

### コード例 1: Factory Method（TypeScript）

```typescript
// Product インタフェース
interface Notification {
  send(message: string): void;
}

// Concrete Products
class EmailNotification implements Notification {
  send(message: string): void {
    console.log(`Email: ${message}`);
  }
}

class SlackNotification implements Notification {
  send(message: string): void {
    console.log(`Slack: ${message}`);
  }
}

// Creator
abstract class NotificationService {
  abstract createNotification(): Notification;

  notify(message: string): void {
    const notification = this.createNotification();
    notification.send(message);
  }
}

// Concrete Creators
class EmailService extends NotificationService {
  createNotification(): Notification {
    return new EmailNotification();
  }
}

class SlackService extends NotificationService {
  createNotification(): Notification {
    return new SlackNotification();
  }
}

// 使用
const service: NotificationService = new SlackService();
service.notify("デプロイ完了");
```

### コード例 2: Simple Factory（関数ベース）

```typescript
type NotificationType = "email" | "slack" | "sms";

function createNotification(type: NotificationType): Notification {
  switch (type) {
    case "email": return new EmailNotification();
    case "slack": return new SlackNotification();
    case "sms":   return new SmsNotification();
    default:
      throw new Error(`Unknown type: ${type}`);
  }
}

const n = createNotification("email");
n.send("Hello");
```

### コード例 3: Abstract Factory（TypeScript）

```typescript
// 抽象プロダクト
interface Button { render(): string; }
interface Input  { render(): string; }

// 抽象ファクトリ
interface UIFactory {
  createButton(): Button;
  createInput(): Input;
}

// Concrete: Material Design
class MaterialButton implements Button {
  render() { return "<MaterialButton />"; }
}
class MaterialInput implements Input {
  render() { return "<MaterialInput />"; }
}
class MaterialFactory implements UIFactory {
  createButton() { return new MaterialButton(); }
  createInput()  { return new MaterialInput(); }
}

// Concrete: iOS Style
class IOSButton implements Button {
  render() { return "<IOSButton />"; }
}
class IOSInput implements Input {
  render() { return "<IOSInput />"; }
}
class IOSFactory implements UIFactory {
  createButton() { return new IOSButton(); }
  createInput()  { return new IOSInput(); }
}

// 使用: ファクトリを差し替えるだけで UI 全体が変わる
function buildUI(factory: UIFactory) {
  const btn = factory.createButton();
  const input = factory.createInput();
  return `${btn.render()} ${input.render()}`;
}

console.log(buildUI(new MaterialFactory()));
// "<MaterialButton /> <MaterialInput />"
```

### コード例 4: Python — Factory Method

```python
from abc import ABC, abstractmethod

class Serializer(ABC):
    @abstractmethod
    def serialize(self, data: dict) -> str: ...

class JsonSerializer(Serializer):
    def serialize(self, data: dict) -> str:
        import json
        return json.dumps(data)

class XmlSerializer(Serializer):
    def serialize(self, data: dict) -> str:
        return f"<data>{data}</data>"

def get_serializer(fmt: str) -> Serializer:
    factories = {"json": JsonSerializer, "xml": XmlSerializer}
    return factories[fmt]()

s = get_serializer("json")
print(s.serialize({"key": "value"}))
```

### コード例 5: Registry パターン（拡張可能 Factory）

```typescript
type Creator<T> = () => T;

class NotificationRegistry {
  private static registry = new Map<string, Creator<Notification>>();

  static register(type: string, creator: Creator<Notification>) {
    this.registry.set(type, creator);
  }

  static create(type: string): Notification {
    const creator = this.registry.get(type);
    if (!creator) throw new Error(`Unknown: ${type}`);
    return creator();
  }
}

// 登録（各モジュールが自分で登録）
NotificationRegistry.register("email", () => new EmailNotification());
NotificationRegistry.register("slack", () => new SlackNotification());

// OCP 準拠: 新しい型は register するだけ
NotificationRegistry.register("teams", () => new TeamsNotification());

const n = NotificationRegistry.create("teams");
```

---

## 4. Factory の選択フロー

```
生成ロジックを分離したい？
        |
        Yes
        |
  +----- 生成するプロダクトは1種類？
  |              |
  Yes            No (ファミリー)
  |              |
  v              v
Factory Method   Abstract Factory
  |
  +--- 型で分岐するだけ？
  |         |
  Yes       No (テンプレートメソッドが要る)
  |         |
  v         v
Simple     Factory Method
Factory    (サブクラス化)
```

---

## 5. 比較表

### 比較表 1: Factory Method vs Abstract Factory

| 観点 | Factory Method | Abstract Factory |
|------|---------------|-----------------|
| 意図 | 1つのプロダクトの生成を委譲 | 関連するプロダクト群の生成 |
| クラス数 | 少ない | 多い |
| 拡張方法 | Creator のサブクラス追加 | Factory + Product 群の追加 |
| 使用場面 | 単一軸のバリエーション | テーマ/プラットフォーム等の直交軸 |
| 複雑度 | 低〜中 | 中〜高 |

### 比較表 2: Factory vs その他の生成パターン

| パターン | 目的 | 生成の自由度 | 複雑度 |
|---------|------|:---:|:---:|
| Simple Factory | 条件分岐で生成 | 低 | 低 |
| Factory Method | サブクラスに委譲 | 中 | 中 |
| Abstract Factory | ファミリー生成 | 高 | 高 |
| Builder | 段階的構築 | 高 | 中 |
| Prototype | クローンで生成 | 中 | 低 |

---

## 6. アンチパターン

### アンチパターン 1: 万能 Factory

```typescript
// BAD: あらゆる型を1つの Factory で処理
class UniversalFactory {
  create(type: string): any {  // any は危険信号
    if (type === "user") return new User();
    if (type === "order") return new Order();
    if (type === "product") return new Product();
    // ... 50行の if-else
  }
}
```

**問題**: OCP 違反。新しい型追加のたびにこのクラスを変更する必要がある。型安全性もない。

**改善**: Registry パターンや個別の Factory に分割する。

### アンチパターン 2: 不要な Abstract Factory

```typescript
// BAD: プロダクトが1種類なのに Abstract Factory を使う
interface ShapeFactory {
  createShape(): Shape;  // 1メソッドだけ
}
```

**問題**: Factory Method で十分な場面に過剰な抽象化を持ち込んでいる。

**改善**: YAGNI 原則に従い、複数プロダクトが必要になるまで Factory Method を使う。

---

## 7. FAQ

### Q1: Simple Factory と Factory Method の違いは？

Simple Factory は単なる関数やクラスメソッドで条件分岐し生成します。Factory Method はサブクラス化により生成をオーバーライドする GoF パターンです。多くの実務では Simple Factory で十分です。

### Q2: Factory を使うべき判断基準は？

(1) `new` の呼び出し箇所が複数に散らばっている、(2) 生成ロジックが複雑、(3) テストでモック差し替えが必要、のいずれかに該当すれば Factory の導入を検討します。

### Q3: DI コンテナがあれば Factory は不要ですか？

DI コンテナは起動時に依存を解決しますが、**実行時に動的に型を切り替える** 場合は Factory が必要です。両者は補完関係にあります。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| Factory Method | 1プロダクトの生成をサブクラスに委譲 |
| Abstract Factory | 関連プロダクト群をまとめて生成 |
| Simple Factory | 最も軽量、関数1つで実現 |
| Registry | OCP 準拠の拡張可能 Factory |
| 判断基準 | 生成の複雑さ・バリエーション数で選択 |

---

## 次に読むべきガイド

- [Builder パターン](./02-builder.md) — 複雑なオブジェクト構築
- [Prototype パターン](./03-prototype.md) — クローンによる生成
- [Strategy パターン](../02-behavioral/01-strategy.md) — アルゴリズムの交換

---

## 参考文献

1. Gamma, E. et al. (1994). *Design Patterns: Elements of Reusable Object-Oriented Software*. Addison-Wesley.
2. Freeman, E. et al. (2004). *Head First Design Patterns*. O'Reilly Media.
3. Refactoring.Guru — Factory Method. https://refactoring.guru/design-patterns/factory-method
