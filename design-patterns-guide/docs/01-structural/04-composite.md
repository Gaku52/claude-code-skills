# Composite パターン

> オブジェクトを **ツリー構造** に構成し、個別オブジェクトとその集合を同一インタフェースで扱えるようにする構造パターン。

---

## この章で学ぶこと

1. Composite パターンの構造と再帰的構成の仕組み
2. ファイルシステム・UI ツリー・組織図などへの実践的適用
3. 透過性と安全性のトレードオフ、適切な設計判断

---

## 1. Composite の構造

```
+------------------+
|   Component      |
|   (interface)    |
+------------------+
| + operation()    |
| + add(c)         |
| + remove(c)      |
| + getChild(i)    |
+------------------+
     ^          ^
     |          |
+--------+ +------------+
|  Leaf  | | Composite  |
+--------+ +------------+
|+operation| | -children[]|
+--------+ | +operation()|  -- 子に委譲して再帰
           | +add(c)     |
           | +remove(c)  |
           +------------+
```

---

## 2. ツリー構造の図解

```
        Composite (root)
        /       \
   Composite    Leaf C
   /      \
Leaf A   Leaf B

operation() の呼び出し:
root.operation()
  ├── composite.operation()
  │     ├── leafA.operation()
  │     └── leafB.operation()
  └── leafC.operation()
```

---

## 3. コード例

### コード例 1: ファイルシステム

```typescript
interface FileSystemNode {
  getName(): string;
  getSize(): number;
  print(indent?: string): void;
}

class File implements FileSystemNode {
  constructor(private name: string, private size: number) {}

  getName(): string { return this.name; }
  getSize(): number { return this.size; }
  print(indent = ""): void {
    console.log(`${indent}${this.name} (${this.size}KB)`);
  }
}

class Directory implements FileSystemNode {
  private children: FileSystemNode[] = [];

  constructor(private name: string) {}

  add(node: FileSystemNode): this {
    this.children.push(node);
    return this;
  }

  remove(node: FileSystemNode): void {
    this.children = this.children.filter(c => c !== node);
  }

  getName(): string { return this.name; }

  getSize(): number {
    return this.children.reduce((sum, c) => sum + c.getSize(), 0);
  }

  print(indent = ""): void {
    console.log(`${indent}${this.name}/`);
    this.children.forEach(c => c.print(indent + "  "));
  }
}

// 使用
const root = new Directory("src");
const components = new Directory("components");
components.add(new File("Button.tsx", 3));
components.add(new File("Modal.tsx", 5));
root.add(components);
root.add(new File("index.ts", 1));

root.print();
// src/
//   components/
//     Button.tsx (3KB)
//     Modal.tsx (5KB)
//   index.ts (1KB)

console.log(root.getSize()); // 9
```

### コード例 2: UI コンポーネントツリー

```typescript
interface UIComponent {
  render(): string;
  getBoundingBox(): { width: number; height: number };
}

class TextElement implements UIComponent {
  constructor(private text: string) {}
  render(): string { return `<span>${this.text}</span>`; }
  getBoundingBox() { return { width: this.text.length * 8, height: 16 }; }
}

class Container implements UIComponent {
  private children: UIComponent[] = [];

  constructor(private tag: string) {}

  add(child: UIComponent): this {
    this.children.push(child);
    return this;
  }

  render(): string {
    const inner = this.children.map(c => c.render()).join("");
    return `<${this.tag}>${inner}</${this.tag}>`;
  }

  getBoundingBox() {
    const width = Math.max(...this.children.map(c => c.getBoundingBox().width));
    const height = this.children.reduce((h, c) => h + c.getBoundingBox().height, 0);
    return { width, height };
  }
}

const page = new Container("div")
  .add(new Container("header").add(new TextElement("Title")))
  .add(new Container("main").add(new TextElement("Content")));

console.log(page.render());
```

### コード例 3: 価格計算（商品とバンドル）

```typescript
interface PriceItem {
  getPrice(): number;
  getDescription(): string;
}

class Product implements PriceItem {
  constructor(private name: string, private price: number) {}
  getPrice(): number { return this.price; }
  getDescription(): string { return this.name; }
}

class Bundle implements PriceItem {
  private items: PriceItem[] = [];
  constructor(private name: string, private discount: number = 0) {}

  add(item: PriceItem): this {
    this.items.push(item);
    return this;
  }

  getPrice(): number {
    const total = this.items.reduce((s, i) => s + i.getPrice(), 0);
    return total * (1 - this.discount);
  }

  getDescription(): string {
    const details = this.items.map(i => i.getDescription()).join(", ");
    return `${this.name} [${details}]`;
  }
}

const bundle = new Bundle("Starter Pack", 0.1)
  .add(new Product("Mouse", 3000))
  .add(new Product("Keyboard", 8000))
  .add(new Bundle("Cable Set", 0)
    .add(new Product("USB-C", 500))
    .add(new Product("HDMI", 800)));

console.log(bundle.getPrice()); // (3000+8000+500+800)*0.9 = 11070
```

### コード例 4: Python — 組織図

```python
from abc import ABC, abstractmethod

class OrganizationUnit(ABC):
    @abstractmethod
    def get_salary_cost(self) -> float: ...

    @abstractmethod
    def print_structure(self, indent: int = 0) -> None: ...

class Employee(OrganizationUnit):
    def __init__(self, name: str, salary: float):
        self.name = name
        self.salary = salary

    def get_salary_cost(self) -> float:
        return self.salary

    def print_structure(self, indent: int = 0) -> None:
        print(" " * indent + f"{self.name} ({self.salary})")

class Department(OrganizationUnit):
    def __init__(self, name: str):
        self.name = name
        self._members: list[OrganizationUnit] = []

    def add(self, unit: OrganizationUnit) -> None:
        self._members.append(unit)

    def get_salary_cost(self) -> float:
        return sum(m.get_salary_cost() for m in self._members)

    def print_structure(self, indent: int = 0) -> None:
        print(" " * indent + f"[{self.name}]")
        for m in self._members:
            m.print_structure(indent + 2)
```

### コード例 5: 条件式ツリー（仕様パターン）

```typescript
interface Specification<T> {
  isSatisfiedBy(item: T): boolean;
}

class AndSpec<T> implements Specification<T> {
  constructor(private specs: Specification<T>[]) {}

  isSatisfiedBy(item: T): boolean {
    return this.specs.every(s => s.isSatisfiedBy(item));
  }
}

class OrSpec<T> implements Specification<T> {
  constructor(private specs: Specification<T>[]) {}

  isSatisfiedBy(item: T): boolean {
    return this.specs.some(s => s.isSatisfiedBy(item));
  }
}

class NotSpec<T> implements Specification<T> {
  constructor(private spec: Specification<T>) {}

  isSatisfiedBy(item: T): boolean {
    return !this.spec.isSatisfiedBy(item);
  }
}

// Leaf
class PriceBelow implements Specification<Product> {
  constructor(private max: number) {}
  isSatisfiedBy(item: Product): boolean {
    return item.price < this.max;
  }
}

// 組み合わせ: 安くてかつ在庫あり、または新商品
const spec = new OrSpec([
  new AndSpec([new PriceBelow(1000), new InStock()]),
  new IsNew(),
]);
```

---

## 4. 再帰処理の流れ

```
getSize() の呼び出しスタック:

root.getSize()
  │
  ├─ components.getSize()
  │     ├─ Button.getSize() → 3
  │     └─ Modal.getSize()  → 5
  │     return 3 + 5 = 8
  │
  └─ index.getSize() → 1
  │
  return 8 + 1 = 9
```

---

## 5. 比較表

### 比較表 1: Composite vs 通常のコレクション

| 観点 | Composite | 配列/リスト |
|------|:---:|:---:|
| ネスト | 再帰的 | フラット |
| 統一インタフェース | Yes | No |
| 操作の委譲 | 自動（再帰） | 手動ループ |
| 型安全性 | 高い | 要キャスト |
| 適用場面 | 階層構造 | 均一なコレクション |

### 比較表 2: 透過設計 vs 安全設計

| 方式 | Leaf にも add/remove | 型安全性 | 透過性 |
|------|:---:|:---:|:---:|
| 透過設計 | Yes（例外スロー） | 低い | 高い |
| 安全設計 | No（Composite のみ） | 高い | 低い |
| 推奨 | - | 安全設計 | - |

---

## 6. アンチパターン

### アンチパターン 1: 無限再帰の許容

```typescript
// BAD: 循環参照を防止していない
class BadComposite {
  add(child: BadComposite): void {
    this.children.push(child); // 自分自身を追加できてしまう
  }
}

const a = new BadComposite();
a.add(a); // 無限ループ！
a.getSize(); // スタックオーバーフロー
```

**改善**: add() 時に循環参照チェックを行う。

### アンチパターン 2: Composite に Leaf 固有のロジック

```typescript
// BAD: Composite が子の型を知っている
class Directory {
  getSize(): number {
    return this.children.reduce((sum, c) => {
      if (c instanceof File) {        // 型チェック
        return sum + c.getRawSize();   // Leaf 固有メソッド
      }
      return sum + c.getSize();
    }, 0);
  }
}
```

**改善**: Component インタフェースの `getSize()` に統一し、型チェックを排除する。

---

## 7. FAQ

### Q1: React の仮想 DOM は Composite パターンですか？

はい。React のコンポーネントツリーは Composite パターンの典型例です。各コンポーネントは `render()` を持ち、子コンポーネントに再帰的に処理を委譲します。

### Q2: Visitor パターンと Composite は併用できますか？

はい。Composite でツリー構造を表現し、Visitor で操作を追加するのは GoF でも推奨される組み合わせです。新しい操作を追加する際にツリーのクラスを変更する必要がなくなります。

### Q3: 深いネストはパフォーマンス問題になりますか？

再帰の深さが数百レベルを超えるとスタックオーバーフローのリスクがあります。その場合は反復（イテレーティブ）なトラバーサルに変更するか、末尾呼出し最適化を利用します。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| 目的 | ツリー構造を統一インタフェースで操作 |
| 構成 | Component, Leaf, Composite |
| 典型例 | ファイルシステム、UI、組織図 |
| 注意 | 循環参照防止、安全設計推奨 |
| 再帰 | Composite が子に操作を委譲 |

---

## 次に読むべきガイド

- [Iterator パターン](../02-behavioral/04-iterator.md) — ツリー走査の抽象化
- [Decorator パターン](./01-decorator.md) — 動的な機能追加
- [合成優先の原則](../../../clean-code-principles/docs/03-practices-advanced/01-composition-over-inheritance.md)

---

## 参考文献

1. Gamma, E. et al. (1994). *Design Patterns: Elements of Reusable Object-Oriented Software*. Addison-Wesley.
2. Freeman, E. et al. (2004). *Head First Design Patterns*. O'Reilly Media.
3. Refactoring.Guru — Composite. https://refactoring.guru/design-patterns/composite
