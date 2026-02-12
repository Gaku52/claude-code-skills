# Composite パターン

> オブジェクトを **ツリー構造** に構成し、個別オブジェクト（Leaf）とその集合（Composite）を同一インタフェースで扱えるようにする構造パターン。クライアントは再帰的なツリー構造全体を統一的に操作でき、個別要素か集合かを意識する必要がなくなる。

---

## この章で学ぶこと

1. Composite パターンの構造と再帰的構成の仕組み、GoF による設計意図を深く理解する
2. ファイルシステム、UIツリー、組織図、メニュー構造、数式ツリーなど実践的な適用場面を習得する
3. 透過性と安全性のトレードオフ、循環参照防止、パフォーマンス最適化の設計判断ができるようになる

---

## 前提知識

このガイドを読む前に、以下の概念を理解しておくことを推奨します。

| 前提知識 | 説明 | 参照リンク |
|---------|------|-----------|
| インタフェースとポリモーフィズム | 共通のインタフェースを通じて異なる型を統一的に扱う概念 | [SOLID 原則](../../../clean-code-principles/docs/00-principles/01-solid.md) |
| 再帰 | 関数やデータ構造が自分自身を参照する概念 | [CS 基礎](../../../01-cs-fundamentals/) |
| ツリーデータ構造 | ノードとエッジで構成される階層的なデータ構造 | [データ構造](../../../01-cs-fundamentals/) |
| 合成（Composition）と継承 | オブジェクトの組み立て方法の違い | [合成優先の原則](../../../clean-code-principles/docs/03-practices-advanced/01-composition-over-inheritance.md) |

---

## 1. Composite パターンとは何か

### 1.1 解決する問題

ソフトウェア開発では「部分と全体」の関係を扱う場面が頻繁にある。例えば:

- ファイルとフォルダ（フォルダはファイルとサブフォルダを含む）
- UIの基本要素とコンテナ（コンテナはボタンやテキストやサブコンテナを含む）
- 組織の従業員と部門（部門は従業員やサブ部門を含む）

これらに共通するのは「**個別の要素と、要素のグループを同じように扱いたい**」という要求である。Composite パターンなしでは、クライアントコードが「これは Leaf か？ Composite か？」を常にチェックする必要があり、コードが複雑化する。

### 1.2 パターンの意図

GoF の定義:

> Compose objects into tree structures to represent part-whole hierarchies. Composite lets clients treat individual objects and compositions of objects uniformly.

日本語に訳すと:

> オブジェクトをツリー構造に組み立てて部分-全体の階層を表現する。Composite パターンにより、クライアントは個々のオブジェクトとオブジェクトの合成物を統一的に扱うことができる。

### 1.3 WHY: なぜ Composite パターンが必要なのか

根本的な理由は **抽象化のレベルを揃える** ことにある。

1. **クライアントコードの簡素化**: `if (isLeaf)` のような分岐が不要になり、再帰的に統一操作を適用できる
2. **Open/Closed Principle の遵守**: 新しい Leaf 型や Composite 型を追加しても、既存のクライアントコードを変更する必要がない
3. **再帰的構造の自然な表現**: ツリー構造を言語の型システムで直接表現でき、操作も自然に再帰的に定義できる

---

## 2. Composite の構造

### 2.1 クラス図

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

### 2.2 構成要素の役割

| 構成要素 | 役割 | 例（ファイルシステム） |
|---------|------|---------------------|
| Component | 統一インタフェース | FileSystemNode |
| Leaf | 子を持たない末端ノード | File |
| Composite | 子を持つノード、操作を子に委譲 | Directory |
| Client | Component を通じて操作 | アプリケーション |

### 2.3 ツリー構造の図解

```
        Composite (root)
        /       \
   Composite    Leaf C
   /      \
Leaf A   Leaf B

operation() の呼び出し:
root.operation()
  +-- composite.operation()
  |     +-- leafA.operation()
  |     +-- leafB.operation()
  +-- leafC.operation()
```

### 2.4 再帰処理の詳細フロー

```
getSize() の呼び出しスタック:

root.getSize()
  |
  +- components.getSize()
  |     +- Button.getSize() -> 3
  |     +- Modal.getSize()  -> 5
  |     return 3 + 5 = 8
  |
  +- index.getSize() -> 1
  |
  return 8 + 1 = 9

[コールスタックの深さ]
depth 0: root.getSize()
depth 1: components.getSize(), index.getSize()
depth 2: Button.getSize(), Modal.getSize()

時間計算量: O(N) — 全ノードを1回ずつ訪問
空間計算量: O(D) — D はツリーの最大深さ（再帰スタック）
```

---

## 3. コード例

### コード例 1: ファイルシステム（TypeScript）

```typescript
// file-system.ts — Composite パターンの典型例
interface FileSystemNode {
  getName(): string;
  getSize(): number;
  print(indent?: string): void;
  find(predicate: (node: FileSystemNode) => boolean): FileSystemNode[];
}

class File implements FileSystemNode {
  constructor(
    private name: string,
    private size: number,
    private extension: string = ''
  ) {}

  getName(): string { return this.name; }
  getSize(): number { return this.size; }

  print(indent = ""): void {
    console.log(`${indent}${this.name} (${this.size}KB)`);
  }

  find(predicate: (node: FileSystemNode) => boolean): FileSystemNode[] {
    return predicate(this) ? [this] : [];
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

  getChildren(): readonly FileSystemNode[] {
    return this.children;
  }

  getName(): string { return this.name; }

  getSize(): number {
    return this.children.reduce((sum, c) => sum + c.getSize(), 0);
  }

  print(indent = ""): void {
    console.log(`${indent}${this.name}/`);
    this.children.forEach(c => c.print(indent + "  "));
  }

  find(predicate: (node: FileSystemNode) => boolean): FileSystemNode[] {
    const results: FileSystemNode[] = [];
    if (predicate(this)) results.push(this);
    for (const child of this.children) {
      results.push(...child.find(predicate));
    }
    return results;
  }
}

// --- 使用例 ---
const root = new Directory("src");
const components = new Directory("components");
components.add(new File("Button.tsx", 3, "tsx"));
components.add(new File("Modal.tsx", 5, "tsx"));
const utils = new Directory("utils");
utils.add(new File("format.ts", 2, "ts"));
root.add(components);
root.add(utils);
root.add(new File("index.ts", 1, "ts"));

root.print();
// 出力:
// src/
//   components/
//     Button.tsx (3KB)
//     Modal.tsx (5KB)
//   utils/
//     format.ts (2KB)
//   index.ts (1KB)

console.log(`Total size: ${root.getSize()}KB`); // 11

// find で条件に合うノードを検索
const largeFiles = root.find(node => node.getSize() > 3);
console.log(largeFiles.map(f => f.getName()));
// ["components", "Modal.tsx"] — Directory は合計サイズで評価される
```

### コード例 2: UI コンポーネントツリー（TypeScript）

```typescript
// ui-component.ts — UI のレンダリングツリー
interface UIComponent {
  render(): string;
  getBoundingBox(): { width: number; height: number };
  findById(id: string): UIComponent | null;
}

class TextElement implements UIComponent {
  constructor(
    private id: string,
    private text: string,
    private fontSize: number = 16
  ) {}

  render(): string {
    return `<span id="${this.id}" style="font-size:${this.fontSize}px">${this.text}</span>`;
  }

  getBoundingBox() {
    return { width: this.text.length * (this.fontSize * 0.6), height: this.fontSize };
  }

  findById(id: string): UIComponent | null {
    return this.id === id ? this : null;
  }
}

class ImageElement implements UIComponent {
  constructor(
    private id: string,
    private src: string,
    private width: number,
    private height: number
  ) {}

  render(): string {
    return `<img id="${this.id}" src="${this.src}" width="${this.width}" height="${this.height}" />`;
  }

  getBoundingBox() {
    return { width: this.width, height: this.height };
  }

  findById(id: string): UIComponent | null {
    return this.id === id ? this : null;
  }
}

class Container implements UIComponent {
  private children: UIComponent[] = [];

  constructor(
    private id: string,
    private tag: string,
    private layout: 'vertical' | 'horizontal' = 'vertical'
  ) {}

  add(child: UIComponent): this {
    this.children.push(child);
    return this;
  }

  render(): string {
    const inner = this.children.map(c => c.render()).join("\n");
    return `<${this.tag} id="${this.id}">\n${inner}\n</${this.tag}>`;
  }

  getBoundingBox() {
    if (this.layout === 'horizontal') {
      const width = this.children.reduce((w, c) => w + c.getBoundingBox().width, 0);
      const height = Math.max(0, ...this.children.map(c => c.getBoundingBox().height));
      return { width, height };
    } else {
      const width = Math.max(0, ...this.children.map(c => c.getBoundingBox().width));
      const height = this.children.reduce((h, c) => h + c.getBoundingBox().height, 0);
      return { width, height };
    }
  }

  findById(id: string): UIComponent | null {
    if (this.id === id) return this;
    for (const child of this.children) {
      const found = child.findById(id);
      if (found) return found;
    }
    return null;
  }
}

// --- 使用例 ---
const page = new Container("page", "div")
  .add(new Container("header", "header", "horizontal")
    .add(new ImageElement("logo", "/logo.png", 100, 50))
    .add(new TextElement("title", "My App", 24))
  )
  .add(new Container("main", "main")
    .add(new TextElement("content", "Welcome to my application", 16))
    .add(new ImageElement("hero", "/hero.jpg", 800, 400))
  );

console.log(page.render());
console.log(page.getBoundingBox());

const logo = page.findById("logo");
console.log(logo !== null); // true
```

### コード例 3: 価格計算 ── 商品とバンドル（TypeScript）

```typescript
// pricing.ts — ECサイトのバンドル商品の価格計算
interface PriceItem {
  getPrice(): number;
  getDescription(): string;
  getItemCount(): number;
  toJSON(): object;
}

class Product implements PriceItem {
  constructor(
    private name: string,
    private price: number,
    private quantity: number = 1
  ) {}

  getPrice(): number { return this.price * this.quantity; }
  getDescription(): string { return `${this.name} x${this.quantity}`; }
  getItemCount(): number { return this.quantity; }

  toJSON(): object {
    return { type: 'product', name: this.name, price: this.price, quantity: this.quantity };
  }
}

class Bundle implements PriceItem {
  private items: PriceItem[] = [];

  constructor(
    private name: string,
    private discount: number = 0  // 0.0 ~ 1.0
  ) {}

  add(item: PriceItem): this {
    this.items.push(item);
    return this;
  }

  getPrice(): number {
    const total = this.items.reduce((s, i) => s + i.getPrice(), 0);
    return Math.round(total * (1 - this.discount));
  }

  getDescription(): string {
    const details = this.items.map(i => i.getDescription()).join(", ");
    const discountLabel = this.discount > 0 ? ` (${this.discount * 100}%OFF)` : '';
    return `${this.name}${discountLabel} [${details}]`;
  }

  getItemCount(): number {
    return this.items.reduce((count, i) => count + i.getItemCount(), 0);
  }

  toJSON(): object {
    return {
      type: 'bundle',
      name: this.name,
      discount: this.discount,
      items: this.items.map(i => i.toJSON()),
      totalPrice: this.getPrice()
    };
  }
}

// --- 使用例 ---
const starterPack = new Bundle("Starter Pack", 0.1)
  .add(new Product("Mouse", 3000))
  .add(new Product("Keyboard", 8000))
  .add(new Bundle("Cable Set", 0)
    .add(new Product("USB-C Cable", 500))
    .add(new Product("HDMI Cable", 800)));

console.log(starterPack.getDescription());
// "Starter Pack (10%OFF) [Mouse x1, Keyboard x1, Cable Set [USB-C Cable x1, HDMI Cable x1]]"

console.log(starterPack.getPrice());
// (3000 + 8000 + 500 + 800) * 0.9 = 11070

console.log(starterPack.getItemCount()); // 4

console.log(JSON.stringify(starterPack.toJSON(), null, 2));
```

### コード例 4: Python ── 組織図

```python
# organization.py — 組織構造の Composite パターン
from abc import ABC, abstractmethod
from typing import Iterator


class OrganizationUnit(ABC):
    """組織の構成単位（Component）"""

    @abstractmethod
    def get_salary_cost(self) -> float:
        """人件費の合計を返す"""
        ...

    @abstractmethod
    def get_headcount(self) -> int:
        """所属人数を返す"""
        ...

    @abstractmethod
    def print_structure(self, indent: int = 0) -> None:
        """組織構造を表示する"""
        ...

    @abstractmethod
    def find_by_name(self, name: str) -> list["OrganizationUnit"]:
        """名前で検索する"""
        ...


class Employee(OrganizationUnit):
    """個人（Leaf）"""

    def __init__(self, name: str, role: str, salary: float):
        self.name = name
        self.role = role
        self.salary = salary

    def get_salary_cost(self) -> float:
        return self.salary

    def get_headcount(self) -> int:
        return 1

    def print_structure(self, indent: int = 0) -> None:
        prefix = " " * indent
        print(f"{prefix}- {self.name} ({self.role}, {self.salary:,.0f})")

    def find_by_name(self, name: str) -> list["OrganizationUnit"]:
        return [self] if name.lower() in self.name.lower() else []


class Department(OrganizationUnit):
    """部門（Composite）"""

    def __init__(self, name: str):
        self.name = name
        self._members: list[OrganizationUnit] = []

    def add(self, unit: OrganizationUnit) -> "Department":
        self._members.append(unit)
        return self

    def remove(self, unit: OrganizationUnit) -> None:
        self._members.remove(unit)

    def get_salary_cost(self) -> float:
        return sum(m.get_salary_cost() for m in self._members)

    def get_headcount(self) -> int:
        return sum(m.get_headcount() for m in self._members)

    def print_structure(self, indent: int = 0) -> None:
        prefix = " " * indent
        cost = self.get_salary_cost()
        count = self.get_headcount()
        print(f"{prefix}[{self.name}] ({count}名, 人件費: {cost:,.0f})")
        for m in self._members:
            m.print_structure(indent + 2)

    def find_by_name(self, name: str) -> list["OrganizationUnit"]:
        results: list[OrganizationUnit] = []
        if name.lower() in self.name.lower():
            results.append(self)
        for m in self._members:
            results.extend(m.find_by_name(name))
        return results

    def __iter__(self) -> Iterator[OrganizationUnit]:
        return iter(self._members)


# --- 使用例 ---
eng = Department("Engineering")
eng.add(Employee("Alice", "Tech Lead", 800_000))
eng.add(Employee("Bob", "Senior Engineer", 700_000))
eng.add(Employee("Charlie", "Engineer", 550_000))

design = Department("Design")
design.add(Employee("Diana", "Design Lead", 750_000))
design.add(Employee("Eve", "Designer", 600_000))

product = Department("Product")
product.add(eng)
product.add(design)
product.add(Employee("Frank", "Product Manager", 850_000))

company = Department("Acme Corp")
company.add(product)
company.add(Employee("Grace", "CEO", 1_200_000))

company.print_structure()
# [Acme Corp] (7名, 人件費: 5,450,000)
#   [Product] (6名, 人件費: 4,250,000)
#     [Engineering] (3名, 人件費: 2,050,000)
#       - Alice (Tech Lead, 800,000)
#       - Bob (Senior Engineer, 700,000)
#       - Charlie (Engineer, 550,000)
#     [Design] (2名, 人件費: 1,350,000)
#       - Diana (Design Lead, 750,000)
#       - Eve (Designer, 600,000)
#     - Frank (Product Manager, 850,000)
#   - Grace (CEO, 1,200,000)

print(f"Total cost: {company.get_salary_cost():,.0f}")
# Total cost: 5,450,000

results = company.find_by_name("engineer")
for r in results:
    print(r.name if hasattr(r, 'name') else r.name)
# Engineering, Bob, Charlie
```

### コード例 5: 条件式ツリー（仕様パターン）

```typescript
// specification.ts — Composite + Specification パターン
interface Specification<T> {
  isSatisfiedBy(item: T): boolean;
  and(other: Specification<T>): Specification<T>;
  or(other: Specification<T>): Specification<T>;
  not(): Specification<T>;
  toString(): string;
}

abstract class BaseSpec<T> implements Specification<T> {
  abstract isSatisfiedBy(item: T): boolean;
  abstract toString(): string;

  and(other: Specification<T>): Specification<T> {
    return new AndSpec([this, other]);
  }

  or(other: Specification<T>): Specification<T> {
    return new OrSpec([this, other]);
  }

  not(): Specification<T> {
    return new NotSpec(this);
  }
}

class AndSpec<T> extends BaseSpec<T> {
  constructor(private specs: Specification<T>[]) { super(); }

  isSatisfiedBy(item: T): boolean {
    return this.specs.every(s => s.isSatisfiedBy(item));
  }

  toString(): string {
    return `(${this.specs.map(s => s.toString()).join(' AND ')})`;
  }
}

class OrSpec<T> extends BaseSpec<T> {
  constructor(private specs: Specification<T>[]) { super(); }

  isSatisfiedBy(item: T): boolean {
    return this.specs.some(s => s.isSatisfiedBy(item));
  }

  toString(): string {
    return `(${this.specs.map(s => s.toString()).join(' OR ')})`;
  }
}

class NotSpec<T> extends BaseSpec<T> {
  constructor(private spec: Specification<T>) { super(); }

  isSatisfiedBy(item: T): boolean {
    return !this.spec.isSatisfiedBy(item);
  }

  toString(): string {
    return `NOT(${this.spec.toString()})`;
  }
}

// --- Leaf Specification の例 ---
interface Product {
  name: string;
  price: number;
  category: string;
  inStock: boolean;
  isNew: boolean;
}

class PriceBelow extends BaseSpec<Product> {
  constructor(private max: number) { super(); }
  isSatisfiedBy(item: Product): boolean { return item.price < this.max; }
  toString(): string { return `price < ${this.max}`; }
}

class InCategory extends BaseSpec<Product> {
  constructor(private category: string) { super(); }
  isSatisfiedBy(item: Product): boolean { return item.category === this.category; }
  toString(): string { return `category = "${this.category}"`; }
}

class InStock extends BaseSpec<Product> {
  isSatisfiedBy(item: Product): boolean { return item.inStock; }
  toString(): string { return `inStock`; }
}

class IsNew extends BaseSpec<Product> {
  isSatisfiedBy(item: Product): boolean { return item.isNew; }
  toString(): string { return `isNew`; }
}

// --- 使用例 ---
// 「安くてかつ在庫あり」OR「新商品」
const spec = new PriceBelow(1000)
  .and(new InStock())
  .or(new IsNew());

console.log(spec.toString());
// "((price < 1000 AND inStock) OR isNew)"

const products: Product[] = [
  { name: "A", price: 500, category: "food", inStock: true, isNew: false },
  { name: "B", price: 1500, category: "food", inStock: true, isNew: true },
  { name: "C", price: 800, category: "drink", inStock: false, isNew: false },
  { name: "D", price: 200, category: "drink", inStock: true, isNew: false },
];

const matching = products.filter(p => spec.isSatisfiedBy(p));
console.log(matching.map(p => p.name));
// ["A", "B", "D"]
// A: 500 < 1000 かつ在庫あり -> true
// B: 新商品 -> true
// C: 800 < 1000 だが在庫なし、新商品でもない -> false
// D: 200 < 1000 かつ在庫あり -> true
```

### コード例 6: 数式ツリー（AST）

```typescript
// expression.ts — 数式の抽象構文木 (AST) を Composite で構築
interface Expression {
  evaluate(): number;
  toString(): string;
  simplify(): Expression;
}

// Leaf: リテラル値
class NumberLiteral implements Expression {
  constructor(private value: number) {}

  evaluate(): number { return this.value; }
  toString(): string { return String(this.value); }
  simplify(): Expression { return this; }
}

// Leaf: 変数参照
class Variable implements Expression {
  constructor(
    private name: string,
    private env: Map<string, number>
  ) {}

  evaluate(): number {
    const val = this.env.get(this.name);
    if (val === undefined) throw new Error(`Undefined variable: ${this.name}`);
    return val;
  }

  toString(): string { return this.name; }
  simplify(): Expression { return this; }
}

// Composite: 二項演算
class BinaryOp implements Expression {
  constructor(
    private op: '+' | '-' | '*' | '/',
    private left: Expression,
    private right: Expression
  ) {}

  evaluate(): number {
    const l = this.left.evaluate();
    const r = this.right.evaluate();
    switch (this.op) {
      case '+': return l + r;
      case '-': return l - r;
      case '*': return l * r;
      case '/':
        if (r === 0) throw new Error('Division by zero');
        return l / r;
    }
  }

  toString(): string {
    return `(${this.left.toString()} ${this.op} ${this.right.toString()})`;
  }

  simplify(): Expression {
    const left = this.left.simplify();
    const right = this.right.simplify();

    // 定数畳み込み: 両辺がリテラルなら計算結果に置換
    if (left instanceof NumberLiteral && right instanceof NumberLiteral) {
      return new NumberLiteral(
        new BinaryOp(this.op, left, right).evaluate()
      );
    }

    // x + 0 = x, x * 1 = x 等の簡略化
    if (this.op === '+' && right instanceof NumberLiteral && right.evaluate() === 0) {
      return left;
    }
    if (this.op === '*' && right instanceof NumberLiteral && right.evaluate() === 1) {
      return left;
    }
    if (this.op === '*' && right instanceof NumberLiteral && right.evaluate() === 0) {
      return new NumberLiteral(0);
    }

    return new BinaryOp(this.op, left, right);
  }
}

// Composite: 関数呼び出し
class FunctionCall implements Expression {
  constructor(
    private fnName: string,
    private args: Expression[]
  ) {}

  evaluate(): number {
    const argValues = this.args.map(a => a.evaluate());
    switch (this.fnName) {
      case 'max': return Math.max(...argValues);
      case 'min': return Math.min(...argValues);
      case 'abs': return Math.abs(argValues[0]);
      case 'sqrt': return Math.sqrt(argValues[0]);
      default: throw new Error(`Unknown function: ${this.fnName}`);
    }
  }

  toString(): string {
    return `${this.fnName}(${this.args.map(a => a.toString()).join(', ')})`;
  }

  simplify(): Expression {
    const simplified = this.args.map(a => a.simplify());
    return new FunctionCall(this.fnName, simplified);
  }
}

// --- 使用例: (x + 3) * max(y, 5) ---
const env = new Map<string, number>([["x", 7], ["y", 2]]);

const expr = new BinaryOp('*',
  new BinaryOp('+', new Variable('x', env), new NumberLiteral(3)),
  new FunctionCall('max', [new Variable('y', env), new NumberLiteral(5)])
);

console.log(expr.toString());
// "((x + 3) * max(y, 5))"

console.log(expr.evaluate());
// (7 + 3) * max(2, 5) = 10 * 5 = 50

// 簡略化のテスト
const simpleExpr = new BinaryOp('+',
  new BinaryOp('*', new NumberLiteral(2), new NumberLiteral(3)),
  new NumberLiteral(0)
);
console.log(simpleExpr.simplify().toString());
// "6" — 定数畳み込み + 0 の除去
```

### コード例 7: メニュー構造

```typescript
// menu.ts — レストランのメニュー（Composite パターン）
interface MenuItem {
  getName(): string;
  getPrice(): number | null; // カテゴリには価格がない
  isVegetarian(): boolean;
  print(indent?: string): void;
}

class Dish implements MenuItem {
  constructor(
    private name: string,
    private price: number,
    private vegetarian: boolean,
    private description: string
  ) {}

  getName(): string { return this.name; }
  getPrice(): number { return this.price; }
  isVegetarian(): boolean { return this.vegetarian; }

  print(indent = ""): void {
    const veg = this.vegetarian ? " [V]" : "";
    console.log(`${indent}${this.name}${veg} - ${this.price}円`);
    console.log(`${indent}  ${this.description}`);
  }
}

class MenuCategory implements MenuItem {
  private items: MenuItem[] = [];

  constructor(private name: string, private description: string = "") {}

  add(item: MenuItem): this {
    this.items.push(item);
    return this;
  }

  getName(): string { return this.name; }

  getPrice(): null {
    return null; // カテゴリ自体に価格はない
  }

  isVegetarian(): boolean {
    return this.items.every(item => item.isVegetarian());
  }

  getVegetarianItems(): MenuItem[] {
    return this.items.filter(item => item.isVegetarian());
  }

  print(indent = ""): void {
    console.log(`${indent}=== ${this.name} ===`);
    if (this.description) {
      console.log(`${indent}${this.description}`);
    }
    this.items.forEach(item => item.print(indent + "  "));
  }
}

// --- 使用例 ---
const menu = new MenuCategory("Grand Menu")
  .add(new MenuCategory("Appetizers")
    .add(new Dish("Caesar Salad", 850, true, "ロメインレタス、パルメザン"))
    .add(new Dish("Bruschetta", 600, true, "トマト、バジル、オリーブオイル"))
    .add(new Dish("Carpaccio", 1200, false, "牛フィレ薄切り"))
  )
  .add(new MenuCategory("Main Courses")
    .add(new Dish("Margherita Pizza", 1400, true, "モッツァレラ、バジル"))
    .add(new Dish("Grilled Salmon", 1800, false, "ノルウェーサーモン"))
  );

menu.print();
// === Grand Menu ===
//   === Appetizers ===
//     Caesar Salad [V] - 850円
//       ロメインレタス、パルメザン
//     Bruschetta [V] - 600円
//       トマト、バジル、オリーブオイル
//     Carpaccio - 1200円
//       牛フィレ薄切り
//   === Main Courses ===
//     Margherita Pizza [V] - 1400円
//       モッツァレラ、バジル
//     Grilled Salmon - 1800円
//       ノルウェーサーモン
```

---

## 4. 透過設計 vs 安全設計：深い考察

Composite パターンの設計において、最も重要なトレードオフが「透過性（Transparency）」と「安全性（Safety）」の選択である。

### 4.1 透過設計（GoF 原書の提案）

```typescript
// 透過設計: Component に add/remove を定義
interface Component {
  operation(): void;
  add(child: Component): void;    // Leaf にも存在
  remove(child: Component): void; // Leaf にも存在
  getChild(index: number): Component | null;
}

class Leaf implements Component {
  operation(): void { /* ... */ }

  // Leaf では例外をスロー
  add(child: Component): void {
    throw new Error("Leaf cannot have children");
  }

  remove(child: Component): void {
    throw new Error("Leaf cannot have children");
  }

  getChild(index: number): Component | null {
    return null;
  }
}
```

**利点**: クライアントは Component 型だけを扱えば良い。Leaf か Composite かの判定が不要。
**欠点**: Leaf に対して add() を呼ぶと実行時エラー。型安全性が低い。

### 4.2 安全設計（現代の推奨）

```typescript
// 安全設計: add/remove は Composite のみに定義
interface Component {
  operation(): void;
}

// Composite を判定するユーティリティ
function isComposite(c: Component): c is Composite {
  return 'add' in c && typeof (c as any).add === 'function';
}

class Leaf implements Component {
  operation(): void { /* ... */ }
  // add/remove は存在しない
}

class Composite implements Component {
  private children: Component[] = [];

  operation(): void {
    this.children.forEach(c => c.operation());
  }

  add(child: Component): void {
    this.children.push(child);
  }

  remove(child: Component): void {
    this.children = this.children.filter(c => c !== child);
  }
}
```

**利点**: 型安全。Leaf に対して add() を呼ぶことがコンパイル時に検出される。
**欠点**: Composite 固有の操作を使うにはダウンキャストや型ガードが必要。

### 4.3 比較表

```
透過設計 vs 安全設計の判断フロー:

             クライアントは add/remove を頻繁に使うか？
                    /              \
                  Yes               No
                  /                   \
         透過設計を検討          安全設計を採用
                |
    型安全性は重要か？
          /        \
        Yes         No
        /             \
   安全設計を採用    透過設計を採用
```

| 方式 | Leaf にも add/remove | 型安全性 | 透過性 | コンパイル時検出 |
|------|:---:|:---:|:---:|:---:|
| 透過設計 | Yes（例外スロー） | 低い | 高い | 不可 |
| 安全設計 | No（Composite のみ） | 高い | 低い | 可能 |
| 推奨 | - | **安全設計** | - | - |

---

## 5. 循環参照防止の実装

Composite パターンで最も危険な問題が循環参照である。以下に堅牢な循環参照チェックの実装を示す。

```typescript
// safe-composite.ts — 循環参照を防止する Composite
interface SafeComponent {
  getName(): string;
  getSize(): number;
  isAncestorOf(node: SafeComponent): boolean;
}

class SafeComposite implements SafeComponent {
  private children: SafeComponent[] = [];
  private parent: SafeComposite | null = null;

  constructor(private name: string) {}

  getName(): string { return this.name; }

  getSize(): number {
    return this.children.reduce((sum, c) => sum + c.getSize(), 0);
  }

  add(child: SafeComponent): this {
    // 循環参照チェック
    if (child === this) {
      throw new Error(`Cannot add "${this.name}" to itself`);
    }

    if (child instanceof SafeComposite) {
      if (child.isAncestorOf(this)) {
        throw new Error(
          `Cannot add "${child.name}": it is an ancestor of "${this.name}"`
        );
      }
      // 既存の親から切り離す
      if (child.parent) {
        child.parent.remove(child);
      }
      child.parent = this;
    }

    this.children.push(child);
    return this;
  }

  remove(child: SafeComponent): void {
    const index = this.children.indexOf(child);
    if (index >= 0) {
      this.children.splice(index, 1);
      if (child instanceof SafeComposite) {
        child.parent = null;
      }
    }
  }

  isAncestorOf(node: SafeComponent): boolean {
    for (const child of this.children) {
      if (child === node) return true;
      if (child instanceof SafeComposite && child.isAncestorOf(node)) {
        return true;
      }
    }
    return false;
  }

  getPath(): string[] {
    const path: string[] = [];
    let current: SafeComposite | null = this;
    while (current) {
      path.unshift(current.name);
      current = current.parent;
    }
    return path;
  }
}

// --- 使用例: 循環参照の検出 ---
const a = new SafeComposite("A");
const b = new SafeComposite("B");
const c = new SafeComposite("C");

a.add(b);
b.add(c);

try {
  c.add(a); // Error: Cannot add "A": it is an ancestor of "C"
} catch (e) {
  console.log((e as Error).message);
}

try {
  a.add(a); // Error: Cannot add "A" to itself
} catch (e) {
  console.log((e as Error).message);
}

console.log(c.getPath()); // ["A", "B", "C"]
```

---

## 6. パフォーマンス最適化：キャッシュ戦略

深いツリー構造で集計操作（getSize(), getCount() 等）を頻繁に呼び出す場合、毎回再帰的に計算するとパフォーマンスが問題になる。キャッシュを使って最適化する方法を示す。

```typescript
// cached-composite.ts — キャッシュ付き Composite
interface CachedComponent {
  getName(): string;
  getSize(): number;
  invalidateCache(): void;
}

class CachedDirectory implements CachedComponent {
  private children: CachedComponent[] = [];
  private sizeCache: number | null = null;

  constructor(private name: string) {}

  getName(): string { return this.name; }

  add(child: CachedComponent): this {
    this.children.push(child);
    this.invalidateCache(); // キャッシュを無効化
    return this;
  }

  remove(child: CachedComponent): void {
    this.children = this.children.filter(c => c !== child);
    this.invalidateCache();
  }

  getSize(): number {
    if (this.sizeCache === null) {
      this.sizeCache = this.children.reduce((sum, c) => sum + c.getSize(), 0);
    }
    return this.sizeCache;
  }

  invalidateCache(): void {
    this.sizeCache = null;
    // 親のキャッシュも無効化する必要がある
    // （親への参照を持つ場合）
  }
}

// --- パフォーマンス比較 ---
// キャッシュなし: getSize() を N 回呼ぶと O(N * M) — M はノード数
// キャッシュあり: 初回は O(M)、以降は O(1)、変更時は O(D) — D は深さ
```

```
キャッシュの無効化伝搬:

  変更されたノード     親に伝搬      ルートまで伝搬
       [D]  --------> [B]  -------> [root]
       cache=null      cache=null     cache=null

  次に root.getSize() が呼ばれたとき:
  root → B を再計算 → D を再計算
  root → C は変更されていない場合、キャッシュからそのまま返す（差分更新）
```

---

## 7. Composite パターンと Visitor パターンの併用

Composite で構造を表現し、Visitor で操作を追加するのは GoF でも推奨される強力な組み合わせである。

```typescript
// visitor.ts — Composite + Visitor パターン
interface FileSystemVisitor {
  visitFile(file: VisitableFile): void;
  visitDirectory(dir: VisitableDirectory): void;
}

interface VisitableNode {
  accept(visitor: FileSystemVisitor): void;
}

class VisitableFile implements VisitableNode {
  constructor(
    public readonly name: string,
    public readonly size: number,
    public readonly extension: string
  ) {}

  accept(visitor: FileSystemVisitor): void {
    visitor.visitFile(this);
  }
}

class VisitableDirectory implements VisitableNode {
  public readonly children: VisitableNode[] = [];

  constructor(public readonly name: string) {}

  add(child: VisitableNode): this {
    this.children.push(child);
    return this;
  }

  accept(visitor: FileSystemVisitor): void {
    visitor.visitDirectory(this);
    this.children.forEach(c => c.accept(visitor));
  }
}

// --- 操作1: ファイルサイズ集計 ---
class SizeCalculator implements FileSystemVisitor {
  totalSize = 0;

  visitFile(file: VisitableFile): void {
    this.totalSize += file.size;
  }

  visitDirectory(_dir: VisitableDirectory): void {
    // ディレクトリ自体にはサイズなし
  }
}

// --- 操作2: 拡張子別ファイル一覧 ---
class ExtensionGrouper implements FileSystemVisitor {
  groups = new Map<string, string[]>();

  visitFile(file: VisitableFile): void {
    if (!this.groups.has(file.extension)) {
      this.groups.set(file.extension, []);
    }
    this.groups.get(file.extension)!.push(file.name);
  }

  visitDirectory(_dir: VisitableDirectory): void {}
}

// --- 操作3: 大きなファイルの検出 ---
class LargeFileFinder implements FileSystemVisitor {
  largeFiles: { name: string; size: number }[] = [];

  constructor(private threshold: number) {}

  visitFile(file: VisitableFile): void {
    if (file.size > this.threshold) {
      this.largeFiles.push({ name: file.name, size: file.size });
    }
  }

  visitDirectory(_dir: VisitableDirectory): void {}
}

// --- 使用例 ---
const root = new VisitableDirectory("project")
  .add(new VisitableDirectory("src")
    .add(new VisitableFile("app.ts", 10, "ts"))
    .add(new VisitableFile("style.css", 50, "css")))
  .add(new VisitableFile("README.md", 3, "md"));

// 操作1: サイズ集計
const calc = new SizeCalculator();
root.accept(calc);
console.log(`Total: ${calc.totalSize}KB`); // 63

// 操作2: 拡張子グルーピング
const grouper = new ExtensionGrouper();
root.accept(grouper);
console.log(grouper.groups);
// Map { 'ts' => ['app.ts'], 'css' => ['style.css'], 'md' => ['README.md'] }

// 操作3: 大きなファイル検出
const finder = new LargeFileFinder(5);
root.accept(finder);
console.log(finder.largeFiles);
// [{ name: 'app.ts', size: 10 }, { name: 'style.css', size: 50 }]
```

**Visitor との併用の利点**:
- ツリー構造のクラスを変更せずに新しい操作を追加できる
- Single Responsibility Principle を満たす（構造と操作の分離）
- 同じツリーに対して複数の操作を独立して定義できる

---

## 8. 実世界での Composite パターン

### 8.1 React の仮想 DOM

React のコンポーネントツリーは Composite パターンの典型例である。

```
ReactElement ツリー:
  <App>                    ← Composite
    <Header>               ← Composite
      <Logo />             ← Leaf
      <Nav>                ← Composite
        <NavItem />        ← Leaf
        <NavItem />        ← Leaf
      </Nav>
    </Header>
    <Main>                 ← Composite
      <Article />          ← Leaf
    </Main>
  </App>

render() の再帰的呼び出し:
  App.render()
    -> Header.render()
      -> Logo.render()
      -> Nav.render()
        -> NavItem.render()
        -> NavItem.render()
    -> Main.render()
      -> Article.render()
```

### 8.2 DOM API

ブラウザの DOM 自体が Composite パターンである。

```
Node (Component)
  +-- Text (Leaf)
  +-- Comment (Leaf)
  +-- Element (Composite)
       +-- children: Node[]
       +-- appendChild()
       +-- removeChild()
       +-- textContent    ← 再帰的に取得
       +-- innerHTML      ← 再帰的に取得
```

### 8.3 JSONパーサー

JSON の値は Composite 構造である。

```
JsonValue (Component)
  +-- JsonString (Leaf)     "hello"
  +-- JsonNumber (Leaf)     42
  +-- JsonBoolean (Leaf)    true
  +-- JsonNull (Leaf)       null
  +-- JsonArray (Composite) [1, "a", [2]]
  +-- JsonObject (Composite) {"key": "value"}
```

---

## 9. 比較表

### 比較表 1: Composite vs 通常のコレクション

| 観点 | Composite | 配列/リスト |
|------|:---:|:---:|
| ネスト | 再帰的（ツリー） | フラット |
| 統一インタフェース | Yes | No |
| 操作の委譲 | 自動（再帰） | 手動ループ |
| 型安全性 | 高い | 要キャスト |
| 適用場面 | 階層構造 | 均一なコレクション |
| 柔軟性 | 高い（深さ任意） | 固定（1レベル） |
| 実装コスト | 中 | 低い |

### 比較表 2: 透過設計 vs 安全設計

| 方式 | Leaf にも add/remove | 型安全性 | 透過性 | 実行時エラーリスク |
|------|:---:|:---:|:---:|:---:|
| 透過設計 | Yes（例外スロー） | 低い | 高い | 高い |
| 安全設計 | No（Composite のみ） | 高い | 低い | 低い |
| ハイブリッド | Optional メソッド | 中 | 中 | 中 |
| 推奨 | - | **安全設計** | - | - |

### 比較表 3: 関連パターンとの比較

| パターン | 目的 | 構造 | 再帰 | 典型例 |
|---------|------|------|:---:|-------|
| **Composite** | 部分-全体の統一操作 | ツリー | Yes | ファイルシステム |
| **Decorator** | 動的な機能追加 | チェーン | 可能 | ストリーム |
| **Chain of Responsibility** | 処理の委譲チェーン | リスト | No | ミドルウェア |
| **Iterator** | 走査の抽象化 | - | No | コレクション |
| **Visitor** | 構造と操作の分離 | - | Yes | AST 処理 |

---

## 10. アンチパターン

### アンチパターン 1: 循環参照の許容

```typescript
// NG: 循環参照を防止していない
class BadComposite {
  private children: BadComposite[] = [];

  add(child: BadComposite): void {
    this.children.push(child); // 自分自身を追加できてしまう！
  }

  getSize(): number {
    let size = 1;
    for (const child of this.children) {
      size += child.getSize(); // 循環参照があるとスタックオーバーフロー
    }
    return size;
  }
}

const a = new BadComposite();
const b = new BadComposite();
a.add(b);
b.add(a); // 循環参照！
// a.getSize() -> b.getSize() -> a.getSize() -> ... スタックオーバーフロー

// OK: 循環参照チェック付き
class GoodComposite {
  private children: GoodComposite[] = [];
  private parent: GoodComposite | null = null;

  add(child: GoodComposite): void {
    // 自分自身のチェック
    if (child === this) {
      throw new Error("Cannot add self as child");
    }

    // 祖先チェック（child が自分の祖先でないことを確認）
    let current: GoodComposite | null = this;
    while (current !== null) {
      if (current === child) {
        throw new Error("Circular reference detected");
      }
      current = current.parent;
    }

    // 既存の親から切り離す
    if (child.parent) {
      child.parent.children = child.parent.children.filter(c => c !== child);
    }

    child.parent = this;
    this.children.push(child);
  }

  getSize(): number {
    let size = 1;
    for (const child of this.children) {
      size += child.getSize();
    }
    return size;
  }
}
```

### アンチパターン 2: Composite が Leaf 固有のロジックに依存

```typescript
// NG: Composite が子の具象型を知っている
class BadDirectory {
  private children: FileSystemNode[] = [];

  getSize(): number {
    return this.children.reduce((sum, c) => {
      if (c instanceof File) {        // 型チェック！
        return sum + c.getRawSize();   // Leaf 固有メソッド呼び出し
      }
      if (c instanceof Directory) {
        return sum + c.getSize();
      }
      return sum; // 新しい型が来たら対応漏れ
    }, 0);
  }
}

// OK: Component インタフェースの getSize() に統一
class GoodDirectory {
  private children: FileSystemNode[] = [];

  getSize(): number {
    // 型チェックなし、インタフェースに委譲
    return this.children.reduce((sum, c) => sum + c.getSize(), 0);
  }
}
```

### アンチパターン 3: 不要な Composite パターンの適用

```typescript
// NG: フラットなリストに Composite を適用（過剰設計）
interface Task {
  getName(): string;
  getDuration(): number;
}

class SimpleTask implements Task {
  constructor(private name: string, private duration: number) {}
  getName(): string { return this.name; }
  getDuration(): number { return this.duration; }
}

class TaskGroup implements Task {
  private tasks: Task[] = [];
  constructor(private name: string) {}
  add(task: Task): void { this.tasks.push(task); }
  getName(): string { return this.name; }
  getDuration(): number {
    return this.tasks.reduce((sum, t) => sum + t.getDuration(), 0);
  }
}

// もしネストする必要がないなら、単純な配列で十分:
// const tasks: SimpleTask[] = [...];
// const total = tasks.reduce((sum, t) => sum + t.getDuration(), 0);

// OK: 実際にツリー構造が必要な場合のみ Composite を使う
// 判断基準: ネストの深さが2以上になり得るか？
```

---

## 11. 実践演習

### 演習 1: 基礎 ── ファイルシステムの実装

**課題**: 以下の要件を満たすファイルシステムを Composite パターンで実装せよ。

1. `FileSystemNode` インタフェースに `getName()`, `getSize()`, `print()` メソッドを定義
2. `File` クラス（Leaf）: 名前、サイズ、拡張子を保持
3. `Directory` クラス（Composite）: 子ノードを保持、再帰的にサイズを計算
4. ツリーの文字列表現を返す `toString()` メソッドを追加

**テストケース**:

```typescript
const root = new Directory("project");
const src = new Directory("src");
src.add(new File("main.ts", 15, "ts"));
src.add(new File("utils.ts", 8, "ts"));
const tests = new Directory("tests");
tests.add(new File("main.test.ts", 10, "ts"));
root.add(src);
root.add(tests);
root.add(new File("package.json", 2, "json"));

console.log(root.getSize()); // 35
root.print();
```

**期待される出力**:

```
project/
  src/
    main.ts (15KB)
    utils.ts (8KB)
  tests/
    main.test.ts (10KB)
  package.json (2KB)
```

---

### 演習 2: 応用 ── 権限管理ツリー

**課題**: 組織の権限管理システムを Composite パターンで実装せよ。

要件:
1. `Permission` インタフェース: `hasPermission(action: string): boolean`
2. `Role` クラス（Leaf）: 特定の権限セットを保持
3. `RoleGroup` クラス（Composite）: 複数のロールを階層的に組み合わせ
4. 子のいずれかが権限を持っていれば `true` を返す（OR 結合）
5. 権限の一覧を取得する `getAllPermissions(): string[]` メソッド

**テストケース**:

```typescript
const reader = new Role("reader", ["read", "list"]);
const writer = new Role("writer", ["write", "update"]);
const admin = new Role("admin", ["delete", "manage"]);

const editor = new RoleGroup("editor");
editor.add(reader);
editor.add(writer);

const superAdmin = new RoleGroup("superAdmin");
superAdmin.add(editor);
superAdmin.add(admin);

console.log(editor.hasPermission("read"));     // true
console.log(editor.hasPermission("delete"));   // false
console.log(superAdmin.hasPermission("delete")); // true
console.log(superAdmin.getAllPermissions());
// ["read", "list", "write", "update", "delete", "manage"]
```

**期待される出力**: 上記のコメント通り。

---

### 演習 3: 発展 ── Visitor 付きの数式評価器

**課題**: Composite パターンと Visitor パターンを組み合わせて、数式の抽象構文木（AST）を構築し、複数の操作を適用できる評価器を実装せよ。

要件:
1. `Expression` インタフェース（Component）: `accept(visitor)` メソッド
2. `NumberLiteral`, `Variable`（Leaf）
3. `BinaryOp`, `UnaryOp`（Composite）
4. Visitor 1: `Evaluator` — 式を評価して数値を返す
5. Visitor 2: `PrettyPrinter` — 式を文字列に整形する
6. Visitor 3: `Simplifier` — 式を簡略化する（0の加算、1の乗算の除去等）

**テストケース**:

```typescript
// 式: (x + 0) * (1 * y)  →  簡略化後: x * y
const env = new Map([["x", 3], ["y", 7]]);

const expr = new BinaryOp('*',
  new BinaryOp('+', new Variable('x'), new NumberLiteral(0)),
  new BinaryOp('*', new NumberLiteral(1), new Variable('y'))
);

const evaluator = new Evaluator(env);
expr.accept(evaluator);
console.log(evaluator.getResult()); // 21

const printer = new PrettyPrinter();
expr.accept(printer);
console.log(printer.getResult()); // "((x + 0) * (1 * y))"

const simplifier = new Simplifier();
expr.accept(simplifier);
const simplified = simplifier.getResult();
const printer2 = new PrettyPrinter();
simplified.accept(printer2);
console.log(printer2.getResult()); // "(x * y)"
```

**期待される出力**: 上記のコメント通り。

---

## 12. FAQ

### Q1: React の仮想 DOM は Composite パターンですか？

はい。React のコンポーネントツリーは Composite パターンの典型例です。各コンポーネントは `render()` を持ち、子コンポーネントに再帰的に処理を委譲します。JSX の `<Parent><Child /></Parent>` という構文は、Composite の add() に相当します。React 18 の Suspense や Server Components も、このツリー構造の上に構築されています。

### Q2: Visitor パターンと Composite は併用できますか？

はい。Composite でツリー構造を表現し、Visitor で操作を追加するのは GoF でも推奨される組み合わせです。新しい操作を追加する際にツリーのクラスを変更する必要がなくなります。ただし、ツリーのノード型を頻繁に追加する場合は Visitor の `visit` メソッドも追加が必要になるため、変更の方向を考慮して選択してください。

### Q3: 深いネストはパフォーマンス問題になりますか？

再帰の深さが数百レベルを超えるとスタックオーバーフローのリスクがあります。対策として:
1. **反復（イテレーティブ）なトラバーサル**: 明示的なスタックを使って再帰をループに変換
2. **末尾呼出し最適化**: 言語がサポートする場合（Scala の `@tailrec` 等）
3. **遅延評価**: 必要なノードのみを展開する
4. **キャッシュ**: 集計結果をキャッシュして再計算を避ける

### Q4: Composite パターンで子ノードの順序は保証すべきですか？

用途によります。ファイルシステムでは名前順のソートが一般的です。UI ツリーではレイアウト順（描画順）が重要です。数式 AST では演算子の左右が意味を持ちます。一般的には、用途に応じてソート済みコレクション（`SortedSet`）や順序付きリスト（`ArrayList`）を使い分けます。

### Q5: Composite と Decorator の違いは何ですか？

構造は似ていますが意図が異なります。Composite はツリー構造で「部分-全体」を表現し、子の集合に操作を委譲します。Decorator はチェーン構造で単一オブジェクトに機能を動的に追加します。Composite は「1対多」、Decorator は「1対1」の関係です。

---

## 13. まとめ

| 項目 | ポイント |
|------|---------|
| 目的 | ツリー構造を統一インタフェースで操作 |
| 構成 | Component（統一IF）, Leaf（末端）, Composite（集合） |
| 再帰 | Composite が子に操作を委譲し、結果を集約 |
| 典型例 | ファイルシステム、UI ツリー、組織図、数式 AST |
| 透過 vs 安全 | 現代は安全設計（add/remove は Composite のみ）推奨 |
| 循環参照 | 親参照と祖先チェックで防止 |
| パフォーマンス | キャッシュで再帰的集計を最適化 |
| Visitor 併用 | 構造を変えずに操作を追加する強力な組み合わせ |

---

## 次に読むべきガイド

- [Iterator パターン](../02-behavioral/04-iterator.md) -- ツリー走査の抽象化
- [Decorator パターン](./01-decorator.md) -- 動的な機能追加
- [Visitor パターン](../02-behavioral/) -- Composite との併用
- [合成優先の原則](../../../clean-code-principles/docs/03-practices-advanced/01-composition-over-inheritance.md) -- 継承より合成を選ぶ理由
- [SOLID 原則](../../../clean-code-principles/docs/00-principles/01-solid.md) -- Open/Closed Principle

---

## 参考文献

1. Gamma, E., Helm, R., Johnson, R., Vlissides, J. (1994). *Design Patterns: Elements of Reusable Object-Oriented Software*. Addison-Wesley. -- Composite パターンの原典。透過設計と安全設計のトレードオフについて詳しい。
2. Freeman, E., Robson, E. (2020). *Head First Design Patterns* (2nd Edition). O'Reilly Media. -- 視覚的に Composite パターンを学べる。
3. Refactoring.Guru -- Composite. https://refactoring.guru/design-patterns/composite -- 図解と多言語の実装例。
4. Martin, R.C. (2008). *Clean Architecture: A Craftsman's Guide to Software Structure and Design*. Prentice Hall. -- SOLID 原則と Composite の関連。
5. React Documentation -- Composition vs Inheritance. https://react.dev/learn/thinking-in-react -- React におけるコンポーネント合成の考え方。
