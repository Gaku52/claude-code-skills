# Prototype パターン

> 既存のオブジェクトを **クローン** して新しいオブジェクトを生成し、コンストラクタの再実行コストを回避する生成パターン。

---

## この章で学ぶこと

1. Prototype パターンの目的と、浅いコピー（Shallow Copy）・深いコピー（Deep Copy）の違い
2. 各言語でのクローン実装方法とプロトタイプレジストリの活用
3. クローンに伴うリスクと適切な利用場面の判断

---

## 1. Prototype の構造

```
+-------------------+
|   Prototype       |
|   (interface)     |
+-------------------+
| + clone(): Self   |
+-------------------+
        ^
        |
+-------------------+       clone()       +-------------------+
| ConcretePrototype |------------------->>| コピーされた       |
+-------------------+                     | オブジェクト       |
| - field1          |                     +-------------------+
| - field2          |
| + clone(): Self   |
+-------------------+
```

---

## 2. 浅いコピー vs 深いコピー

```
Shallow Copy (浅いコピー)
┌──────────┐    clone    ┌──────────┐
│ Original │ ──────────> │  Clone   │
│ name: "A"│             │ name: "A"│
│ list: ───┼──┐          │ list: ───┼──┐
└──────────┘  │          └──────────┘  │
              v                        v
         ┌────────┐  <-- 同じ参照！
         │[1,2,3] │
         └────────┘

Deep Copy (深いコピー)
┌──────────┐    clone    ┌──────────┐
│ Original │ ──────────> │  Clone   │
│ name: "A"│             │ name: "A"│
│ list: ───┼──┐          │ list: ───┼──┐
└──────────┘  │          └──────────┘  │
              v                        v
         ┌────────┐             ┌────────┐
         │[1,2,3] │             │[1,2,3] │  <-- 別のオブジェクト
         └────────┘             └────────┘
```

---

## 3. コード例

### コード例 1: TypeScript — Prototype インタフェース

```typescript
interface Cloneable<T> {
  clone(): T;
}

class Shape implements Cloneable<Shape> {
  constructor(
    public x: number,
    public y: number,
    public color: string
  ) {}

  clone(): Shape {
    return new Shape(this.x, this.y, this.color);
  }
}

class Circle extends Shape {
  constructor(x: number, y: number, color: string, public radius: number) {
    super(x, y, color);
  }

  clone(): Circle {
    return new Circle(this.x, this.y, this.color, this.radius);
  }
}

const original = new Circle(10, 20, "red", 50);
const copy = original.clone();
copy.color = "blue";

console.log(original.color); // "red" — 独立
console.log(copy.color);     // "blue"
```

### コード例 2: Deep Copy（structuredClone）

```typescript
class Document implements Cloneable<Document> {
  constructor(
    public title: string,
    public sections: Section[]
  ) {}

  // structuredClone で深いコピー（Node 17+ / ブラウザ対応）
  clone(): Document {
    return structuredClone(this);
  }
}

class Section {
  constructor(
    public heading: string,
    public content: string
  ) {}
}

const doc = new Document("Report", [
  new Section("Intro", "..."),
  new Section("Body", "..."),
]);

const copy = doc.clone();
copy.sections[0].heading = "Changed";

console.log(doc.sections[0].heading);  // "Intro" — 独立
console.log(copy.sections[0].heading); // "Changed"
```

### コード例 3: Prototype Registry

```typescript
class PrototypeRegistry {
  private prototypes = new Map<string, Cloneable<any>>();

  register(key: string, prototype: Cloneable<any>): void {
    this.prototypes.set(key, prototype);
  }

  create<T>(key: string): T {
    const proto = this.prototypes.get(key);
    if (!proto) throw new Error(`Prototype "${key}" not found`);
    return proto.clone() as T;
  }
}

// 登録
const registry = new PrototypeRegistry();
registry.register("red-circle", new Circle(0, 0, "red", 10));
registry.register("blue-rect", new Rectangle(0, 0, "blue", 100, 50));

// 使用: プロトタイプからクローン生成
const c1 = registry.create<Circle>("red-circle");
const c2 = registry.create<Circle>("red-circle");
console.log(c1 !== c2); // true — 別インスタンス
```

### コード例 4: Python — copy モジュール

```python
import copy

class GameState:
    def __init__(self, level: int, inventory: list[str]):
        self.level = level
        self.inventory = inventory

    def shallow_clone(self) -> "GameState":
        return copy.copy(self)

    def deep_clone(self) -> "GameState":
        return copy.deepcopy(self)

state = GameState(5, ["sword", "shield"])
save = state.deep_clone()

state.inventory.append("potion")
print(save.inventory)   # ["sword", "shield"] — 独立
print(state.inventory)  # ["sword", "shield", "potion"]
```

### コード例 5: Java — Cloneable

```java
public class Spreadsheet implements Cloneable {
    private String name;
    private List<List<String>> cells;

    @Override
    public Spreadsheet clone() {
        try {
            Spreadsheet copy = (Spreadsheet) super.clone();
            // Deep copy of cells
            copy.cells = new ArrayList<>();
            for (List<String> row : this.cells) {
                copy.cells.add(new ArrayList<>(row));
            }
            return copy;
        } catch (CloneNotSupportedException e) {
            throw new AssertionError();
        }
    }
}
```

---

## 4. クローン手法の選択フロー

```
オブジェクトをコピーしたい
        |
  全フィールドがプリミティブ？
  |                         |
  Yes                       No
  |                         |
  v                         v
Shallow Copy で十分     参照型フィールドあり
                        |
                  独立性が必要？
                  |            |
                  Yes          No
                  |            |
                  v            v
              Deep Copy    Shallow Copy
              |
       +------+------+
       |             |
  structuredClone  手動再帰コピー
  (JS/TS)          (複雑な場合)
```

---

## 5. 比較表

### 比較表 1: Shallow Copy vs Deep Copy

| 観点 | Shallow Copy | Deep Copy |
|------|:---:|:---:|
| コピー速度 | 速い | 遅い |
| メモリ使用量 | 少ない | 多い |
| 参照共有 | する | しない |
| 安全性 | 変更が波及 | 完全独立 |
| 実装難易度 | 低い | 中〜高 |
| 使用場面 | 不変データ、読み取り専用 | 可変データ、独立した変更 |

### 比較表 2: Prototype vs Factory vs コンストラクタ

| 観点 | Prototype | Factory | コンストラクタ |
|------|:---:|:---:|:---:|
| 生成コスト | 低い（コピー） | 中 | 高い（初期化） |
| 事前設定の保持 | Yes | 要設定 | No |
| 動的な型決定 | Yes | Yes | No |
| 実装複雑度 | 中（clone） | 中 | 低 |

---

## 6. アンチパターン

### アンチパターン 1: Shallow Copy で可変オブジェクトを共有

```typescript
// BAD: Shallow Copy で配列を共有
class Config {
  constructor(public plugins: string[]) {}

  clone(): Config {
    return Object.assign(new Config([]), this);
    // plugins は同じ配列を参照！
  }
}

const a = new Config(["auth", "logger"]);
const b = a.clone();
b.plugins.push("cache");

console.log(a.plugins); // ["auth", "logger", "cache"] — 意図しない変更！
```

**改善**: 参照型フィールドは明示的に Deep Copy する。

### アンチパターン 2: clone() でコンストラクタの不変条件を迂回

```typescript
// BAD: clone() がバリデーションをスキップ
class Age {
  constructor(private value: number) {
    if (value < 0 || value > 150) throw new Error("Invalid age");
  }

  clone(): Age {
    // Object.create でバリデーションを迂回してしまう
    const copy = Object.create(Age.prototype);
    copy.value = this.value;
    return copy;
  }
}
```

**改善**: clone() 内でもコンストラクタまたはファクトリメソッド経由で生成し、不変条件を維持する。

---

## 7. FAQ

### Q1: JavaScript の `structuredClone` はいつ使うべきですか？

DOM ノードや関数、Symbol を含まないプレーンなデータオブジェクトを深くコピーしたい場合に最適です。クラスインスタンスのメソッドは失われるため、メソッドを持つオブジェクトにはカスタム `clone()` を実装してください。

### Q2: Prototype パターンと JavaScript の prototype チェーンは同じですか？

名前は似ていますが別概念です。GoF の Prototype パターンはオブジェクトのクローン生成に関するパターンです。JavaScript の prototype チェーンはプロパティ検索の委譲メカニズムです。

### Q3: 不変データ構造を使っていれば Prototype パターンは不要ですか？

不変データ（Immutable.js、Immer 等）を使う場合、構造共有（structural sharing）により効率的にコピーが作成されるため、明示的な clone() は不要になることが多いです。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| 目的 | 既存オブジェクトをクローンして新規生成 |
| Shallow Copy | 高速だが参照を共有する |
| Deep Copy | 完全独立だがコスト高 |
| Registry | プロトタイプをカタログ管理 |
| JS/TS 推奨 | structuredClone + カスタム clone() |
| 注意 | clone() でも不変条件を維持する |

---

## 次に読むべきガイド

- [Singleton パターン](./00-singleton.md) — インスタンス数の制御
- [Decorator パターン](../01-structural/01-decorator.md) — 動的な機能追加
- [不変性](../../../clean-code-principles/docs/03-practices-advanced/00-immutability.md) — イミュータブルデータ構造

---

## 参考文献

1. Gamma, E. et al. (1994). *Design Patterns: Elements of Reusable Object-Oriented Software*. Addison-Wesley.
2. MDN Web Docs — structuredClone(). https://developer.mozilla.org/en-US/docs/Web/API/structuredClone
3. Python Documentation — copy module. https://docs.python.org/3/library/copy.html
