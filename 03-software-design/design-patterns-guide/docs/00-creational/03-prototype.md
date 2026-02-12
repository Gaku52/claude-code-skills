# Prototype パターン

> 既存のオブジェクトを **クローン** して新しいオブジェクトを生成し、コンストラクタの再実行コストを回避する生成パターン。

---

## 前提知識

| トピック | 必要レベル | 参照先 |
|---------|-----------|--------|
| オブジェクト指向プログラミング | 基礎 | [OOP基礎](../../../02-programming/oop-guide/docs/) |
| インタフェースと抽象クラス | 基礎 | [インタフェース設計](../../../02-programming/oop-guide/docs/) |
| 参照型とプリミティブ型の違い | 理解 | 各言語リファレンス |
| Generics（ジェネリクス） | 基礎 | [TypeScript Generics](../../../02-programming/typescript-guide/docs/) |
| メモリモデル（スタック・ヒープ） | 理解 | [CS基礎](../../../01-cs-fundamentals/) |

---

## この章で学ぶこと

1. Prototype パターンの**目的**と、なぜコンストラクタではなくクローンで生成するのか
2. **浅いコピー（Shallow Copy）** と **深いコピー（Deep Copy）** の違い・使い分け・実装上の罠
3. 各言語でのクローン実装方法（TypeScript / Python / Java / Go / Kotlin）
4. **Prototype Registry** パターンによるプロトタイプのカタログ管理
5. クローンに伴うリスク・アンチパターンと適切な利用場面の判断基準

---

## なぜ Prototype パターンが必要なのか（WHY）

### 問題: コンストラクタ再実行のコストと制約

オブジェクトの生成には、コンストラクタの実行が伴います。以下のようなケースでは、コンストラクタ経由の生成がボトルネックや設計上の制約になります。

```
[問題1: 生成コストが高い]
  DB接続、外部API呼び出し、大量データの読み込みなどを
  コンストラクタで行うオブジェクトを複数作る場合
  → 毎回同じ初期化を繰り返すのは無駄

[問題2: 実行時に型が決まる]
  どのクラスのインスタンスを生成すべきか、
  コンパイル時ではなく実行時に決まる場合
  → new ConcreteClass() とハードコードできない

[問題3: 複雑な初期状態の再現]
  多数のプロパティを持つオブジェクトの「ある状態」を
  別のオブジェクトにも再現したい場合
  → コンストラクタ引数を全部渡すのは煩雑で脆い

[問題4: フレームワーク/ライブラリの制約]
  外部ライブラリが提供するオブジェクトの内部構造は
  非公開だが、そのコピーが必要な場合
  → private フィールドにはアクセスできない
```

### 解決: クローンによる生成

Prototype パターンは「**既存の完成したオブジェクトをコピーして新しいオブジェクトを作る**」というアプローチです。

```
従来のアプローチ:
  設計図（クラス） → new → オブジェクト → 初期化 → 設定 → 完成

Prototype アプローチ:
  完成済みオブジェクト → clone() → 独立した新オブジェクト
                                     ↓
                                   即座に使用可能
```

このパターンにより:
- 初期化コストを**1回だけ**に限定できる
- 実行時にオブジェクトの型を知らなくてもコピーできる
- 複雑な状態を正確に再現できる
- private フィールドも含めてコピーできる（同クラス内の clone() メソッドからアクセス可能）

---

## 1. Prototype の構造

### クラス図

```
+-------------------+
|   <<interface>>   |
|    Prototype      |
+-------------------+
| + clone(): Self   |
+-------------------+
        △
        |
+-------------------+       clone()       +-------------------+
| ConcretePrototype |───────────────────>>| コピーされた       |
+-------------------+                     | オブジェクト       |
| - field1: T       |                     +-------------------+
| - field2: U       |                     | - field1: T (copy)|
| - nested: V       |                     | - field2: U (copy)|
| + clone(): Self   |                     | - nested: V (???) |
+-------------------+                     +-------------------+
                                              ↑
                                     Shallow? Deep? が設計判断
```

### Prototype Registry 構造

```
+-------------------+          +-------------------+
|     Client        |          | PrototypeRegistry |
+-------------------+          +-------------------+
| + operation()     |--------->| - prototypes: Map |
+-------------------+          | + register(key,p) |
                               | + get(key): Proto |
                               +-------------------+
                                         |
                          +--------------+--------------+
                          |              |              |
                   +----------+   +----------+   +----------+
                   | Proto A  |   | Proto B  |   | Proto C  |
                   +----------+   +----------+   +----------+
                   | clone()  |   | clone()  |   | clone()  |
                   +----------+   +----------+   +----------+
```

### シーケンス図

```
Client          Registry         Prototype(original)    Clone
  |                |                    |                  |
  |--get("typeA")-->|                   |                  |
  |                |--clone()---------->|                  |
  |                |                    |--new(copy)------>|
  |                |                    |    (Deep Copy)   |
  |                |<---clone instance--|                  |
  |<--返却---------|                    |                  |
  |                                                       |
  |--modify()------------------------------------------------>|
  |                                     |                  |
  |             (original は影響を受けない)                  |
```

---

## 2. 浅いコピー vs 深いコピー

### 概念図

```
=== Shallow Copy（浅いコピー）===

┌──────────────┐    clone    ┌──────────────┐
│   Original   │ ──────────> │    Clone     │
│ name: "A"    │             │ name: "A"    │  ← プリミティブはコピー
│ age: 25      │             │ age: 25      │  ← プリミティブはコピー
│ tags: ───────┼──┐          │ tags: ───────┼──┐
│ addr: ───────┼──┼──┐       │ addr: ───────┼──┼──┐
└──────────────┘  │  │       └──────────────┘  │  │
                  v  │                         v  │
            ┌────────┐│                  同じ参照！ │
            │["a","b"]│                            │
            └────────┘                             │
                  ┌────────────┐                   │
                  │{city:"NYC"}│ <── 同じオブジェクト！
                  └────────────┘


=== Deep Copy（深いコピー）===

┌──────────────┐    clone    ┌──────────────┐
│   Original   │ ──────────> │    Clone     │
│ name: "A"    │             │ name: "A"    │
│ age: 25      │             │ age: 25      │
│ tags: ───────┼──┐          │ tags: ───────┼──┐
│ addr: ───────┼──┼──┐       │ addr: ───────┼──┼──┐
└──────────────┘  │  │       └──────────────┘  │  │
                  v  │                         v  │
            ┌────────┐│               ┌────────┐  │
            │["a","b"]│               │["a","b"]│  │ ← 別の配列
            └────────┘                └────────┘  │
                  ┌────────────┐  ┌────────────┐  │
                  │{city:"NYC"}│  │{city:"NYC"}│<─┘ ← 別のオブジェクト
                  └────────────┘  └────────────┘
```

### いつどちらを使うか

```
オブジェクトをコピーしたい
        |
  全フィールドがプリミティブ or イミュータブル？
  |                                            |
  Yes                                          No
  |                                            |
  v                                            v
Shallow Copy で十分                      参照型フィールドあり
（String, number, boolean,               |
 readonly, frozen）                  コピー後に変更する？
                                    |                    |
                                    Yes                  No（読み取り専用）
                                    |                    |
                                    v                    v
                                Deep Copy が必須      Shallow Copy + 注意
                                    |
                            +-------+--------+
                            |                |
                      structuredClone    手動再帰コピー
                      JSON.parse/stringify (循環参照、
                      copy.deepcopy      特殊型対応)
```

---

## 3. コード例

### コード例 1: TypeScript — 基本的な Prototype インタフェース

```typescript
// Cloneable インタフェース
// 自身と同じ型のオブジェクトを返す clone() メソッドを定義
interface Cloneable<T> {
  clone(): T;
}

// Shape 基底クラス
class Shape implements Cloneable<Shape> {
  constructor(
    public x: number,
    public y: number,
    public color: string
  ) {}

  clone(): Shape {
    // プリミティブのみなので Shallow Copy で十分
    return new Shape(this.x, this.y, this.color);
  }

  toString(): string {
    return `Shape(${this.x}, ${this.y}, ${this.color})`;
  }
}

// Circle: Shape を拡張
class Circle extends Shape {
  constructor(
    x: number,
    y: number,
    color: string,
    public radius: number
  ) {
    super(x, y, color);
  }

  // 戻り値型を Circle に特化（共変戻り値型）
  clone(): Circle {
    return new Circle(this.x, this.y, this.color, this.radius);
  }

  toString(): string {
    return `Circle(${this.x}, ${this.y}, ${this.color}, r=${this.radius})`;
  }
}

// Rectangle: Shape を拡張
class Rectangle extends Shape {
  constructor(
    x: number,
    y: number,
    color: string,
    public width: number,
    public height: number
  ) {
    super(x, y, color);
  }

  clone(): Rectangle {
    return new Rectangle(this.x, this.y, this.color, this.width, this.height);
  }
}

// 使用例
const original = new Circle(10, 20, "red", 50);
const copy = original.clone();
copy.color = "blue";
copy.x = 100;

console.log(original.toString()); // Circle(10, 20, red, r=50)  — 独立
console.log(copy.toString());     // Circle(100, 20, blue, r=50) — 独立
console.log(original !== copy);   // true — 別インスタンス
console.log(copy instanceof Circle); // true — 型も保持
```

**ポイント**: clone() メソッド内でコンストラクタを呼ぶことで、型情報とフィールド値の両方を正確にコピーしています。サブクラスごとに clone() をオーバーライドすることで、共変戻り値型（covariant return type）を実現しています。

---

### コード例 2: Deep Copy（structuredClone と手動実装の比較）

```typescript
// === Deep Copy が必要なケース: ネストされたオブジェクト ===

class Section {
  constructor(
    public heading: string,
    public content: string,
    public subsections: Section[] = []
  ) {}

  clone(): Section {
    return new Section(
      this.heading,
      this.content,
      this.subsections.map(s => s.clone()) // 再帰的に Deep Copy
    );
  }
}

class Document implements Cloneable<Document> {
  constructor(
    public title: string,
    public sections: Section[],
    public metadata: Map<string, string> = new Map()
  ) {}

  // 方法1: 手動 Deep Copy（推奨: メソッド・型情報を完全に保持）
  clone(): Document {
    const clonedSections = this.sections.map(s => s.clone());
    const clonedMetadata = new Map(this.metadata);
    return new Document(this.title, clonedSections, clonedMetadata);
  }

  // 方法2: structuredClone（メソッドが失われる点に注意）
  cloneWithStructuredClone(): Document {
    // 注意: structuredClone はプレーンデータのみコピー
    // クラスメソッド、Map、Set のカスタム処理が失われる場合がある
    const data = structuredClone({
      title: this.title,
      sections: this.sections.map(s => ({
        heading: s.heading,
        content: s.content,
        subsections: s.subsections
      })),
      metadata: Object.fromEntries(this.metadata)
    });
    return new Document(
      data.title,
      data.sections.map(
        (s: any) => new Section(s.heading, s.content, s.subsections)
      ),
      new Map(Object.entries(data.metadata))
    );
  }

  // 方法3: JSON.parse/stringify（最もシンプルだが制約あり）
  cloneWithJson(): Document {
    // 制約: Date, Map, Set, undefined, 関数, 循環参照に非対応
    const plain = JSON.parse(JSON.stringify({
      title: this.title,
      sections: this.sections
    }));
    return new Document(
      plain.title,
      plain.sections.map(
        (s: any) => new Section(s.heading, s.content, s.subsections || [])
      )
    );
  }
}

// 動作確認
const doc = new Document("Report", [
  new Section("Intro", "Introduction text", [
    new Section("Background", "Background detail")
  ]),
  new Section("Body", "Main content"),
]);
doc.metadata.set("author", "Taro");
doc.metadata.set("version", "1.0");

const copy = doc.clone();
copy.sections[0].heading = "Changed Intro";
copy.sections[0].subsections[0].content = "Modified";
copy.metadata.set("version", "2.0");

console.log(doc.sections[0].heading);                  // "Intro" — 独立
console.log(doc.sections[0].subsections[0].content);   // "Background detail" — 独立
console.log(doc.metadata.get("version"));              // "1.0" — 独立
```

**Deep Copy 手法の比較**:

| 手法 | メソッド保持 | 循環参照 | Map/Set | Date | パフォーマンス |
|------|:---:|:---:|:---:|:---:|:---:|
| 手動 clone() | Yes | 対応可能 | Yes | Yes | 最速 |
| structuredClone | No | Yes | Yes | Yes | 中速 |
| JSON.parse/stringify | No | No | No | No | 遅い |
| lodash.cloneDeep | No | Yes | Yes | Yes | 中速 |

---

### コード例 3: Prototype Registry パターン

```typescript
// プロトタイプをキーで管理し、必要に応じてクローンを提供するレジストリ
class PrototypeRegistry<T extends Cloneable<T>> {
  private prototypes = new Map<string, T>();

  // プロトタイプの登録
  register(key: string, prototype: T): void {
    this.prototypes.set(key, prototype);
  }

  // 登録解除
  unregister(key: string): boolean {
    return this.prototypes.delete(key);
  }

  // クローンの取得
  create(key: string): T {
    const proto = this.prototypes.get(key);
    if (!proto) {
      throw new Error(
        `Prototype "${key}" not found. Available: ${[...this.prototypes.keys()].join(", ")}`
      );
    }
    return proto.clone();
  }

  // 登録済みキー一覧
  keys(): string[] {
    return [...this.prototypes.keys()];
  }

  // 登録済みかチェック
  has(key: string): boolean {
    return this.prototypes.has(key);
  }
}

// 使用例: 図形のプロトタイプレジストリ
const shapeRegistry = new PrototypeRegistry<Shape>();

// デフォルトプロトタイプを登録
shapeRegistry.register("small-red-circle", new Circle(0, 0, "red", 10));
shapeRegistry.register("large-blue-circle", new Circle(0, 0, "blue", 100));
shapeRegistry.register("standard-rect", new Rectangle(0, 0, "gray", 200, 100));

// 使用: プロトタイプからクローン生成
const c1 = shapeRegistry.create("small-red-circle");
const c2 = shapeRegistry.create("small-red-circle");

console.log(c1 !== c2);          // true — 別インスタンス
console.log(c1.toString());      // Circle(0, 0, red, r=10)

// カスタマイズ
c1.x = 50;
c1.y = 100;
console.log(c1.toString());      // Circle(50, 100, red, r=10)
```

**Registry パターンの利点**:
- プロトタイプの**中央管理**: 全てのテンプレートが1箇所に集約
- **実行時の動的登録**: 設定ファイルやAPIレスポンスからプロトタイプを登録可能
- **クラス名の隠蔽**: クライアントは具象クラスを知る必要がない
- **Factory との組み合わせ**: Factory パターンの内部実装として Prototype を使える

---

### コード例 4: Python — copy モジュールと __copy__ / __deepcopy__

```python
import copy
from dataclasses import dataclass, field
from typing import Self

# === 方法1: copy モジュールのデフォルト動作 ===

class GameState:
    """ゲームの状態を管理するクラス"""
    def __init__(self, level: int, inventory: list[str], stats: dict[str, int]):
        self.level = level
        self.inventory = inventory
        self.stats = stats

    def shallow_clone(self) -> "GameState":
        """浅いコピー: inventory と stats は同じオブジェクトを参照"""
        return copy.copy(self)

    def deep_clone(self) -> "GameState":
        """深いコピー: 全てのネストされたオブジェクトも再帰的にコピー"""
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        return f"GameState(lv={self.level}, inv={self.inventory}, stats={self.stats})"


# 動作確認
state = GameState(5, ["sword", "shield"], {"hp": 100, "mp": 50})

# 浅いコピーの罠
shallow = state.shallow_clone()
shallow.inventory.append("potion")
print(state.inventory)    # ["sword", "shield", "potion"] ← 元も変わる！
print(shallow.inventory)  # ["sword", "shield", "potion"]

# 深いコピーの安全性
state2 = GameState(5, ["sword", "shield"], {"hp": 100, "mp": 50})
deep = state2.deep_clone()
deep.inventory.append("potion")
deep.stats["hp"] = 80
print(state2.inventory)   # ["sword", "shield"] ← 独立
print(state2.stats)       # {"hp": 100, "mp": 50} ← 独立


# === 方法2: __copy__ / __deepcopy__ のカスタマイズ ===

class CachedResource:
    """キャッシュ付きリソース: clone 時にキャッシュはリセットしたい"""
    def __init__(self, url: str, data: dict, cache: dict | None = None):
        self.url = url
        self.data = data
        self._cache = cache or {}
        self._fetch_count = 0

    def __copy__(self) -> "CachedResource":
        """浅いコピーをカスタマイズ: キャッシュとカウンタをリセット"""
        new = CachedResource(self.url, self.data)
        new._cache = {}  # キャッシュはリセット
        new._fetch_count = 0
        return new

    def __deepcopy__(self, memo: dict) -> "CachedResource":
        """深いコピーをカスタマイズ: data は Deep Copy、キャッシュはリセット"""
        new = CachedResource(
            copy.deepcopy(self.url, memo),
            copy.deepcopy(self.data, memo)
        )
        new._cache = {}
        new._fetch_count = 0
        return new


resource = CachedResource("https://api.example.com", {"key": "value"}, {"cached": True})
cloned = copy.deepcopy(resource)
print(cloned._cache)        # {} ← キャッシュがリセットされている
print(cloned.data)           # {"key": "value"} ← データはコピーされている
print(resource.data is cloned.data)  # False ← 独立したオブジェクト


# === 方法3: dataclass + カスタム clone ===

@dataclass
class Config:
    """設定クラス（dataclass版）"""
    host: str
    port: int
    options: dict[str, str] = field(default_factory=dict)

    def clone(self) -> "Config":
        """Deep Copy で独立したコピーを生成"""
        return Config(
            host=self.host,
            port=self.port,
            options=dict(self.options)  # dict の浅いコピー（値が str なので十分）
        )
```

---

### コード例 5: Java — Cloneable と Copy Constructor

```java
// === 方法1: Cloneable インタフェース ===
// 注意: Java の Cloneable は設計上の問題が多く、現在は非推奨の傾向

public class Spreadsheet implements Cloneable {
    private String name;
    private List<List<String>> cells;
    private Map<String, String> metadata;

    public Spreadsheet(String name, List<List<String>> cells) {
        this.name = name;
        this.cells = cells;
        this.metadata = new HashMap<>();
    }

    @Override
    public Spreadsheet clone() {
        try {
            Spreadsheet copy = (Spreadsheet) super.clone();
            // Deep copy of cells（2次元リスト）
            copy.cells = new ArrayList<>();
            for (List<String> row : this.cells) {
                copy.cells.add(new ArrayList<>(row));
            }
            // Deep copy of metadata
            copy.metadata = new HashMap<>(this.metadata);
            return copy;
        } catch (CloneNotSupportedException e) {
            throw new AssertionError("Cloneable を実装しているので到達しない");
        }
    }
}


// === 方法2: Copy Constructor（推奨） ===
// Effective Java で推奨されている方法

public class SpreadsheetV2 {
    private final String name;
    private final List<List<String>> cells;
    private final Map<String, String> metadata;

    // 通常のコンストラクタ
    public SpreadsheetV2(String name, List<List<String>> cells) {
        this.name = name;
        this.cells = cells;
        this.metadata = new HashMap<>();
    }

    // Copy Constructor
    public SpreadsheetV2(SpreadsheetV2 other) {
        this.name = other.name;
        this.cells = new ArrayList<>();
        for (List<String> row : other.cells) {
            this.cells.add(new ArrayList<>(row));
        }
        this.metadata = new HashMap<>(other.metadata);
    }

    // Static Factory Method 形式
    public static SpreadsheetV2 copyOf(SpreadsheetV2 other) {
        return new SpreadsheetV2(other);
    }
}


// === 方法3: Serialization による Deep Copy ===
import java.io.*;

public class DeepCopyUtil {
    @SuppressWarnings("unchecked")
    public static <T extends Serializable> T deepCopy(T original) {
        try {
            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            ObjectOutputStream oos = new ObjectOutputStream(bos);
            oos.writeObject(original);
            oos.close();

            ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
            ObjectInputStream ois = new ObjectInputStream(bis);
            return (T) ois.readObject();
        } catch (IOException | ClassNotFoundException e) {
            throw new RuntimeException("Deep copy failed", e);
        }
    }
}


// 使用例
SpreadsheetV2 original = new SpreadsheetV2("Budget", List.of(
    new ArrayList<>(List.of("Item", "Cost")),
    new ArrayList<>(List.of("Server", "500"))
));

SpreadsheetV2 copy = new SpreadsheetV2(original); // Copy Constructor
// または
SpreadsheetV2 copy2 = SpreadsheetV2.copyOf(original); // Static Factory
```

**Java Cloneable の問題点（Effective Java より）**:
1. `Cloneable` はメソッドを定義しないマーカーインタフェースだが、`Object.clone()` の動作を変える
2. `clone()` の戻り値型は `Object`（キャストが必要）
3. `CloneNotSupportedException` のチェック例外が煩雑
4. `super.clone()` は Shallow Copy のみ
5. final フィールドへの代入ができない

**推奨**: Copy Constructor または Static Factory Method を使う

---

### コード例 6: Go — インタフェースベースの Clone

```go
package main

import "fmt"

// Cloneable インタフェース
type Cloneable[T any] interface {
    Clone() T
}

// Shape 構造体
type Shape struct {
    X, Y  int
    Color string
    Tags  []string
}

// Clone: Deep Copy を実装
func (s *Shape) Clone() *Shape {
    // Tags スライスを新しくコピー
    tagsCopy := make([]string, len(s.Tags))
    copy(tagsCopy, s.Tags)

    return &Shape{
        X:     s.X,
        Y:     s.Y,
        Color: s.Color,
        Tags:  tagsCopy,
    }
}

// Circle 構造体
type Circle struct {
    Shape  // 埋め込み
    Radius float64
}

// Clone: 埋め込み構造体も含めて Deep Copy
func (c *Circle) Clone() *Circle {
    shapeCopy := c.Shape.Clone()
    return &Circle{
        Shape:  *shapeCopy,
        Radius: c.Radius,
    }
}

// Document 構造体（ネストあり）
type Document struct {
    Title    string
    Sections []Section
    Meta     map[string]string
}

type Section struct {
    Heading string
    Content string
}

func (d *Document) Clone() *Document {
    // Sections の Deep Copy
    sections := make([]Section, len(d.Sections))
    for i, s := range d.Sections {
        sections[i] = Section{
            Heading: s.Heading,
            Content: s.Content,
        }
    }

    // Map の Deep Copy
    meta := make(map[string]string, len(d.Meta))
    for k, v := range d.Meta {
        meta[k] = v
    }

    return &Document{
        Title:    d.Title,
        Sections: sections,
        Meta:     meta,
    }
}

func main() {
    original := &Circle{
        Shape:  Shape{X: 10, Y: 20, Color: "red", Tags: []string{"important"}},
        Radius: 50,
    }

    cloned := original.Clone()
    cloned.Color = "blue"
    cloned.Tags = append(cloned.Tags, "modified")

    fmt.Println(original.Color, original.Tags) // red [important] — 独立
    fmt.Println(cloned.Color, cloned.Tags)     // blue [important modified]
}
```

---

### コード例 7: Kotlin — data class の copy() と手動 Clone

```kotlin
// === 方法1: data class の copy()（Shallow Copy）===

data class Address(
    val city: String,
    val street: String
)

data class User(
    val name: String,
    val age: Int,
    val address: Address,    // 参照型
    val tags: MutableList<String>  // ミュータブルなコレクション
)

fun main() {
    val original = User(
        name = "Taro",
        age = 25,
        address = Address("Tokyo", "Shibuya"),
        tags = mutableListOf("admin", "user")
    )

    // data class の copy() は Shallow Copy
    val shallow = original.copy(name = "Jiro")

    // Address は不変（val + data class）なので安全
    println(original.address === shallow.address) // true（同じ参照だが不変なので問題なし）

    // MutableList は浅いコピーなので危険！
    shallow.tags.add("editor")
    println(original.tags) // [admin, user, editor] ← 元も変わる！
}


// === 方法2: Deep Copy を手動実装 ===

data class UserV2(
    val name: String,
    val age: Int,
    val address: Address,
    val tags: List<String>  // 不変リストにする（設計で解決）
) {
    // Deep Copy メソッド
    fun deepCopy(): UserV2 = UserV2(
        name = this.name,
        age = this.age,
        address = this.address.copy(), // data class の copy() で OK（全フィールドが val String）
        tags = this.tags.toList()      // 新しいリストを生成
    )
}


// === 方法3: sealed interface + clone ===

sealed interface ShapeK {
    fun clone(): ShapeK
}

data class CircleK(
    val x: Int,
    val y: Int,
    val radius: Double,
    val color: String
) : ShapeK {
    override fun clone(): CircleK = this.copy()
}

data class RectangleK(
    val x: Int,
    val y: Int,
    val width: Int,
    val height: Int,
    val color: String
) : ShapeK {
    override fun clone(): RectangleK = this.copy()
}

// 使用例: 多態的なクローン
fun duplicateShapes(shapes: List<ShapeK>): List<ShapeK> {
    return shapes.map { it.clone() }
}
```

---

### コード例 8: Prototype + Factory パターンの組み合わせ

```typescript
// Prototype を内部的に使う Factory
// クライアントにはファクトリインタフェースだけを公開

interface Notification {
  title: string;
  body: string;
  priority: "low" | "medium" | "high";
  channels: string[];
  clone(): Notification;
}

class EmailNotification implements Notification {
  constructor(
    public title: string,
    public body: string,
    public priority: "low" | "medium" | "high",
    public channels: string[],
    public templateId: string
  ) {}

  clone(): EmailNotification {
    return new EmailNotification(
      this.title,
      this.body,
      this.priority,
      [...this.channels],
      this.templateId
    );
  }
}

class SlackNotification implements Notification {
  constructor(
    public title: string,
    public body: string,
    public priority: "low" | "medium" | "high",
    public channels: string[],
    public webhookUrl: string
  ) {}

  clone(): SlackNotification {
    return new SlackNotification(
      this.title,
      this.body,
      this.priority,
      [...this.channels],
      this.webhookUrl
    );
  }
}

// NotificationFactory: Prototype を内部で管理
class NotificationFactory {
  private prototypes = new Map<string, Notification>();

  constructor() {
    // デフォルトテンプレートを登録
    this.prototypes.set("welcome-email", new EmailNotification(
      "Welcome!",
      "Welcome to our service.",
      "medium",
      ["email"],
      "tmpl-welcome-001"
    ));
    this.prototypes.set("alert-slack", new SlackNotification(
      "Alert",
      "System alert detected.",
      "high",
      ["#alerts"],
      "https://hooks.slack.com/xxx"
    ));
  }

  // Factory メソッド: Prototype をクローンしてカスタマイズ
  create(type: string, overrides?: Partial<Notification>): Notification {
    const proto = this.prototypes.get(type);
    if (!proto) throw new Error(`Unknown notification type: ${type}`);

    const notification = proto.clone();
    if (overrides) {
      Object.assign(notification, overrides);
    }
    return notification;
  }

  // 新しいテンプレートを動的に登録
  registerTemplate(name: string, prototype: Notification): void {
    this.prototypes.set(name, prototype);
  }
}

// 使用例
const factory = new NotificationFactory();

const welcome = factory.create("welcome-email", {
  title: "Welcome, Taro!",
  body: "Your account has been created."
});

const alert = factory.create("alert-slack", {
  body: "CPU usage exceeded 90%"
});

console.log(welcome); // EmailNotification with customized title/body
console.log(alert);   // SlackNotification with customized body
```

---

### コード例 9: Undo/Redo のための状態クローン（Memento + Prototype）

```typescript
// エディタの状態をクローンして Undo/Redo スタックに保存

interface EditorState {
  content: string;
  cursorPosition: number;
  selections: Array<{ start: number; end: number }>;
  clone(): EditorState;
}

class TextEditorState implements EditorState {
  constructor(
    public content: string,
    public cursorPosition: number,
    public selections: Array<{ start: number; end: number }>
  ) {}

  clone(): TextEditorState {
    return new TextEditorState(
      this.content,
      this.cursorPosition,
      this.selections.map(s => ({ ...s })) // Deep Copy
    );
  }
}

class TextEditor {
  private state: TextEditorState;
  private undoStack: TextEditorState[] = [];
  private redoStack: TextEditorState[] = [];
  private readonly maxHistory = 50;

  constructor() {
    this.state = new TextEditorState("", 0, []);
  }

  // 状態変更前にクローンを保存
  private saveState(): void {
    this.undoStack.push(this.state.clone());
    if (this.undoStack.length > this.maxHistory) {
      this.undoStack.shift(); // 古い履歴を削除
    }
    this.redoStack = []; // Redo スタックをクリア
  }

  type(text: string): void {
    this.saveState();
    const before = this.state.content.slice(0, this.state.cursorPosition);
    const after = this.state.content.slice(this.state.cursorPosition);
    this.state.content = before + text + after;
    this.state.cursorPosition += text.length;
  }

  undo(): void {
    if (this.undoStack.length === 0) return;
    this.redoStack.push(this.state.clone());
    this.state = this.undoStack.pop()!;
  }

  redo(): void {
    if (this.redoStack.length === 0) return;
    this.undoStack.push(this.state.clone());
    this.state = this.redoStack.pop()!;
  }

  getContent(): string {
    return this.state.content;
  }

  getCursorPosition(): number {
    return this.state.cursorPosition;
  }
}

// 使用例
const editor = new TextEditor();
editor.type("Hello");
editor.type(", World!");
console.log(editor.getContent()); // "Hello, World!"

editor.undo();
console.log(editor.getContent()); // "Hello"

editor.undo();
console.log(editor.getContent()); // ""

editor.redo();
console.log(editor.getContent()); // "Hello"
```

---

### コード例 10: 循環参照を含むオブジェクトの Deep Copy

```typescript
// 循環参照がある場合の Deep Copy は特別な処理が必要

class TreeNode {
  children: TreeNode[] = [];
  parent: TreeNode | null = null;

  constructor(public name: string, public value: number) {}

  addChild(child: TreeNode): void {
    child.parent = this;
    this.children.push(child);
  }

  // 循環参照対応の Deep Copy
  // visited マップで既にクローン済みのノードを追跡
  clone(visited = new Map<TreeNode, TreeNode>()): TreeNode {
    // 既にクローン済みならそれを返す（循環参照の解決）
    if (visited.has(this)) {
      return visited.get(this)!;
    }

    // 新しいノードを作成し、visited に登録
    const cloned = new TreeNode(this.name, this.value);
    visited.set(this, cloned);

    // 子ノードを再帰的にクローン
    for (const child of this.children) {
      const clonedChild = child.clone(visited);
      clonedChild.parent = cloned;
      cloned.children.push(clonedChild);
    }

    return cloned;
  }

  toString(indent = 0): string {
    const prefix = "  ".repeat(indent);
    let result = `${prefix}${this.name}(${this.value})`;
    for (const child of this.children) {
      result += "\n" + child.toString(indent + 1);
    }
    return result;
  }
}

// 使用例
const root = new TreeNode("root", 0);
const a = new TreeNode("A", 1);
const b = new TreeNode("B", 2);
const c = new TreeNode("C", 3);

root.addChild(a);
root.addChild(b);
a.addChild(c);

const clonedRoot = root.clone();
clonedRoot.children[0].name = "A-modified";
clonedRoot.children[0].value = 999;

console.log(root.toString());
// root(0)
//   A(1)
//     C(3)
//   B(2)

console.log(clonedRoot.toString());
// root(0)
//   A-modified(999)  ← 独立
//     C(3)
//   B(2)

// 親子関係の確認
console.log(clonedRoot.children[0].parent === clonedRoot); // true
console.log(clonedRoot.children[0].parent === root);       // false（独立）
```

---

## 4. 比較表

### 比較表 1: Shallow Copy vs Deep Copy

| 観点 | Shallow Copy | Deep Copy |
|------|:---:|:---:|
| コピー速度 | **高速**（O(n) フィールド数） | **低速**（O(N) 全ノード数） |
| メモリ使用量 | **少ない**（参照共有） | **多い**（全て複製） |
| 参照共有 | **する**（副作用リスク） | **しない**（完全独立） |
| 安全性 | **低い**（変更が波及） | **高い**（完全独立） |
| 実装難易度 | **低い** | **中〜高**（循環参照対応等） |
| 不変データでの使用 | **安全**（変更しないため） | **不要**（コピー自体が無駄） |
| 使用場面 | 不変データ、読み取り専用、パフォーマンス重視 | 可変データ、独立した変更が必要 |

### 比較表 2: Prototype vs Factory vs Constructor vs Copy Constructor

| 観点 | Prototype(clone) | Factory Method | Constructor | Copy Constructor |
|------|:---:|:---:|:---:|:---:|
| 生成コスト | 低い（コピー） | 中 | 高い（初期化） | 低い（コピー） |
| 事前設定の保持 | **Yes** | 要設定 | No | **Yes** |
| 動的な型決定 | **Yes** | **Yes** | No | No |
| 型安全性 | 中 | 高 | 高 | 高 |
| private フィールド | **アクセス可** | 要 getter | N/A | **アクセス可** |
| 言語サポート | 全言語 | 全言語 | 全言語 | Java/C++/Kotlin |
| 推奨度 | 中 | 高 | 基本 | 高（Java） |

### 比較表 3: 言語別クローン実装の比較

| 言語 | 標準手法 | Deep Copy サポート | 推奨アプローチ |
|------|---------|:---:|---------|
| TypeScript | カスタム clone() | structuredClone | 手動 clone() + structuredClone 併用 |
| Python | copy.copy/deepcopy | **組み込み** | copy.deepcopy + __deepcopy__ カスタマイズ |
| Java | Cloneable.clone() | Serialization | **Copy Constructor**（Effective Java推奨） |
| Go | 手動実装 | なし | 構造体ごとに Clone() メソッド |
| Kotlin | data class copy() | なし | data class copy() + 不変設計 |
| C# | ICloneable.Clone() | なし | 手動 Deep Copy + record 型 |
| Rust | Clone trait | Clone derive | `#[derive(Clone)]` |

---

## 5. アンチパターン

### アンチパターン 1: Shallow Copy で可変オブジェクトを共有

```typescript
// NG: Shallow Copy で配列・オブジェクトを共有してしまう
class Config {
  constructor(
    public name: string,
    public plugins: string[],
    public settings: Record<string, unknown>
  ) {}

  clone(): Config {
    // Object.assign は Shallow Copy！
    return Object.assign(new Config("", [], {}), this);
    // plugins と settings は同じ参照を共有
  }
}

const a = new Config("prod", ["auth", "logger"], { debug: false });
const b = a.clone();
b.plugins.push("cache");
b.settings.debug = true;

console.log(a.plugins);        // ["auth", "logger", "cache"] ← 意図しない変更！
console.log(a.settings.debug); // true ← 意図しない変更！
```

```typescript
// OK: 参照型フィールドを明示的に Deep Copy
class Config {
  constructor(
    public name: string,
    public plugins: string[],
    public settings: Record<string, unknown>
  ) {}

  clone(): Config {
    return new Config(
      this.name,
      [...this.plugins],                    // 配列のスプレッドコピー
      structuredClone(this.settings)        // ネストされたオブジェクトの Deep Copy
    );
  }
}
```

---

### アンチパターン 2: clone() でコンストラクタの不変条件（invariant）を迂回

```typescript
// NG: clone() がバリデーションをスキップ
class PositiveNumber {
  private value: number;

  constructor(value: number) {
    if (value <= 0) throw new Error("Value must be positive");
    this.value = value;
  }

  getValue(): number { return this.value; }

  clone(): PositiveNumber {
    // Object.create でコンストラクタを迂回
    const copy = Object.create(PositiveNumber.prototype);
    copy.value = this.value;
    return copy;
    // 問題: 将来 value を変更する setter が追加された場合、
    // バリデーションなしでオブジェクトが作成される可能性
  }
}
```

```typescript
// OK: コンストラクタ経由で不変条件を維持
class PositiveNumber {
  private value: number;

  constructor(value: number) {
    if (value <= 0) throw new Error("Value must be positive");
    this.value = value;
  }

  getValue(): number { return this.value; }

  clone(): PositiveNumber {
    // コンストラクタを通すことでバリデーションが実行される
    return new PositiveNumber(this.value);
  }
}
```

---

### アンチパターン 3: clone() で一意識別子もコピーしてしまう

```typescript
// NG: ID もそのままコピー → 一意性が壊れる
class Entity {
  constructor(
    public id: string,     // UUID — 一意であるべき
    public name: string,
    public data: unknown
  ) {}

  clone(): Entity {
    return new Entity(this.id, this.name, structuredClone(this.data));
    // id が同じ → データベースで衝突！
  }
}
```

```typescript
// OK: clone() で新しい ID を生成
import { randomUUID } from "crypto";

class Entity {
  constructor(
    public id: string,
    public name: string,
    public data: unknown
  ) {}

  clone(): Entity {
    return new Entity(
      randomUUID(),  // 新しい ID を生成
      this.name,
      structuredClone(this.data)
    );
  }

  // 「どのフィールドをコピーし、どのフィールドを新規生成するか」を
  // 明示的に設計することが重要
}
```

---

## 6. エッジケースと注意点

### エッジケース 1: structuredClone の制約

```typescript
// structuredClone がコピーできないもの
class Example {
  method(): void {}  // 関数 → コピーされない
}

const obj = {
  fn: () => "hello",           // ❌ 関数
  symbol: Symbol("id"),        // ❌ Symbol
  dom: document.createElement("div"), // ❌ DOM ノード
  error: new Error("test"),    // ✅ コピー可能
  date: new Date(),            // ✅ コピー可能
  regex: /test/gi,             // ✅ コピー可能
  map: new Map([["a", 1]]),    // ✅ コピー可能
  set: new Set([1, 2, 3]),     // ✅ コピー可能
  arrayBuffer: new ArrayBuffer(8), // ✅ コピー可能
};

// クラスインスタンスの場合
const instance = new Example();
const cloned = structuredClone(instance);
console.log(typeof cloned.method); // "undefined" — メソッドが失われる！
console.log(cloned instanceof Example); // false — 型情報も失われる！
```

### エッジケース 2: イベントリスナーやコールバックの扱い

```typescript
class EventEmitterWidget {
  private listeners = new Map<string, Function[]>();

  on(event: string, handler: Function): void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event)!.push(handler);
  }

  // clone() でリスナーはどうするか？
  // 選択肢1: リスナーもコピー → 同じハンドラが2つのオブジェクトで発火
  // 選択肢2: リスナーをリセット → クローン後に再登録が必要
  // 選択肢3: リスナーの浅いコピー → 同じ関数参照を共有（通常はOK）

  clone(copyListeners = false): EventEmitterWidget {
    const cloned = new EventEmitterWidget();
    if (copyListeners) {
      for (const [event, handlers] of this.listeners) {
        cloned.listeners.set(event, [...handlers]); // 浅いコピー
      }
    }
    return cloned;
  }
}
```

### エッジケース 3: 循環参照の検出と処理

```typescript
// 循環参照があるとスタックオーバーフローを起こす
const a: any = { name: "A" };
const b: any = { name: "B", ref: a };
a.ref = b; // 循環参照

// NG: 無限再帰
function naiveDeepCopy(obj: any): any {
  const copy: any = {};
  for (const key of Object.keys(obj)) {
    copy[key] = typeof obj[key] === "object"
      ? naiveDeepCopy(obj[key]) // ← 無限ループ！
      : obj[key];
  }
  return copy;
}

// OK: visited マップで循環を検出
function safeDeepCopy(obj: any, visited = new WeakMap()): any {
  if (obj === null || typeof obj !== "object") return obj;
  if (visited.has(obj)) return visited.get(obj); // 既訪問なら返す

  const copy: any = Array.isArray(obj) ? [] : {};
  visited.set(obj, copy); // 先に登録（循環参照対策）

  for (const key of Object.keys(obj)) {
    copy[key] = safeDeepCopy(obj[key], visited);
  }
  return copy;
}

// structuredClone は循環参照を自動で処理する
const cloned = structuredClone(a); // OK！
console.log(cloned.ref.ref === cloned); // true — 循環参照が正しく再現
```

### エッジケース 4: Prototype チェーンとシリアライゼーション

```typescript
// JSON.stringify/parse で失われるもの一覧
const original = {
  date: new Date("2024-01-01"),         // → 文字列になる
  undefined: undefined,                  // → 消える
  nan: NaN,                              // → null になる
  infinity: Infinity,                    // → null になる
  regex: /test/g,                        // → {} になる
  map: new Map([["a", 1]]),              // → {} になる
  set: new Set([1, 2]),                  // → {} になる
  fn: () => "hello",                     // → 消える
  symbol: Symbol("id"),                  // → 消える
  bigint: BigInt(42),                    // → TypeError!
};
```

---

## 7. トレードオフ分析

### Prototype パターンを使うべき場面

```
✅ 使うべき場面:
┌─────────────────────────────────────────────────┐
│ 1. 初期化コストが高いオブジェクトを多数生成する   │
│    例: DB接続設定、ML モデル設定、ゲームキャラ     │
│                                                  │
│ 2. 実行時に型が決まるオブジェクトのコピーが必要   │
│    例: プラグインシステム、設定テンプレート       │
│                                                  │
│ 3. オブジェクトの状態をスナップショットとして保存  │
│    例: Undo/Redo、バージョン管理、テスト fixture  │
│                                                  │
│ 4. プロトタイプレジストリでテンプレート管理       │
│    例: UIコンポーネント、通知テンプレート         │
│                                                  │
│ 5. 外部ライブラリのオブジェクトのコピーが必要     │
│    例: 非公開フィールドを含むオブジェクトの複製   │
└─────────────────────────────────────────────────┘

❌ 使うべきでない場面:
┌─────────────────────────────────────────────────┐
│ 1. 生成コストが低いオブジェクト                   │
│    → new で十分。clone() の実装コストが上回る     │
│                                                  │
│ 2. 不変データ構造を使っている場合                 │
│    → 構造共有（structural sharing）が効率的       │
│    例: Immutable.js, Immer                       │
│                                                  │
│ 3. 循環参照が複雑すぎるオブジェクトグラフ         │
│    → Deep Copy の実装が困難、バグの温床           │
│                                                  │
│ 4. クローン後にほとんどのフィールドを変更する     │
│    → コンストラクタで直接生成した方が明確         │
└─────────────────────────────────────────────────┘
```

### パフォーマンス特性

```
生成方式のパフォーマンス比較（概算）:

方式              | 小オブジェクト | 大オブジェクト | ネストあり
                  | (5 fields)     | (50 fields)    | (3 levels deep)
─────────────────|────────────────|────────────────|────────────────
new + 初期化      |     基準       |     基準       |     基準
Shallow Clone     |    0.1x        |    0.1x        |    0.1x
Deep Clone(手動)  |    0.3x        |    0.5x        |    0.8x
structuredClone   |    2.0x        |    1.5x        |    1.2x
JSON parse/strfy  |    5.0x        |    3.0x        |    2.5x

※ new + 初期化 に DB接続やAPI呼び出しが含まれる場合、
  clone は圧倒的に高速（100x〜1000x以上の差）
```

---

## 8. 演習問題

### 演習 1（基礎）: Shape の Clone 実装

以下の要件を満たす Shape クラス階層を実装してください。

**要件**:
- `Shape` 基底クラスに `clone()` メソッドを定義
- `Circle`、`Rectangle`、`Triangle` のサブクラスを作成
- 各クラスの `clone()` が正しく Deep Copy を返すことを確認
- `describe()` メソッドで図形の情報を文字列で返す

```typescript
// テスト
const circle = new Circle(0, 0, "red", 25);
const clonedCircle = circle.clone();
clonedCircle.color = "blue";

console.log(circle.describe());       // "Circle(x=0, y=0, color=red, r=25)"
console.log(clonedCircle.describe()); // "Circle(x=0, y=0, color=blue, r=25)"
console.log(circle !== clonedCircle); // true
```

**期待される出力**:
```
Circle(x=0, y=0, color=red, r=25)
Circle(x=0, y=0, color=blue, r=25)
true
```

<details>
<summary>解答例</summary>

```typescript
interface Cloneable<T> {
  clone(): T;
}

abstract class Shape implements Cloneable<Shape> {
  constructor(
    public x: number,
    public y: number,
    public color: string
  ) {}

  abstract clone(): Shape;
  abstract describe(): string;
}

class Circle extends Shape {
  constructor(x: number, y: number, color: string, public radius: number) {
    super(x, y, color);
  }

  clone(): Circle {
    return new Circle(this.x, this.y, this.color, this.radius);
  }

  describe(): string {
    return `Circle(x=${this.x}, y=${this.y}, color=${this.color}, r=${this.radius})`;
  }
}

class Rectangle extends Shape {
  constructor(
    x: number, y: number, color: string,
    public width: number, public height: number
  ) {
    super(x, y, color);
  }

  clone(): Rectangle {
    return new Rectangle(this.x, this.y, this.color, this.width, this.height);
  }

  describe(): string {
    return `Rectangle(x=${this.x}, y=${this.y}, color=${this.color}, w=${this.width}, h=${this.height})`;
  }
}

class Triangle extends Shape {
  constructor(
    x: number, y: number, color: string,
    public base: number, public height2: number
  ) {
    super(x, y, color);
  }

  clone(): Triangle {
    return new Triangle(this.x, this.y, this.color, this.base, this.height2);
  }

  describe(): string {
    return `Triangle(x=${this.x}, y=${this.y}, color=${this.color}, base=${this.base}, h=${this.height2})`;
  }
}

// テスト
const circle = new Circle(0, 0, "red", 25);
const clonedCircle = circle.clone();
clonedCircle.color = "blue";

console.log(circle.describe());       // "Circle(x=0, y=0, color=red, r=25)"
console.log(clonedCircle.describe()); // "Circle(x=0, y=0, color=blue, r=25)"
console.log(circle !== clonedCircle); // true
```
</details>

---

### 演習 2（応用）: Prototype Registry + Deep Copy

ゲームのキャラクターテンプレートを管理する Prototype Registry を実装してください。

**要件**:
- `Character` クラス: name, hp, mp, skills(配列), equipment(オブジェクト)
- `CharacterRegistry`: テンプレートの登録・クローン取得
- Deep Copy であること（skills と equipment が独立）
- クローン時に新しい ID を自動付与

```typescript
// テスト
const registry = new CharacterRegistry();

registry.register("warrior", new Character(
  "Warrior Template",
  100, 20,
  ["slash", "block"],
  { weapon: "sword", armor: "plate" }
));

const player1 = registry.create("warrior", "Hero Taro");
const player2 = registry.create("warrior", "Hero Jiro");

player1.skills.push("charge");
player1.equipment.weapon = "legendary-sword";

console.log(player1.name);            // "Hero Taro"
console.log(player2.name);            // "Hero Jiro"
console.log(player1.id !== player2.id); // true
console.log(player1.skills);          // ["slash", "block", "charge"]
console.log(player2.skills);          // ["slash", "block"] ← 独立
console.log(player2.equipment.weapon); // "sword" ← 独立
```

**期待される出力**:
```
Hero Taro
Hero Jiro
true
["slash", "block", "charge"]
["slash", "block"]
sword
```

<details>
<summary>解答例</summary>

```typescript
import { randomUUID } from "crypto";

interface Cloneable<T> {
  clone(): T;
}

class Character implements Cloneable<Character> {
  public id: string;

  constructor(
    public name: string,
    public hp: number,
    public mp: number,
    public skills: string[],
    public equipment: Record<string, string>
  ) {
    this.id = randomUUID();
  }

  clone(): Character {
    const cloned = new Character(
      this.name,
      this.hp,
      this.mp,
      [...this.skills],          // 配列の Deep Copy
      { ...this.equipment }      // オブジェクトの Shallow Copy（値が string なので十分）
    );
    // id は new Character() 内で自動生成される
    return cloned;
  }
}

class CharacterRegistry {
  private templates = new Map<string, Character>();

  register(key: string, template: Character): void {
    this.templates.set(key, template);
  }

  create(key: string, name?: string): Character {
    const template = this.templates.get(key);
    if (!template) {
      throw new Error(`Template "${key}" not found`);
    }
    const character = template.clone();
    if (name) {
      character.name = name;
    }
    return character;
  }

  listTemplates(): string[] {
    return [...this.templates.keys()];
  }
}

// テスト
const registry = new CharacterRegistry();

registry.register("warrior", new Character(
  "Warrior Template", 100, 20,
  ["slash", "block"],
  { weapon: "sword", armor: "plate" }
));

registry.register("mage", new Character(
  "Mage Template", 60, 100,
  ["fireball", "heal"],
  { weapon: "staff", armor: "robe" }
));

const player1 = registry.create("warrior", "Hero Taro");
const player2 = registry.create("warrior", "Hero Jiro");

player1.skills.push("charge");
player1.equipment.weapon = "legendary-sword";

console.log(player1.name);              // "Hero Taro"
console.log(player2.name);              // "Hero Jiro"
console.log(player1.id !== player2.id); // true
console.log(player1.skills);            // ["slash", "block", "charge"]
console.log(player2.skills);            // ["slash", "block"]
console.log(player2.equipment.weapon);  // "sword"
```
</details>

---

### 演習 3（上級）: 汎用 Deep Clone ユーティリティ

以下の全てのデータ型を正しく Deep Clone できる汎用ユーティリティ関数を実装してください。

**要件**:
- プリミティブ型（string, number, boolean, null, undefined）
- Date, RegExp, Map, Set
- 配列とプレーンオブジェクト
- 循環参照の検出と正しい処理
- `clone()` メソッドを持つオブジェクトはそれを優先使用
- TypeScript の型安全な実装

```typescript
// テスト
const original = {
  str: "hello",
  num: 42,
  bool: true,
  date: new Date("2024-01-01"),
  regex: /test/gi,
  map: new Map([["a", { nested: true }]]),
  set: new Set([1, 2, { x: 3 }]),
  arr: [1, [2, [3]]],
  circular: null as any,
};
original.circular = original; // 循環参照

const cloned = deepClone(original);

console.log(cloned.date instanceof Date);     // true
console.log(cloned.date !== original.date);   // true
console.log(cloned.regex instanceof RegExp);  // true
console.log(cloned.map.get("a")!.nested);     // true
console.log(cloned.map.get("a") !== original.map.get("a")); // true
console.log(cloned.circular === cloned);      // true（循環参照が正しく再現）
console.log(cloned.circular !== original);    // true（独立）
```

**期待される出力**:
```
true
true
true
true
true
true
true
```

<details>
<summary>解答例</summary>

```typescript
function deepClone<T>(obj: T, visited = new WeakMap()): T {
  // プリミティブと null/undefined はそのまま返す
  if (obj === null || obj === undefined) return obj;
  if (typeof obj !== "object" && typeof obj !== "function") return obj;

  // 循環参照チェック
  if (visited.has(obj as any)) {
    return visited.get(obj as any);
  }

  // clone() メソッドを持つオブジェクトはそれを使う
  if (typeof (obj as any).clone === "function") {
    const cloned = (obj as any).clone();
    visited.set(obj as any, cloned);
    return cloned;
  }

  let result: any;

  // Date
  if (obj instanceof Date) {
    result = new Date(obj.getTime());
    visited.set(obj as any, result);
    return result;
  }

  // RegExp
  if (obj instanceof RegExp) {
    result = new RegExp(obj.source, obj.flags);
    result.lastIndex = obj.lastIndex;
    visited.set(obj as any, result);
    return result;
  }

  // Map
  if (obj instanceof Map) {
    result = new Map();
    visited.set(obj as any, result); // 先に登録（循環参照対策）
    for (const [key, value] of obj) {
      result.set(deepClone(key, visited), deepClone(value, visited));
    }
    return result;
  }

  // Set
  if (obj instanceof Set) {
    result = new Set();
    visited.set(obj as any, result);
    for (const value of obj) {
      result.add(deepClone(value, visited));
    }
    return result;
  }

  // Array
  if (Array.isArray(obj)) {
    result = [];
    visited.set(obj as any, result);
    for (let i = 0; i < obj.length; i++) {
      result[i] = deepClone(obj[i], visited);
    }
    return result;
  }

  // ArrayBuffer
  if (obj instanceof ArrayBuffer) {
    result = obj.slice(0);
    visited.set(obj as any, result);
    return result;
  }

  // TypedArray (Uint8Array, Float32Array, etc.)
  if (ArrayBuffer.isView(obj)) {
    const typedArray = obj as any;
    result = new typedArray.constructor(deepClone(typedArray.buffer, visited));
    visited.set(obj as any, result);
    return result;
  }

  // プレーンオブジェクト
  result = Object.create(Object.getPrototypeOf(obj));
  visited.set(obj as any, result);

  for (const key of Reflect.ownKeys(obj as any)) {
    const descriptor = Object.getOwnPropertyDescriptor(obj, key as any);
    if (descriptor) {
      if ("value" in descriptor) {
        descriptor.value = deepClone(descriptor.value, visited);
      }
      Object.defineProperty(result, key, descriptor);
    }
  }

  return result;
}

// テスト
const original = {
  str: "hello",
  num: 42,
  bool: true,
  date: new Date("2024-01-01"),
  regex: /test/gi,
  map: new Map([["a", { nested: true }]]),
  set: new Set([1, 2, { x: 3 }]),
  arr: [1, [2, [3]]],
  circular: null as any,
};
original.circular = original;

const cloned = deepClone(original);

console.log(cloned.date instanceof Date);     // true
console.log(cloned.date !== original.date);   // true
console.log(cloned.regex instanceof RegExp);  // true
console.log(cloned.map.get("a")!.nested);     // true
console.log(cloned.map.get("a") !== original.map.get("a")); // true
console.log(cloned.circular === cloned);      // true
console.log(cloned.circular !== original);    // true
```
</details>

---

## 9. FAQ

### Q1: JavaScript の `structuredClone` はいつ使うべきですか？

DOM ノードや関数、Symbol を含まないプレーンなデータオブジェクトを深くコピーしたい場合に最適です。クラスインスタンスのメソッドやプロトタイプチェーンは失われるため、メソッドを持つオブジェクトにはカスタム `clone()` を実装してください。`structuredClone` は循環参照を自動で処理できる点が `JSON.parse(JSON.stringify(...))` より優れています。

### Q2: Prototype パターンと JavaScript の prototype チェーンは同じですか？

名前は似ていますが**全く別の概念**です。

| | GoF Prototype パターン | JavaScript prototype チェーン |
|--|--|--|
| 目的 | オブジェクトの**クローン生成** | プロパティの**委譲検索** |
| 操作 | clone() で新しいオブジェクトを作る | `obj.prop` でプロトタイプを辿る |
| 結果 | 独立したコピー | 共有された振る舞い |

JavaScript の `Object.create()` は GoF Prototype パターンに近い概念ですが、プロパティのコピーではなくプロトタイプチェーンの設定を行う点が異なります。

### Q3: 不変データ構造（Immutable Data Structure）を使えば clone() は不要ですか？

多くの場合、不変データ構造を使えば明示的な `clone()` は不要になります。Immutable.js や Immer では、**構造共有（structural sharing）** により変更されていない部分の参照を共有するため、Deep Copy よりも遥かに効率的です。ただし、以下のケースでは依然として Prototype パターンが有用です:
- 既存のミュータブルなクラスとの互換性が必要
- 構造共有のオーバーヘッドが問題になる小さなオブジェクト
- サードパーティライブラリのオブジェクトのコピー

### Q4: Java の `Cloneable` はなぜ「壊れた」インタフェースと言われるのですか？

Josh Bloch（`java.util.Collection` の設計者）が Effective Java で詳細に指摘しています:

1. **マーカーインタフェース**なのにメソッドがない: `clone()` は `Object` に定義されており、`Cloneable` にはない
2. **protected**: `Object.clone()` は protected なので、外部から呼べない（public にオーバーライドが必要）
3. **Shallow Copy のみ**: `super.clone()` は Shallow Copy しか行わない
4. **final フィールド非互換**: `clone()` 後に final フィールドを再設定できない
5. **例外が不適切**: `CloneNotSupportedException` はチェック例外だが、実際にはほぼ発生しない

**結論**: Java では Copy Constructor か static factory method(`copyOf()`) を使うべきです。

### Q5: Prototype パターンと Flyweight パターンの関係は？

| | Prototype | Flyweight |
|--|--|--|
| 目的 | オブジェクトの**複製** | オブジェクトの**共有** |
| メモリ | **増加**（コピー分） | **削減**（共有分） |
| 独立性 | **完全に独立** | 内在状態を共有 |

両者は対照的ですが、組み合わせることもあります。例えば、Prototype でクローンしたオブジェクトの内部で Flyweight を使って重いデータ（テクスチャ、フォントなど）を共有する設計です。

### Q6: テストでの Prototype パターンの活用方法は？

テストでは「テストフィクスチャ（test fixture）」の作成に Prototype パターンが非常に有効です:

```typescript
// テストの基本フィクスチャをプロトタイプとして定義
const baseUser = new User("test-user", "test@example.com", {
  role: "user",
  settings: { theme: "dark", notifications: true }
});

// 各テストケースでクローンしてカスタマイズ
test("admin can access settings", () => {
  const admin = baseUser.clone();
  admin.role = "admin";
  // ... テスト
});

test("user with notifications off", () => {
  const user = baseUser.clone();
  user.settings.notifications = false;
  // ... テスト
});
```

---

## まとめ

| 項目 | ポイント |
|------|---------|
| **目的** | 既存オブジェクトをクローンして新規生成（コンストラクタコストを回避） |
| **Shallow Copy** | 高速だが参照型フィールドを共有する（不変データなら安全） |
| **Deep Copy** | 完全独立だがコスト高（循環参照に注意） |
| **Registry** | プロトタイプをカタログ管理し、キーでクローンを取得 |
| **JS/TS 推奨** | 手動 clone() + structuredClone の併用 |
| **Python 推奨** | copy.deepcopy + __deepcopy__ カスタマイズ |
| **Java 推奨** | Copy Constructor（Cloneable は非推奨） |
| **Go 推奨** | 構造体ごとに Clone() メソッドを実装 |
| **Kotlin 推奨** | data class copy() + 不変設計 |
| **最重要注意** | clone() でも不変条件を維持する、一意IDは再生成する |
| **活用場面** | Undo/Redo、テストフィクスチャ、設定テンプレート、ゲーム状態保存 |

---

## 次に読むべきガイド

- [Singleton パターン](./00-singleton.md) — インスタンス数の制御と Prototype との対比
- [Factory パターン](./01-factory.md) — オブジェクト生成の抽象化（Prototype と併用可能）
- [Builder パターン](./02-builder.md) — 複雑なオブジェクトの段階的構築
- [Decorator パターン](../01-structural/01-decorator.md) — 動的な機能追加
- [Memento パターン](../02-behavioral/05-memento.md) — 状態の保存と復元（Prototype と関連）
- [不変性](../../../03-software-design/clean-code-principles/docs/03-practices-advanced/00-immutability.md) — イミュータブルデータ構造

---

## 参考文献

1. Gamma, E. et al. (1994). *Design Patterns: Elements of Reusable Object-Oriented Software*. Addison-Wesley.
2. Bloch, J. (2018). *Effective Java* (3rd ed.). Addison-Wesley. — Item 13: Override clone judiciously
3. MDN Web Docs — structuredClone(). https://developer.mozilla.org/en-US/docs/Web/API/structuredClone
4. Python Documentation — copy module. https://docs.python.org/3/library/copy.html
5. Refactoring.Guru — Prototype. https://refactoring.guru/design-patterns/prototype
6. Freeman, E. et al. (2004). *Head First Design Patterns*. O'Reilly Media.
