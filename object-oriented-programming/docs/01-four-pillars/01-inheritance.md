# 継承

> 継承は「既存クラスの機能を引き継いで新しいクラスを作る」仕組み。強力だが誤用しやすく、モダンOOPでは「継承よりコンポジション」が原則となっている。

## この章で学ぶこと

- [ ] 継承の仕組みとメモリ上の表現を理解する
- [ ] 継承の適切な使い方と落とし穴を把握する
- [ ] 抽象クラスとインターフェースの違いを学ぶ

---

## 1. 継承の基本

```
継承（Inheritance）:
  → 親クラス（スーパークラス）のフィールドとメソッドを
    子クラス（サブクラス）が引き継ぐ仕組み

  Animal（親クラス）
  ├── name: string
  ├── sound(): string
  └── move(): void
       ↑ 継承
  ┌─────┴──────┐
  Dog          Cat
  ├── breed    ├── indoor
  └── fetch()  └── purr()

  Dog は Animal の name, sound(), move() を自動的に持つ
  + 独自の breed, fetch() を追加
```

```typescript
// TypeScript: 基本的な継承
class Animal {
  constructor(
    protected name: string,
    protected age: number,
  ) {}

  speak(): string {
    return `${this.name}が鳴いています`;
  }

  toString(): string {
    return `${this.name} (${this.age}歳)`;
  }
}

class Dog extends Animal {
  constructor(name: string, age: number, private breed: string) {
    super(name, age); // 親のコンストラクタ呼び出し
  }

  // オーバーライド（親のメソッドを上書き）
  speak(): string {
    return `${this.name}「ワン！」`;
  }

  fetch(): string {
    return `${this.name}がボールを取ってきた`;
  }
}

class Cat extends Animal {
  speak(): string {
    return `${this.name}「ニャー」`;
  }
}

const dog = new Dog("ポチ", 3, "柴犬");
const cat = new Cat("タマ", 5);
console.log(dog.speak()); // ポチ「ワン！」
console.log(cat.speak()); // タマ「ニャー」
```

---

## 2. メソッドオーバーライドと super

```
オーバーライド（Override）:
  → 親クラスのメソッドを子クラスで再定義
  → 動的ディスパッチ: 実行時に実際の型のメソッドが呼ばれる

super の役割:
  → 親クラスのコンストラクタ/メソッドを明示的に呼ぶ
  → 「親の処理 + 追加の処理」パターン
```

```python
# Python: super() の使い方
class Shape:
    def __init__(self, color: str = "black"):
        self.color = color

    def area(self) -> float:
        raise NotImplementedError

    def describe(self) -> str:
        return f"{self.color}の{type(self).__name__}"

class Circle(Shape):
    def __init__(self, radius: float, color: str = "black"):
        super().__init__(color)  # 親の初期化
        self.radius = radius

    def area(self) -> float:
        return 3.14159 * self.radius ** 2

    def describe(self) -> str:
        return f"{super().describe()}, 半径{self.radius}"

class Rectangle(Shape):
    def __init__(self, width: float, height: float, color: str = "black"):
        super().__init__(color)
        self.width = width
        self.height = height

    def area(self) -> float:
        return self.width * self.height
```

---

## 3. 継承の種類

```
単一継承（Single Inheritance）:
  → 1つの親クラスのみ継承可能
  → Java, C#, Swift, Kotlin, Ruby
  → シンプルだが表現力に制限

多重継承（Multiple Inheritance）:
  → 複数の親クラスを継承可能
  → C++, Python
  → 強力だがダイヤモンド問題が発生

  ┌─────────┐
  │ Animal  │ ← ダイヤモンド問題
  └────┬────┘
  ┌────┴────┐
  ▼         ▼
┌─────┐  ┌──────┐
│ Fly │  │ Swim │
└──┬──┘  └──┬───┘
   └────┬───┘
        ▼
  ┌──────────┐
  │ FlyFish  │ ← Animal のメソッドをどちらから継承？
  └──────────┘
```

```python
# Python: MRO（Method Resolution Order）でダイヤモンド問題を解決
class Animal:
    def move(self):
        return "移動"

class Flyer(Animal):
    def move(self):
        return "飛ぶ"

class Swimmer(Animal):
    def move(self):
        return "泳ぐ"

class FlyingFish(Flyer, Swimmer):
    pass

fish = FlyingFish()
print(fish.move())  # "飛ぶ"（MRO: FlyingFish → Flyer → Swimmer → Animal）

# MROの確認
print(FlyingFish.__mro__)
# (FlyingFish, Flyer, Swimmer, Animal, object)
# → C3線形化アルゴリズムで順序を決定
```

---

## 4. 継承の落とし穴

```
問題1: 脆い基底クラス問題（Fragile Base Class）
  → 親クラスの変更が子クラスを壊す

問題2: 不適切な is-a 関係
  → 正方形 is-a 長方形？（リスコフの置換原則に違反）

問題3: 深い継承階層
  → 3段階以上の継承は理解困難
  → Entity → LivingEntity → Animal → Mammal → Dog → GuideDog
  → 各レイヤーの変更が下位全てに影響

問題4: 継承によるカプセル化の破壊
  → 子クラスが親の実装詳細に依存
  → protected フィールドへの直接アクセス

問題5: ゴリラ・バナナ問題
  「バナナが欲しいだけなのに、バナナを持ったゴリラと
   ジャングル全体がついてきた」
  → 継承すると不要な機能も全てついてくる
```

```java
// 脆い基底クラス問題の例
public class HashSet<E> {
    private int addCount = 0;

    public boolean add(E e) {
        addCount++;
        // ... 実際の追加処理
        return true;
    }

    public boolean addAll(Collection<E> c) {
        // 内部で add() を呼ぶ実装
        for (E e : c) add(e);
        return true;
    }

    public int getAddCount() { return addCount; }
}

// 問題のあるサブクラス
public class InstrumentedHashSet<E> extends HashSet<E> {
    private int addCount = 0;

    @Override
    public boolean add(E e) {
        addCount++;
        return super.add(e);
    }

    @Override
    public boolean addAll(Collection<E> c) {
        addCount += c.size();
        return super.addAll(c); // super.addAll() が add() を呼ぶ！
    }
    // addAll({a, b, c}) → addCount = 6（期待は3）
    // → super.addAll() が内部で add() を呼び、二重カウント
}
```

---

## 5. 抽象クラス

```
抽象クラス:
  → インスタンス化できないクラス
  → 共通の実装 + サブクラスへの実装義務を定義
  → 「テンプレートメソッドパターン」の基盤

用途:
  - 共通のフィールドと一部のメソッド実装を提供
  - サブクラスが実装すべきメソッドを強制
  - is-a 関係が明確な場合
```

```typescript
// TypeScript: 抽象クラス
abstract class DatabaseConnection {
  protected connected: boolean = false;

  // 共通実装
  async query(sql: string): Promise<any[]> {
    if (!this.connected) {
      await this.connect();
    }
    return this.executeQuery(sql);
  }

  // サブクラスが実装すべき抽象メソッド
  abstract connect(): Promise<void>;
  abstract disconnect(): Promise<void>;
  protected abstract executeQuery(sql: string): Promise<any[]>;
}

class PostgresConnection extends DatabaseConnection {
  async connect(): Promise<void> {
    // PostgreSQL固有の接続処理
    this.connected = true;
  }

  async disconnect(): Promise<void> {
    this.connected = false;
  }

  protected async executeQuery(sql: string): Promise<any[]> {
    // PostgreSQL固有のクエリ実行
    return [];
  }
}
```

---

## 6. 継承の適切な使い方

```
継承を使うべき場面:
  ✓ 明確な is-a 関係がある
  ✓ サブクラスが親のインターフェースを完全に満たす（LSP）
  ✓ フレームワークが継承を要求する（Androidの Activity等）
  ✓ テンプレートメソッドパターン

継承を避けるべき場面:
  ✗ コードの再利用だけが目的
  ✗ has-a 関係（コンポジションを使う）
  ✗ 3段階以上の継承が必要
  ✗ 親クラスの一部のメソッドだけ使いたい

判断基準:
  「このサブクラスは、親クラスが使える全ての場面で
   代替として使えるか？」
  → Yes → 継承が適切
  → No  → コンポジションを使う
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| 継承 | 親の機能を引き継いで拡張 |
| オーバーライド | 親メソッドの再定義 |
| 多重継承 | ダイヤモンド問題に注意 |
| 抽象クラス | 共通実装 + 実装義務 |
| 原則 | 「継承よりコンポジション」 |

---

## 次に読むべきガイド
→ [[02-polymorphism.md]] — ポリモーフィズム

---

## 参考文献
1. Bloch, J. "Effective Java." Item 18: Favor composition over inheritance. 2018.
2. Gamma, E. et al. "Design Patterns." Addison-Wesley, 1994.
