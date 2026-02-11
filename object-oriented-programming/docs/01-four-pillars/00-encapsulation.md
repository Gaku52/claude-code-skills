# カプセル化

> カプセル化は「データとそれを操作するメソッドを1つの単位にまとめ、内部の実装詳細を隠蔽する」原則。OOPの4つの柱の中で最も基本的かつ重要。

## この章で学ぶこと

- [ ] カプセル化の2つの側面（バンドリングと情報隠蔽）を理解する
- [ ] アクセス修飾子の使い分けを把握する
- [ ] 不変オブジェクトの設計を学ぶ

---

## 1. カプセル化の2つの側面

```
カプセル化 = バンドリング + 情報隠蔽

  バンドリング（Bundling）:
    → 関連するデータとメソッドを1つのクラスにまとめる
    → 「このデータはこのメソッドで操作する」という意図を明確化

  情報隠蔽（Information Hiding）:
    → 内部の実装詳細を外部から見えなくする
    → 外部には必要最小限のインターフェースのみ公開
    → 内部実装を変更しても外部に影響しない

  ┌──────────────────────────────────┐
  │         BankAccount              │
  │  ┌──────────────────────────┐   │
  │  │ private:                 │   │
  │  │   balance: number        │   │  内部（隠蔽）
  │  │   transactions: Log[]    │   │
  │  │   validate(amount)       │   │
  │  └──────────────────────────┘   │
  │  ┌──────────────────────────┐   │
  │  │ public:                  │   │
  │  │   deposit(amount)        │   │  外部（公開API）
  │  │   withdraw(amount)       │   │
  │  │   getBalance()           │   │
  │  └──────────────────────────┘   │
  └──────────────────────────────────┘
```

---

## 2. アクセス修飾子

```
┌──────────────┬───────────┬──────────┬───────────┬──────────┐
│ 修飾子       │ クラス内  │ サブクラス│ パッケージ│ 外部     │
├──────────────┼───────────┼──────────┼───────────┼──────────┤
│ private      │ ○        │ ×       │ ×        │ ×       │
│ protected    │ ○        │ ○       │ △(Java)  │ ×       │
│ package      │ ○        │ ×       │ ○        │ ×       │
│ public       │ ○        │ ○       │ ○        │ ○       │
└──────────────┴───────────┴──────────┴───────────┴──────────┘

原則: 最も制限的なアクセスレベルを選ぶ
  → まず private にして、必要に応じて公開範囲を広げる
```

### 各言語のアクセス制御

```typescript
// TypeScript
class User {
  public name: string;        // どこからでもアクセス可
  protected email: string;    // サブクラスからアクセス可
  private password: string;   // クラス内のみ
  readonly id: string;        // 読み取り専用

  constructor(name: string, email: string, password: string) {
    this.id = crypto.randomUUID();
    this.name = name;
    this.email = email;
    this.password = password;
  }
}
```

```python
# Python: 規約ベース（強制ではない）
class User:
    def __init__(self, name: str, email: str, password: str):
        self.name = name          # public（規約）
        self._email = email       # protected（規約: アンダースコア1つ）
        self.__password = password # private（名前マングリング）

    @property
    def email(self) -> str:       # プロパティでアクセス制御
        return self._email

    @email.setter
    def email(self, value: str) -> None:
        if "@" not in value:
            raise ValueError("Invalid email")
        self._email = value
```

```java
// Java: 厳格なアクセス制御
public class User {
    private final String id;           // private + final = 不変
    private String name;
    private String email;

    public User(String name, String email) {
        this.id = UUID.randomUUID().toString();
        this.name = name;
        this.email = email;
    }

    // getter: 読み取りのみ公開
    public String getName() { return name; }
    public String getEmail() { return email; }

    // setter: バリデーション付き
    public void setEmail(String email) {
        if (!email.contains("@")) {
            throw new IllegalArgumentException("Invalid email");
        }
        this.email = email;
    }

    // id の setter は提供しない → 外部から変更不可
}
```

---

## 3. ゲッター/セッター論争

```
「全フィールドに getter/setter を付ける」は悪い慣習:

  悪い例（Anemic Domain Model）:
    class User {
      getName() / setName()
      getAge() / setAge()
      getEmail() / setEmail()
      getBalance() / setBalance()
    }
    → 単なるデータの器。振る舞いが外部に漏れ出す
    → カプセル化の意味がない

  良い例（Rich Domain Model）:
    class BankAccount {
      deposit(amount)     ← ビジネスロジックを内包
      withdraw(amount)    ← バリデーション含む
      getBalance()        ← 読み取りのみ
      // setBalance() は存在しない
    }
    → オブジェクトが自分の責任で状態を管理

指針:
  getter: 必要なものだけ公開
  setter: 原則として作らない。代わりにビジネスメソッドを提供
  → 「Tell, Don't Ask」原則
```

```typescript
// Tell, Don't Ask の例

// ❌ Ask（状態を聞いてから外部で判断）
if (account.getBalance() >= amount) {
  account.setBalance(account.getBalance() - amount);
}

// ✅ Tell（オブジェクトに指示する）
account.withdraw(amount); // 内部でバリデーション + 更新
```

---

## 4. 不変オブジェクト

```
不変オブジェクト（Immutable Object）:
  → 生成後に状態が変化しないオブジェクト
  → スレッドセーフ（ロック不要）
  → 予測可能（副作用なし）
  → ハッシュキーとして安全

作り方:
  1. 全フィールドを final/readonly にする
  2. setter を提供しない
  3. コンストラクタで全ての値を設定
  4. ミュータブルな参照を外部に漏らさない
```

```typescript
// TypeScript: 不変オブジェクト
class Money {
  constructor(
    public readonly amount: number,
    public readonly currency: string,
  ) {
    if (amount < 0) throw new Error("金額は0以上");
  }

  // 変更メソッドは新しいオブジェクトを返す
  add(other: Money): Money {
    if (this.currency !== other.currency) {
      throw new Error("通貨が異なります");
    }
    return new Money(this.amount + other.amount, this.currency);
  }

  multiply(factor: number): Money {
    return new Money(this.amount * factor, this.currency);
  }

  toString(): string {
    return `${this.amount} ${this.currency}`;
  }
}

const price = new Money(1000, "JPY");
const tax = price.multiply(0.1);      // 新しいオブジェクト
const total = price.add(tax);         // 新しいオブジェクト
// price は変わらない（不変）
```

```kotlin
// Kotlin: data class（不変オブジェクトの簡潔な記法）
data class Point(val x: Double, val y: Double) {
    fun distanceTo(other: Point): Double =
        sqrt((x - other.x).pow(2) + (y - other.y).pow(2))

    // copy() で一部だけ変えた新しいオブジェクトを生成
    fun translate(dx: Double, dy: Double): Point =
        copy(x = x + dx, y = y + dy)
}

val p1 = Point(1.0, 2.0)
val p2 = p1.translate(3.0, 4.0)  // Point(4.0, 6.0)
// p1 は Point(1.0, 2.0) のまま
```

---

## 5. カプセル化のアンチパターン

```
1. 全公開（public フィールド）:
   → 内部実装への依存が発生
   → 変更すると利用者コードが全て壊れる

2. 過剰な getter/setter:
   → Anemic Domain Model
   → カプセル化の意味がない

3. 内部コレクションの漏洩:
   class Team {
     getMembers(): Member[] { return this.members; }
   }
   → 外部で members.push() されると内部状態が壊れる
   → 防衛的コピーまたは ReadonlyArray を返す

4. フレンドクラスの乱用（C++）:
   → カプセル化の境界を曖昧にする

5. リフレクションによるアクセス:
   → private を無視してアクセス可能
   → テスト以外では使わない
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| バンドリング | データとメソッドを1つの単位に |
| 情報隠蔽 | 内部実装を隠し、公開APIのみ提供 |
| アクセス修飾子 | 最も制限的なレベルを選ぶ |
| getter/setter | setterは原則不要。ビジネスメソッドを提供 |
| 不変オブジェクト | 変更時は新しいオブジェクトを返す |

---

## 次に読むべきガイド
→ [[01-inheritance.md]] — 継承

---

## 参考文献
1. Bloch, J. "Effective Java." Item 16: Favor immutability. 2018.
2. Fowler, M. "Anemic Domain Model." martinfowler.com, 2003.
