# OOP vs 他パラダイム

> OOPは万能ではない。手続き型、関数型、リアクティブ、アクターモデルなど、各パラダイムの強みと弱みを理解し、適切に使い分けることが現代のエンジニアに求められる。

## この章で学ぶこと

- [ ] 主要パラダイムの特徴と適用領域を把握する
- [ ] OOPと関数型の根本的な設計思想の違いを理解する
- [ ] マルチパラダイムの実践的な使い分けを学ぶ

---

## 1. パラダイム比較

```
┌──────────────┬──────────────┬──────────────┬──────────────┐
│              │ 手続き型     │ OOP          │ 関数型       │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ 中心概念      │ 手順・命令   │ オブジェクト │ 関数・変換   │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ データ管理    │ グローバル変数│ カプセル化   │ イミュータブル│
├──────────────┼──────────────┼──────────────┼──────────────┤
│ コード再利用  │ 関数         │ 継承/合成    │ 高階関数     │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ 状態管理      │ 変数の変更   │ メソッドで変更│ 新しい値を返す│
├──────────────┼──────────────┼──────────────┼──────────────┤
│ 副作用        │ どこでも発生 │ メソッド内   │ 最小限に制限 │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ テスト容易性  │ △           │ ○           │ ◎           │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ 並行処理      │ 困難         │ 困難(共有状態)│ 容易(不変)   │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ 学習コスト    │ 低           │ 中           │ 高           │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

---

## 2. 同じ問題を各パラダイムで解く

### 問題: ユーザーリストから成人のメールアドレスを取得

```python
# === 手続き型 ===
users = [
    {"name": "田中", "age": 25, "email": "tanaka@example.com"},
    {"name": "山田", "age": 17, "email": "yamada@example.com"},
    {"name": "佐藤", "age": 30, "email": "sato@example.com"},
]

adult_emails = []
for user in users:
    if user["age"] >= 18:
        adult_emails.append(user["email"])
# → 手順を逐次的に記述
```

```python
# === OOP ===
class User:
    def __init__(self, name: str, age: int, email: str):
        self.name = name
        self.age = age
        self.email = email

    def is_adult(self) -> bool:
        return self.age >= 18

class UserRepository:
    def __init__(self, users: list[User]):
        self._users = users

    def get_adult_emails(self) -> list[str]:
        return [u.email for u in self._users if u.is_adult()]

# → 責任をオブジェクトに委譲
repo = UserRepository([
    User("田中", 25, "tanaka@example.com"),
    User("山田", 17, "yamada@example.com"),
    User("佐藤", 30, "sato@example.com"),
])
adult_emails = repo.get_adult_emails()
```

```python
# === 関数型 ===
from dataclasses import dataclass
from typing import Callable

@dataclass(frozen=True)  # イミュータブル
class User:
    name: str
    age: int
    email: str

def is_adult(user: User) -> bool:
    return user.age >= 18

def get_email(user: User) -> str:
    return user.email

users = [
    User("田中", 25, "tanaka@example.com"),
    User("山田", 17, "yamada@example.com"),
    User("佐藤", 30, "sato@example.com"),
]

adult_emails = list(map(get_email, filter(is_adult, users)))
# → データ変換のパイプライン
```

---

## 3. OOP vs 関数型: 根本的な違い

```
Expression Problem（表現問題）:

  OOP: 新しい「型」の追加が容易、新しい「操作」の追加が困難
  FP:  新しい「操作」の追加が容易、新しい「型」の追加が困難

  例: 図形の描画

  OOP:
    Shape（抽象クラス）
    ├── Circle.draw()      ← 新しい型（Triangle）を追加するのは簡単
    ├── Rectangle.draw()      各クラスにdraw()を実装するだけ
    └── Triangle.draw()    ← でも新しい操作（area()）を追加すると
                              全クラスを修正する必要がある

  FP:
    draw(shape) = match shape with    ← 新しい操作（area()）を追加するのは簡単
    | Circle r -> ...                    新しい関数を定義するだけ
    | Rectangle w h -> ...            ← でも新しい型（Triangle）を追加すると
                                         全関数を修正する必要がある

  → どちらが良いかは「何が頻繁に変わるか」による
  → 型が増える → OOP
  → 操作が増える → FP
```

### 状態管理の違い

```
OOP: 可変状態をカプセル化
  account.deposit(1000)   ← オブジェクトの内部状態が変わる
  account.withdraw(500)   ← 同じオブジェクトが変化し続ける

FP: 不変データを変換
  newAccount = deposit(account, 1000)   ← 新しいオブジェクトを返す
  finalAccount = withdraw(newAccount, 500) ← 元のaccountは変わらない

  FPの利点:
    - 並行処理が安全（共有状態がない）
    - タイムトラベルデバッグが可能
    - テストが容易（同じ入力 → 同じ出力）

  OOPの利点:
    - 直感的（現実世界のモデルに近い）
    - GUIやゲームなど、状態変化が本質的な領域に適合
    - メモリ効率が良い（オブジェクトを更新するだけ）
```

---

## 4. マルチパラダイムの実践

```
現代のベストプラクティス:

  「ドメインモデル」→ OOP
    ビジネスエンティティの表現にクラスを使う
    例: User, Order, Product

  「データ変換」→ FP
    データのフィルタリング・変換に関数を使う
    例: map, filter, reduce

  「副作用の管理」→ FP
    I/O、DB操作を関数の境界に押しやる

  「状態を持つUI」→ OOP + リアクティブ
    コンポーネントの状態管理

実践例（TypeScript）:
```

```typescript
// ドメインモデル: OOP
class Order {
  constructor(
    public readonly id: string,
    public readonly items: OrderItem[],
    public readonly status: OrderStatus,
    public readonly createdAt: Date,
  ) {}

  get totalPrice(): number {
    // FP: データ変換
    return this.items
      .map(item => item.price * item.quantity)
      .reduce((sum, price) => sum + price, 0);
  }

  canCancel(): boolean {
    return this.status === "pending" || this.status === "confirmed";
  }
}

// データ変換: FP
const getRecentHighValueOrders = (orders: Order[]): Order[] =>
  orders
    .filter(order => order.totalPrice > 10000)
    .filter(order => order.createdAt > thirtyDaysAgo())
    .sort((a, b) => b.totalPrice - a.totalPrice);

// 副作用の管理: 関数の境界に分離
async function processOrders(repo: OrderRepository): Promise<void> {
  const orders = await repo.findAll();          // 副作用（DB）
  const highValue = getRecentHighValueOrders(orders); // 純粋関数
  await notifyAdmins(highValue);                // 副作用（通知）
}
```

---

## 5. 選択指針

```
OOPを使うべき場面:
  ✓ ビジネスドメインのモデリング（エンティティが多い）
  ✓ GUIフレームワーク（ウィジェット階層）
  ✓ ゲームのエンティティシステム
  ✓ フレームワーク/ライブラリの公開API設計
  ✓ チーム開発（構造の共通理解が重要）

関数型を使うべき場面:
  ✓ データ処理パイプライン
  ✓ 並行・分散処理
  ✓ 数学的・科学的計算
  ✓ コンパイラ・パーサー
  ✓ 状態を持たない変換ロジック

手続き型を使うべき場面:
  ✓ シンプルなスクリプト
  ✓ シェルスクリプト的な処理
  ✓ プロトタイプ・使い捨てコード
  ✓ ハードウェア制御

マルチパラダイム（推奨）:
  → ドメインモデル = OOP
  → データ変換 = FP
  → 設定・スクリプト = 手続き型
  → 並行処理 = アクターモデル / CSP
```

---

## まとめ

| パラダイム | 核心 | 得意分野 | 弱点 |
|-----------|------|---------|------|
| 手続き型 | 手順の記述 | シンプル・スクリプト | 大規模で破綻 |
| OOP | オブジェクト | ドメインモデル・GUI | 共有状態・並行処理 |
| 関数型 | データ変換 | 並行処理・パイプライン | GUI・状態管理 |
| マルチ | 適材適所 | 現代の大規模開発 | 判断力が必要 |

---

## 次に読むべきガイド
→ [[03-class-and-object.md]] — クラスとオブジェクト

---

## 参考文献
1. Wadler, P. "The Expression Problem." 1998.
2. Martin, R. "Clean Architecture." Prentice Hall, 2017.
