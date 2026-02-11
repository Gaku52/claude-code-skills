# イミュータビリティ（不変性）の原則

> なぜ不変データが安全なコードを生むのか。言語別の実装パターン、パフォーマンスへの影響、マルチスレッド環境での恩恵まで、イミュータビリティの理論と実践を網羅する。

---

## この章で学ぶこと

1. **イミュータビリティの理論的根拠**を理解し、なぜ不変データが安全で予測可能なコードを生むか説明できる
2. **言語別の不変性実装パターン**（Java、TypeScript、Python、Rust、Kotlin）を習得し、実務に適用できる
3. **パフォーマンスとのトレードオフ**を理解し、構造共有やコピーオンライトなどの最適化手法を使い分けられる

---

## 1. イミュータビリティとは何か

### 1.1 ミュータブル vs イミュータブル

```
ミュータブル（可変）                 イミュータブル（不変）
─────────────────                ─────────────────

  user.name = "田中"               newUser = user.copy(name="田中")
       │                                │
       v                                v
  ┌──────────┐                    ┌──────────┐  ┌──────────┐
  │ user     │                    │ user     │  │ newUser  │
  │ name:"田中"│  ← 元が変わる     │ name:"鈴木"│  │ name:"田中"│
  │ age: 30  │                    │ age: 30  │  │ age: 30  │
  └──────────┘                    └──────────┘  └──────────┘
                                   元は不変        新しいコピー

  問題: 誰がいつ変えた？           利点: 変更履歴が明確
  共有参照で予期せぬ変更           共有しても安全
```

### 1.2 不変性がもたらす利点

```
┌───────────────────────────────────────────────────┐
│            イミュータビリティの5つの利点              │
├───────────────────────────────────────────────────┤
│                                                   │
│  1. 予測可能性                                     │
│     値が変わらない → 関数の結果が常に同じ            │
│                                                   │
│  2. スレッド安全性                                  │
│     変更がない → ロック不要 → デッドロックなし       │
│                                                   │
│  3. デバッグ容易性                                  │
│     状態変化がない → 問題の再現が容易               │
│                                                   │
│  4. 変更検知の効率化                                │
│     参照比較だけで変更判定 → O(1)                   │
│                                                   │
│  5. 履歴管理（Undo/Redo）                          │
│     古い状態がそのまま残る → タイムトラベル可能      │
│                                                   │
└───────────────────────────────────────────────────┘
```

---

## 2. 言語別イミュータビリティ実装

### 2.1 TypeScript / JavaScript

```typescript
// TypeScript: イミュータブルなデータ操作

// 1. readonly と as const
interface User {
  readonly id: string;
  readonly name: string;
  readonly age: number;
  readonly address: Readonly<Address>;
}

interface Address {
  readonly prefecture: string;
  readonly city: string;
}

// as const で深い不変性
const CONFIG = {
  api: {
    baseUrl: "https://api.example.com",
    timeout: 5000,
  },
  features: ["auth", "logging"] as const,
} as const;

// 2. Readonly ユーティリティ型
type DeepReadonly<T> = {
  readonly [P in keyof T]: T[P] extends object
    ? DeepReadonly<T[P]>
    : T[P];
};

// 3. 不変な更新パターン
function updateUserName(user: User, newName: string): User {
  // スプレッド構文で新しいオブジェクトを生成
  return { ...user, name: newName };
}

// 4. 配列の不変操作
function addItem<T>(items: readonly T[], item: T): readonly T[] {
  return [...items, item];  // 新しい配列を返す
}

function removeItem<T>(items: readonly T[], index: number): readonly T[] {
  return [...items.slice(0, index), ...items.slice(index + 1)];
}

function updateItem<T>(
  items: readonly T[], index: number, updater: (item: T) => T
): readonly T[] {
  return items.map((item, i) => (i === index ? updater(item) : item));
}

// 5. Map/Set の不変操作
function addToMap<K, V>(map: ReadonlyMap<K, V>, key: K, value: V): ReadonlyMap<K, V> {
  const newMap = new Map(map);
  newMap.set(key, value);
  return newMap;
}
```

### 2.2 Java

```java
// Java: イミュータブルクラスの設計パターン

// 1. 基本的なイミュータブルクラス
public final class Money {  // final で継承禁止
    private final int amount;       // final で再代入禁止
    private final String currency;

    public Money(int amount, String currency) {
        this.amount = amount;
        this.currency = Objects.requireNonNull(currency);
    }

    // ゲッターのみ（セッターなし）
    public int getAmount() { return amount; }
    public String getCurrency() { return currency; }

    // 変更は新しいインスタンスを返す
    public Money add(Money other) {
        if (!this.currency.equals(other.currency)) {
            throw new IllegalArgumentException("通貨が異なります");
        }
        return new Money(this.amount + other.amount, this.currency);
    }

    public Money multiply(int factor) {
        return new Money(this.amount * factor, this.currency);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Money money)) return false;
        return amount == money.amount
            && currency.equals(money.currency);
    }

    @Override
    public int hashCode() {
        return Objects.hash(amount, currency);
    }
}

// 2. Java 16+ Record（自動的にイミュータブル）
public record User(
    String id,
    String name,
    int age,
    Address address
) {
    // バリデーション付きコンストラクタ
    public User {
        Objects.requireNonNull(id, "IDは必須です");
        Objects.requireNonNull(name, "名前は必須です");
        if (age < 0) throw new IllegalArgumentException("年齢は0以上");
    }

    // Wither パターン
    public User withName(String newName) {
        return new User(id, newName, age, address);
    }

    public User withAge(int newAge) {
        return new User(id, name, newAge, address);
    }
}

// 3. 不変コレクション（Java 9+）
var immutableList = List.of("a", "b", "c");
var immutableMap = Map.of("key1", "value1", "key2", "value2");
var immutableSet = Set.of(1, 2, 3);
// immutableList.add("d"); // UnsupportedOperationException
```

### 2.3 Python

```python
# Python: イミュータビリティの実装パターン

# 1. frozen dataclass
from dataclasses import dataclass, replace

@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def translate(self, dx: float, dy: float) -> "Point":
        """新しいPointを返す（元は変更しない）"""
        return replace(self, x=self.x + dx, y=self.y + dy)

    def distance_to(self, other: "Point") -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

p1 = Point(1.0, 2.0)
p2 = p1.translate(3.0, 4.0)
# p1.x = 5.0  # FrozenInstanceError

# 2. NamedTuple（軽量なイミュータブル型）
from typing import NamedTuple

class Color(NamedTuple):
    r: int
    g: int
    b: int
    a: float = 1.0

    def with_alpha(self, alpha: float) -> "Color":
        return self._replace(a=alpha)

# 3. 不変辞書パターン
from types import MappingProxyType

def create_config(overrides: dict = None) -> MappingProxyType:
    """変更不可な設定辞書を作成"""
    defaults = {
        "debug": False,
        "log_level": "INFO",
        "max_retries": 3,
    }
    if overrides:
        defaults.update(overrides)
    return MappingProxyType(defaults)  # 読み取り専用ビュー

config = create_config({"debug": True})
# config["debug"] = False  # TypeError

# 4. Pydantic v2のイミュータブルモデル
from pydantic import BaseModel, ConfigDict

class UserModel(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    email: str

    def update_name(self, new_name: str) -> "UserModel":
        return self.model_copy(update={"name": new_name})
```

### 2.4 Rust（言語レベルでの不変性）

```rust
// Rust: デフォルトがイミュータブル

// 1. 変数はデフォルトで不変
fn main() {
    let x = 5;
    // x = 6;  // コンパイルエラー！

    let mut y = 5;  // mut を明示的に付ける
    y = 6;          // OK

    // 2. 構造体の不変性
    let user = User::new("tanaka".to_string(), 30);
    // user.name = "suzuki".to_string();  // コンパイルエラー！
    let new_user = user.with_name("suzuki".to_string());  // OK
}

#[derive(Clone, Debug)]
struct User {
    name: String,
    age: u32,
}

impl User {
    fn new(name: String, age: u32) -> Self {
        Self { name, age }
    }

    // Builder パターンで不変更新
    fn with_name(self, name: String) -> Self {
        Self { name, ..self }
    }

    fn with_age(self, age: u32) -> Self {
        Self { age, ..self }
    }
}

// 3. 共有参照は不変（&T）、排他参照は可変（&mut T）
fn print_user(user: &User) {       // 読み取りのみ
    println!("{:?}", user);
}

fn update_age(user: &mut User) {   // 変更可能（排他的）
    user.age += 1;
}
```

---

## 3. パフォーマンスと最適化

### 3.1 構造共有（Structural Sharing）

```
構造共有: 変更されていない部分を共有する

  元のツリー          nameを変更後のツリー
  ──────────          ─────────────────

      root                  newRoot
     /    \                /    \
    A      B            newA     B  ← 共有（コピーなし）
   / \    / \           / \    / \
  a1  a2 b1  b2      a1* a2 b1  b2  ← B以下は全て共有
                       ↑
                   変更された部分のみ新規作成

  メモリ効率: O(log n) の新規ノードで済む
  変更検知: ルートの参照が異なれば変更あり → O(1)
```

### 3.2 コピーオンライトとの比較

| 戦略 | メモリ効率 | CPU効率 | 実装複雑度 | 適用場面 |
|------|-----------|---------|-----------|---------|
| 毎回フルコピー | 低 | 低 | 簡単 | 小さなデータ |
| 構造共有 | 高 | 高 | 複雑 | 永続データ構造 |
| コピーオンライト | 高 | 中 | 中 | OS/ランタイムレベル |
| Immer (JS) | 中 | 中 | 簡単 | Reactアプリ |
| 永続データ構造 | 高 | 高 | 非常に複雑 | 関数型言語 |

### 3.3 Immer.jsによる効率的な不変更新

```typescript
// Immer: ミュータブルな記法でイミュータブルな更新
import { produce, Draft } from "immer";

interface AppState {
  users: User[];
  selectedId: string | null;
  filters: {
    status: string;
    search: string;
  };
}

// produce で「ドラフト」に直接変更を加える
// → Immerが自動的に不変な新しい状態を生成
const nextState = produce(currentState, (draft: Draft<AppState>) => {
  // ミュータブルな記法で書ける！
  const user = draft.users.find(u => u.id === "123");
  if (user) {
    user.name = "新しい名前";  // 直接変更OK（ドラフト上）
  }
  draft.filters.search = "検索ワード";
});

// currentState は変更されていない（不変）
// nextState は新しいオブジェクト
console.log(currentState === nextState); // false
console.log(currentState.users[1] === nextState.users[1]); // true（変更なし→共有）

// React useReducer との組み合わせ
import { useImmerReducer } from "use-immer";

type Action =
  | { type: "UPDATE_USER"; id: string; name: string }
  | { type: "ADD_USER"; user: User }
  | { type: "DELETE_USER"; id: string };

function reducer(draft: Draft<AppState>, action: Action): void {
  switch (action.type) {
    case "UPDATE_USER": {
      const user = draft.users.find(u => u.id === action.id);
      if (user) user.name = action.name;
      break;
    }
    case "ADD_USER":
      draft.users.push(action.user);
      break;
    case "DELETE_USER":
      draft.users = draft.users.filter(u => u.id !== action.id);
      break;
  }
}
```

---

## 4. マルチスレッド環境での恩恵

### 4.1 可変データの並行処理の危険

```
スレッド1              共有可変データ           スレッド2
─────────             ──────────────          ─────────

read(balance)          balance = 1000         read(balance)
  → 1000                                       → 1000

balance -= 500         balance = ???           balance -= 300
  → 500                                        → 700

write(balance)                                write(balance)
  → balance = 500       ← レースコンディション   → balance = 700

期待値: 200 (1000 - 500 - 300)
実際値: 500 or 700（後にwriteした方が勝つ）
```

### 4.2 不変データならロック不要

```
スレッド1              不変データ              スレッド2
─────────             ──────────             ─────────

read(account)          account{balance:1000}  read(account)
  → {balance:1000}     (変更されない)           → {balance:1000}

new1 = withdraw(500)   account{balance:1000}  new2 = withdraw(300)
  → {balance:500}      (元は不変)              → {balance:700}

CAS(account, new1)     ← Compare-And-Swap     CAS(account, new2)
  → 成功                                       → 失敗→リトライ

                                              new3 = withdraw(300)
                                                from account(500)
                                              CAS → 成功

最終結果: {balance: 200} ← 正しい！
```

---

## 5. 実践パターン

### 5.1 イベントソーシングとの親和性

```python
# イベントソーシング: 不変イベントの蓄積で状態を管理
from dataclasses import dataclass
from typing import Union
from datetime import datetime

@dataclass(frozen=True)
class AccountCreated:
    account_id: str
    owner: str
    timestamp: datetime

@dataclass(frozen=True)
class MoneyDeposited:
    account_id: str
    amount: int
    timestamp: datetime

@dataclass(frozen=True)
class MoneyWithdrawn:
    account_id: str
    amount: int
    timestamp: datetime

Event = Union[AccountCreated, MoneyDeposited, MoneyWithdrawn]

@dataclass(frozen=True)
class AccountState:
    """不変な口座状態"""
    account_id: str
    owner: str
    balance: int

    @staticmethod
    def apply(state: "AccountState | None", event: Event) -> "AccountState":
        """イベントを適用して新しい状態を返す"""
        match event:
            case AccountCreated(account_id, owner, _):
                return AccountState(account_id, owner, 0)
            case MoneyDeposited(_, amount, _):
                return AccountState(state.account_id, state.owner, state.balance + amount)
            case MoneyWithdrawn(_, amount, _):
                if state.balance < amount:
                    raise ValueError("残高不足")
                return AccountState(state.account_id, state.owner, state.balance - amount)

# 使用例: イベント列から状態を復元
events = [
    AccountCreated("acc-1", "田中", datetime.now()),
    MoneyDeposited("acc-1", 10000, datetime.now()),
    MoneyWithdrawn("acc-1", 3000, datetime.now()),
]

state = None
for event in events:
    state = AccountState.apply(state, event)

print(state)  # AccountState(account_id='acc-1', owner='田中', balance=7000)
```

---

## 6. アンチパターン

### 6.1 アンチパターン：浅いコピーの罠

```python
# NG: 浅いコピーでネストされたオブジェクトが共有される
original = {"user": {"name": "田中", "scores": [90, 85]}}
copied = original.copy()  # 浅いコピー

copied["user"]["name"] = "鈴木"
print(original["user"]["name"])  # "鈴木" ← 元も変わってしまう！

# OK: 深いコピーまたは不変データ構造を使用
import copy
deep_copied = copy.deepcopy(original)
deep_copied["user"]["name"] = "鈴木"
print(original["user"]["name"])  # "田中" ← 元は変わらない

# より良い: frozen dataclass で根本的に防止
@dataclass(frozen=True)
class User:
    name: str
    scores: tuple[int, ...]  # tupleは不変
```

**問題点**: 浅いコピーはネストされた参照を共有するため、意図しない変更が伝播する。深いコピーか不変データ構造で対処する。

### 6.2 アンチパターン：全てをイミュータブルにする

```python
# NG: パフォーマンスクリティカルな処理でも不変性を強制
def process_large_dataset(data: tuple) -> tuple:
    result = data
    for i in range(len(data)):
        # 毎回タプル全体をコピー → O(n^2) の計算量
        result = result[:i] + (transform(result[i]),) + result[i+1:]
    return result

# OK: 内部処理は可変、外部インターフェースは不変
def process_large_dataset(data: tuple) -> tuple:
    # 内部ではリスト（可変）で効率的に処理
    work_list = list(data)
    for i in range(len(work_list)):
        work_list[i] = transform(work_list[i])
    # 結果はタプル（不変）で返す
    return tuple(work_list)
```

**問題点**: 全てを不変にするとパフォーマンスが劣化する場合がある。境界を明確にし、内部実装は可変でもよい。

---

## 7. FAQ

### Q1: イミュータビリティはパフォーマンスに悪影響か？

**A**: 小〜中規模データでは影響は無視できる。大規模データでは構造共有（Persistent Data Structures）やImmer.jsのようなライブラリで効率的に処理できる。むしろ、変更検知がO(1)になるため、React等のUIフレームワークではパフォーマンス向上に寄与する。ボトルネックが確認された場合のみ、局所的に可変データを使う。

### Q2: データベースとの連携でイミュータビリティは維持できるか？

**A**: アプリケーション層でイミュータブルに扱い、永続化層（Repository/DAO）で変換するのが一般的。イベントソーシングやCQRSパターンを採用すれば、データベース層でも不変性を活かせる。ORMの遅延ロードとの相性は要注意。

### Q3: チームにイミュータビリティを導入するにはどうすればよいか？

**A**: (1) まず値オブジェクト（Money、Date等）から始める、(2) 新規コードに `readonly`/`final`/`frozen` を適用、(3) Lintルールで可変操作を警告（`no-param-reassign` 等）、(4) コードレビューで不変パターンを推奨。段階的に広げることで抵抗なく導入できる。

---

## 8. まとめ

| カテゴリ | ポイント |
|---------|---------|
| 原則 | デフォルトを不変に、可変は明示的に |
| スレッド安全 | 不変データはロック不要で並行処理が安全 |
| 予測可能性 | 値が変わらない → デバッグ・テストが容易 |
| 変更検知 | 参照比較O(1)でUI更新の効率化 |
| パフォーマンス | 構造共有/Immerで大規模データも効率的 |
| 言語選択 | Rustはデフォルト不変、他言語はライブラリ/規約で対応 |
| 導入戦略 | 値オブジェクトから段階的に、Lint支援で定着 |

---

## 次に読むべきガイド

- [01-composition-over-inheritance.md](./01-composition-over-inheritance.md) — 継承より合成の原則
- [02-functional-principles.md](./02-functional-principles.md) — 関数型プログラミングの原則
- 並行プログラミング — スレッド安全性の詳細

---

## 参考文献

1. Joshua Bloch, "Effective Java" 第3版 — Item 17: Minimize mutability
2. Michael Feathers, "Working Effectively with Legacy Code" — Immutability as a tool for safety
3. Immer.js ドキュメント — https://immerjs.github.io/immer/
4. Rust Book, "Understanding Ownership" — https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html
5. Eric Evans, "Domain-Driven Design" — Value Objects
