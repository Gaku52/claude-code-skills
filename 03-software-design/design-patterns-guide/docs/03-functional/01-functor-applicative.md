# ファンクタとアプリカティブ

> map と ap の抽象化を理解し、コンテキスト内の値に対する関数適用と合成を型安全に実現する

## この章で学ぶこと

1. **ファンクタの本質と法則** — map によるコンテキスト内の値の変換、ファンクタ則（恒等則・合成則）の意味と検証方法
2. **アプリカティブの理論と実践** — 複数のコンテキスト付き値に対する関数適用、エラー蓄積型バリデーション、並列計算
3. **型クラス階層の全体像** — Functor < Applicative < Monad の関係性、各レベルの能力と限界、実務での使い分け判断基準
4. **圏論との接点** — プログラミングにおける圏論的概念の直感的理解と、なぜこの抽象化が有用なのか

---

## 前提知識

このガイドを読む前に、以下の知識を持っていることを推奨します。

| 前提知識 | 参照先 |
|---|---|
| TypeScript/Rust の基本的な型システム | [02-programming カテゴリ](../../../../02-programming/) |
| ジェネリクスとインターフェース | [02-programming カテゴリ](../../../../02-programming/) |
| 高階関数（map, filter, reduce） | [関数型パターン](./02-fp-patterns.md) |
| モナドの基礎（flatMap/bind） | [モナド](./00-monad.md) |
| クリーンコードの原則 | [clean-code-principles](../../clean-code-principles/) |

---

## 1. ファンクタの本質

### 1.1 ファンクタとは何か — WHY から理解する

プログラミングにおいて、値は様々な「コンテキスト（文脈）」に包まれて存在します。

```
コンテキスト（文脈）の例
===========================

値が「存在しないかもしれない」 → Maybe / Option
値が「エラーかもしれない」     → Result / Either
値が「複数あるかもしれない」   → Array / List
値が「未来に届くかもしれない」 → Promise / Future
値が「副作用を伴うかもしれない」→ IO
値が「環境に依存するかもしれない」→ Reader

問題: これらのコンテキスト内の値に対して
      同じ変換ロジックを適用したい
      → ファンクタが解決する
```

**WHY**: なぜファンクタが必要なのか？

素朴なアプローチでは、コンテキストごとに別々の変換コードを書く必要があります。

```typescript
// コンテキストなし
const doubled = value * 2;

// Maybe コンテキスト — null チェックが必要
if (maybeValue !== null) {
  const doubled = maybeValue * 2;
}

// Array コンテキスト — ループが必要
const doubled = [];
for (const v of array) {
  doubled.push(v * 2);
}

// Promise コンテキスト — コールバックが必要
promise.then(value => value * 2);
```

これらはすべて「中の値に関数を適用する」という同じパターンです。ファンクタはこのパターンを `map` という統一的なインターフェースで抽象化します。

```
ファンクタ = map を持つ型
===========================

通常の関数適用:
  f : A -> B
  f(a)  -->  b

ファンクタでの関数適用:
  F[A].map(f)  -->  F[B]

  Maybe[3].map(x => x * 2)  -->  Maybe[6]
  [1,2,3].map(x => x * 2)   -->  [2,4,6]
  Promise[data].then(parse)  -->  Promise[parsed]

つまり:
  「コンテキストを維持したまま中身だけを変換する」
  これがファンクタの本質
```

### 1.2 ファンクタ則 — なぜ法則が重要なのか

ファンクタと名乗るには、`map` メソッドが2つの法則を満たす必要があります。法則を満たさない `map` は予測不可能な動作を引き起こし、リファクタリングの安全性を損ないます。

```
ファンクタ則:
  1. 恒等則 (Identity Law):
     fa.map(id) === fa
     「何もしない関数で map しても変化しない」

  2. 合成則 (Composition Law):
     fa.map(f).map(g) === fa.map(x => g(f(x)))
     「2回 map するのと、合成した関数で1回 map するのは同じ」

なぜ法則が重要か:
  - 恒等則: リファクタリング時に map(id) を安全に削除できる
  - 合成則: パフォーマンス最適化で map の連鎖を1回にまとめられる
  - 両方: コードの振る舞いを予測可能にする（等式推論）
```

### 1.3 ファンクタの図解

```
┌─────────────────────────────────────────────────────┐
│  ファンクタの動作イメージ                            │
│                                                     │
│  通常の関数:                                         │
│    f: A → B                                         │
│    3  ──f(×2)──▶  6                                 │
│                                                     │
│  ファンクタの map:                                   │
│    map(f): F[A] → F[B]                              │
│                                                     │
│    ┌─────┐              ┌─────┐                     │
│    │Maybe│              │Maybe│                     │
│    │  3  │──map(×2)──▶  │  6  │                     │
│    └─────┘              └─────┘                     │
│                                                     │
│    ┌─────────┐              ┌───────────┐           │
│    │ Array   │              │  Array    │           │
│    │ [1,2,3] │──map(×2)──▶  │ [2,4,6]  │           │
│    └─────────┘              └───────────┘           │
│                                                     │
│    ┌─────────┐              ┌───────────┐           │
│    │ Nothing │──map(×2)──▶  │  Nothing  │           │
│    └─────────┘              └───────────┘           │
│    (コンテキストが維持される)                         │
│                                                     │
│  ポイント:                                           │
│  - 箱（コンテキスト）の形は変わらない                │
│  - 中身だけが変換される                              │
│  - Nothing の場合は何もしない（安全にスキップ）      │
└─────────────────────────────────────────────────────┘
```

```
┌─────────────────────────────────────────────────────┐
│  ファンクタ則の図解                                   │
│                                                     │
│  恒等則: map(id) = id                                │
│                                                     │
│  ┌──────┐  map(id)  ┌──────┐                        │
│  │ F[3] │──────────▶│ F[3] │  何も変わらない         │
│  └──────┘           └──────┘                        │
│                                                     │
│  合成則: map(g).map(f) = map(g∘f)                    │
│                                                     │
│  方法1（2回map）:                                    │
│  ┌──────┐  map(f)  ┌──────┐  map(g)  ┌──────┐      │
│  │ F[3] │────────▶│ F[6] │────────▶│ F[7] │      │
│  └──────┘          └──────┘          └──────┘      │
│                                                     │
│  方法2（合成して1回map）:                             │
│  ┌──────┐  map(g∘f) ┌──────┐                        │
│  │ F[3] │──────────▶│ F[7] │  同じ結果               │
│  └──────┘           └──────┘                        │
│                                                     │
│  → パフォーマンス最適化に利用可能                     │
└─────────────────────────────────────────────────────┘
```

### コード例 1: TypeScript での完全な Maybe ファンクタ実装

```typescript
// === Maybe ファンクタの完全実装 ===

class Maybe<T> {
  private constructor(private readonly value: T | null | undefined) {}

  static of<T>(value: T): Maybe<T> {
    return new Maybe(value);
  }

  static nothing<T>(): Maybe<T> {
    return new Maybe<T>(null);
  }

  static fromNullable<T>(value: T | null | undefined): Maybe<T> {
    return value == null ? Maybe.nothing<T>() : Maybe.of(value);
  }

  isNothing(): boolean {
    return this.value == null;
  }

  isJust(): boolean {
    return this.value != null;
  }

  // ファンクタの核心: map
  map<U>(fn: (value: T) => U): Maybe<U> {
    if (this.value == null) return Maybe.nothing<U>();
    return Maybe.of(fn(this.value));
  }

  getOrElse(defaultValue: T): T {
    return this.value == null ? defaultValue : this.value;
  }

  get(): T {
    if (this.value == null) throw new Error("Cannot get value of Nothing");
    return this.value;
  }

  toString(): string {
    return this.isNothing() ? "Nothing" : `Just(${this.value})`;
  }
}

// --- 使用例 ---

// Array はファンクタ
const nums = [1, 2, 3];
const doubled = nums.map(x => x * 2);       // [2, 4, 6]
const strings = nums.map(x => x.toString()); // ["1", "2", "3"]

// Promise はファンクタ
const data = fetch("/api/users")
  .then(res => res.json())    // map
  .then(users => users[0]);   // map

// Maybe はファンクタ
Maybe.of(5)
  .map(x => x * 2)    // Maybe(10)
  .map(x => x + 1);   // Maybe(11)

Maybe.nothing<number>()
  .map(x => x * 2)    // Nothing (スキップ)
  .map(x => x + 1);   // Nothing (スキップ)

// --- ファンクタ則の検証 ---

// 恒等関数
const id = <T>(x: T): T => x;

// 1. 恒等則: fa.map(id) === fa
const fa = Maybe.of(42);
const result1 = fa.map(id);
console.log(fa.toString());       // Just(42)
console.log(result1.toString());  // Just(42) ← 同じ

const nothing = Maybe.nothing<number>();
const result2 = nothing.map(id);
console.log(nothing.toString());  // Nothing
console.log(result2.toString()); // Nothing ← 同じ

// 2. 合成則: fa.map(f).map(g) === fa.map(x => g(f(x)))
const f = (x: number) => x * 2;
const g = (x: number) => x + 1;

const left  = fa.map(f).map(g);
const right = fa.map(x => g(f(x)));
console.log(left.toString());  // Just(85)
console.log(right.toString()); // Just(85) ← 同じ
```

### コード例 2: Rust での Option/Result ファンクタ

```rust
// Rust では Option と Result が標準でファンクタ（map を持つ型）

fn main() {
    // === Option はファンクタ ===
    let x: Option<i32> = Some(5);
    let y = x.map(|n| n * 2);        // Some(10)
    let z = x.map(|n| n.to_string()); // Some("5")

    let none: Option<i32> = None;
    let w = none.map(|n| n * 2);      // None — 安全にスキップ

    println!("y = {:?}", y);  // y = Some(10)
    println!("z = {:?}", z);  // z = Some("5")
    println!("w = {:?}", w);  // w = None

    // === Result はファンクタ ===
    let ok: Result<i32, String> = Ok(42);
    let mapped = ok.map(|n| n * 2);   // Ok(84)

    let err: Result<i32, String> = Err("failed".to_string());
    let err_mapped = err.map(|n| n * 2);  // Err("failed") — エラーは保持

    println!("mapped = {:?}", mapped);       // mapped = Ok(84)
    println!("err_mapped = {:?}", err_mapped); // err_mapped = Err("failed")

    // === ファンクタ則の検証 ===
    // 恒等則
    let id_fn = |x: i32| x;
    assert_eq!(Some(5).map(id_fn), Some(5));
    assert_eq!(None::<i32>.map(id_fn), None);

    // 合成則
    let f = |x: i32| x * 2;
    let g = |x: i32| x + 1;
    assert_eq!(Some(5).map(f).map(g), Some(5).map(|x| g(f(x))));
    // Some(11) == Some(11)

    // === 実践的なチェーン ===
    let config = get_config_value("database.port")
        .map(|s| s.trim().to_string())
        .map(|s| s.parse::<u16>())
        .and_then(|r| r.ok());
    // Option<u16> — 安全な型変換のチェーン

    println!("config = {:?}", config);
}

fn get_config_value(key: &str) -> Option<String> {
    match key {
        "database.port" => Some("5432 ".to_string()),
        _ => None,
    }
}
```

### コード例 3: Haskell でのファンクタ型クラス

```haskell
-- Haskell では Functor は型クラスとして定義される
-- これがファンクタの理論的な原点

class Functor f where
    fmap :: (a -> b) -> f a -> f b

-- Maybe のファンクタインスタンス
instance Functor Maybe where
    fmap _ Nothing  = Nothing
    fmap f (Just a) = Just (f a)

-- リストのファンクタインスタンス
instance Functor [] where
    fmap = map

-- 使用例
example1 = fmap (*2) (Just 5)      -- Just 10
example2 = fmap (*2) Nothing       -- Nothing
example3 = fmap (*2) [1, 2, 3]     -- [2, 4, 6]

-- <$> は fmap の中置記法
example4 = (*2) <$> Just 5         -- Just 10
example5 = show <$> [1, 2, 3]      -- ["1", "2", "3"]

-- ファンクタ則の検証
-- 恒等則: fmap id x == x
prop_identity :: (Functor f, Eq (f a)) => f a -> Bool
prop_identity x = fmap id x == x

-- 合成則: fmap (g . f) x == (fmap g . fmap f) x
prop_composition :: (Functor f, Eq (f c)) =>
                    (b -> c) -> (a -> b) -> f a -> Bool
prop_composition g f x = fmap (g . f) x == (fmap g . fmap f) x
```

### 1.4 身近なファンクタの例

私たちが日常的に使っているファンクタを整理します。

```typescript
// 1. Array — 最も身近なファンクタ
const numbers = [1, 2, 3, 4, 5];
const doubled = numbers.map(x => x * 2); // [2, 4, 6, 8, 10]

// 2. Promise — 非同期のファンクタ
// .then() が map に相当（正確にはモナドの bind でもある）
const userPromise = fetch("/api/user")
  .then(res => res.json())       // map
  .then(data => data.name);      // map

// 3. DOM NodeList を Array に変換して map
const elements = Array.from(document.querySelectorAll(".item"));
const texts = elements.map(el => el.textContent);

// 4. Map オブジェクト — entries を通じてファンクタ的に使える
const prices = new Map([["apple", 100], ["banana", 200]]);
const discounted = new Map(
  Array.from(prices.entries()).map(([k, v]) => [k, v * 0.9])
);

// 5. TypeScript の Record 型を map する汎用関数
function mapRecord<K extends string, A, B>(
  record: Record<K, A>,
  fn: (a: A) => B
): Record<K, B> {
  const result = {} as Record<K, B>;
  for (const key in record) {
    result[key] = fn(record[key]);
  }
  return result;
}

const inventory = { apple: 10, banana: 20, cherry: 5 };
const doubled2 = mapRecord(inventory, x => x * 2);
// { apple: 20, banana: 40, cherry: 10 }
```

### 1.5 ファンクタではないものとの比較

ファンクタ則を満たさない、あるいは `map` の意味論が正しくない例を見ることで、ファンクタの理解を深めます。

```typescript
// === Set は（厳密には）ファンクタではない ===
// 理由: map で重複が除去される可能性があり、構造が保存されない

const s = new Set([1, 2, 3]);
// Set には map がないため Array に変換
const mapped = new Set(Array.from(s).map(x => x % 2));
// Set {1, 0} — 要素数が3→2に変わった!
// ファンクタは構造を保存するべきだが、Set は構造（要素数）を変えうる

// === EventEmitter/Observable の subscribe は map ではない ===
// 副作用を持つため、ファンクタ則を満たさない
// ただし、RxJS の Observable.pipe(map(...)) は
// 適切に実装されればファンクタ則を満たす

// === 反例: 恒等則を破る悪い map ===
class BadContainer<T> {
  constructor(public value: T, public count: number = 0) {}

  map<U>(fn: (value: T) => U): BadContainer<U> {
    // count をインクリメント — これは副作用!
    return new BadContainer(fn(this.value), this.count + 1);
  }
}

const c = new BadContainer(5);
const c2 = c.map(x => x); // { value: 5, count: 1 }
// c と c2 は count が異なる → 恒等則違反!
```

---

## 2. アプリカティブの本質

### 2.1 アプリカティブが解決する問題

```
アプリカティブ = ap を持つファンクタ
=====================================

問題: map では引数が1つの関数しか適用できない

  add は 2引数: add(a, b) = a + b
  Maybe[3].map(add) → Maybe[(b) => 3 + b]
  ↑ 関数が Maybe の中に閉じ込められた!
  この Maybe[(b) => 3 + b] を Maybe[5] に適用したい
  → ファンクタの map だけでは不可能

解決: アプリカティブの ap (apply)
  F[A → B].ap(F[A])  -->  F[B]

  Maybe[add].ap(Maybe[3]).ap(Maybe[5])  -->  Maybe[8]

  コンテキストに閉じ込められた関数を
  コンテキストに閉じ込められた値に適用できる!

独立した値の組み合わせ:
  ファンクタ:      1つの値を変換
  アプリカティブ:  複数の独立した値を組み合わせ
  モナド:          前の結果に依存した次の計算
```

**WHY**: なぜアプリカティブが必要なのか？

ファンクタの `map` は1引数関数しか受け取れません。しかし実際のプログラミングでは、2つ以上の値を組み合わせて新しい値を作ることが頻繁にあります。

- ユーザー名 + メールアドレス + 年齢 → ユーザーオブジェクト
- 価格 + 数量 → 合計金額
- 複数の API 結果 → 統合されたレスポンス

これらの値がそれぞれコンテキスト（Maybe, Result, Promise）に包まれている場合、ファンクタだけでは対応できません。アプリカティブがこの問題を解決します。

### 2.2 アプリカティブの図解

```
┌──────────────────────────────────────────────────────────────┐
│  ファンクタ vs アプリカティブ vs モナド                        │
│                                                              │
│  ■ ファンクタ (map): 1つの値を変換                            │
│                                                              │
│    F[A] ──map(f)──▶ F[B]                                    │
│                                                              │
│    Maybe[3] ──map(×2)──▶ Maybe[6]                           │
│                                                              │
│  ■ アプリカティブ (ap): 複数の独立した値を組み合わせ           │
│                                                              │
│    F[A→B→C]                                                  │
│      │                                                       │
│      ├── ap(F[A]) ──▶ F[B→C]                                │
│      │                    │                                   │
│      │                    ├── ap(F[B]) ──▶ F[C]              │
│      │                                                       │
│    Maybe[add] ── ap(Maybe[3]) ── ap(Maybe[5]) ──▶ Maybe[8]  │
│                                                              │
│  ■ モナド (bind/flatMap): 依存する計算を連鎖                  │
│                                                              │
│    F[A] ──bind(A→F[B])──▶ F[B] ──bind(B→F[C])──▶ F[C]     │
│                                                              │
│    Maybe[userId]                                             │
│      │                                                       │
│      ├── bind(findUser) ──▶ Maybe[User]                     │
│      │                          │                             │
│      │                          ├── bind(getOrders)          │
│      │                          │     ──▶ Maybe[Orders]     │
│      │                                                       │
│    前の結果が次の計算に必要（依存関係あり）                    │
└──────────────────────────────────────────────────────────────┘
```

```
┌──────────────────────────────────────────────────────────────┐
│  アプリカティブ vs モナドのエラーハンドリング比較              │
│                                                              │
│  ■ モナド (Either/Result): 最初のエラーで停止                 │
│                                                              │
│    validate(name)  ──Err──▶ ここで停止                       │
│    validate(email) ──────▶ 実行されない                      │
│    validate(age)   ──────▶ 実行されない                      │
│    結果: Err("名前が不正")                                    │
│                                                              │
│  ■ アプリカティブ (Validation): 全エラーを蓄積                │
│                                                              │
│    validate(name)  ──Err1──┐                                 │
│    validate(email) ──Err2──┼──▶ 全エラーを結合               │
│    validate(age)   ──Err3──┘                                 │
│    結果: Err(["名前が不正", "メールが不正", "年齢が不正"])     │
│                                                              │
│  → フォームバリデーションでは                                 │
│    アプリカティブが圧倒的に便利                               │
└──────────────────────────────────────────────────────────────┘
```

### コード例 4: アプリカティブの完全実装（Maybe）

```typescript
// === Maybe のファンクタ + アプリカティブ + モナド 完全実装 ===

class Maybe<T> {
  private constructor(private readonly value: T | null | undefined) {}

  static of<T>(value: T): Maybe<T> {
    return new Maybe(value);
  }

  static nothing<T>(): Maybe<T> {
    return new Maybe<T>(null);
  }

  static fromNullable<T>(value: T | null | undefined): Maybe<T> {
    return value == null ? Maybe.nothing<T>() : Maybe.of(value);
  }

  isNothing(): boolean {
    return this.value == null;
  }

  // --- ファンクタ ---
  map<U>(fn: (value: T) => U): Maybe<U> {
    if (this.value == null) return Maybe.nothing<U>();
    return Maybe.of(fn(this.value));
  }

  // --- アプリカティブ ---
  // ap: Maybe に包まれた関数を Maybe に包まれた値に適用する
  ap<U>(maybeFn: Maybe<(value: T) => U>): Maybe<U> {
    if (this.value == null || maybeFn.isNothing()) return Maybe.nothing<U>();
    return Maybe.of(maybeFn.get()(this.value));
  }

  // --- モナド ---
  flatMap<U>(fn: (value: T) => Maybe<U>): Maybe<U> {
    if (this.value == null) return Maybe.nothing<U>();
    return fn(this.value);
  }

  // --- ユーティリティ ---
  getOrElse(defaultValue: T): T {
    return this.value == null ? defaultValue : this.value;
  }

  get(): T {
    if (this.value == null) throw new Error("Cannot get value of Nothing");
    return this.value;
  }

  toString(): string {
    return this.isNothing() ? "Nothing" : `Just(${this.value})`;
  }
}

// === liftA2, liftA3: 多引数関数をアプリカティブに持ち上げ ===

function liftA2<A, B, C>(
  fn: (a: A, b: B) => C,
  ma: Maybe<A>,
  mb: Maybe<B>
): Maybe<C> {
  // カリー化して ap で適用
  return mb.ap(ma.map(a => (b: B) => fn(a, b)));
}

function liftA3<A, B, C, D>(
  fn: (a: A, b: B, c: C) => D,
  ma: Maybe<A>,
  mb: Maybe<B>,
  mc: Maybe<C>
): Maybe<D> {
  return mc.ap(mb.ap(ma.map(a => (b: B) => (c: C) => fn(a, b, c))));
}

// === 使用例 ===

// 2つの Maybe 値を組み合わせ
const price = Maybe.of(100);
const quantity = Maybe.of(3);
const total = liftA2((p, q) => p * q, price, quantity);
console.log(total.toString()); // Just(300)

// 片方が Nothing なら結果も Nothing
const noPrice = Maybe.nothing<number>();
const noTotal = liftA2((p, q) => p * q, noPrice, quantity);
console.log(noTotal.toString()); // Nothing

// 3つの Maybe 値を組み合わせてユーザーオブジェクトを作成
interface User {
  name: string;
  email: string;
  age: number;
}

const createUser = (name: string, email: string, age: number): User => ({
  name,
  email,
  age,
});

const userName = Maybe.of("Taro");
const userEmail = Maybe.of("taro@example.com");
const userAge = Maybe.of(30);

const user = liftA3(createUser, userName, userEmail, userAge);
console.log(user.toString());
// Just({name: "Taro", email: "taro@example.com", age: 30})

// 1つでも Nothing なら全体が Nothing
const noEmail = Maybe.nothing<string>();
const noUser = liftA3(createUser, userName, noEmail, userAge);
console.log(noUser.toString()); // Nothing
```

### コード例 5: アプリカティブバリデーション（エラー蓄積）

```typescript
// === Validation: アプリカティブの最大の利点 ===
// モナド (flatMap) では最初のエラーで停止するが、
// アプリカティブではすべてのエラーを収集できる

type Validation<E, A> =
  | { tag: "Success"; value: A }
  | { tag: "Failure"; errors: E[] };

function success<E, A>(value: A): Validation<E, A> {
  return { tag: "Success", value };
}

function failure<E, A>(errors: E[]): Validation<E, A> {
  return { tag: "Failure", errors };
}

function failOne<E, A>(error: E): Validation<E, A> {
  return { tag: "Failure", errors: [error] };
}

// map (ファンクタ)
function mapV<E, A, B>(
  va: Validation<E, A>,
  fn: (a: A) => B
): Validation<E, B> {
  return va.tag === "Success" ? success(fn(va.value)) : va;
}

// ap (アプリカティブ) — エラーを蓄積する
function apV<E, A, B>(
  vf: Validation<E, (a: A) => B>,
  va: Validation<E, A>
): Validation<E, B> {
  if (vf.tag === "Failure" && va.tag === "Failure") {
    return failure([...vf.errors, ...va.errors]);  // 両方のエラーを蓄積!
  }
  if (vf.tag === "Failure") return failure(vf.errors);
  if (va.tag === "Failure") return failure(va.errors);
  return success(vf.value(va.value));
}

// liftA2V, liftA3V
function liftA2V<E, A, B, C>(
  fn: (a: A, b: B) => C,
  va: Validation<E, A>,
  vb: Validation<E, B>
): Validation<E, C> {
  return apV(mapV(va, (a: A) => (b: B) => fn(a, b)), vb);
}

function liftA3V<E, A, B, C, D>(
  fn: (a: A, b: B, c: C) => D,
  va: Validation<E, A>,
  vb: Validation<E, B>,
  vc: Validation<E, C>
): Validation<E, D> {
  return apV(apV(mapV(va, (a: A) => (b: B) => (c: C) => fn(a, b, c)), vb), vc);
}

// === バリデーション関数群 ===

interface ValidationError {
  field: string;
  message: string;
}

function validateName(name: string): Validation<ValidationError, string> {
  if (name.length === 0) {
    return failOne({ field: "name", message: "名前は必須です" });
  }
  if (name.length < 2) {
    return failOne({ field: "name", message: "名前は2文字以上で入力してください" });
  }
  if (name.length > 50) {
    return failOne({ field: "name", message: "名前は50文字以下で入力してください" });
  }
  return success(name);
}

function validateEmail(email: string): Validation<ValidationError, string> {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  if (!emailRegex.test(email)) {
    return failOne({ field: "email", message: "有効なメールアドレスを入力してください" });
  }
  return success(email);
}

function validateAge(age: number): Validation<ValidationError, number> {
  if (!Number.isInteger(age)) {
    return failOne({ field: "age", message: "年齢は整数で入力してください" });
  }
  if (age < 0 || age > 150) {
    return failOne({ field: "age", message: "年齢は0〜150の範囲で入力してください" });
  }
  if (age < 18) {
    return failOne({ field: "age", message: "18歳以上である必要があります" });
  }
  return success(age);
}

// === ユーザー登録フォームの検証 ===

interface UserRegistration {
  name: string;
  email: string;
  age: number;
}

function validateUserRegistration(input: {
  name: string;
  email: string;
  age: number;
}): Validation<ValidationError, UserRegistration> {
  const vName = validateName(input.name);
  const vEmail = validateEmail(input.email);
  const vAge = validateAge(input.age);

  return liftA3V(
    (name, email, age) => ({ name, email, age }),
    vName,
    vEmail,
    vAge
  );
}

// === 実行例 ===

// 全てのフィールドが不正 → エラーが全て蓄積される
const result1 = validateUserRegistration({
  name: "",
  email: "invalid",
  age: 10,
});
console.log(result1);
// {
//   tag: "Failure",
//   errors: [
//     { field: "name", message: "名前は必須です" },
//     { field: "email", message: "有効なメールアドレスを入力してください" },
//     { field: "age", message: "18歳以上である必要があります" }
//   ]
// }
// → モナドなら "名前は必須です" だけで停止していた!

// 全て有効な場合
const result2 = validateUserRegistration({
  name: "Taro",
  email: "taro@example.com",
  age: 25,
});
console.log(result2);
// { tag: "Success", value: { name: "Taro", email: "taro@example.com", age: 25 } }
```

### コード例 6: Promise.all はアプリカティブ

```typescript
// === Promise.all はアプリカティブの ap に相当する ===

// WHY: 独立した非同期処理を並列実行できる
// → 逐次実行に比べて大幅な高速化が可能

// --- アプリカティブ（Promise.all）: 独立した処理を並列実行 ---
async function getUserDashboard(userId: string) {
  // 3つの API 呼び出しは互いに独立
  const [user, orders, settings] = await Promise.all([
    fetchUser(userId),        // 200ms
    fetchOrders(userId),      // 300ms
    fetchSettings(userId),    // 150ms
  ]);
  // 合計: max(200, 300, 150) = 300ms（並列実行）

  return {
    userName: user.name,
    orderCount: orders.length,
    theme: settings.theme,
  };
}

// --- モナド（async/await逐次）: 依存する処理を直列実行 ---
async function getOrderDetails(userId: string) {
  const user = await fetchUser(userId);                   // 200ms
  const orders = await fetchOrders(user.id);              // 300ms（user に依存）
  const details = await fetchOrderDetails(orders[0].id);  // 100ms（orders に依存）
  // 合計: 200 + 300 + 100 = 600ms（逐次実行）

  return details;
}

// --- Promise.allSettled: 失敗しても全結果を取得 ---
async function getUserDataSafe(userId: string) {
  const results = await Promise.allSettled([
    fetchUser(userId),
    fetchOrders(userId),
    fetchSettings(userId),
  ]);

  return results.map((r, i) => {
    if (r.status === "fulfilled") {
      return { success: true, data: r.value };
    } else {
      return { success: false, error: r.reason, index: i };
    }
  });
}

// --- 実用的な例: API リクエストの並列バッチ処理 ---
async function batchFetch<T>(
  urls: string[],
  concurrency: number = 5
): Promise<T[]> {
  const results: T[] = [];
  for (let i = 0; i < urls.length; i += concurrency) {
    const batch = urls.slice(i, i + concurrency);
    const batchResults = await Promise.all(
      batch.map(url => fetch(url).then(r => r.json() as Promise<T>))
    );
    results.push(...batchResults);
  }
  return results;
}
```

### コード例 7: Rust でのアプリカティブ的操作

```rust
// Rust にはアプリカティブの直接的な構文はないが、
// 同等のパターンは実現可能

fn main() {
    // === Option の zip: アプリカティブ的な組み合わせ ===
    let x: Option<i32> = Some(3);
    let y: Option<i32> = Some(5);
    let sum = x.zip(y).map(|(a, b)| a + b);
    println!("sum = {:?}", sum); // Some(8)

    let z: Option<i32> = None;
    let no_sum = x.zip(z).map(|(a, b)| a + b);
    println!("no_sum = {:?}", no_sum); // None

    // === 実践: 設定の並列パース ===
    let config = parse_config("8080", "localhost", "mydb");
    println!("config = {:?}", config);
    // Some(ServerConfig { port: 8080, host: "localhost", db: "mydb" })

    // === エラー蓄積型バリデーション ===
    let result = validate_user_input("", "bad-email", -5);
    println!("validation = {:?}", result);
    // Err(["名前は必須です", "無効なメールアドレス", "年齢は0以上"])
}

#[derive(Debug)]
struct ServerConfig {
    port: u16,
    host: String,
    db: String,
}

fn parse_config(port_str: &str, host: &str, db: &str) -> Option<ServerConfig> {
    let port = port_str.parse::<u16>().ok()?;
    Some(ServerConfig {
        port,
        host: host.to_string(),
        db: db.to_string(),
    })
}

// エラー蓄積型バリデーション（アプリカティブ的）
fn validate_user_input(
    name: &str,
    email: &str,
    age: i32,
) -> Result<(String, String, i32), Vec<String>> {
    let mut errors = Vec::new();

    if name.is_empty() {
        errors.push("名前は必須です".to_string());
    }
    if !email.contains('@') {
        errors.push("無効なメールアドレス".to_string());
    }
    if age < 0 {
        errors.push("年齢は0以上".to_string());
    }

    if errors.is_empty() {
        Ok((name.to_string(), email.to_string(), age))
    } else {
        Err(errors)
    }
}

// === Iterator の zip: アプリカティブ的操作 ===
fn applicative_iterators() {
    let names = vec!["Alice", "Bob", "Charlie"];
    let ages = vec![30, 25, 35];
    let scores = vec![95, 87, 92];

    // 3つのイテレータを zip で組み合わせ
    let students: Vec<_> = names.iter()
        .zip(ages.iter())
        .zip(scores.iter())
        .map(|((name, age), score)| {
            format!("{}: age={}, score={}", name, age, score)
        })
        .collect();

    for s in &students {
        println!("{}", s);
    }
    // Alice: age=30, score=95
    // Bob: age=25, score=87
    // Charlie: age=35, score=92
}
```

---

## 3. 型クラス階層

### 3.1 Functor < Applicative < Monad の関係

```
┌─────────────────────────────────────────────────────────────┐
│  型クラス階層（圏論的な包含関係）                              │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Monad                                              │   │
│  │  bind/flatMap: F[A] → (A → F[B]) → F[B]            │   │
│  │  依存する計算の連鎖                                  │   │
│  │                                                     │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │  Applicative                                │   │   │
│  │  │  ap:   F[A → B] → F[A] → F[B]              │   │   │
│  │  │  pure: A → F[A]                             │   │   │
│  │  │  独立した値の組み合わせ                      │   │   │
│  │  │                                             │   │   │
│  │  │  ┌─────────────────────────────────────┐   │   │   │
│  │  │  │  Functor                            │   │   │   │
│  │  │  │  map/fmap: F[A] → (A → B) → F[B]   │   │   │   │
│  │  │  │  1つの値の変換                       │   │   │   │
│  │  │  └─────────────────────────────────────┘   │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  重要: すべてのモナドはアプリカティブであり、                  │
│        すべてのアプリカティブはファンクタである                 │
│                                                             │
│  しかし逆は成り立たない:                                      │
│  - Validation はアプリカティブだがモナドではない               │
│    (エラー蓄積にはアプリカティブが必要)                        │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 使い分け比較表

| 特性 | Functor (map) | Applicative (ap) | Monad (bind) |
|---|---|---|---|
| **1値の変換** | 可能 | 可能 | 可能 |
| **複数値の組み合わせ** | 不可 | 可能 | 可能 |
| **エラー蓄積** | 不可 | 可能 | 不可（最初で停止） |
| **依存する計算** | 不可 | 不可 | 可能 |
| **並列実行** | — | 可能 | 不可（逐次） |
| **計算の静的解析** | 可能 | 可能 | 不可（動的） |
| **日常での例** | Array.map | Promise.all | async/await |
| **Haskell** | fmap / <$> | <*> | >>= / do |
| **TypeScript** | .map() | Promise.all() | .then() / await |
| **Rust** | .map() | .zip() | .and_then() / ? |

### 3.3 アプリカティブとモナドの選択基準

```
┌─────────────────────────────────────────────────────────┐
│  選択フローチャート                                      │
│                                                         │
│  「計算 B は計算 A の結果に依存するか？」                 │
│        │                                                │
│        ├── YES → モナド（flatMap / bind / await）        │
│        │         例: fetchUser → fetchOrders(user.id)   │
│        │                                                │
│        └── NO  → アプリカティブ（ap / Promise.all）      │
│                  例: fetchUser + fetchProducts + fetchAds│
│                                                         │
│  さらに:                                                 │
│  「エラーを全て収集したいか？」                           │
│        │                                                │
│        ├── YES → Validation（アプリカティブ）             │
│        │         例: フォームバリデーション               │
│        │                                                │
│        └── NO  → Either/Result（モナド）                 │
│                  例: 最初のエラーで中断して早期リターン   │
│                                                         │
│  パフォーマンスの観点:                                    │
│  - アプリカティブは並列実行が可能                         │
│  - モナドは逐次実行（前の結果が次に必要なため）           │
│  - 可能な限りアプリカティブを選ぶと高速                   │
└─────────────────────────────────────────────────────────┘
```

### コード例 8: 各レベルの能力の違いを示す実践例

```typescript
// === ファンクタレベル: 1つの値の変換 ===

// ユーザーIDから表示名を取得
const displayName = Maybe.fromNullable(user)
  .map(u => u.firstName + " " + u.lastName)
  .map(name => name.trim())
  .map(name => name.toUpperCase());
// Maybe<string>: Just("TARO YAMADA") or Nothing

// === アプリカティブレベル: 独立した値の組み合わせ ===

// フォームの3つのフィールドを独立に検証して結合
const validatedForm = liftA3V(
  (name, email, age) => ({ name, email, age }),
  validateName(formData.name),
  validateEmail(formData.email),
  validateAge(formData.age)
);
// Validation<Error[], FormData>
// エラーは全フィールド分蓄積される

// === モナドレベル: 依存する計算の連鎖 ===

// ユーザー取得 → 権限チェック → データ取得
const result = Maybe.fromNullable(userId)
  .flatMap(id => findUser(id))           // Maybe<User>
  .flatMap(user => checkPermission(user)) // Maybe<Permission>
  .flatMap(perm => fetchData(perm));      // Maybe<Data>
// 各ステップが前のステップの結果に依存

// === 誤った選択の例 ===

// [NG] 独立した処理をモナドで書く（不必要に逐次実行）
const user2 = await fetchUser(userId);      // 200ms
const products = await fetchProducts();      // 300ms（user に依存しない!）
const ads = await fetchAds();                // 100ms（上記に依存しない!）
// 合計: 600ms

// [OK] アプリカティブで書く（並列実行）
const [user3, products2, ads2] = await Promise.all([
  fetchUser(userId),    // 200ms
  fetchProducts(),      // 300ms
  fetchAds(),           // 100ms
]);
// 合計: 300ms（2倍高速!）
```

### 3.4 日常での対応表

| 抽象化 | Array | Promise | Option/Maybe | Result/Either | IO |
|---|---|---|---|---|---|
| **Functor (map)** | `.map()` | `.then()` | `.map()` | `.map()` | `.map()` |
| **Applicative** | `zip`/スプレッド | `Promise.all()` | `liftA2` | `Validation` | `liftA2` |
| **Monad (bind)** | `.flatMap()` | `async/await` | `.flatMap()`/`?.` | `?` 演算子 | `do` 記法 |

---

## 4. 圏論との接点 — 直感的な理解

### 4.1 プログラマのための圏論

```
┌──────────────────────────────────────────────────────────────┐
│  圏論の概念とプログラミングの対応                              │
│                                                              │
│  圏論           プログラミング                                │
│  ──────         ─────────────                                │
│  圏(Category)   型の世界                                      │
│  対象(Object)   型（Int, String, User, ...）                  │
│  射(Morphism)   関数（A → B）                                │
│  合成(∘)         関数合成 (f ∘ g)(x) = f(g(x))               │
│  恒等射(id)     恒等関数 id(x) = x                           │
│                                                              │
│  ファンクタ     圏から圏への「構造を保つ写像」                 │
│                 F: C → D                                     │
│                 - 対象を対象に:  A → F[A]                     │
│                 - 射を射に:     (A→B) → (F[A]→F[B])          │
│                 これが map!                                   │
│                                                              │
│  自然変換       ファンクタ間の「構造を保つ変換」               │
│                 Maybe[A] → List[A]                           │
│                 例: maybeToList(Just 5) = [5]                 │
│                     maybeToList(Nothing) = []                 │
│                                                              │
│  モナド         自己ファンクタの圏のモノイド                   │
│                 (join: F[F[A]] → F[A] と pure: A → F[A])     │
│                 flatMap = join ∘ map                          │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 なぜ圏論が役立つのか

圏論を知ることの実用的な価値:

1. **API 設計**: ファンクタ則を満たす `map` を設計すれば、ユーザーが安心してリファクタリングできる
2. **パフォーマンス最適化**: 合成則により、`map` チェーンを1回の `map` にまとめられる
3. **ライブラリ理解**: fp-ts, cats, scalaz 等のライブラリの API が直感的に理解できる
4. **パターン発見**: 「この型に map を実装できるか？」と考えることで、新しい抽象化を発見できる

---

## 5. 高度なトピック

### 5.1 Traversable — ファンクタの中のアプリカティブ

```typescript
// Traversable: コンテキストの順序を入れ替える
// 問題: Array<Maybe<number>> があるが、Maybe<Array<number>> が欲しい
// つまり「1つでも Nothing があれば全体を Nothing にしたい」

// sequence: [F[A]] → F[[A]]
function sequence<A>(maybes: Maybe<A>[]): Maybe<A[]> {
  return maybes.reduce<Maybe<A[]>>(
    (acc, maybe) =>
      liftA2((arr, val) => [...arr, val], acc, maybe),
    Maybe.of([] as A[])
  );
}

// traverse: (A → F[B]) → [A] → F[[B]]
// map してから sequence する（より効率的）
function traverse<A, B>(
  fn: (a: A) => Maybe<B>,
  arr: A[]
): Maybe<B[]> {
  return arr.reduce<Maybe<B[]>>(
    (acc, item) =>
      liftA2((arr, val) => [...arr, val], acc, fn(item)),
    Maybe.of([] as B[])
  );
}

// 使用例
const ids = [1, 2, 3, 4, 5];

// 全ユーザーが見つかれば Just([users])、1人でも見つからなければ Nothing
const allUsers = traverse(id => findUserById(id), ids);
// Maybe<User[]>

// sequence の使用例
const maybeNumbers: Maybe<number>[] = [
  Maybe.of(1),
  Maybe.of(2),
  Maybe.of(3),
];
const seqResult = sequence(maybeNumbers);
// Just([1, 2, 3])

const withNothing: Maybe<number>[] = [
  Maybe.of(1),
  Maybe.nothing(),
  Maybe.of(3),
];
const seqResult2 = sequence(withNothing);
// Nothing — 1つでも Nothing なら全体が Nothing
```

### 5.2 Contravariant Functor — 反変ファンクタ

```typescript
// 通常のファンクタは「出力」側に map する（共変: Covariant）
// 反変ファンクタは「入力」側に map する（反変: Contravariant）

// 通常のファンクタ: F[A] → (A → B) → F[B]
// 反変ファンクタ:   F[A] → (B → A) → F[B]

// 例: 比較関数（Comparator）は反変ファンクタ
interface Comparator<A> {
  compare: (a1: A, a2: A) => number;
}

// contramap: 入力側を変換
function contramap<A, B>(
  comp: Comparator<A>,
  fn: (b: B) => A
): Comparator<B> {
  return {
    compare: (b1, b2) => comp.compare(fn(b1), fn(b2)),
  };
}

// 数値比較器
const numberComparator: Comparator<number> = {
  compare: (a, b) => a - b,
};

// 文字列の長さで比較（反変マッピング）
const byLength = contramap(numberComparator, (s: string) => s.length);
// Comparator<string>

// ユーザーを年齢で比較（反変マッピング）
interface Person {
  name: string;
  age: number;
}

const byAge = contramap(numberComparator, (p: Person) => p.age);
// Comparator<Person>

// 使用
const people: Person[] = [
  { name: "Alice", age: 30 },
  { name: "Bob", age: 25 },
  { name: "Charlie", age: 35 },
];

const sorted = [...people].sort(byAge.compare);
// [Bob(25), Alice(30), Charlie(35)]

// 反変ファンクタの法則:
// 1. 恒等則: contramap(id) = id
// 2. 合成則: contramap(f).contramap(g) = contramap(g . f)
//            (注意: 通常のファンクタとは合成の順序が逆!)
```

### 5.3 Free Applicative パターン

```typescript
// Free Applicative: アプリカティブな操作を
// データ構造として構築し、後から解釈する

// 宣言的な API リクエスト定義
type ApiRequest<A> =
  | { tag: "Pure"; value: A }
  | { tag: "Fetch"; url: string; parse: (data: unknown) => A }
  | { tag: "Ap"; fn: ApiRequest<(a: any) => A>; arg: ApiRequest<any> };

function pureReq<A>(value: A): ApiRequest<A> {
  return { tag: "Pure", value };
}

function fetchReq<A>(url: string, parse: (data: unknown) => A): ApiRequest<A> {
  return { tag: "Fetch", url, parse };
}

function apReq<A, B>(
  fn: ApiRequest<(a: A) => B>,
  arg: ApiRequest<A>
): ApiRequest<B> {
  return { tag: "Ap", fn, arg };
}

function mapReq<A, B>(req: ApiRequest<A>, fn: (a: A) => B): ApiRequest<B> {
  return apReq(pureReq(fn), req);
}

function liftA2Req<A, B, C>(
  fn: (a: A, b: B) => C,
  ra: ApiRequest<A>,
  rb: ApiRequest<B>
): ApiRequest<C> {
  return apReq(mapReq(ra, (a: A) => (b: B) => fn(a, b)), rb);
}

// リクエストの宣言（この時点ではまだ実行されない）
const userReq = fetchReq("/api/user/1", (d: any) => d as User);
const ordersReq = fetchReq("/api/orders?user=1", (d: any) => d as Order[]);

const dashboardReq = liftA2Req(
  (user, orders) => ({ user, orders }),
  userReq,
  ordersReq
);

// === インタプリタ1: 並列実行 ===
async function runParallel<A>(req: ApiRequest<A>): Promise<A> {
  if (req.tag === "Pure") return req.value;
  if (req.tag === "Fetch") {
    const res = await fetch(req.url);
    const data = await res.json();
    return req.parse(data);
  }
  // Ap: 関数と引数を並列実行
  const [fn, arg] = await Promise.all([
    runParallel(req.fn),
    runParallel(req.arg),
  ]);
  return fn(arg);
}

// === インタプリタ2: URL 収集（テストやログ用） ===
function collectUrls<A>(req: ApiRequest<A>): string[] {
  if (req.tag === "Pure") return [];
  if (req.tag === "Fetch") return [req.url];
  return [...collectUrls(req.fn), ...collectUrls(req.arg)];
}

// 使用例
const urls = collectUrls(dashboardReq);
// ["/api/user/1", "/api/orders?user=1"]

const dashboard = await runParallel(dashboardReq);
// { user: ..., orders: [...] }
```

### 5.4 React におけるアプリカティブパターン

```tsx
// === React Query (TanStack Query) での並列クエリ ===

function Dashboard({ userId }: { userId: string }) {
  // 3つのクエリを並列実行（アプリカティブ）
  const userQuery = useQuery({
    queryKey: ["user", userId],
    queryFn: () => fetchUser(userId),
  });
  const ordersQuery = useQuery({
    queryKey: ["orders", userId],
    queryFn: () => fetchOrders(userId),
  });
  const settingsQuery = useQuery({
    queryKey: ["settings", userId],
    queryFn: () => fetchSettings(userId),
  });

  // 全クエリのローディング状態を組み合わせ
  if (userQuery.isLoading || ordersQuery.isLoading || settingsQuery.isLoading) {
    return <Loading />;
  }

  // エラーの蓄積（アプリカティブ的）
  const errors = [userQuery.error, ordersQuery.error, settingsQuery.error]
    .filter(Boolean);
  if (errors.length > 0) {
    return <ErrorList errors={errors} />;
  }

  // 全データが揃った時のみレンダリング
  return (
    <DashboardView
      user={userQuery.data!}
      orders={ordersQuery.data!}
      settings={settingsQuery.data!}
    />
  );
}

// === フォームバリデーション ===
function useFormValidation<T extends Record<string, unknown>>(
  validators: Record<keyof T, (value: unknown) => Validation<string, unknown>>
) {
  const [errors, setErrors] = useState<Record<string, string[]>>({});

  const validate = (formData: Record<string, unknown>): boolean => {
    const allErrors: Record<string, string[]> = {};
    let hasError = false;

    for (const [field, validator] of Object.entries(validators)) {
      const result = (validator as Function)(formData[field]);
      if (result.tag === "Failure") {
        allErrors[field] = result.errors;
        hasError = true;
      }
    }

    setErrors(allErrors);
    return !hasError;
  };

  return { validate, errors };
}
```

### 5.5 Haskell での型クラス階層

```haskell
-- Haskell ではファンクタ・アプリカティブ・モナドは
-- 型クラスとして明示的に定義される

-- === ファンクタ ===
class Functor f where
    fmap :: (a -> b) -> f a -> f b

-- === アプリカティブ ===
class Functor f => Applicative f where
    pure  :: a -> f a
    (<*>) :: f (a -> b) -> f a -> f b

-- === モナド ===
class Applicative m => Monad m where
    (>>=) :: m a -> (a -> m b) -> m b

-- === 使用例 ===

-- ファンクタ
ex1 = fmap (+1) (Just 5)              -- Just 6
ex2 = fmap (+1) [1, 2, 3]             -- [2, 3, 4]

-- アプリカティブ
ex3 = pure (+) <*> Just 3 <*> Just 5  -- Just 8
ex4 = pure (+) <*> Nothing <*> Just 5 -- Nothing

-- リストのアプリカティブ（直積 = 全組み合わせ）
ex5 = pure (+) <*> [1, 2] <*> [10, 20]
-- [11, 21, 12, 22]

-- アプリカティブスタイルでのユーザー作成
data User = User String String Int deriving (Show)

mkUser :: String -> String -> Int -> User
mkUser = User

validUser = mkUser <$> validateName "Taro"
                   <*> validateEmail "taro@example.com"
                   <*> validateAge 25

-- モナド（do 記法）
ex6 = do
    user  <- findUser userId
    perms <- checkPermissions user
    fetchData perms
```

---

## アンチパターン

### 1. モナドを使うべき場面でアプリカティブを使う

```typescript
// [NG] 前の計算結果に依存する処理をアプリカティブで無理に書く
// 問題: order は user.id に依存しているのに並列実行しようとしている

// fetchOrders は userId ではなく user.id（取得結果に含まれるID）を必要とする
const [user, orders] = await Promise.all([
  fetchUser(userId),
  fetchOrders(userId),  // 本来は user.id が必要だが、userId で代用してしまっている
]);

// [OK] 依存関係がある場合はモナド（async/await）を使う
const user = await fetchUser(userId);
const orders = await fetchOrders(user.id);  // user.id に依存
const details = await fetchOrderDetails(orders[0].id);  // orders に依存

// 判断基準: 「後続の処理が前の処理の"結果"を引数にしているか？」
// YES → モナド（逐次）、NO → アプリカティブ（並列可能）
```

### 2. ファンクタ則を破る map の実装

```typescript
// [NG] map 内で副作用を行う
class BadMaybe<T> {
  private value: T | null;

  map<U>(fn: (value: T) => U): BadMaybe<U> {
    console.log("mapping!");  // 副作用!
    // 恒等則: badMaybe.map(id) で "mapping!" が出力される
    // → id を適用しただけなのに観測可能な副作用が発生 → 恒等則違反
    if (this.value == null) return BadMaybe.nothing();
    return BadMaybe.of(fn(this.value));
  }
}

// [OK] map は純粋な変換のみ、副作用は別メソッドで
class GoodMaybe<T> {
  private value: T | null;

  map<U>(fn: (value: T) => U): GoodMaybe<U> {
    if (this.value == null) return GoodMaybe.nothing();
    return GoodMaybe.of(fn(this.value));
  }

  // 副作用用の明示的なメソッド
  tap(fn: (value: T) => void): GoodMaybe<T> {
    if (this.value != null) fn(this.value);
    return this; // 元の値を返す
  }
}

// 使用例
GoodMaybe.of(42)
  .tap(x => console.log(`値: ${x}`))  // 副作用は tap で
  .map(x => x * 2);                    // map は純粋に
```

### 3. Validation を Either/Result と混同する

```typescript
// [NG] Either (モナド) でバリデーション → 最初のエラーで停止
function validateWithEither(input: FormData): Either<string, User> {
  const name = validateName(input.name);  // Err なら次へ進めない
  if (name.isLeft()) return name;         // "名前が短すぎます" だけが返る

  const email = validateEmail(input.email); // ここに到達しない可能性
  if (email.isLeft()) return email;

  const age = validateAge(input.age);       // ここにも到達しない可能性
  if (age.isLeft()) return age;

  return Right(createUser(name.get(), email.get(), age.get()));
}
// → ユーザーはエラーを1つずつしか修正できない（UX が悪い）

// [OK] Validation (アプリカティブ) → 全エラーを収集
function validateWithApplicative(input: FormData): Validation<string[], User> {
  return liftA3V(
    createUser,
    validateName(input.name),    // 全て独立に実行される
    validateEmail(input.email),  // 全て独立に実行される
    validateAge(input.age)       // 全て独立に実行される
  );
  // ["名前が短すぎます", "無効なメールアドレス", "年齢が不正"] が全て返る
  // → ユーザーは一度に全てのエラーを確認できる（UX が良い）
}
```

### 4. 過度な抽象化 — YAGNI

```typescript
// [NG] 小さなプロジェクトでフルスペックの型クラス階層を実装
// TypeScript の型システムでは高カインド型をうまく表現できない
interface Functor<F> {
  map<A, B>(fa: F, fn: (a: A) => B): F;  // F の型パラメータが失われている
}
interface Applicative<F> extends Functor<F> {
  pure<A>(a: A): F;
  ap<A, B>(ff: F, fa: F): F;
}
// → 複雑なだけで実用的なメリットが薄い

// [OK] 必要な箇所だけシンプルに実装
// Maybe.map, Maybe.flatMap を直接実装し、
// 型クラスの概念を「設計指針」として頭の中で活用する

// 本格的に型クラス階層が必要なら fp-ts ライブラリを使う
import * as O from "fp-ts/Option";
import { pipe } from "fp-ts/function";

const result = pipe(
  O.some(5),
  O.map(x => x * 2),
  O.getOrElse(() => 0)
);
```

---

## 実践演習

### 演習1（基礎）: Maybe ファンクタの実装とファンクタ則の検証

**課題**: 以下の `Maybe` クラスの `map` を実装し、ファンクタ則（恒等則・合成則）をテストコードで検証してください。

```typescript
class Maybe<T> {
  private constructor(private readonly value: T | null) {}

  static of<T>(value: T): Maybe<T> {
    return new Maybe(value);
  }

  static nothing<T>(): Maybe<T> {
    return new Maybe<T>(null);
  }

  isNothing(): boolean {
    return this.value === null;
  }

  // TODO: map を実装してください
  map<U>(fn: (value: T) => U): Maybe<U> {
    // ここを実装
  }

  getOrElse(defaultValue: T): T {
    return this.value === null ? defaultValue : this.value;
  }

  equals(other: Maybe<T>): boolean {
    if (this.isNothing() && other.isNothing()) return true;
    if (this.isNothing() || other.isNothing()) return false;
    return this.value === other.getOrElse(null as any);
  }
}

// TODO: 以下のテストが全て PASS するようにしてください
const id = <T>(x: T): T => x;
const f = (x: number) => x * 2;
const g = (x: number) => x + 1;

// テスト1: 恒等則（Just）
console.assert(Maybe.of(42).map(id).equals(Maybe.of(42)));
// テスト2: 恒等則（Nothing）
console.assert(Maybe.nothing<number>().map(id).equals(Maybe.nothing<number>()));
// テスト3: 合成則（Just）
console.assert(Maybe.of(5).map(f).map(g).equals(Maybe.of(5).map(x => g(f(x)))));
// テスト4: 合成則（Nothing）
console.assert(Maybe.nothing<number>().map(f).map(g).equals(Maybe.nothing<number>().map(x => g(f(x)))));
```

**期待される出力**:

```
恒等則 (Just): PASS — Just(42).map(id) === Just(42)
恒等則 (Nothing): PASS — Nothing.map(id) === Nothing
合成則 (Just): PASS — Just(5).map(f).map(g) === Just(5).map(g∘f)
合成則 (Nothing): PASS — Nothing.map(f).map(g) === Nothing.map(g∘f)
全テスト通過!
```

### 演習2（応用）: アプリカティブバリデーションの実装

**課題**: 以下の要件を満たすアプリカティブバリデーションシステムを構築してください。

要件:
1. `Validation<E, A>` 型を実装する（Success/Failure）
2. `map`, `ap` を実装する（`ap` はエラーを蓄積する）
3. 以下のバリデーション関数を実装する:
   - `validateUsername`: 3文字以上20文字以下、英数字のみ
   - `validatePassword`: 8文字以上、大文字・小文字・数字を各1つ以上含む
   - `validateConfirmPassword`: パスワードと一致すること
4. `liftA3V` を使って全バリデーションを組み合わせる
5. 全フィールドが不正な場合にすべてのエラーが蓄積されることを検証する

```typescript
// TODO: Validation 型, map, ap, liftA2V, liftA3V を実装

// TODO: バリデーション関数を実装

// テストケース
const result1 = validateRegistration({
  username: "ab",           // 短すぎる
  password: "weak",         // 条件不足
  confirmPassword: "wrong", // 不一致
});
// 期待: Failure(["ユーザー名は3文字以上...", "パスワードは8文字以上...", ...])

const result2 = validateRegistration({
  username: "validuser",
  password: "Str0ngPass",
  confirmPassword: "Str0ngPass",
});
// 期待: Success({ username: "validuser", password: "Str0ngPass" })
```

**期待される出力**:

```
テスト1 (全フィールド不正):
  Failure:
    - ユーザー名は3文字以上20文字以下で入力してください
    - パスワードは8文字以上で入力してください
    - パスワードに大文字を含めてください
    - パスワードに数字を含めてください
    - パスワードが一致しません

テスト2 (全フィールド有効):
  Success: { username: "validuser", password: "Str0ngPass" }
```

### 演習3（発展）: Free Applicative による宣言的 API クライアント

**課題**: Free Applicative パターンを使って、宣言的 API クライアントを実装してください。

要件:
1. `ApiRequest<A>` 型を定義する（Pure, Fetch, Ap の3つのケース）
2. `map`, `ap`, `liftA2` を実装する
3. 以下の2つのインタプリタを実装する:
   - `runParallel`: リクエストを並列実行する（Promise.all を使用）
   - `collectUrls`: 実行せずに全 URL を収集する
4. モックサーバーを使ってテストする

```typescript
// TODO: ApiRequest<A> 型を定義

// TODO: map, ap, liftA2 を実装

// TODO: runParallel, collectUrls インタプリタを実装

// テストケース
const dashboardReq = liftA2Req(
  (user, orders) => ({ user, orders }),
  fetchReq<User>("/api/user/1", data => data as User),
  fetchReq<Order[]>("/api/orders?user=1", data => data as Order[]),
);

// URL収集（実行せずに解析）
const urls = collectUrls(dashboardReq);
console.log(urls); // ["/api/user/1", "/api/orders?user=1"]

// 並列実行
const result = await runParallel(dashboardReq);
console.log(result); // { user: {...}, orders: [...] }
```

**期待される出力**:

```
収集されたURL:
  /api/user/1
  /api/orders?user=1

並列実行結果:
  リクエスト: GET /api/user/1 ... 200 OK (120ms)
  リクエスト: GET /api/orders?user=1 ... 200 OK (85ms)
  結合結果: { user: { id: 1, name: "Taro" }, orders: [{ id: 101, ... }] }
```

---

## FAQ

### Q1: アプリカティブはいつ使うべきですか？

**A**: 「複数の独立した計算の結果を組み合わせたい」場合に使います。典型的な3つのケース:

1. **フォームバリデーション**: 全フィールドのエラーを一度に表示したい場合。モナドでは最初のエラーで停止するため、全エラーを収集するにはアプリカティブ（Validation型）が必要
2. **並列 API 呼び出し**: `Promise.all` は独立した非同期処理を並列実行するアプリカティブの代表例
3. **パーサーコンビネータ**: 独立したフィールドのパースを組み合わせる場合

### Q2: map と flatMap の違いを簡潔に説明すると？

**A**: `map` は「箱の中身を変換して箱に戻す」、`flatMap` は「箱の中身を変換し、結果が二重の箱になったら一重にする」です。

```typescript
// map: (A → B) を F[A] に適用 → F[B]
[1, 2].map(x => [x, x])     // [[1,1], [2,2]] — 二重配列

// flatMap: (A → F[B]) を F[A] に適用 → F[B]（flat + map）
[1, 2].flatMap(x => [x, x]) // [1, 1, 2, 2]   — 平坦化される

// Maybe の場合
Maybe.of(5).map(x => Maybe.of(x * 2))     // Maybe(Maybe(10)) — 二重 Maybe
Maybe.of(5).flatMap(x => Maybe.of(x * 2)) // Maybe(10)         — 平坦化
```

### Q3: Functor/Applicative/Monad を意識してコードを書く必要がありますか？

**A**: 明示的に意識する必要はありません。`Array.map`、`Promise.all`、`async/await` を使う時点で既にこれらのパターンを活用しています。理論を知ることで以下のメリットがあります:

- **設計判断**: 「この処理は並列実行できるか？（→ アプリカティブ）」「前の結果に依存するか？（→ モナド）」という判断が的確になる
- **API 理解**: 新しいライブラリの `map`, `flatMap`, `ap` 等のメソッドの意味が直感的に分かる
- **バグ予防**: ファンクタ則を意識することで、`map` 内での副作用を避けるようになる

### Q4: TypeScript で高カインド型（HKT）は使えますか？

**A**: TypeScript の型システムでは高カインド型を直接サポートしていません。fp-ts ライブラリがブランド型を使ったエミュレーションを提供しています。小規模なプロジェクトでは、各型（Maybe, Either, Validation）に直接 map/flatMap を実装するのが実用的です。

### Q5: アプリカティブと並列処理の関係は？

**A**: アプリカティブの `ap` は「計算間に依存関係がない」ことを型レベルで保証します。依存関係がないということは、理論的に並列実行が可能です。`Promise.all` はまさにこの性質を利用しています。ただし、アプリカティブ = 必ず並列実行ではなく、「並列実行が可能」という情報を提供するだけで、実際に並列にするかはインタプリタの実装次第です。

### Q6: Validation はモナドにならないのはなぜですか？

**A**: モナドの `bind/flatMap` は「前の計算の結果を使って次の計算を決める」ため、エラーの場合は短絡評価（Short-circuit）するしかありません。一方、アプリカティブの `ap` は「両方の計算を独立に実行して結果を組み合わせる」ため、両方がエラーの場合にエラーを蓄積できます。この「エラー蓄積」と「モナドの短絡評価」は本質的に両立しません。

```haskell
-- モナドの bind は前の結果に依存する
-- エラーの場合、次の計算に渡す値がないため停止するしかない
bind (Failure errs) f = Failure errs  -- f を呼べない
bind (Success a)    f = f a

-- アプリカティブの ap は両方を独立に評価できる
ap (Failure e1) (Failure e2) = Failure (e1 ++ e2)  -- 両方のエラーを結合
```

---

## まとめ

| 項目 | 要点 |
|---|---|
| ファンクタ | `map` でコンテキスト内の値を変換。Array, Option, Promise, Result 等 |
| ファンクタ則 | 恒等則: `map(id) = id`、合成則: `map(f).map(g) = map(g . f)` |
| アプリカティブ | `ap` で独立した複数の値を組み合わせ。エラー蓄積・並列実行が可能 |
| Validation | アプリカティブの実践例。モナドと違い全エラーを収集できる |
| モナド | `bind/flatMap` で依存する計算を連鎖。前の結果に基づく次の計算 |
| 型クラス階層 | Functor < Applicative < Monad（包含関係） |
| 選択基準 | 独立 → Applicative（並列可能）、依存 → Monad（逐次実行） |
| Promise.all | アプリカティブの代表例。独立した非同期処理の並列実行 |
| Traversable | `sequence`/`traverse` でコンテキストの順序を入れ替え |
| 反変ファンクタ | 「入力」側に map する。Comparator, Predicate 等 |
| 圏論との関係 | ファンクタは「構造を保つ写像」。法則が安全なリファクタリングを保証 |
| 実践指針 | 最小限の抽象化を選ぶ（YAGNI）。TypeScript では fp-ts が実用的 |

---

## 次に読むべきガイド

- [モナド](./00-monad.md) — flatMap/bind の詳細と応用、do 記法
- [関数型パターン](./02-fp-patterns.md) — カリー化、パイプライン、レンズとの統合
- [クリーンコードの原則](../../clean-code-principles/docs/00-principles/) — 関数設計の基本原則
- [ビヘイビアパターン](../02-behavioral/) — OOP のパターンとの比較
- [アーキテクチャパターン](../04-architectural/) — 大規模設計での関数型アプローチ

---

## 参考文献

1. **Haskell Wiki**: [Typeclassopedia](https://wiki.haskell.org/Typeclassopedia) — 型クラス階層の包括的ガイド。Functor, Applicative, Monad の関係を圏論的に解説
2. **Giulio Canti**: [fp-ts](https://gcanti.github.io/fp-ts/) — TypeScript の関数型プログラミングライブラリ。HKT のエミュレーション手法が参考になる
3. **Bartosz Milewski**: [Category Theory for Programmers](https://bartoszmilewski.com/2014/10/28/category-theory-for-programmers-the-preface/) — プログラマ向けの圏論入門。ファンクタの数学的背景を理解できる
4. **Conor McBride, Ross Paterson**: [Applicative Programming with Effects](http://www.staff.city.ac.uk/~ross/papers/Applicative.html) — アプリカティブファンクタの原論文。理論的背景を深く知りたい場合
5. **Brian Lonsdorf**: [Professor Frisby's Mostly Adequate Guide](https://mostly-adequate.gitbook.io/mostly-adequate-guide/) — JavaScript での関数型プログラミングの実践入門
