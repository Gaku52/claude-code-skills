# ファンクタとアプリカティブ

> map と ap の抽象化を理解し、コンテキスト内の値に対する関数適用と合成を型安全に実現する

## この章で学ぶこと

1. **ファンクタ** — map によるコンテキスト内の値の変換と、ファンクタ則
2. **アプリカティブ** — 複数のコンテキスト付き値に対する関数適用と並列検証
3. **型クラス階層** — Functor < Applicative < Monad の関係と使い分け

---

## 1. ファンクタの本質

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

ファンクタ則:
  1. 恒等則:  fa.map(id) === fa
  2. 合成則:  fa.map(f).map(g) === fa.map(x => g(f(x)))
```

### コード例 1: 各言語のファンクタ

```typescript
// TypeScript: 共通のファンクタインターフェース
interface Functor<A> {
  map<B>(fn: (a: A) => B): Functor<B>;
}

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

Maybe.nothing()
  .map(x => x * 2)    // Nothing (スキップ)
  .map(x => x + 1);   // Nothing (スキップ)
```

```rust
// Rust: Option と Result はファンクタ
let x: Option<i32> = Some(5);
let y = x.map(|n| n * 2);        // Some(10)
let z = x.map(|n| n.to_string()); // Some("5")

let none: Option<i32> = None;
let w = none.map(|n| n * 2);     // None

// Result のファンクタ操作
let ok: Result<i32, String> = Ok(42);
let mapped = ok.map(|n| n * 2);   // Ok(84)

let err: Result<i32, String> = Err("failed".to_string());
let mapped = err.map(|n| n * 2);  // Err("failed")
```

---

## 2. アプリカティブの本質

```
アプリカティブ = ap を持つファンクタ
=====================================

問題: map では引数が1つの関数しか適用できない

  Maybe[3].map(add)  -->  Maybe[add(3)]  -->  ???
                         関数がコンテキスト内に閉じ込められた

アプリカティブの ap (apply):
  F[A -> B].ap(F[A])  -->  F[B]

  Maybe[add].ap(Maybe[3]).ap(Maybe[5])  -->  Maybe[8]

独立した値の組み合わせ:
  ファンクタ:      1つの値を変換
  アプリカティブ:  複数の独立した値を組み合わせ
  モナド:          前の結果に依存した次の計算
```

### コード例 2: アプリカティブの実装

```typescript
class Maybe<T> {
  // ... (前章のファンクタ実装)

  // ap: Maybe<(a: T) => U> を Maybe<T> に適用
  ap<U>(maybeFn: Maybe<(value: T) => U>): Maybe<U> {
    if (this.value == null || maybeFn.isNothing()) return Maybe.nothing();
    return Maybe.of(maybeFn.get()(this.value));
  }

  // liftA2: 2引数関数をアプリカティブに持ち上げ
  static liftA2<A, B, C>(
    fn: (a: A, b: B) => C,
    ma: Maybe<A>,
    mb: Maybe<B>
  ): Maybe<C> {
    return ma.flatMap(a => mb.map(b => fn(a, b)));
  }
}

// 使用例: 2つの Maybe 値を組み合わせ
const price = Maybe.of(100);
const quantity = Maybe.of(3);
const total = Maybe.liftA2((p, q) => p * q, price, quantity);
// Maybe(300)

const noPrice = Maybe.nothing<number>();
const noTotal = Maybe.liftA2((p, q) => p * q, noPrice, quantity);
// Nothing
```

### コード例 3: アプリカティブバリデーション

```typescript
// アプリカティブの最大の利点: 独立した検証の並列実行
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

function mapV<E, A, B>(
  va: Validation<E, A>,
  fn: (a: A) => B
): Validation<E, B> {
  return va.tag === "Success" ? success(fn(va.value)) : va;
}

// ap: エラーを蓄積する
function apV<E, A, B>(
  vf: Validation<E, (a: A) => B>,
  va: Validation<E, A>
): Validation<E, B> {
  if (vf.tag === "Failure" && va.tag === "Failure") {
    return failure([...vf.errors, ...va.errors]);  // エラー蓄積!
  }
  if (vf.tag === "Failure") return vf;
  if (va.tag === "Failure") return va;
  return success(vf.value(va.value));
}

// 使用例
function validateName(name: string): Validation<string, string> {
  return name.length >= 2 ? success(name) : failure(["Name too short"]);
}

function validateEmail(email: string): Validation<string, string> {
  return email.includes("@") ? success(email) : failure(["Invalid email"]);
}

function validateAge(age: number): Validation<string, number> {
  return age >= 18 ? success(age) : failure(["Must be 18+"]);
}

// 全エラーを一度に収集
type User = { name: string; email: string; age: number };

function validateUser(input: { name: string; email: string; age: number }): Validation<string, User> {
  const vName = validateName(input.name);
  const vEmail = validateEmail(input.email);
  const vAge = validateAge(input.age);

  const mkUser = (name: string) => (email: string) => (age: number): User =>
    ({ name, email, age });

  return apV(apV(mapV(vName, mkUser), vEmail), vAge);
}

// validateUser({ name: "", email: "bad", age: 10 })
// --> Failure(["Name too short", "Invalid email", "Must be 18+"])
// モナドなら "Name too short" だけで停止していた
```

---

## 3. 型クラス階層

```
Functor < Applicative < Monad
================================

Functor     :  map   : F[A] -> (A -> B) -> F[B]
Applicative :  ap    : F[A -> B] -> F[A] -> F[B]
              pure  : A -> F[A]
Monad       :  bind  : F[A] -> (A -> F[B]) -> F[B]

能力の強さ:
  Functor     < Applicative  < Monad
  1値を変換     独立した値を     依存する計算を
               組み合わせ       連鎖

Monad は Applicative の機能を含む（上位互換）
Applicative は Functor の機能を含む（上位互換）
```

### 使い分け比較表

| 特性 | Functor (map) | Applicative (ap) | Monad (bind) |
|---|---|---|---|
| **1値の変換** | 可能 | 可能 | 可能 |
| **複数値の組み合わせ** | 不可 | 可能 | 可能 |
| **エラー蓄積** | 不可 | 可能 | 不可（最初で停止） |
| **依存する計算** | 不可 | 不可 | 可能 |
| **並列実行** | - | 可能 | 不可（逐次） |
| **日常での例** | Array.map | Promise.all | async/await |

### コード例 4: Promise.all はアプリカティブ

```typescript
// Promise.all = アプリカティブの ap
// 独立した非同期処理を並列実行して結果を組み合わせ

// [アプリカティブ] 独立した処理を並列実行
const [user, orders, settings] = await Promise.all([
  fetchUser(userId),      // 独立
  fetchOrders(userId),    // 独立
  fetchSettings(userId),  // 独立
]);
// 3つが並列実行される

// [モナド] 依存する処理を逐次実行
const user = await fetchUser(userId);
const orders = await fetchOrders(user.id);    // user に依存
const details = await fetchOrderDetails(orders[0].id); // orders に依存
// 逐次実行される（依存関係があるため）
```

### コード例 5: Rust での Iterator はファンクタ

```rust
// Iterator チェーンはファンクタ + αの操作

let data = vec![1, 2, 3, 4, 5];

// map: ファンクタ
let doubled: Vec<i32> = data.iter().map(|x| x * 2).collect();

// filter + map: ファンクタの拡張
let result: Vec<String> = data.iter()
    .filter(|&&x| x > 2)        // 述語でフィルタ
    .map(|x| x.to_string())     // 変換
    .collect();

// flat_map: モナド的操作
let nested = vec![vec![1, 2], vec![3, 4]];
let flat: Vec<i32> = nested.into_iter()
    .flat_map(|v| v.into_iter())
    .collect();
// [1, 2, 3, 4]

// zip: アプリカティブ的操作（2つのイテレータを組み合わせ）
let names = vec!["Alice", "Bob"];
let ages = vec![30, 25];
let pairs: Vec<_> = names.iter().zip(ages.iter())
    .map(|(n, a)| format!("{}: {}", n, a))
    .collect();
// ["Alice: 30", "Bob: 25"]
```

---

## 日常での対応表

| 抽象化 | Array | Promise | Option/Maybe | Result/Either |
|---|---|---|---|---|
| **Functor (map)** | `.map()` | `.then()` | `.map()` | `.map()` |
| **Applicative** | `zip` / スプレッド | `Promise.all()` | `liftA2` | `Validation` |
| **Monad (bind)** | `.flatMap()` | `async/await` | `.flatMap()` / `?.` | `?` 演算子 |

---

## アンチパターン

### 1. モナドを使うべき場面でアプリカティブを使う

**問題**: 前の計算結果に依存する処理をアプリカティブで書こうとすると、型が合わず不自然なコードになる。

**対策**: 「前の結果に依存するか？」を判断基準にする。依存する場合はモナド（flatMap/bind）、独立している場合はアプリカティブ（ap/Promise.all）を使う。

### 2. ファンクタ則を破る map の実装

**問題**: `map` 内で副作用（ログ出力、状態変更）を行うと、ファンクタ則（恒等則・合成則）が破れ、予測不可能な動作になる。

**対策**: `map` は純粋な変換のみに使用する。副作用が必要な場合は `forEach` や専用のメソッドを使う。

---

## FAQ

### Q1: アプリカティブはいつ使うべきですか？

**A**: 「複数の独立した計算の結果を組み合わせたい」場合に使います。典型的な例はフォームバリデーション（全フィールドのエラーを一度に表示）と並列 API 呼び出し（`Promise.all`）です。

### Q2: map と flatMap の違いを簡潔に説明すると？

**A**: `map` は「箱の中身を変換」、`flatMap` は「箱の中身を変換し、二重の箱を一重にする」です。`[1,2].map(x => [x, x])` は `[[1,1],[2,2]]`、`[1,2].flatMap(x => [x, x])` は `[1,1,2,2]` になります。

### Q3: Functor/Applicative/Monad を意識してコードを書く必要がありますか？

**A**: 明示的に意識する必要はありません。`Array.map`、`Promise.all`、`async/await` を使う時点で既にこれらのパターンを活用しています。理論を知ることで「なぜこの API がこう設計されているか」を理解でき、新しいライブラリの API を直感的に把握できるようになります。

---

## まとめ

| 項目 | 要点 |
|---|---|
| ファンクタ | `map` でコンテキスト内の値を変換。Array, Option, Promise 等 |
| アプリカティブ | `ap` で独立した複数の値を組み合わせ。エラー蓄積・並列実行 |
| モナド | `bind` で依存する計算を連鎖。前の結果に基づく次の計算 |
| 選択基準 | 独立 -> Applicative、依存 -> Monad |
| Promise.all | アプリカティブの代表例。並列非同期処理 |
| Validation | アプリカティブの実践例。全エラーの収集 |

## 次に読むべきガイド

- [モナド](./00-monad.md) — flatMap/bind の詳細と応用
- [関数型パターン](./02-fp-patterns.md) — カリー化、パイプラインとの統合

## 参考文献

1. **Haskell Wiki**: [Typeclassopedia](https://wiki.haskell.org/Typeclassopedia) — 型クラス階層の包括的ガイド
2. **Giulio Canti**: [fp-ts](https://gcanti.github.io/fp-ts/) — TypeScript の関数型プログラミングライブラリ
3. **Bartosz Milewski**: [Functors](https://bartoszmilewski.com/2015/01/20/functors/) — 圏論の観点からのファンクタ解説
