# モナド

> Maybe/Either、IO、Promise などのモナドパターンを理解し、副作用を制御しつつ合成可能なデータフローを構築する

## この章で学ぶこと

1. **モナドの本質** — bind (flatMap) による計算の連鎖と、コンテキスト内での値の変換
2. **実用的なモナド** — Maybe/Option、Either/Result、Promise/Future の活用
3. **モナド合成** — モナドトランスフォーマー、do 記法、async/await との関係

---

## 1. モナドの基本概念

```
モナドの3つの法則
==================

型 M に対して:

1. unit (return, of): a -> M a
   値をモナドコンテキストに入れる

2. bind (flatMap, >>=): M a -> (a -> M b) -> M b
   コンテキスト内の値に関数を適用し、結果をフラットにする

3. モナド則:
   左単位元: unit(a).bind(f) === f(a)
   右単位元: m.bind(unit)    === m
   結合則:   m.bind(f).bind(g) === m.bind(x => f(x).bind(g))

図解:
  値 a  --[unit]--> M a  --[bind(f)]--> M b  --[bind(g)]--> M c

  map  : M a -> (a -> b)   -> M b    (値を変換)
  bind : M a -> (a -> M b) -> M b    (コンテキストを変換)
```

### コード例 1: Maybe/Option モナド

```typescript
// TypeScript での Maybe モナド実装
class Maybe<T> {
  private constructor(private value: T | null) {}

  static of<T>(value: T): Maybe<T> {
    return new Maybe(value);
  }

  static nothing<T>(): Maybe<T> {
    return new Maybe<T>(null);
  }

  static fromNullable<T>(value: T | null | undefined): Maybe<T> {
    return value == null ? Maybe.nothing() : Maybe.of(value);
  }

  map<U>(fn: (value: T) => U): Maybe<U> {
    return this.value == null ? Maybe.nothing() : Maybe.of(fn(this.value));
  }

  flatMap<U>(fn: (value: T) => Maybe<U>): Maybe<U> {
    return this.value == null ? Maybe.nothing() : fn(this.value);
  }

  getOrElse(defaultValue: T): T {
    return this.value ?? defaultValue;
  }

  filter(predicate: (value: T) => boolean): Maybe<T> {
    if (this.value == null) return Maybe.nothing();
    return predicate(this.value) ? this : Maybe.nothing();
  }
}

// 使用例: null チェックの連鎖を安全に
interface User {
  name: string;
  address?: { city?: string; zip?: string };
}

function getUserCity(user: User | null): string {
  return Maybe.fromNullable(user)
    .flatMap(u => Maybe.fromNullable(u.address))
    .flatMap(a => Maybe.fromNullable(a.city))
    .getOrElse("Unknown");
}

// [NG] null チェックのネスト地獄
function getUserCityBad(user: User | null): string {
  if (user != null) {
    if (user.address != null) {
      if (user.address.city != null) {
        return user.address.city;
      }
    }
  }
  return "Unknown";
}
```

---

## 2. Either/Result モナド

```
Either モナドの構造
====================

Either<L, R>
  |
  +-- Left(error)   : エラー値を保持、bind をスキップ
  |
  +-- Right(value)  : 正常値を保持、bind で処理を継続

  Right(5).map(x => x * 2)     --> Right(10)
  Left("err").map(x => x * 2)  --> Left("err")  (スキップ)

鉄道指向プログラミング (Railway Oriented):

  Input --> [validate] --> [process] --> [save] --> Output
               |              |           |
               v              v           v
             Left           Left        Left
           (error)        (error)     (error)
```

### コード例 2: Either/Result モナド

```typescript
type Either<L, R> = { tag: "Left"; value: L } | { tag: "Right"; value: R };

const Left = <L>(value: L): Either<L, never> => ({ tag: "Left", value });
const Right = <R>(value: R): Either<never, R> => ({ tag: "Right", value });

function map<L, R, U>(either: Either<L, R>, fn: (r: R) => U): Either<L, U> {
  return either.tag === "Right" ? Right(fn(either.value)) : either;
}

function flatMap<L, R, U>(
  either: Either<L, R>,
  fn: (r: R) => Either<L, U>
): Either<L, U> {
  return either.tag === "Right" ? fn(either.value) : either;
}

// 実践例: バリデーションパイプライン
type ValidationError = string;

function validateEmail(email: string): Either<ValidationError, string> {
  return email.includes("@") ? Right(email) : Left("Invalid email format");
}

function validateAge(age: number): Either<ValidationError, number> {
  return age >= 18 ? Right(age) : Left("Must be 18 or older");
}

function validateName(name: string): Either<ValidationError, string> {
  return name.length >= 2 ? Right(name) : Left("Name too short");
}

// パイプラインで連鎖
function validateUser(input: { name: string; email: string; age: number }) {
  const name = validateName(input.name);
  const email = flatMap(name, () => validateEmail(input.email));
  const age = flatMap(email, () => validateAge(input.age));
  return map(age, () => ({ ...input }));
}
```

### コード例 3: Rust の Result モナド

```rust
use std::num::ParseIntError;

#[derive(Debug)]
enum AppError {
    Parse(ParseIntError),
    Validation(String),
    NotFound(String),
}

impl From<ParseIntError> for AppError {
    fn from(e: ParseIntError) -> Self {
        AppError::Parse(e)
    }
}

// ? 演算子 = flatMap (bind) の糖衣構文
fn process_order(input: &str) -> Result<Order, AppError> {
    let order_id: u64 = input.parse()?;  // ParseIntError -> AppError

    let order = find_order(order_id)
        .ok_or_else(|| AppError::NotFound(format!("Order {}", order_id)))?;

    if order.total <= 0.0 {
        return Err(AppError::Validation("Invalid total".to_string()));
    }

    Ok(order)
}

// and_then = flatMap
fn pipeline(input: &str) -> Result<String, AppError> {
    input.parse::<u64>()
        .map_err(AppError::from)
        .and_then(find_order_result)
        .and_then(validate_order)
        .map(|order| format!("Processed: {}", order.id))
}
```

---

## 3. IO モナドと Promise

```
IO モナドの目的
================

純粋関数の世界     IO モナドの境界     副作用の世界
+-----------+     +------------+     +-----------+
| 計算ロジック |     | IO<A>      |     | ファイル   |
| (テスト容易)|     | (実行を遅延)|     | ネットワーク|
|           | --> | bind で合成 | --> | DB        |
+-----------+     +------------+     +-----------+

Promise/Future も IO モナドの一種:
  非同期副作用を値として扱い、then/await で合成
```

### コード例 4: Promise モナドとしての async/await

```typescript
// Promise = 非同期 IO モナド
// then = flatMap (bind)
// async/await = do 記法の糖衣構文

// [モナディックスタイル]
function fetchUserOrders(userId: string): Promise<OrderSummary> {
  return fetchUser(userId)
    .then(user => fetchOrders(user.id))           // flatMap
    .then(orders => orders.filter(o => o.active))  // map
    .then(orders => ({
      userId,
      totalOrders: orders.length,
      totalAmount: orders.reduce((sum, o) => sum + o.amount, 0)
    }));
}

// [async/await = do 記法]
async function fetchUserOrders(userId: string): Promise<OrderSummary> {
  const user = await fetchUser(userId);            // bind
  const orders = await fetchOrders(user.id);       // bind
  const activeOrders = orders.filter(o => o.active); // 純粋計算

  return {
    userId,
    totalOrders: activeOrders.length,
    totalAmount: activeOrders.reduce((sum, o) => sum + o.amount, 0)
  };
}
```

### コード例 5: モナド合成パターン

```typescript
// 複数のモナドを組み合わせる: Maybe + Either + Promise

// Result 型の定義
type Result<T, E = Error> = { ok: true; value: T } | { ok: false; error: E };

// 安全な非同期操作
async function safeAsync<T>(
  fn: () => Promise<T>
): Promise<Result<T>> {
  try {
    return { ok: true, value: await fn() };
  } catch (error) {
    return { ok: false, error: error as Error };
  }
}

// パイプライン関数
function pipe<T>(...fns: Array<(arg: any) => any>): (arg: T) => any {
  return (arg: T) => fns.reduce((acc, fn) => fn(acc), arg);
}

// 実践的な組み合わせ
async function processRequest(rawInput: string): Promise<Result<Response>> {
  const parsed = parseInput(rawInput);          // Either<Error, Input>
  if (!parsed.ok) return parsed;

  const validated = validateInput(parsed.value); // Either<Error, ValidInput>
  if (!validated.ok) return validated;

  const result = await safeAsync(() =>           // Promise<Result<Data>>
    fetchData(validated.value)
  );
  if (!result.ok) return result;

  const transformed = transformData(result.value); // Maybe<Output>
  if (transformed == null) {
    return { ok: false, error: new Error("Transform failed") };
  }

  return { ok: true, value: buildResponse(transformed) };
}
```

---

## 4. モナド比較

### 主要モナド比較表

| モナド | コンテキスト | bind の動作 | 言語での対応 |
|---|---|---|---|
| **Maybe/Option** | 値の有無 | None ならスキップ | Rust: `Option`, TS: `?.` |
| **Either/Result** | 成功/失敗 | Err ならスキップ | Rust: `Result`, Go: `error` |
| **Promise/Future** | 非同期 | 完了後に次を実行 | JS: `Promise`, Rust: `Future` |
| **IO** | 副作用 | 実行を遅延 | Haskell: `IO` |
| **List** | 複数の値 | 各要素に適用・結合 | JS: `Array.flatMap` |
| **Reader** | 依存注入 | 環境を引き回す | 関数の合成 |
| **Writer** | ログ蓄積 | 値＋ログを伝搬 | ロギングミドルウェア |
| **State** | 状態管理 | 状態を引き回す | React: `useState` |

### モナドパターンの言語対応表

| 概念 | Haskell | Rust | TypeScript | Python |
|---|---|---|---|---|
| Maybe bind | `>>=` | `and_then()` | `?.` / `flatMap` | `or None` |
| Either bind | `>>=` | `?` 演算子 | `then` / `catch` | 例外 |
| do 記法 | `do` | `?` 演算子 | `async/await` | `async/await` |
| for 内包表記 | `do` | `Iterator` chain | `Array.flatMap` | リスト内包表記 |

---

## アンチパターン

### 1. モナドの過剰使用

**問題**: すべてに Maybe/Either を適用し、コードが過度に複雑になる。特にモナドの概念に馴染みのないチームでは保守が困難。

**対策**: 言語が提供するネイティブ機能を優先する。Rust なら `?` 演算子、TypeScript なら Optional Chaining `?.` と Nullish Coalescing `??` を活用する。自前のモナド実装は本当に必要な場合のみ。

### 2. モナドの型を無視した unwrap

**問題**: `Option.unwrap()` や `Result.unwrap()` を安易に使うと、実行時パニックが発生する。モナドの安全性が台無しになる。

```rust
// [NG]
let value = some_option.unwrap();  // None でパニック

// [OK]
let value = some_option.unwrap_or_default();
let value = some_option.unwrap_or_else(|| compute_default());
let value = match some_option {
    Some(v) => process(v),
    None => handle_missing(),
};
```

---

## FAQ

### Q1: モナドは実際の開発で役に立ちますか？

**A**: はい、ただし「モナド」という名前を意識する必要はありません。Rust の `?` 演算子、JavaScript の `async/await`、`Array.flatMap` はすべてモナドパターンです。既に日常的に使っている概念に理論的な基盤を与えるのがモナドの価値です。

### Q2: Haskell を学ばないとモナドは理解できませんか？

**A**: いいえ。モナドの概念は言語非依存です。TypeScript や Rust で `flatMap/and_then` のパターンを実践すれば十分理解できます。Haskell は型クラスでモナドを最も体系的に表現しますが、実用的な理解には不要です。

### Q3: モナドトランスフォーマーとは何ですか？

**A**: 複数のモナドを重ねて使う仕組みです。例えば `MaybeT (Either Error)` は「失敗するかもしれない計算で、さらに None の可能性がある」を表現します。実用的には async/await で Promise + Result を組み合わせるパターンが最も身近なモナドトランスフォーマーです。

---

## まとめ

| 項目 | 要点 |
|---|---|
| モナドの本質 | `bind` (flatMap) によるコンテキスト付き計算の合成 |
| Maybe/Option | null 安全性。値がないかもしれない計算の連鎖 |
| Either/Result | エラーハンドリング。鉄道指向プログラミング |
| Promise/Future | 非同期処理。async/await は do 記法の糖衣構文 |
| IO | 副作用の分離。純粋関数と副作用の境界を明確化 |
| 実践指針 | 言語のネイティブ機能を優先。unwrap を避ける |

## 次に読むべきガイド

- [ファンクタ・アプリカティブ](./01-functor-applicative.md) — モナドの前提となる抽象化
- [関数型パターン](./02-fp-patterns.md) — カリー化、パイプラインとの組み合わせ

## 参考文献

1. **Bartosz Milewski**: [Category Theory for Programmers](https://bartoszmilewski.com/2014/10/28/category-theory-for-programmers-the-preface/) — 圏論とモナドの理論的基盤
2. **Scott Wlaschin**: [Railway Oriented Programming](https://fsharpforfunandprofit.com/rop/) — Either モナドの実践的解説
3. **Rust by Example**: [Error Handling](https://doc.rust-lang.org/rust-by-example/error.html) — Result モナドの実践ガイド
