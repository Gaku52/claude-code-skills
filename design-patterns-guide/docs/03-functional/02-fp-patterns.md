# 関数型パターン

> カリー化、パイプライン、レンズなど、関数型プログラミングの実践的パターンを習得し、合成可能で保守性の高いコードを書く

## この章で学ぶこと

1. **関数合成の基本** — カリー化、部分適用、ポイントフリースタイル
2. **パイプラインパターン** — データ変換チェーン、ミドルウェア合成
3. **不変データ操作** — レンズ、トランスデューサー、メモ化

---

## 1. カリー化と部分適用

```
カリー化と部分適用
====================

通常の関数:
  add(a, b) = a + b
  add(3, 5)  --> 8

カリー化:
  add = a => b => a + b
  add(3)     --> b => 3 + b   (部分適用された関数)
  add(3)(5)  --> 8

部分適用:
  add3 = add(3)    // 引数を1つ固定
  add3(5)  --> 8
  add3(10) --> 13

利点: 関数の再利用と合成が容易になる
```

### コード例 1: カリー化の実装と活用

```typescript
// 汎用カリー化関数
function curry<A, B, C>(fn: (a: A, b: B) => C): (a: A) => (b: B) => C {
  return (a: A) => (b: B) => fn(a, b);
}

function curry3<A, B, C, D>(
  fn: (a: A, b: B, c: C) => D
): (a: A) => (b: B) => (c: C) => D {
  return (a: A) => (b: B) => (c: C) => fn(a, b, c);
}

// 実用例
const multiply = curry((a: number, b: number) => a * b);
const double = multiply(2);
const triple = multiply(3);

console.log(double(5));  // 10
console.log(triple(5));  // 15

// フィルタの部分適用
const filterBy = curry3(
  <T>(key: keyof T, value: T[keyof T], items: T[]) =>
    items.filter(item => item[key] === value)
);

const filterByStatus = filterBy("status");
const getActiveItems = filterByStatus("active");
const getPendingItems = filterByStatus("pending");

// 再利用可能なフィルタ関数
const activeOrders = getActiveItems(orders);
const pendingOrders = getPendingItems(orders);
```

---

## 2. パイプラインパターン

```
パイプラインの考え方
=====================

データ --> [変換1] --> [変換2] --> [変換3] --> 結果

Unix パイプ:
  cat file | grep "error" | sort | uniq -c

関数パイプライン:
  pipe(getData, filter(isActive), map(toDTO), sortBy('name'))
```

### コード例 2: パイプライン関数

```typescript
// pipe: 左から右に関数を合成
function pipe<T>(...fns: Array<(arg: any) => any>): (arg: T) => any {
  return (arg: T) => fns.reduce((acc, fn) => fn(acc), arg as any);
}

// compose: 右から左に関数を合成（数学的な合成順序）
function compose<T>(...fns: Array<(arg: any) => any>): (arg: T) => any {
  return (arg: T) => fns.reduceRight((acc, fn) => fn(acc), arg as any);
}

// 実用例: データ変換パイプライン
interface RawUser {
  first_name: string;
  last_name: string;
  age: number;
  status: string;
}

const processUsers = pipe<RawUser[]>(
  users => users.filter(u => u.status === "active"),
  users => users.filter(u => u.age >= 18),
  users => users.map(u => ({
    fullName: `${u.first_name} ${u.last_name}`,
    age: u.age,
  })),
  users => users.sort((a, b) => a.fullName.localeCompare(b.fullName)),
);

const result = processUsers(rawUsers);
```

### コード例 3: ミドルウェアパターン

```typescript
// Express/Koa スタイルのミドルウェア合成
type Middleware<T> = (ctx: T, next: () => Promise<void>) => Promise<void>;

function composeMiddleware<T>(...middlewares: Middleware<T>[]): Middleware<T> {
  return (ctx: T, next: () => Promise<void>) => {
    let index = -1;
    function dispatch(i: number): Promise<void> {
      if (i <= index) return Promise.reject(new Error("next() called multiple times"));
      index = i;
      const fn = i === middlewares.length ? next : middlewares[i];
      return fn(ctx, () => dispatch(i + 1));
    }
    return dispatch(0);
  };
}

// 使用例
const logger: Middleware<Context> = async (ctx, next) => {
  const start = Date.now();
  await next();
  console.log(`${ctx.method} ${ctx.path} - ${Date.now() - start}ms`);
};

const auth: Middleware<Context> = async (ctx, next) => {
  if (!ctx.headers.authorization) throw new Error("Unauthorized");
  await next();
};

const app = composeMiddleware(logger, auth, handler);
```

---

## 3. 不変データ操作（レンズ）

```
レンズ: 不変データ構造の部分的な読み書き
=========================================

深いネストのオブジェクト更新:

[NG] ミュータブルな直接更新
  user.address.city = "Tokyo";

[NG] スプレッド地獄
  { ...user, address: { ...user.address, city: "Tokyo" } }

[OK] レンズで宣言的に更新
  set(addressCityLens, "Tokyo", user)

レンズの型:
  Lens<S, A>
    get: S -> A          (全体から部分を取得)
    set: (A, S) -> S     (部分を更新して新しい全体を返す)
```

### コード例 4: レンズの実装

```typescript
interface Lens<S, A> {
  get: (s: S) => A;
  set: (a: A, s: S) => S;
}

function lens<S, A>(
  get: (s: S) => A,
  set: (a: A, s: S) => S
): Lens<S, A> {
  return { get, set };
}

// レンズの合成
function composeLens<S, A, B>(outer: Lens<S, A>, inner: Lens<A, B>): Lens<S, B> {
  return {
    get: (s: S) => inner.get(outer.get(s)),
    set: (b: B, s: S) => outer.set(inner.set(b, outer.get(s)), s),
  };
}

// over: レンズを通して関数を適用
function over<S, A>(l: Lens<S, A>, fn: (a: A) => A, s: S): S {
  return l.set(fn(l.get(s)), s);
}

// 使用例
interface Address { city: string; zip: string; }
interface User { name: string; address: Address; }

const addressLens = lens<User, Address>(
  u => u.address,
  (a, u) => ({ ...u, address: a })
);

const cityLens = lens<Address, string>(
  a => a.city,
  (c, a) => ({ ...a, city: c })
);

const userCityLens = composeLens(addressLens, cityLens);

const user: User = { name: "Taro", address: { city: "Osaka", zip: "530" } };
const updated = userCityLens.set("Tokyo", user);
// { name: "Taro", address: { city: "Tokyo", zip: "530" } }

const uppercased = over(userCityLens, c => c.toUpperCase(), user);
// { name: "Taro", address: { city: "OSAKA", zip: "530" } }
```

---

## 4. メモ化

### コード例 5: メモ化の実装

```typescript
// 汎用メモ化関数
function memoize<Args extends unknown[], R>(
  fn: (...args: Args) => R,
  keyFn: (...args: Args) => string = (...args) => JSON.stringify(args)
): (...args: Args) => R {
  const cache = new Map<string, R>();

  return (...args: Args): R => {
    const key = keyFn(...args);
    if (cache.has(key)) return cache.get(key)!;
    const result = fn(...args);
    cache.set(key, result);
    return result;
  };
}

// LRU キャッシュ付きメモ化
function memoizeLRU<Args extends unknown[], R>(
  fn: (...args: Args) => R,
  maxSize: number = 100
): (...args: Args) => R {
  const cache = new Map<string, R>();

  return (...args: Args): R => {
    const key = JSON.stringify(args);
    if (cache.has(key)) {
      const value = cache.get(key)!;
      cache.delete(key);
      cache.set(key, value);  // LRU: アクセスしたら末尾に移動
      return value;
    }
    const result = fn(...args);
    cache.set(key, result);
    if (cache.size > maxSize) {
      const oldest = cache.keys().next().value;
      cache.delete(oldest!);
    }
    return result;
  };
}

// 使用例: フィボナッチ
const fibonacci = memoize((n: number): number =>
  n <= 1 ? n : fibonacci(n - 1) + fibonacci(n - 2)
);

fibonacci(50);  // 瞬時（メモ化なしでは数十秒）
```

---

## 関数型パターン一覧比較表

| パターン | 目的 | 言語での対応 |
|---|---|---|
| **カリー化** | 部分適用で関数を再利用 | Haskell: デフォルト、JS: ライブラリ |
| **パイプライン** | データ変換の宣言的記述 | `|>` 演算子、`pipe()` 関数 |
| **レンズ** | 不変データの部分更新 | Haskell: lens、JS: Ramda |
| **メモ化** | 計算結果のキャッシュ | React: `useMemo`, Python: `@lru_cache` |
| **トランスデューサー** | 中間配列なしの変換合成 | Clojure: transducers |
| **パターンマッチ** | データ構造の分解と分岐 | Rust: `match`, TS: 型ガード |

### 関数型 vs 手続き型比較表

| 側面 | 関数型 | 手続き型 |
|---|---|---|
| **状態管理** | 不変データ + 新しい値を返す | ミュータブル変数を直接変更 |
| **制御フロー** | 再帰、高階関数 | ループ、条件分岐 |
| **副作用** | 分離して管理 | どこでも発生 |
| **テスト容易性** | 高い（参照透過性） | 低い（状態依存） |
| **並行性** | 安全（共有状態なし） | 危険（競合状態） |
| **デバッグ** | 値の追跡が容易 | 状態の追跡が困難 |

---

## アンチパターン

### 1. 過度なポイントフリースタイル

**問題**: 引数名を省略しすぎて可読性が著しく低下する。

```typescript
// [NG] 過度なポイントフリー
const process = pipe(filter(propEq("active", true)), map(pick(["id", "name"])), sortBy(prop("name")));
// 何を処理しているか読み取りにくい

// [OK] 適度な命名
const getActiveUserNames = (users: User[]) =>
  users
    .filter(u => u.active)
    .map(u => ({ id: u.id, name: u.name }))
    .sort((a, b) => a.name.localeCompare(b.name));
```

### 2. 不適切なメモ化

**問題**: 副作用のある関数や、引数空間が巨大な関数をメモ化するとバグやメモリリークが発生する。

**対策**: メモ化は純粋関数にのみ適用する。LRU キャッシュでメモリ使用量を制限する。時間依存の処理（`Date.now()` を含む関数など）はメモ化しない。

---

## FAQ

### Q1: カリー化は JavaScript/TypeScript で実用的ですか？

**A**: 部分適用は実用的ですが、フルカリー化は TypeScript の型推論と相性が悪い場合があります。`lodash/fp` の `curry` や arrow function での手動部分適用が実用的です。

### Q2: パイプライン演算子 (`|>`) は使えますか？

**A**: JavaScript の TC39 提案（Stage 2）として進行中ですが、2026年時点では標準化されていません。代わりに `pipe()` ユーティリティ関数か、メソッドチェーンを使用してください。

### Q3: Immutable.js や Immer は必要ですか？

**A**: 小規模なオブジェクトならスプレッド構文で十分です。深いネストや大規模データでは Immer（構造共有による効率的な不変更新）が推奨です。レンズパターンも選択肢ですが、学習コストとのトレードオフを考慮してください。

---

## まとめ

| 項目 | 要点 |
|---|---|
| カリー化 | 引数を1つずつ受け取る関数に変換。部分適用で再利用性向上 |
| パイプライン | データ変換を宣言的に記述。可読性と保守性の向上 |
| レンズ | 不変データ構造の部分的な読み書きを合成可能に |
| メモ化 | 純粋関数の計算結果をキャッシュ。再計算を回避 |
| ミドルウェア | 関数合成による横断的関心事の分離 |
| 実践指針 | 可読性とのバランスが重要。過度な抽象化を避ける |

## 次に読むべきガイド

- [モナド](./00-monad.md) — 関数合成のより高度な抽象化
- [ファンクタ・アプリカティブ](./01-functor-applicative.md) — map と ap の理論的基盤

## 参考文献

1. **Eric Elliott**: [Composing Software](https://medium.com/javascript-scene/composing-software-an-introduction-27b72500d6ea) — JavaScript での関数型プログラミング
2. **Brian Lonsdorf**: [Professor Frisby's Mostly Adequate Guide](https://mostly-adequate.gitbook.io/mostly-adequate-guide/) — 関数型プログラミングの入門ガイド
3. **Ramda Documentation**: [Ramda](https://ramdajs.com/) — JavaScript の関数型ユーティリティライブラリ
