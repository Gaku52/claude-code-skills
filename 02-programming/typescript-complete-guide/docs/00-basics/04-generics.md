# ジェネリクス

> 型パラメータを使い、再利用可能かつ型安全なコードを書く。制約、条件型、型推論の仕組みを理解する。

## この章で学ぶこと

1. **ジェネリクスの基本** -- 型パラメータ、型引数の推論、ジェネリック関数・クラス・インターフェース
2. **型制約（constraints）** -- extends による型パラメータの制約、複数制約
3. **ジェネリクスの応用** -- 条件型との組み合わせ、デフォルト型パラメータ、型推論（infer）

---

## 1. ジェネリクスの基本

### コード例1: ジェネリック関数

```typescript
// 型パラメータ T を使った汎用関数
function identity<T>(value: T): T {
  return value;
}

// 型引数の明示的指定
const str = identity<string>("hello");  // 型: string
const num = identity<number>(42);       // 型: number

// 型引数の推論（多くの場合、明示不要）
const inferred = identity("hello");     // 型: string（推論される）
```

### コード例2: 実用的なジェネリック関数

```typescript
// 配列の最初の要素を返す
function first<T>(arr: T[]): T | undefined {
  return arr[0];
}

first([1, 2, 3]);       // 型: number | undefined
first(["a", "b"]);      // 型: string | undefined

// ペアを作る
function pair<A, B>(a: A, b: B): [A, B] {
  return [a, b];
}

const p = pair("name", 42);  // 型: [string, number]

// オブジェクトからキーの値を取得
function getProperty<T, K extends keyof T>(obj: T, key: K): T[K] {
  return obj[key];
}

const user = { name: "Alice", age: 30 };
const name = getProperty(user, "name");  // 型: string
const age = getProperty(user, "age");    // 型: number
// getProperty(user, "email");            // エラー: "email" は keyof User にない
```

### 型パラメータの流れ

```
  呼び出し: identity("hello")
                |
                v
  型推論エンジン:
    T = string （引数 "hello" の型から推論）
                |
                v
  インスタンス化: identity<string>(value: string): string
                |
                v
  結果の型: string
```

---

## 2. ジェネリクスの様々な形

### コード例3: ジェネリックインターフェースとクラス

```typescript
// ジェネリックインターフェース
interface Repository<T> {
  findById(id: string): Promise<T | null>;
  findAll(): Promise<T[]>;
  save(entity: T): Promise<T>;
  delete(id: string): Promise<void>;
}

// ジェネリッククラス
class InMemoryRepository<T extends { id: string }> implements Repository<T> {
  private items: Map<string, T> = new Map();

  async findById(id: string): Promise<T | null> {
    return this.items.get(id) ?? null;
  }

  async findAll(): Promise<T[]> {
    return Array.from(this.items.values());
  }

  async save(entity: T): Promise<T> {
    this.items.set(entity.id, entity);
    return entity;
  }

  async delete(id: string): Promise<void> {
    this.items.delete(id);
  }
}

// 使用例
interface User {
  id: string;
  name: string;
}

const userRepo = new InMemoryRepository<User>();
await userRepo.save({ id: "1", name: "Alice" });
const user = await userRepo.findById("1");  // 型: User | null
```

### コード例4: ジェネリック型エイリアス

```typescript
// 非同期操作の結果型
type AsyncResult<T> = {
  data: T | null;
  loading: boolean;
  error: Error | null;
};

// APIレスポンス
type ApiResponse<T> = {
  status: number;
  data: T;
  timestamp: string;
};

// ページネーション付きレスポンス
type Paginated<T> = {
  items: T[];
  total: number;
  page: number;
  pageSize: number;
  hasNext: boolean;
};

// 使用例
type UserListResponse = ApiResponse<Paginated<User>>;
```

### ジェネリクスの適用箇所

```
+------------------+---------------------------+---------------------+
| 適用箇所         | 構文                      | 例                  |
+------------------+---------------------------+---------------------+
| 関数             | function fn<T>(x: T): T   | identity<T>         |
| アロー関数       | const fn = <T>(x: T): T   | <T>(x: T) => T     |
| インターフェース | interface I<T> { ... }    | Repository<T>       |
| クラス           | class C<T> { ... }        | Stack<T>            |
| 型エイリアス     | type T<U> = { ... }       | Result<T>           |
| メソッド         | method<T>(x: T): T        | Array#map           |
+------------------+---------------------------+---------------------+
```

---

## 3. 型制約（Constraints）

### コード例5: extends による制約

```typescript
// T は { length: number } を持つ型に制約される
function logLength<T extends { length: number }>(value: T): T {
  console.log(`Length: ${value.length}`);
  return value;
}

logLength("hello");       // OK: string は length を持つ
logLength([1, 2, 3]);     // OK: number[] は length を持つ
logLength({ length: 10 }); // OK
// logLength(42);          // エラー: number は length を持たない

// keyof による制約
function pick<T, K extends keyof T>(obj: T, keys: K[]): Pick<T, K> {
  const result = {} as Pick<T, K>;
  for (const key of keys) {
    result[key] = obj[key];
  }
  return result;
}

const user = { id: 1, name: "Alice", email: "alice@test.com", age: 30 };
const picked = pick(user, ["name", "email"]);
// 型: { name: string; email: string }
```

### コード例6: 複数の型パラメータと制約

```typescript
// マージ関数
function merge<
  T extends Record<string, unknown>,
  U extends Record<string, unknown>
>(target: T, source: U): T & U {
  return { ...target, ...source };
}

const merged = merge(
  { name: "Alice" },
  { age: 30 }
);
// 型: { name: string } & { age: number }

// デフォルト型パラメータ
interface FetchOptions<T = unknown> {
  url: string;
  method?: "GET" | "POST";
  body?: T;
}

// T を指定しなければ unknown になる
const opts: FetchOptions = { url: "/api/users" };
const opts2: FetchOptions<{ name: string }> = {
  url: "/api/users",
  method: "POST",
  body: { name: "Alice" },
};
```

### 制約の階層図

```
  <T>                     制約なし（全ての型を受け入れる）
    |
    v
  <T extends object>     オブジェクト型に制限
    |
    v
  <T extends { id: number }>   id プロパティを持つ型に制限
    |
    v
  <T extends User>        User 型を満たす型に制限
    |
    v
  <T extends Admin>       Admin（extends User）を満たす型に制限
```

---

## 4. ジェネリクスの応用

### コード例7: 型推論の活用

```typescript
// Promiseの中身を取り出す型
type Unwrap<T> = T extends Promise<infer U> ? U : T;

type A = Unwrap<Promise<string>>;  // string
type B = Unwrap<Promise<number>>;  // number
type C = Unwrap<string>;           // string（Promiseでなければそのまま）

// 関数の戻り値型を取り出す（ReturnType相当）
type MyReturnType<T> = T extends (...args: any[]) => infer R ? R : never;

type D = MyReturnType<() => string>;            // string
type E = MyReturnType<(x: number) => boolean>;  // boolean

// 配列の要素型を取り出す
type ElementOf<T> = T extends (infer U)[] ? U : never;

type F = ElementOf<string[]>;     // string
type G = ElementOf<number[]>;     // number
```

### コード例8: ジェネリクスとマップ型

```typescript
// 全プロパティをオプショナルかつnullableにする
type NullablePartial<T> = {
  [K in keyof T]?: T[K] | null;
};

interface User {
  id: number;
  name: string;
  email: string;
}

type UpdateUserInput = NullablePartial<User>;
// { id?: number | null; name?: string | null; email?: string | null }

// イベントマップからハンドラ型を生成
type EventMap = {
  click: { x: number; y: number };
  keypress: { key: string; code: number };
  scroll: { offset: number };
};

type EventHandlers<T> = {
  [K in keyof T as `on${Capitalize<string & K>}`]: (event: T[K]) => void;
};

type Handlers = EventHandlers<EventMap>;
// {
//   onClick: (event: { x: number; y: number }) => void;
//   onKeypress: (event: { key: string; code: number }) => void;
//   onScroll: (event: { offset: number }) => void;
// }
```

---

## 型制約パターン比較

| パターン | 構文 | 用途 |
|----------|------|------|
| 上界制約 | `<T extends U>` | T を U のサブタイプに制限 |
| keyof制約 | `<K extends keyof T>` | K を T のキーに制限 |
| 複数制約 | `<T extends A & B>` | T を A かつ B を満たす型に制限 |
| デフォルト型 | `<T = DefaultType>` | 型引数省略時のデフォルト |
| 条件型 | `T extends U ? X : Y` | 型レベルの条件分岐 |
| 推論 | `T extends X<infer U>` | 構造から型を抽出 |

---

## アンチパターン

### アンチパターン1: 不要なジェネリクス

```typescript
// BAD: T を使っていないのにジェネリクスにしている
function greet<T>(name: string): string {
  return `Hello, ${name}`;
}

// BAD: T がそのまま返されるだけ（identity以外で意味なし）
function wrap<T>(value: T): { value: T } {
  return { value };
}
// ↑ これは良い例。BAD なのは以下のようなケース:

// BAD: ジェネリクスが過剰
function getLength<T extends { length: number }>(x: T): number {
  return x.length;
}
// GOOD: ジェネリクス不要（T を返さないので）
function getLength(x: { length: number }): number {
  return x.length;
}
```

### アンチパターン2: any で制約を回避

```typescript
// BAD: 制約エラーを any で黙らせる
function merge<T>(a: T, b: T): T {
  return { ...(a as any), ...(b as any) } as T;
}

// GOOD: 適切な制約をつける
function merge<T extends Record<string, unknown>>(a: T, b: Partial<T>): T {
  return { ...a, ...b };
}
```

---

## FAQ

### Q1: `<T>` と `<T extends unknown>` は同じですか？

**A:** はい、実質同じです。全ての型は `unknown` のサブタイプなので、`<T>` と `<T extends unknown>` は等価です。ただし、`<T extends object>` とすると null, undefined, プリミティブは除外されます。

### Q2: ジェネリクスの型パラメータ名の慣例は？

**A:** 一般的な慣例:
- `T` (Type): 汎用的な型
- `K` (Key): キーの型
- `V` (Value): 値の型
- `E` (Element / Error): 要素型やエラー型
- `R` (Return / Result): 戻り値型

複雑な場合は `TInput`, `TOutput` のように説明的な名前を使うことも推奨されます。

### Q3: アロー関数でジェネリクスを使うとJSXと衝突しませんか？

**A:** `.tsx` ファイルでは `<T>` が JSX タグと誤認される場合があります。回避策として `<T,>` や `<T extends unknown>` を使います。
```typescript
// .tsx ファイルでの回避策
const identity = <T,>(value: T): T => value;
const identity = <T extends unknown>(value: T): T => value;
```

---

## まとめ

| 項目 | 内容 |
|------|------|
| ジェネリクスとは | 型をパラメータ化し、再利用可能な型安全コードを書く仕組み |
| 型推論 | 多くの場合、TypeScriptが型引数を自動推論する |
| 制約 (extends) | 型パラメータが満たすべき条件を指定する |
| デフォルト型 | 型引数を省略した場合の既定値を設定 |
| infer | 条件型の中で型を抽出するキーワード |
| 適用箇所 | 関数、クラス、インターフェース、型エイリアス |
| 設計指針 | 不要なジェネリクスは避け、型を返す場合に使う |

---

## 次に読むべきガイド

- [../01-advanced-types/00-conditional-types.md](../01-advanced-types/00-conditional-types.md) -- 条件型（詳細）
- [../01-advanced-types/01-mapped-types.md](../01-advanced-types/01-mapped-types.md) -- マップ型（詳細）

---

## 参考文献

1. **TypeScript Handbook: Generics** -- https://www.typescriptlang.org/docs/handbook/2/generics.html
2. **TypeScript Deep Dive: Generics** -- https://basarat.gitbook.io/typescript/type-system/generics
3. **Effective TypeScript, Item 26: Understand How Context Is Used in Type Inference** -- Dan Vanderkam著, O'Reilly
