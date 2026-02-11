# マップ型（Mapped Types）

> 既存の型を変換して新しい型を生成する。Partial, Required, Readonly, Record などのユーティリティ型の仕組みを理解し、独自のマップ型を構築する。

## この章で学ぶこと

1. **マップ型の基本** -- `{ [K in keyof T]: ... }` 構文、プロパティの変換
2. **組み込みユーティリティ型** -- Partial, Required, Readonly, Record, Pick, Omit の内部実装
3. **Key Remapping** -- `as` 句によるキー名の変換、フィルタリング

---

## 1. マップ型の基本

### コード例1: 基本構文

```typescript
// T の全プロパティを string 型にする
type Stringify<T> = {
  [K in keyof T]: string;
};

interface User {
  id: number;
  name: string;
  active: boolean;
}

type StringUser = Stringify<User>;
// { id: string; name: string; active: string }

// T の全プロパティをオプショナルにする（Partial相当）
type MyPartial<T> = {
  [K in keyof T]?: T[K];
};

type PartialUser = MyPartial<User>;
// { id?: number; name?: string; active?: boolean }
```

### マップ型の変換イメージ

```
  元の型 T                    マップ型 { [K in keyof T]: F(T[K]) }
+------------------+        +------------------+
| id: number       |  --->  | id: F(number)    |
| name: string     |  --->  | name: F(string)  |
| active: boolean  |  --->  | active: F(bool)  |
+------------------+        +------------------+

  各プロパティに変換関数 F を適用するイメージ
```

### コード例2: 修飾子の追加・除去

```typescript
// readonly の追加
type MyReadonly<T> = {
  readonly [K in keyof T]: T[K];
};

// readonly の除去（-readonly）
type Mutable<T> = {
  -readonly [K in keyof T]: T[K];
};

// オプショナルの除去（-?）
type MyRequired<T> = {
  [K in keyof T]-?: T[K];
};

// 例
interface Config {
  readonly host: string;
  readonly port: number;
  debug?: boolean;
}

type MutableConfig = Mutable<Config>;
// { host: string; port: number; debug?: boolean }

type RequiredConfig = MyRequired<Config>;
// { readonly host: string; readonly port: number; debug: boolean }
```

---

## 2. 組み込みユーティリティ型

### コード例3: 主要ユーティリティ型の実装

```typescript
// Partial<T>: 全プロパティをオプショナルに
type Partial<T> = { [K in keyof T]?: T[K] };

// Required<T>: 全プロパティを必須に
type Required<T> = { [K in keyof T]-?: T[K] };

// Readonly<T>: 全プロパティをreadonly に
type Readonly<T> = { readonly [K in keyof T]: T[K] };

// Record<K, V>: キーの集合からオブジェクト型を生成
type Record<K extends keyof any, T> = { [P in K]: T };

// Pick<T, K>: 特定のプロパティのみ取り出す
type Pick<T, K extends keyof T> = { [P in K]: T[P] };

// Omit<T, K>: 特定のプロパティを除外する
type Omit<T, K extends keyof any> = Pick<T, Exclude<keyof T, K>>;
```

### コード例4: ユーティリティ型の活用

```typescript
interface User {
  id: number;
  name: string;
  email: string;
  password: string;
  createdAt: Date;
}

// 更新時は全フィールドオプショナル
type UpdateUser = Partial<Omit<User, "id" | "createdAt">>;
// { name?: string; email?: string; password?: string }

// 公開APIのレスポンス（パスワードを除外）
type PublicUser = Omit<User, "password">;
// { id: number; name: string; email: string; createdAt: Date }

// ステータスマップ
type StatusMap = Record<"pending" | "active" | "inactive", number>;
// { pending: number; active: number; inactive: number }

// 部分的にReadonly
type UserWithReadonlyId = Readonly<Pick<User, "id">> & Omit<User, "id">;
```

### ユーティリティ型の関係

```
  元の型 T
    |
    +---> Partial<T>     全プロパティ ? 付与
    |
    +---> Required<T>    全プロパティ ? 除去
    |
    +---> Readonly<T>    全プロパティ readonly 付与
    |
    +---> Pick<T, K>     指定プロパティのみ抽出
    |       |
    |       +---> Omit<T, K> = Pick<T, Exclude<keyof T, K>>
    |
    +---> Record<K, V>   キー集合 → オブジェクト型生成
```

---

## 3. Key Remapping（キー再マッピング）

### コード例5: as 句によるキー変換

```typescript
// プロパティ名にプレフィックスを追加
type Prefixed<T, P extends string> = {
  [K in keyof T as `${P}${Capitalize<string & K>}`]: T[K];
};

interface User {
  name: string;
  age: number;
}

type PrefixedUser = Prefixed<User, "get">;
// { getName: string; getAge: number }

// getter メソッドに変換
type Getters<T> = {
  [K in keyof T as `get${Capitalize<string & K>}`]: () => T[K];
};

type UserGetters = Getters<User>;
// { getName: () => string; getAge: () => number }

// 特定の型のプロパティだけ抽出（フィルタリング）
type OnlyStrings<T> = {
  [K in keyof T as T[K] extends string ? K : never]: T[K];
};

interface Mixed {
  name: string;
  age: number;
  email: string;
  active: boolean;
}

type StringProps = OnlyStrings<Mixed>;
// { name: string; email: string }
```

### コード例6: イベントハンドラ生成

```typescript
// イベント名からハンドラ型を自動生成
type EventMap = {
  click: { x: number; y: number };
  focus: { target: HTMLElement };
  submit: { data: FormData };
};

type EventHandlers<T> = {
  [K in keyof T as `on${Capitalize<string & K>}`]: (event: T[K]) => void;
};

type Handlers = EventHandlers<EventMap>;
// {
//   onClick: (event: { x: number; y: number }) => void;
//   onFocus: (event: { target: HTMLElement }) => void;
//   onSubmit: (event: { data: FormData }) => void;
// }

// 逆マッピング（ハンドラ名からイベント名を取得）
type UnCapitalize<S extends string> =
  S extends `${infer F}${infer R}` ? `${Lowercase<F>}${R}` : S;

type HandlerToEvent<T extends string> =
  T extends `on${infer E}` ? UnCapitalize<E> : never;

type EventName = HandlerToEvent<"onClick">;  // "click"
```

---

## 4. 高度なマップ型

### コード例7: DeepPartial（再帰マップ型）

```typescript
type DeepPartial<T> =
  T extends object
    ? { [K in keyof T]?: DeepPartial<T[K]> }
    : T;

interface Config {
  server: {
    host: string;
    port: number;
    ssl: {
      cert: string;
      key: string;
    };
  };
  logging: {
    level: "debug" | "info" | "error";
    file: string;
  };
}

type PartialConfig = DeepPartial<Config>;
// 全てのネストしたプロパティもオプショナル

// 深い部分だけ更新できる
function updateConfig(config: Config, updates: DeepPartial<Config>): Config {
  return { ...config, ...updates }; // 簡略化した実装
}
```

### コード例8: 実践的なマップ型の組み合わせ

```typescript
// フォームの状態型を自動生成
interface FormFields {
  username: string;
  email: string;
  age: number;
}

// 各フィールドの状態
type FieldState<T> = {
  value: T;
  error: string | null;
  touched: boolean;
  dirty: boolean;
};

// フォーム全体の状態
type FormState<T> = {
  [K in keyof T]: FieldState<T[K]>;
};

type MyFormState = FormState<FormFields>;
// {
//   username: FieldState<string>;
//   email: FieldState<string>;
//   age: FieldState<number>;
// }
```

---

## ユーティリティ型早見表

| 型 | 入力 | 出力 | 用途 |
|----|------|------|------|
| `Partial<T>` | `{ a: string; b: number }` | `{ a?: string; b?: number }` | 部分更新 |
| `Required<T>` | `{ a?: string; b?: number }` | `{ a: string; b: number }` | 必須化 |
| `Readonly<T>` | `{ a: string }` | `{ readonly a: string }` | 不変化 |
| `Record<K,V>` | `"a" \| "b", number` | `{ a: number; b: number }` | 辞書作成 |
| `Pick<T,K>` | `{ a: 1; b: 2; c: 3 }, "a"\|"b"` | `{ a: 1; b: 2 }` | 選択 |
| `Omit<T,K>` | `{ a: 1; b: 2; c: 3 }, "c"` | `{ a: 1; b: 2 }` | 除外 |

---

## 修飾子操作比較

| 操作 | 構文 | 効果 |
|------|------|------|
| optional追加 | `[K in keyof T]?:` | `?` 追加 |
| optional除去 | `[K in keyof T]-?:` | `?` 除去 |
| readonly追加 | `readonly [K in keyof T]:` | `readonly` 追加 |
| readonly除去 | `-readonly [K in keyof T]:` | `readonly` 除去 |

---

## アンチパターン

### アンチパターン1: マップ型で不要な複雑さを追加

```typescript
// BAD: マップ型を使う必要のないケース
type UserKeys = {
  [K in "name" | "email"]: string;
};
// GOOD: シンプルに書ける
interface UserKeys {
  name: string;
  email: string;
}
```

### アンチパターン2: DeepReadonly を全てに適用

```typescript
// BAD: 全てのオブジェクトを DeepReadonly にする
type DeepReadonly<T> = { readonly [K in keyof T]: DeepReadonly<T[K]> };
type State = DeepReadonly<HugeComplexType>;
// コンパイルが遅くなり、エラーメッセージが読めなくなる

// GOOD: 必要な境界でのみ Readonly を使う
function getConfig(): Readonly<Config> {
  return config;
}
```

---

## FAQ

### Q1: `keyof T` と `keyof T & string` の違いは？

**A:** `keyof T` は `string | number | symbol` のいずれかです。テンプレートリテラル型で `${K}` のように文字列として使う場合、`K` が `string` でなければならないため、`keyof T & string` として文字列キーのみに限定します。

### Q2: マップ型で一部のプロパティだけ変換することはできますか？

**A:** はい、`Pick` と `Omit` を組み合わせるか、Key Remapping の `as` 句で条件分岐します。
```typescript
type PartialBy<T, K extends keyof T> =
  Omit<T, K> & Partial<Pick<T, K>>;
```

### Q3: `Record<string, T>` と `{ [key: string]: T }` は同じですか？

**A:** ほぼ同じですが、微妙な違いがあります。`Record<string, T>` はマップ型として定義されており、エディタでの表示やエラーメッセージが異なる場合があります。実用上は同じ意味で使えます。

---

## まとめ

| 項目 | 内容 |
|------|------|
| マップ型 | `{ [K in keyof T]: ... }` で型のプロパティを変換 |
| 修飾子 | `?`, `readonly` の追加・除去（`-?`, `-readonly`） |
| Key Remapping | `as` 句でキー名を変換・フィルタリング |
| 組み込み型 | Partial, Required, Readonly, Record, Pick, Omit |
| 再帰マップ型 | DeepPartial, DeepReadonly などネスト構造に対応 |
| 実用パターン | フォーム状態、イベントハンドラ、API型変換 |

---

## 次に読むべきガイド

- [02-template-literal-types.md](./02-template-literal-types.md) -- テンプレートリテラル型
- [03-type-challenges.md](./03-type-challenges.md) -- 型チャレンジ

---

## 参考文献

1. **TypeScript Handbook: Mapped Types** -- https://www.typescriptlang.org/docs/handbook/2/mapped-types.html
2. **TypeScript Handbook: Utility Types** -- https://www.typescriptlang.org/docs/handbook/utility-types.html
3. **Effective TypeScript, Item 14: Use Type Operations and Generics to Avoid Repeating Yourself** -- Dan Vanderkam著, O'Reilly
