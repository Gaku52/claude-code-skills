# マップ型（Mapped Types）

> 既存の型を変換して新しい型を生成する。Partial, Required, Readonly, Record などのユーティリティ型の仕組みを理解し、独自のマップ型を構築する。

## この章で学ぶこと

1. **マップ型の基本** -- `{ [K in keyof T]: ... }` 構文、プロパティの変換
2. **組み込みユーティリティ型** -- Partial, Required, Readonly, Record, Pick, Omit の内部実装
3. **Key Remapping** -- `as` 句によるキー名の変換、フィルタリング
4. **高度なマップ型** -- 再帰マップ型、条件付きマッピング、複合パターン
5. **実務パターン** -- フォーム状態管理、API型変換、イベントシステム設計
6. **パフォーマンスとデバッグ** -- マップ型のコンパイル性能と問題解決

---

## 1. マップ型の基本

### 1.1 基本構文

マップ型は、既存の型のキーをイテレーションして新しい型を構築する。JavaScript の `Array.map()` に似たイメージで、型を変換する。

```typescript
// 基本構文: { [K in keyof T]: 変換後の型 }
// T の各プロパティキー K に対して、変換後の型を指定する

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

// T の全プロパティを Promise でラップする
type Promisified<T> = {
  [K in keyof T]: Promise<T[K]>;
};

type PromiseUser = Promisified<User>;
// { id: Promise<number>; name: Promise<string>; active: Promise<boolean> }

// T の全プロパティを配列にする
type Arrayified<T> = {
  [K in keyof T]: T[K][];
};

type ArrayUser = Arrayified<User>;
// { id: number[]; name: string[]; active: boolean[] }
```

### 1.2 マップ型の変換イメージ

```
  元の型 T                    マップ型 { [K in keyof T]: F(T[K]) }
+------------------+        +------------------+
| id: number       |  --->  | id: F(number)    |
| name: string     |  --->  | name: F(string)  |
| active: boolean  |  --->  | active: F(bool)  |
+------------------+        +------------------+

  各プロパティに変換関数 F を適用するイメージ

  マップ型の構成要素:
  { [K in keyof T]: T[K] }
     ^  ^     ^       ^
     |  |     |       +--- 値の型（変換可能）
     |  |     +----------- イテレーション対象
     |  +----------------- イテレーション変数
     +--------------------- プロパティキー
```

### 1.3 修飾子の追加・除去

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

// 両方の修飾子を同時に操作
type MutableRequired<T> = {
  -readonly [K in keyof T]-?: T[K];
};

type MutableRequiredConfig = MutableRequired<Config>;
// { host: string; port: number; debug: boolean }

// readonly かつ optional に
type ReadonlyPartial<T> = {
  readonly [K in keyof T]?: T[K];
};

type ReadonlyPartialConfig = ReadonlyPartial<Config>;
// { readonly host?: string; readonly port?: number; readonly debug?: boolean }
```

### 1.4 keyof の詳細な挙動

```typescript
// keyof は型のすべてのプロパティキーのユニオンを返す
interface Example {
  name: string;
  age: number;
  0: boolean;
  [Symbol.iterator]: () => Iterator<any>;
}

type Keys = keyof Example;
// "name" | "age" | 0 | typeof Symbol.iterator
// → string | number | symbol のいずれかになりうる

// マップ型で文字列キーのみを使用する場合
type StringKeysOnly<T> = {
  [K in keyof T & string]: T[K];
};

// 数値キーのみを使用する場合
type NumberKeysOnly<T> = {
  [K in keyof T & number]: T[K];
};

// keyof の特殊なケース: インデックスシグネチャ
interface IndexSignature {
  [key: string]: number;
  specific: 42;
}

type ISKeys = keyof IndexSignature;
// string | number（インデックスシグネチャにより）
// "specific" は string に含まれる

// 配列型の keyof
type ArrayKeys = keyof string[];
// number | "length" | "push" | "pop" | "concat" | ... (配列のメソッドも含む)

// タプル型の keyof
type TupleKeys = keyof [string, number];
// "0" | "1" | "length" | "push" | ... (配列メソッドも含む)
```

### 1.5 マップ型と in 演算子

```typescript
// in 演算子はユニオン型のメンバーをイテレーションする
// keyof T だけでなく、任意のユニオン型を使用できる

// 文字列リテラルのユニオンから型を生成
type StatusFlags = {
  [K in "loading" | "error" | "success"]: boolean;
};
// { loading: boolean; error: boolean; success: boolean }

// 数値リテラルのユニオンから型を生成
type DigitMap = {
  [K in 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9]: string;
};

// テンプレートリテラル型と組み合わせ
type CSSVariables = {
  [K in `--color-${"primary" | "secondary" | "accent"}`]: string;
};
// {
//   "--color-primary": string;
//   "--color-secondary": string;
//   "--color-accent": string;
// }
```

---

## 2. 組み込みユーティリティ型

### 2.1 主要ユーティリティ型の実装

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

### 2.2 ユーティリティ型の活用パターン

```typescript
interface User {
  id: number;
  name: string;
  email: string;
  password: string;
  createdAt: Date;
  updatedAt: Date;
}

// 更新時は全フィールドオプショナル（idとcreatedAtは変更不可）
type UpdateUser = Partial<Omit<User, "id" | "createdAt">>;
// { name?: string; email?: string; password?: string; updatedAt?: Date }

// 公開APIのレスポンス（パスワードを除外）
type PublicUser = Omit<User, "password">;
// { id: number; name: string; email: string; createdAt: Date; updatedAt: Date }

// 作成時の型（idとタイムスタンプは自動生成）
type CreateUser = Omit<User, "id" | "createdAt" | "updatedAt">;
// { name: string; email: string; password: string }

// ステータスマップ
type StatusMap = Record<"pending" | "active" | "inactive", number>;
// { pending: number; active: number; inactive: number }

// 部分的にReadonly
type UserWithReadonlyId = Readonly<Pick<User, "id">> & Omit<User, "id">;
// { readonly id: number; name: string; email: string; ... }

// 部分的にPartial
type PartialBy<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;
type UserOptionalEmail = PartialBy<User, "email" | "password">;
// { id: number; name: string; email?: string; password?: string; createdAt: Date; updatedAt: Date }

// 部分的にRequired
type RequiredBy<T, K extends keyof T> = Omit<T, K> & Required<Pick<T, K>>;
```

### 2.3 Record の高度な使い方

```typescript
// Record を使ったルックアップテーブル
type HttpMethod = "GET" | "POST" | "PUT" | "DELETE" | "PATCH";

type EndpointConfig = {
  path: string;
  auth: boolean;
  rateLimit: number;
};

type ApiEndpoints = Record<HttpMethod, EndpointConfig[]>;

// Record とジェネリクスの組み合わせ
type EntityStore<T extends string, E> = Record<T, E[]>;

// 使用例
interface Product {
  id: string;
  name: string;
  price: number;
}

type ProductStore = EntityStore<"electronics" | "clothing" | "food", Product>;
// {
//   electronics: Product[];
//   clothing: Product[];
//   food: Product[];
// }

// Record を使ったキャッシュ型
type CacheEntry<T> = {
  data: T;
  timestamp: number;
  ttl: number;
};

type Cache<Keys extends string, Value> = Record<Keys, CacheEntry<Value> | null>;

// Recordの型安全な初期化
function createRecord<K extends string, V>(
  keys: K[],
  initialValue: V
): Record<K, V> {
  const result = {} as Record<K, V>;
  for (const key of keys) {
    result[key] = initialValue;
  }
  return result;
}

const flags = createRecord(["loading", "error", "success"] as const, false);
// Record<"loading" | "error" | "success", boolean>
```

### 2.4 Pick と Omit の高度なパターン

```typescript
// 条件付き Pick: 特定の型のプロパティだけを選択
type PickByType<T, ValueType> = {
  [K in keyof T as T[K] extends ValueType ? K : never]: T[K];
};

interface UserProfile {
  name: string;
  age: number;
  email: string;
  isActive: boolean;
  score: number;
  bio: string;
}

type StringFields = PickByType<UserProfile, string>;
// { name: string; email: string; bio: string }

type NumberFields = PickByType<UserProfile, number>;
// { age: number; score: number }

// 条件付き Omit: 特定の型のプロパティを除外
type OmitByType<T, ValueType> = {
  [K in keyof T as T[K] extends ValueType ? never : K]: T[K];
};

type NonStringFields = OmitByType<UserProfile, string>;
// { age: number; isActive: boolean; score: number }

// 再帰的 Pick: ネストされたオブジェクトの特定パスのみ抽出
type DeepPick<T, Paths extends string> =
  Paths extends `${infer Key}.${infer Rest}`
    ? Key extends keyof T
      ? { [K in Key]: DeepPick<T[Key], Rest> }
      : never
    : Paths extends keyof T
      ? { [K in Paths]: T[Paths] }
      : never;

interface NestedUser {
  profile: {
    name: string;
    avatar: {
      url: string;
      size: number;
    };
  };
  settings: {
    theme: string;
    notifications: boolean;
  };
}

type JustAvatar = DeepPick<NestedUser, "profile.avatar.url">;
// { profile: { avatar: { url: string } } }
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
    |
    +--- 組み合わせパターン ---+
    |                          |
    +---> PartialBy<T, K>      Omit<T, K> & Partial<Pick<T, K>>
    +---> RequiredBy<T, K>     Omit<T, K> & Required<Pick<T, K>>
    +---> ReadonlyBy<T, K>     Omit<T, K> & Readonly<Pick<T, K>>
    +---> PickByType<T, V>     値の型でフィルタリング
    +---> OmitByType<T, V>     値の型で除外
```

---

## 3. Key Remapping（キー再マッピング）

### 3.1 as 句によるキー変換

TypeScript 4.1 で導入された Key Remapping（`as` 句）を使うと、マップ型のキーを動的に変換できる。

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

// サフィックスを追加
type Suffixed<T, S extends string> = {
  [K in keyof T as `${string & K}${S}`]: T[K];
};

type UserChangedFlags = Suffixed<User, "Changed">;
// { nameChanged: string; ageChanged: number }

// getter メソッドに変換
type Getters<T> = {
  [K in keyof T as `get${Capitalize<string & K>}`]: () => T[K];
};

type UserGetters = Getters<User>;
// { getName: () => string; getAge: () => number }

// setter メソッドに変換
type Setters<T> = {
  [K in keyof T as `set${Capitalize<string & K>}`]: (value: T[K]) => void;
};

type UserSetters = Setters<User>;
// { setName: (value: string) => void; setAge: (value: number) => void }

// getter と setter を合成
type GettersAndSetters<T> = Getters<T> & Setters<T>;

type UserAccessors = GettersAndSetters<User>;
// {
//   getName: () => string;
//   getAge: () => number;
//   setName: (value: string) => void;
//   setAge: (value: number) => void;
// }
```

### 3.2 フィルタリング

```typescript
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

// 関数プロパティだけ抽出
type OnlyFunctions<T> = {
  [K in keyof T as T[K] extends (...args: any[]) => any ? K : never]: T[K];
};

interface Service {
  name: string;
  port: number;
  start: () => void;
  stop: () => void;
  getStatus: () => string;
}

type ServiceMethods = OnlyFunctions<Service>;
// { start: () => void; stop: () => void; getStatus: () => string }

// 関数以外のプロパティを抽出
type OnlyData<T> = {
  [K in keyof T as T[K] extends (...args: any[]) => any ? never : K]: T[K];
};

type ServiceData = OnlyData<Service>;
// { name: string; port: number }

// 特定のプレフィックスを持つプロパティだけ抽出
type WithPrefix<T, P extends string> = {
  [K in keyof T as K extends `${P}${string}` ? K : never]: T[K];
};

interface AppConfig {
  dbHost: string;
  dbPort: number;
  dbName: string;
  apiKey: string;
  apiUrl: string;
  logLevel: string;
}

type DbConfig = WithPrefix<AppConfig, "db">;
// { dbHost: string; dbPort: number; dbName: string }

type ApiConfig = WithPrefix<AppConfig, "api">;
// { apiKey: string; apiUrl: string }

// プレフィックスを除去してリマップ
type StripPrefix<T, P extends string> = {
  [K in keyof T as K extends `${P}${infer Rest}`
    ? Uncapitalize<Rest>
    : never]: T[K];
};

type CleanDbConfig = StripPrefix<AppConfig, "db">;
// { host: string; port: number; name: string }
```

### 3.3 イベントハンドラ生成

```typescript
// イベント名からハンドラ型を自動生成
type EventMap = {
  click: { x: number; y: number };
  focus: { target: HTMLElement };
  submit: { data: FormData };
  resize: { width: number; height: number };
  scroll: { scrollTop: number; scrollLeft: number };
};

type EventHandlers<T> = {
  [K in keyof T as `on${Capitalize<string & K>}`]: (event: T[K]) => void;
};

type Handlers = EventHandlers<EventMap>;
// {
//   onClick: (event: { x: number; y: number }) => void;
//   onFocus: (event: { target: HTMLElement }) => void;
//   onSubmit: (event: { data: FormData }) => void;
//   onResize: (event: { width: number; height: number }) => void;
//   onScroll: (event: { scrollTop: number; scrollLeft: number }) => void;
// }

// イベントリスナー追加/削除メソッドの生成
type EventMethods<T> = {
  [K in keyof T as `add${Capitalize<string & K>}Listener`]:
    (handler: (event: T[K]) => void) => void;
} & {
  [K in keyof T as `remove${Capitalize<string & K>}Listener`]:
    (handler: (event: T[K]) => void) => void;
};

type EventEmitterMethods = EventMethods<EventMap>;
// {
//   addClickListener: (handler: (event: { x: number; y: number }) => void) => void;
//   removeClickListener: (handler: (event: { x: number; y: number }) => void) => void;
//   addFocusListener: ...
//   removeFocusListener: ...
//   ...
// }

// 逆マッピング（ハンドラ名からイベント名を取得）
type HandlerToEvent<T extends string> =
  T extends `on${infer E}` ? Uncapitalize<E> : never;

type EventName = HandlerToEvent<"onClick">;  // "click"
type EventName2 = HandlerToEvent<"onResize">; // "resize"
```

### 3.4 キャメルケース変換

```typescript
// スネークケースからキャメルケースへのキー変換
type SnakeToCamel<S extends string> =
  S extends `${infer Head}_${infer Tail}`
    ? `${Head}${Capitalize<SnakeToCamel<Tail>>}`
    : S;

type CamelizeKeys<T> = {
  [K in keyof T as K extends string ? SnakeToCamel<K> : K]: T[K] extends object
    ? T[K] extends any[]
      ? T[K]
      : CamelizeKeys<T[K]>
    : T[K];
};

// APIレスポンス（スネークケース）
interface ApiResponse {
  user_id: number;
  first_name: string;
  last_name: string;
  email_address: string;
  created_at: string;
  profile_data: {
    avatar_url: string;
    display_name: string;
    bio_text: string | null;
  };
}

type CamelizedResponse = CamelizeKeys<ApiResponse>;
// {
//   userId: number;
//   firstName: string;
//   lastName: string;
//   emailAddress: string;
//   createdAt: string;
//   profileData: {
//     avatarUrl: string;
//     displayName: string;
//     bioText: string | null;
//   };
// }

// キャメルケースからスネークケースへ
type CamelToSnake<S extends string> =
  S extends `${infer First}${infer Rest}`
    ? First extends Uppercase<First>
      ? First extends Lowercase<First>
        ? `${First}${CamelToSnake<Rest>}`
        : `_${Lowercase<First>}${CamelToSnake<Rest>}`
      : `${First}${CamelToSnake<Rest>}`
    : S;

type SnakifyKeys<T> = {
  [K in keyof T as K extends string ? CamelToSnake<K> : K]: T[K] extends object
    ? T[K] extends any[]
      ? T[K]
      : SnakifyKeys<T[K]>
    : T[K];
};
```

---

## 4. 高度なマップ型

### 4.1 DeepPartial と DeepReadonly

```typescript
// DeepPartial: 再帰的に全プロパティをオプショナルにする
type DeepPartial<T> =
  T extends (...args: any[]) => any
    ? T  // 関数はそのまま
    : T extends any[]
      ? DeepPartialArray<T>
      : T extends object
        ? { [K in keyof T]?: DeepPartial<T[K]> }
        : T;

type DeepPartialArray<T extends any[]> = {
  [K in keyof T]?: DeepPartial<T[K]>;
};

interface Config {
  server: {
    host: string;
    port: number;
    ssl: {
      cert: string;
      key: string;
      passphrase: string;
    };
  };
  logging: {
    level: "debug" | "info" | "warn" | "error";
    file: string;
    format: {
      timestamp: boolean;
      colors: boolean;
    };
  };
  features: string[];
}

type PartialConfig = DeepPartial<Config>;
// 全てのネストしたプロパティもオプショナル

// 深い部分だけ更新できるマージ関数
function deepMerge<T extends object>(
  base: T,
  updates: DeepPartial<T>
): T {
  const result = { ...base } as any;
  for (const key of Object.keys(updates) as (keyof T)[]) {
    const updateValue = (updates as any)[key];
    if (
      updateValue !== undefined &&
      typeof updateValue === "object" &&
      !Array.isArray(updateValue) &&
      updateValue !== null
    ) {
      result[key] = deepMerge(result[key] as object, updateValue);
    } else if (updateValue !== undefined) {
      result[key] = updateValue;
    }
  }
  return result;
}

// 使用例
const defaultConfig: Config = {
  server: { host: "localhost", port: 3000, ssl: { cert: "", key: "", passphrase: "" } },
  logging: { level: "info", file: "app.log", format: { timestamp: true, colors: false } },
  features: ["auth", "api"],
};

const updatedConfig = deepMerge(defaultConfig, {
  server: { port: 8080 },           // port だけ変更
  logging: { level: "debug" },       // level だけ変更
});

// DeepReadonly: 再帰的に全プロパティを readonly にする
type DeepReadonly<T> =
  T extends (...args: any[]) => any
    ? T
    : T extends any[]
      ? readonly [...{ [K in keyof T]: DeepReadonly<T[K]> }]
      : T extends object
        ? { readonly [K in keyof T]: DeepReadonly<T[K]> }
        : T;

type FrozenConfig = DeepReadonly<Config>;
// 全てのネストしたプロパティが readonly
// features は readonly string[]

// DeepRequired: 再帰的に全プロパティを必須にする
type DeepRequired<T> =
  T extends (...args: any[]) => any
    ? T
    : T extends any[]
      ? T
      : T extends object
        ? { [K in keyof T]-?: DeepRequired<T[K]> }
        : T;

// DeepMutable: 再帰的に readonly を除去
type DeepMutable<T> =
  T extends (...args: any[]) => any
    ? T
    : T extends any[]
      ? { -readonly [K in keyof T]: DeepMutable<T[K]> }
      : T extends object
        ? { -readonly [K in keyof T]: DeepMutable<T[K]> }
        : T;
```

### 4.2 条件付きマッピング

```typescript
// プロパティの型に応じて異なる変換を適用
type TransformByType<T> = {
  [K in keyof T]: T[K] extends string
    ? { type: "string"; value: T[K]; maxLength: number }
    : T[K] extends number
      ? { type: "number"; value: T[K]; min: number; max: number }
      : T[K] extends boolean
        ? { type: "boolean"; value: T[K] }
        : T[K] extends any[]
          ? { type: "array"; value: T[K]; minItems: number; maxItems: number }
          : { type: "object"; value: T[K] };
};

interface Product {
  name: string;
  price: number;
  inStock: boolean;
  tags: string[];
}

type ProductSchema = TransformByType<Product>;
// {
//   name: { type: "string"; value: string; maxLength: number };
//   price: { type: "number"; value: number; min: number; max: number };
//   inStock: { type: "boolean"; value: boolean };
//   tags: { type: "array"; value: string[]; minItems: number; maxItems: number };
// }

// Nullable なプロパティだけを必須にする（null を除去）
type StrictNonNullable<T> = {
  [K in keyof T]: NonNullable<T[K]>;
};

interface ApiUser {
  name: string;
  email: string | null;
  phone: string | null | undefined;
  age: number | undefined;
}

type StrictUser = StrictNonNullable<ApiUser>;
// { name: string; email: string; phone: string; age: number }

// Nullable なプロパティだけをオプショナルにする
type NullableToOptional<T> = {
  [K in keyof T as null extends T[K] ? never : undefined extends T[K] ? never : K]: T[K];
} & {
  [K in keyof T as null extends T[K] ? K : undefined extends T[K] ? K : never]?: NonNullable<T[K]>;
};

type OptionalizedUser = NullableToOptional<ApiUser>;
// { name: string } & { email?: string; phone?: string; age?: number }
```

### 4.3 テンプレートリテラルとの組み合わせ

```typescript
// CSSプロパティから React スタイルオブジェクトを生成
type CSSProperties = {
  "background-color": string;
  "font-size": string;
  "border-radius": string;
  "margin-top": string;
  "padding-left": string;
};

// ケバブケースからキャメルケースへ
type KebabToCamel<S extends string> =
  S extends `${infer Head}-${infer Tail}`
    ? `${Head}${Capitalize<KebabToCamel<Tail>>}`
    : S;

type ReactStyle = {
  [K in keyof CSSProperties as K extends string ? KebabToCamel<K> : K]:
    CSSProperties[K];
};
// {
//   backgroundColor: string;
//   fontSize: string;
//   borderRadius: string;
//   marginTop: string;
//   paddingLeft: string;
// }

// 環境変数の型定義
type EnvVarName = "DATABASE_URL" | "API_KEY" | "PORT" | "NODE_ENV";

// process.env の型安全なアクセサー
type EnvAccessors = {
  [K in EnvVarName as `get${SnakeToPascal<K>}`]: () => string;
};

type SnakeToPascal<S extends string> =
  S extends `${infer Head}_${infer Tail}`
    ? `${Capitalize<Lowercase<Head>>}${SnakeToPascal<Tail>}`
    : Capitalize<Lowercase<S>>;

// EnvAccessors:
// {
//   getDatabaseUrl: () => string;
//   getApiKey: () => string;
//   getPort: () => string;
//   getNodeEnv: () => string;
// }
```

### 4.4 実践的なマップ型の組み合わせ

```typescript
// フォームの状態型を自動生成
interface FormFields {
  username: string;
  email: string;
  age: number;
  bio: string;
  agreeToTerms: boolean;
}

// 各フィールドの状態
type FieldState<T> = {
  value: T;
  error: string | null;
  touched: boolean;
  dirty: boolean;
  validating: boolean;
};

// フォーム全体の状態
type FormState<T> = {
  fields: {
    [K in keyof T]: FieldState<T[K]>;
  };
  isValid: boolean;
  isSubmitting: boolean;
  submitCount: number;
};

type MyFormState = FormState<FormFields>;
// {
//   fields: {
//     username: FieldState<string>;
//     email: FieldState<string>;
//     age: FieldState<number>;
//     bio: FieldState<string>;
//     agreeToTerms: FieldState<boolean>;
//   };
//   isValid: boolean;
//   isSubmitting: boolean;
//   submitCount: number;
// }

// フォームのバリデーションルール型
type ValidationRule<T> = {
  validate: (value: T) => boolean;
  message: string;
};

type FormValidation<T> = {
  [K in keyof T]?: ValidationRule<T[K]>[];
};

const validationRules: FormValidation<FormFields> = {
  username: [
    { validate: (v) => v.length >= 3, message: "3文字以上必要です" },
    { validate: (v) => /^[a-zA-Z0-9]+$/.test(v), message: "英数字のみ使用できます" },
  ],
  email: [
    { validate: (v) => v.includes("@"), message: "有効なメールアドレスを入力してください" },
  ],
  age: [
    { validate: (v) => v >= 0, message: "0以上の数値を入力してください" },
    { validate: (v) => v <= 150, message: "150以下の数値を入力してください" },
  ],
  agreeToTerms: [
    { validate: (v) => v === true, message: "利用規約に同意してください" },
  ],
};

// フォームアクション型の自動生成
type FormActions<T> = {
  [K in keyof T as `set${Capitalize<string & K>}`]: (value: T[K]) => void;
} & {
  [K in keyof T as `validate${Capitalize<string & K>}`]: () => boolean;
} & {
  [K in keyof T as `reset${Capitalize<string & K>}`]: () => void;
} & {
  submit: () => Promise<void>;
  reset: () => void;
  validateAll: () => boolean;
};

type MyFormActions = FormActions<FormFields>;
// {
//   setUsername: (value: string) => void;
//   setEmail: (value: string) => void;
//   setAge: (value: number) => void;
//   ...
//   validateUsername: () => boolean;
//   validateEmail: () => boolean;
//   ...
//   resetUsername: () => void;
//   ...
//   submit: () => Promise<void>;
//   reset: () => void;
//   validateAll: () => boolean;
// }
```

---

## 5. 実務パターン

### 5.1 API クライアントの型定義

```typescript
// RESTful API のエンドポイント定義
interface ApiEndpoints {
  "/users": {
    GET: { response: User[]; query: { page: number; limit: number } };
    POST: { response: User; body: CreateUser };
  };
  "/users/:id": {
    GET: { response: User; params: { id: string } };
    PUT: { response: User; body: UpdateUser; params: { id: string } };
    DELETE: { response: void; params: { id: string } };
  };
  "/posts": {
    GET: { response: Post[]; query: { page: number; limit: number; userId?: string } };
    POST: { response: Post; body: CreatePost };
  };
}

type HttpMethod = "GET" | "POST" | "PUT" | "DELETE";

// エンドポイントからリクエスト型を生成
type RequestConfig<
  Path extends keyof ApiEndpoints,
  Method extends keyof ApiEndpoints[Path]
> = ApiEndpoints[Path][Method] extends { body: infer B }
  ? { body: B }
  : {} & (ApiEndpoints[Path][Method] extends { query: infer Q }
    ? { query: Q }
    : {}) & (ApiEndpoints[Path][Method] extends { params: infer P }
    ? { params: P }
    : {});

// エンドポイントからレスポンス型を取得
type ResponseType<
  Path extends keyof ApiEndpoints,
  Method extends keyof ApiEndpoints[Path]
> = ApiEndpoints[Path][Method] extends { response: infer R }
  ? R
  : never;

// 型安全な API クライアント
class ApiClient {
  async request<
    Path extends keyof ApiEndpoints,
    Method extends keyof ApiEndpoints[Path] & string
  >(
    method: Method,
    path: Path,
    config?: RequestConfig<Path, Method>
  ): Promise<ResponseType<Path, Method>> {
    // 実装...
    return {} as ResponseType<Path, Method>;
  }
}

// 使用例
const api = new ApiClient();

// 型安全に API を呼び出せる
const users = await api.request("GET", "/users", {
  query: { page: 1, limit: 10 },
});
// users は User[] 型

const newUser = await api.request("POST", "/users", {
  body: { name: "Alice", email: "alice@example.com", password: "secret" },
});
// newUser は User 型
```

### 5.2 状態管理パターン

```typescript
// Zustand 風の型安全なストア定義
type StoreDefinition<T extends object> = {
  state: T;
  actions: {
    [K in keyof T as `set${Capitalize<string & K>}`]: (value: T[K]) => void;
  } & {
    reset: () => void;
  };
  selectors: {
    [K in keyof T as `select${Capitalize<string & K>}`]: () => T[K];
  };
  computed: Record<string, (...args: any[]) => any>;
};

interface AppState {
  user: User | null;
  theme: "light" | "dark";
  language: string;
  notifications: Notification[];
}

type AppStore = StoreDefinition<AppState>;
// {
//   state: AppState;
//   actions: {
//     setUser: (value: User | null) => void;
//     setTheme: (value: "light" | "dark") => void;
//     setLanguage: (value: string) => void;
//     setNotifications: (value: Notification[]) => void;
//     reset: () => void;
//   };
//   selectors: {
//     selectUser: () => User | null;
//     selectTheme: () => "light" | "dark";
//     selectLanguage: () => string;
//     selectNotifications: () => Notification[];
//   };
//   computed: Record<string, (...args: any[]) => any>;
// }

// Immer 風の不変更新パターン
type DraftState<T> = {
  -readonly [K in keyof T]: T[K] extends object
    ? T[K] extends (...args: any[]) => any
      ? T[K]
      : DraftState<T[K]>
    : T[K];
};

function produce<T extends object>(
  state: T,
  recipe: (draft: DraftState<T>) => void
): T {
  // 実装...
  return state;
}
```

### 5.3 データベースモデル型の自動生成

```typescript
// モデル定義からCRUD操作の型を自動生成

interface ModelDefinition {
  User: {
    id: number;
    name: string;
    email: string;
    createdAt: Date;
    updatedAt: Date;
  };
  Post: {
    id: number;
    title: string;
    content: string;
    authorId: number;
    published: boolean;
    createdAt: Date;
    updatedAt: Date;
  };
  Comment: {
    id: number;
    text: string;
    postId: number;
    authorId: number;
    createdAt: Date;
  };
}

// 自動生成されるフィールドを除いた作成用の型
type AutoFields = "id" | "createdAt" | "updatedAt";

type CreateInput<T> = Omit<T, AutoFields>;
type UpdateInput<T> = Partial<Omit<T, AutoFields>>;

// モデルごとのリポジトリインターフェース
type Repository<T> = {
  findById: (id: number) => Promise<T | null>;
  findMany: (where?: Partial<T>) => Promise<T[]>;
  create: (data: CreateInput<T>) => Promise<T>;
  update: (id: number, data: UpdateInput<T>) => Promise<T>;
  delete: (id: number) => Promise<boolean>;
  count: (where?: Partial<T>) => Promise<number>;
};

// 全モデルのリポジトリを一括生成
type Repositories = {
  [K in keyof ModelDefinition as Uncapitalize<string & K>]:
    Repository<ModelDefinition[K]>;
};

// Repositories:
// {
//   user: Repository<ModelDefinition["User"]>;
//   post: Repository<ModelDefinition["Post"]>;
//   comment: Repository<ModelDefinition["Comment"]>;
// }

// 使用例
declare const repos: Repositories;

const user = await repos.user.findById(1);          // User | null
const posts = await repos.post.findMany({ published: true }); // Post[]
const newComment = await repos.comment.create({
  text: "Great post!",
  postId: 1,
  authorId: 1,
});  // Comment
```

### 5.4 国際化（i18n）の型安全な設計

```typescript
// 翻訳キーの型安全な管理
interface Translations {
  common: {
    save: string;
    cancel: string;
    delete: string;
    confirm: string;
  };
  auth: {
    login: string;
    logout: string;
    register: string;
    forgotPassword: string;
  };
  errors: {
    notFound: string;
    unauthorized: string;
    serverError: string;
    validation: {
      required: string;
      minLength: string;
      maxLength: string;
      email: string;
    };
  };
}

// ドットパスで翻訳キーを生成
type DotPath<T, Prefix extends string = ""> =
  T extends string
    ? Prefix
    : {
        [K in keyof T & string]:
          DotPath<T[K], Prefix extends "" ? K : `${Prefix}.${K}`>
      }[keyof T & string];

type TranslationKey = DotPath<Translations>;
// "common.save" | "common.cancel" | "common.delete" | "common.confirm"
// | "auth.login" | "auth.logout" | ...
// | "errors.validation.required" | "errors.validation.minLength" | ...

// 型安全な翻訳関数
function t(key: TranslationKey): string {
  // 実装...
  return "";
}

// 使用例
t("common.save");        // OK
t("auth.login");         // OK
// t("invalid.key");     // コンパイルエラー
// t("common");          // コンパイルエラー（リーフノードのみ許可）
```

### 5.5 テスト用のモック型生成

```typescript
// インターフェースから自動でモック型を生成

// 全メソッドを jest.Mock に変換
type MockedMethods<T> = {
  [K in keyof T as T[K] extends (...args: any[]) => any ? K : never]:
    T[K] extends (...args: infer A) => infer R
      ? jest.Mock<R, A>
      : never;
};

// プロパティはそのまま、メソッドだけモック化
type Mocked<T> = {
  [K in keyof T]: T[K] extends (...args: any[]) => any
    ? T[K] extends (...args: infer A) => infer R
      ? jest.Mock<R, A>
      : T[K]
    : T[K];
};

interface UserService {
  name: string;
  getUser(id: number): Promise<User>;
  createUser(data: CreateUser): Promise<User>;
  deleteUser(id: number): Promise<void>;
  getUserCount(): number;
}

type MockedUserService = Mocked<UserService>;
// {
//   name: string;
//   getUser: jest.Mock<Promise<User>, [id: number]>;
//   createUser: jest.Mock<Promise<User>, [data: CreateUser]>;
//   deleteUser: jest.Mock<Promise<void>, [id: number]>;
//   getUserCount: jest.Mock<number, []>;
// }

// モックファクトリ関数
function createMock<T extends object>(): Mocked<T> {
  return new Proxy({} as Mocked<T>, {
    get: (target, prop) => {
      if (!(prop in target)) {
        (target as any)[prop] = jest.fn();
      }
      return (target as any)[prop];
    },
  });
}

// 使用例
const mockService = createMock<UserService>();
mockService.getUser.mockResolvedValue({ id: 1, name: "Alice" } as User);
```

---

## 6. パフォーマンスとデバッグ

### 6.1 コンパイル性能の最適化

```typescript
// マップ型のパフォーマンスに関する注意点

// BAD: 不必要な再帰
type BadDeepReadonly<T> = {
  readonly [K in keyof T]: T[K] extends object ? BadDeepReadonly<T[K]> : T[K];
};
// 配列やDate、Functionも再帰してしまう

// GOOD: 適切な条件分岐で再帰を制限
type GoodDeepReadonly<T> =
  T extends Function ? T :
  T extends Date ? T :
  T extends RegExp ? T :
  T extends Map<infer K, infer V> ? ReadonlyMap<K, V> :
  T extends Set<infer U> ? ReadonlySet<U> :
  T extends any[] ? readonly [...{ [K in keyof T]: GoodDeepReadonly<T[K]> }] :
  T extends object ? { readonly [K in keyof T]: GoodDeepReadonly<T[K]> } :
  T;

// BAD: 大量のキーでの Key Remapping
// 数百のプロパティを持つ型に対してテンプレートリテラルでリマップ
// → コンパイル時間が大幅に増加

// GOOD: 必要なプロパティだけを処理
type SelectiveRemap<T, Keys extends keyof T> = {
  [K in Keys as `get${Capitalize<string & K>}`]: () => T[K];
} & Omit<T, Keys>;

// BAD: マップ型の重複適用
type Redundant<T> = Readonly<Partial<Required<Partial<T>>>>;
// Partial の後に Required して再び Partial... 無意味

// GOOD: 1回の変換で済ませる
type Clean<T> = { readonly [K in keyof T]?: T[K] };
```

### 6.2 デバッグテクニック

```typescript
// テクニック1: Prettify で型を展開して確認
type Prettify<T> = {
  [K in keyof T]: T[K];
} & {};

// 型をフラットに展開してホバーで確認しやすくする
type ComplexType = Omit<User, "password"> & Partial<Pick<User, "email">>;
type PrettifiedType = Prettify<ComplexType>;
// エディタでホバーすると展開された型が見える

// テクニック2: 型の等値テスト
type Equal<X, Y> =
  (<T>() => T extends X ? 1 : 2) extends
  (<T>() => T extends Y ? 1 : 2) ? true : false;

type Expect<T extends true> = T;

// テストケースの記述
type TestPartial = Expect<Equal<
  MyPartial<{ a: string; b: number }>,
  { a?: string; b?: number }
>>;

// テクニック3: 型エラーの可視化
type ShowError<T, Message extends string> = T & { __error: Message };

// テクニック4: 段階的な型の構築
type Step1 = keyof User;                      // "id" | "name" | "email" | ...
type Step2 = Exclude<Step1, "password">;      // "id" | "name" | "email" | "createdAt" | "updatedAt"
type Step3 = Pick<User, Step2>;               // パスワード以外のUser
type Step4 = Prettify<Step3>;                 // 展開して確認

// テクニック5: ブランド型でのデバッグ
type Debug<T, Label extends string = ""> = T & { __debug: Label; __type: T };
```

### 6.3 よくあるエラーと対処法

```typescript
// エラー1: "Type 'string' cannot be used to index type 'T'"
// 原因: keyof T は string | number | symbol を返す可能性がある
// 解決: & string で制限する
type BadGetters<T> = {
  // [K in keyof T as `get${Capitalize<K>}`]: () => T[K]; // エラー
  [K in keyof T as K extends string ? `get${Capitalize<K>}` : never]: () => T[K]; // OK
};

// エラー2: "Type instantiation is excessively deep and possibly infinite"
// 原因: 再帰が深すぎる
// 解決: 深さカウンターを追加
type SafeDeep<T, Depth extends any[] = []> =
  Depth["length"] extends 10
    ? T  // 深さ制限で停止
    : T extends object
      ? { [K in keyof T]: SafeDeep<T[K], [...Depth, 0]> }
      : T;

// エラー3: "Expression produces a union type that is too complex to represent"
// 原因: マップ型の結果が巨大なユニオンになる
// 解決: 処理対象を限定する

// エラー4: 修飾子の意図しない伝播
interface WithOptional {
  required: string;
  optional?: number;
}

type Mapped = { [K in keyof WithOptional]: string };
// { required: string; optional?: string }
// ← optional の ? が保持される！

// 解決: 明示的に必須にする
type MappedRequired = { [K in keyof WithOptional]-?: string };
// { required: string; optional: string }

// エラー5: readonly の意図しない伝播
interface WithReadonly {
  readonly fixed: string;
  mutable: number;
}

type MappedRO = { [K in keyof WithReadonly]: boolean };
// { readonly fixed: boolean; mutable: boolean }
// ← readonly が保持される！

// 解決: 明示的に除去する
type MappedMutable = { -readonly [K in keyof WithReadonly]: boolean };
// { fixed: boolean; mutable: boolean }
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

### カスタムユーティリティ型早見表

| 型 | 説明 | 用途 |
|----|------|------|
| `DeepPartial<T>` | 再帰的に全プロパティをオプショナル | 設定のマージ |
| `DeepReadonly<T>` | 再帰的に全プロパティを readonly | 不変データ構造 |
| `DeepRequired<T>` | 再帰的に全プロパティを必須 | バリデーション後 |
| `DeepMutable<T>` | 再帰的に readonly を除去 | Draft パターン |
| `PartialBy<T, K>` | 特定プロパティだけオプショナル | 部分的な省略 |
| `RequiredBy<T, K>` | 特定プロパティだけ必須 | 部分的な必須化 |
| `PickByType<T, V>` | 特定の型のプロパティを抽出 | 型フィルタリング |
| `OmitByType<T, V>` | 特定の型のプロパティを除外 | 型フィルタリング |
| `Prettify<T>` | 型をフラットに展開 | デバッグ |
| `Mocked<T>` | メソッドをモックに変換 | テスト |

---

## 修飾子操作比較

| 操作 | 構文 | 効果 | 例 |
|------|------|------|-----|
| optional追加 | `[K in keyof T]?:` | `?` 追加 | `Partial<T>` |
| optional除去 | `[K in keyof T]-?:` | `?` 除去 | `Required<T>` |
| readonly追加 | `readonly [K in keyof T]:` | `readonly` 追加 | `Readonly<T>` |
| readonly除去 | `-readonly [K in keyof T]:` | `readonly` 除去 | `Mutable<T>` |
| 両方追加 | `readonly [K in keyof T]?:` | 両方追加 | `ReadonlyPartial<T>` |
| 両方除去 | `-readonly [K in keyof T]-?:` | 両方除去 | `MutableRequired<T>` |

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

// BAD: 単なるコピー
type Copy<T> = { [K in keyof T]: T[K] };
// T と同じ型を返すだけ（Prettify 目的を除く）

// GOOD: 変換が伴う場合にのみマップ型を使用
type Nullable<T> = { [K in keyof T]: T[K] | null };
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

// GOOD: 入力の境界で不変性を保証
function processData(data: Readonly<InputData>): OutputData {
  // data を変更しない保証
  return transform(data);
}
```

### アンチパターン3: 過度な型レベルプログラミング

```typescript
// BAD: ランタイムで簡単にできることを型レベルで複雑に実装
type TypeLevelSort<T extends number[]> = /* 非常に複雑な型 */;

// GOOD: ランタイムで処理し、型は結果だけ保証
function sortNumbers<T extends number[]>(arr: T): number[] {
  return [...arr].sort((a, b) => a - b);
}

// BAD: 型推論に頼りすぎてコードが読めない
type AbstractFactory<
  T extends Record<string, new (...args: any[]) => any>,
  K extends keyof T = keyof T
> = {
  [P in K as `create${Capitalize<string & P>}`]:
    (...args: ConstructorParameters<T[P]>) => InstanceType<T[P]>;
};

// GOOD: 必要な抽象化レベルに留める
interface Factory {
  createUser(name: string, age: number): User;
  createPost(title: string, content: string): Post;
}
```

### アンチパターン4: Key Remapping の乱用

```typescript
// BAD: 1つの型で多数の変換を行う
type EverythingAtOnce<T> = {
  [K in keyof T as
    T[K] extends string
      ? `str_${string & K}`
      : T[K] extends number
        ? `num_${string & K}`
        : T[K] extends boolean
          ? `bool_${string & K}`
          : `other_${string & K}`
  ]: T[K] extends string
    ? { type: "string"; value: T[K] }
    : T[K] extends number
      ? { type: "number"; value: T[K] }
      : { type: "other"; value: T[K] };
};

// GOOD: 分割して名前をつける
type PrefixByType<T> = {
  [K in keyof T as `${TypePrefix<T[K]>}_${string & K}`]: T[K];
};

type TypePrefix<T> =
  T extends string ? "str" :
  T extends number ? "num" :
  T extends boolean ? "bool" :
  "other";
```

---

## FAQ

### Q1: `keyof T` と `keyof T & string` の違いは？

**A:** `keyof T` は `string | number | symbol` のいずれかを返す可能性があります。テンプレートリテラル型で `${K}` のように文字列として使う場合、`K` が `string` でなければならないため、`keyof T & string` として文字列キーのみに限定します。

```typescript
interface Example {
  name: string;
  0: number;
}

type AllKeys = keyof Example;            // "name" | 0
type StringKeys = keyof Example & string; // "name"
```

### Q2: マップ型で一部のプロパティだけ変換することはできますか？

**A:** はい、`Pick` と `Omit` を組み合わせるか、Key Remapping の `as` 句で条件分岐します。

```typescript
// 方法1: Pick + Omit の組み合わせ
type PartialBy<T, K extends keyof T> =
  Omit<T, K> & Partial<Pick<T, K>>;

// 方法2: Key Remapping + 条件型
type SelectivePartial<T, K extends keyof T> = {
  [P in keyof T as P extends K ? P : never]?: T[P];
} & {
  [P in keyof T as P extends K ? never : P]: T[P];
};
```

### Q3: `Record<string, T>` と `{ [key: string]: T }` は同じですか？

**A:** ほぼ同じですが、微妙な違いがあります。`Record<string, T>` はマップ型として定義されており、エディタでの型表示が異なります。また、`Record` はジェネリクスとの相互作用が良い場合があります。実用上は同じ意味で使えます。

```typescript
// これらはほぼ同等
type A = Record<string, number>;
type B = { [key: string]: number };

// ただし Record<string, T> は string | number キーの両方を受け付ける
// （JavaScript のインデックスシグネチャの仕様による）
const a: A = { foo: 1 };
a[0] = 2;  // OK（number キーも使える）
```

### Q4: マップ型で配列やタプルを扱えますか？

**A:** はい、マップ型は配列やタプルにも適用できます。配列/タプルに対してマップ型を使うと、要素の型が変換されます。

```typescript
type MapArray<T extends any[]> = {
  [K in keyof T]: T[K] extends string ? number : T[K];
};

type Result = MapArray<[string, number, string]>;
// [number, number, number]

// ただし、配列のメソッド（push, pop など）にも適用されるので注意
// 通常はジェネリック制約を使って要素のみを変換する
```

### Q5: マップ型の結果で修飾子（optional, readonly）が意図しない形になるのはなぜ？

**A:** マップ型は元の型のプロパティの修飾子をそのまま引き継ぎます。これは「同型写像（homomorphic mapped type）」と呼ばれる挙動です。修飾子を変更したい場合は、`?`, `-?`, `readonly`, `-readonly` を明示的に使用してください。

```typescript
interface Original {
  readonly a: string;
  b?: number;
}

// 修飾子がそのまま引き継がれる
type Inherited = { [K in keyof Original]: boolean };
// { readonly a: boolean; b?: boolean }

// 修飾子を明示的に制御
type Normalized = { -readonly [K in keyof Original]-?: boolean };
// { a: boolean; b: boolean }
```

### Q6: Omit<T, K> と Pick<T, Exclude<keyof T, K>> の違いは？

**A:** 実装上は同じです。`Omit` は `Pick` と `Exclude` の組み合わせで定義されています。ただし、`Omit` の `K` は `keyof any`（= `string | number | symbol`）制約なので、`T` に存在しないキーを指定してもエラーになりません。一方、`Pick` の `K` は `keyof T` 制約なので、存在しないキーを指定するとエラーになります。

```typescript
interface User {
  name: string;
  age: number;
}

type A = Omit<User, "nonexistent">;       // OK（User と同じ型）
// type B = Pick<User, "nonexistent">;     // エラー: "nonexistent" は keyof User にない
```

---

## まとめ

| 項目 | 内容 |
|------|------|
| マップ型 | `{ [K in keyof T]: ... }` で型のプロパティを変換 |
| 修飾子 | `?`, `readonly` の追加・除去（`-?`, `-readonly`） |
| Key Remapping | `as` 句でキー名を変換・フィルタリング（TS 4.1+） |
| 組み込み型 | Partial, Required, Readonly, Record, Pick, Omit |
| 再帰マップ型 | DeepPartial, DeepReadonly などネスト構造に対応 |
| 条件付きマッピング | プロパティの型に応じた異なる変換 |
| 実用パターン | フォーム状態、API型変換、モック生成、i18n |
| パフォーマンス | 再帰深度制限、不要な再帰の回避 |
| デバッグ | Prettify, 段階的構築, 型テスト |

---

## 次に読むべきガイド

- [02-template-literal-types.md](./02-template-literal-types.md) -- テンプレートリテラル型
- [03-type-challenges.md](./03-type-challenges.md) -- 型チャレンジ
- [00-conditional-types.md](./00-conditional-types.md) -- 条件型

---

## 参考文献

1. **TypeScript Handbook: Mapped Types** -- https://www.typescriptlang.org/docs/handbook/2/mapped-types.html
2. **TypeScript Handbook: Utility Types** -- https://www.typescriptlang.org/docs/handbook/utility-types.html
3. **Effective TypeScript, Item 14: Use Type Operations and Generics to Avoid Repeating Yourself** -- Dan Vanderkam著, O'Reilly
4. **TypeScript 4.1 Release Notes: Key Remapping** -- https://www.typescriptlang.org/docs/handbook/release-notes/typescript-4-1.html
5. **Type-Level TypeScript** -- https://type-level-typescript.com/
6. **Total TypeScript: Mapped Types** -- https://www.totaltypescript.com/
