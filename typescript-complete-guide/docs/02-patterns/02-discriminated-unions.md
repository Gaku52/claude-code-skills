# TypeScript 判別共用体パターン

> 判別共用体（Discriminated Unions）、Redux の型安全な Action 設計、網羅性チェックで堅牢な分岐処理を実現する

## この章で学ぶこと

1. **判別共用体の基礎** -- リテラル型の判別子フィールドを使って、ユニオン型のメンバーを安全に絞り込む方法
2. **Redux / useReducer での活用** -- Action 型を判別共用体で定義し、Reducer を型安全に記述するパターン
3. **網羅性チェック** -- `never` 型を活用し、switch 文で全ケースを処理していることをコンパイル時に保証する技法

---

## 1. 判別共用体の基礎

### 1-1. 構造

```
判別共用体の構成要素:

  共通フィールド (判別子)
       |
       v
  +----------+     +----------+     +----------+
  | type:     |     | type:     |     | type:     |
  | "circle"  |     | "rect"   |     | "triangle"|
  +----------+     +----------+     +----------+
  | radius    |     | width    |     | base     |
  |           |     | height   |     | height   |
  +----------+     +----------+     +----------+

  Shape = Circle | Rect | Triangle
             \       |       /
              判別子: type フィールド
```

```typescript
// 判別共用体の定義
interface Circle {
  readonly type: "circle";
  readonly radius: number;
}

interface Rect {
  readonly type: "rect";
  readonly width: number;
  readonly height: number;
}

interface Triangle {
  readonly type: "triangle";
  readonly base: number;
  readonly height: number;
}

type Shape = Circle | Rect | Triangle;

// 判別子による型の絞り込み
function area(shape: Shape): number {
  switch (shape.type) {
    case "circle":
      // shape は Circle に絞り込まれる
      return Math.PI * shape.radius ** 2;
    case "rect":
      // shape は Rect に絞り込まれる
      return shape.width * shape.height;
    case "triangle":
      // shape は Triangle に絞り込まれる
      return (shape.base * shape.height) / 2;
  }
}
```

### 1-2. コンストラクタ関数

```typescript
// スマートコンストラクタで安全に生成
function circle(radius: number): Circle {
  if (radius <= 0) throw new Error("radius must be positive");
  return { type: "circle", radius };
}

function rect(width: number, height: number): Rect {
  return { type: "rect", width, height };
}

function triangle(base: number, height: number): Triangle {
  return { type: "triangle", base, height };
}

// 使用例
const shapes: Shape[] = [
  circle(5),
  rect(10, 20),
  triangle(6, 8),
];

const totalArea = shapes.reduce((sum, s) => sum + area(s), 0);
```

---

## 2. 網羅性チェック

### 2-1. exhaustive check パターン

```
Switch 文の網羅性:

  case "circle":  --> Circle を処理
  case "rect":    --> Rect を処理
  case "triangle" --> Triangle を処理
  default:        --> shape は never 型
                      (全ケースを処理済みの証明)

  もし新しい Shape を追加して case を書き忘れると:
  default:        --> shape は新しい型
                      never に代入不可 → コンパイルエラー!
```

```typescript
// 網羅性チェック用ヘルパー
function assertNever(value: never): never {
  throw new Error(`Unexpected value: ${JSON.stringify(value)}`);
}

function area(shape: Shape): number {
  switch (shape.type) {
    case "circle":
      return Math.PI * shape.radius ** 2;
    case "rect":
      return shape.width * shape.height;
    case "triangle":
      return (shape.base * shape.height) / 2;
    default:
      // 全ケースを処理していれば shape は never 型
      return assertNever(shape);
  }
}

// 新しい Shape を追加した場合:
interface Pentagon {
  readonly type: "pentagon";
  readonly side: number;
}
type Shape = Circle | Rect | Triangle | Pentagon;

// area() で case "pentagon" を追加し忘れると:
// Error: Argument of type 'Pentagon' is not assignable to parameter of type 'never'
```

### 2-2. satisfies を使った網羅性チェック

```typescript
// オブジェクトマップによる網羅性チェック
const areaCalculators = {
  circle: (s: Circle) => Math.PI * s.radius ** 2,
  rect: (s: Rect) => s.width * s.height,
  triangle: (s: Triangle) => (s.base * s.height) / 2,
} satisfies Record<Shape["type"], (s: any) => number>;

function area(shape: Shape): number {
  return areaCalculators[shape.type](shape as any);
}

// Pentagon を追加すると、satisfies で
// "pentagon" キーがないことがコンパイルエラーになる
```

---

## 3. Redux / useReducer での活用

### 3-1. Action の型設計

```
                   dispatch(action)
                        |
                        v
+---------------------------------------------------+
|                    Reducer                          |
|                                                     |
|  switch (action.type) {                            |
|    case "ADD_TODO":                                |
|      action.payload  → { text: string }            |
|    case "TOGGLE_TODO":                             |
|      action.payload  → { id: number }              |
|    case "DELETE_TODO":                             |
|      action.payload  → { id: number }              |
|    case "SET_FILTER":                              |
|      action.payload  → { filter: FilterType }      |
|    default:                                        |
|      assertNever(action)  → 網羅性チェック          |
|  }                                                 |
+---------------------------------------------------+
```

```typescript
// State
interface TodoState {
  readonly todos: readonly Todo[];
  readonly filter: "all" | "active" | "completed";
}

interface Todo {
  readonly id: number;
  readonly text: string;
  readonly completed: boolean;
}

// Action -- 判別共用体
type TodoAction =
  | { readonly type: "ADD_TODO"; readonly payload: { text: string } }
  | { readonly type: "TOGGLE_TODO"; readonly payload: { id: number } }
  | { readonly type: "DELETE_TODO"; readonly payload: { id: number } }
  | { readonly type: "SET_FILTER"; readonly payload: { filter: TodoState["filter"] } };

// Reducer
function todoReducer(state: TodoState, action: TodoAction): TodoState {
  switch (action.type) {
    case "ADD_TODO":
      return {
        ...state,
        todos: [
          ...state.todos,
          {
            id: Date.now(),
            text: action.payload.text, // { text: string } に絞り込み
            completed: false,
          },
        ],
      };

    case "TOGGLE_TODO":
      return {
        ...state,
        todos: state.todos.map((todo) =>
          todo.id === action.payload.id // { id: number } に絞り込み
            ? { ...todo, completed: !todo.completed }
            : todo
        ),
      };

    case "DELETE_TODO":
      return {
        ...state,
        todos: state.todos.filter(
          (todo) => todo.id !== action.payload.id
        ),
      };

    case "SET_FILTER":
      return {
        ...state,
        filter: action.payload.filter, // { filter: ... } に絞り込み
      };

    default:
      return assertNever(action);
  }
}
```

### 3-2. Action Creator の型安全な定義

```typescript
// Action Creator 型を自動生成
type ActionCreator<A extends { type: string }> = {
  [T in A["type"]]: (
    payload: Extract<A, { type: T }> extends { payload: infer P }
      ? P
      : never
  ) => Extract<A, { type: T }>;
};

// 実装
const todoActions: ActionCreator<TodoAction> = {
  ADD_TODO: (payload) => ({ type: "ADD_TODO", payload }),
  TOGGLE_TODO: (payload) => ({ type: "TOGGLE_TODO", payload }),
  DELETE_TODO: (payload) => ({ type: "DELETE_TODO", payload }),
  SET_FILTER: (payload) => ({ type: "SET_FILTER", payload }),
};

// 使用例 -- payload の型が自動推論される
const action = todoActions.ADD_TODO({ text: "Learn TypeScript" });
// 型: { type: "ADD_TODO"; payload: { text: string } }
```

---

## 4. 高度なパターン

### 4-1. ネストした判別共用体

```typescript
// API レスポンスの型
type ApiResponse<T> =
  | { status: "loading" }
  | { status: "success"; data: T }
  | { status: "error"; error: ApiError };

type ApiError =
  | { code: "NETWORK"; message: string; retryable: true }
  | { code: "AUTH"; message: string; retryable: false }
  | { code: "VALIDATION"; message: string; fields: string[] };

// ネストした判別
function handleResponse<T>(response: ApiResponse<T>): string {
  switch (response.status) {
    case "loading":
      return "Loading...";
    case "success":
      return `Data: ${JSON.stringify(response.data)}`;
    case "error":
      switch (response.error.code) {
        case "NETWORK":
          return `Network error (retryable): ${response.error.message}`;
        case "AUTH":
          return "Please login again";
        case "VALIDATION":
          return `Invalid fields: ${response.error.fields.join(", ")}`;
        default:
          return assertNever(response.error);
      }
    default:
      return assertNever(response);
  }
}
```

### 4-2. 判別共用体とジェネリクスの組み合わせ

```typescript
// イベントシステム
type AppEvent =
  | { kind: "user.created"; payload: { userId: string; name: string } }
  | { kind: "user.deleted"; payload: { userId: string } }
  | { kind: "order.placed"; payload: { orderId: string; total: number } }
  | { kind: "order.shipped"; payload: { orderId: string; trackingId: string } };

// イベントハンドラーの型安全な登録
type EventHandler<E extends AppEvent["kind"]> = (
  payload: Extract<AppEvent, { kind: E }>["payload"]
) => void;

class EventBus {
  private handlers = new Map<string, Set<Function>>();

  on<E extends AppEvent["kind"]>(
    event: E,
    handler: EventHandler<E>
  ): void {
    if (!this.handlers.has(event)) {
      this.handlers.set(event, new Set());
    }
    this.handlers.get(event)!.add(handler);
  }

  emit<E extends AppEvent["kind"]>(
    event: E,
    payload: Extract<AppEvent, { kind: E }>["payload"]
  ): void {
    this.handlers.get(event)?.forEach((handler) => handler(payload));
  }
}

// 使用例
const bus = new EventBus();
bus.on("user.created", (payload) => {
  // payload は { userId: string; name: string } に推論
  console.log(`User ${payload.name} created`);
});

bus.emit("order.placed", { orderId: "123", total: 9800 });
// bus.emit("order.placed", { orderId: "123" }); // エラー: total が必要
```

---

## 比較表

### 型の絞り込み手法の比較

| 手法 | 安全性 | パフォーマンス | 拡張性 | 網羅性チェック |
|------|--------|-------------|--------|-------------|
| 判別共用体 (switch) | 高 | 最高 | 高 | `never` で可能 |
| instanceof | 中 | 高 | 低 | 不完全 |
| in 演算子 | 低 | 高 | 低 | 不可 |
| 型述語 (is) | 高 | 中 | 中 | 不可 |
| zod discriminatedUnion | 最高 | 中 | 高 | ランタイムで可能 |

### 判別子の選択肢

| 判別子 | 例 | 利点 | 注意点 |
|--------|-----|------|--------|
| 文字列リテラル | `type: "circle"` | 最も一般的、可読性が高い | typo のリスク |
| 数値リテラル | `kind: 0 \| 1 \| 2` | switch 最適化 | 可読性が低い |
| enum | `Action.Add` | IDE 補完が優秀 | Tree-shaking 問題 |
| const enum | `const enum` | ランタイムコストゼロ | isolatedModules 非対応 |
| Symbol | `Symbol("circle")` | 一意性保証 | シリアライズ不可 |

---

## アンチパターン

### AP-1: 判別子のないユニオン型

```typescript
// NG: 判別子がなく、絞り込みが困難
type Shape =
  | { radius: number }
  | { width: number; height: number }
  | { base: number; height: number }; // height が重複!

function area(shape: Shape): number {
  if ("radius" in shape) {
    return Math.PI * shape.radius ** 2;
  }
  // width があるかないかでしか区別できない
  if ("width" in shape) {
    return shape.width * shape.height;
  }
  return shape.base * shape.height; // 本当に Triangle?
}

// OK: 判別子を追加
type Shape =
  | { type: "circle"; radius: number }
  | { type: "rect"; width: number; height: number }
  | { type: "triangle"; base: number; height: number };
```

### AP-2: default で握りつぶす

```typescript
// NG: default で新しいケースを見逃す
function describe(shape: Shape): string {
  switch (shape.type) {
    case "circle":
      return "A circle";
    case "rect":
      return "A rectangle";
    default:
      return "Unknown shape"; // Triangle を見逃し、将来の型追加も見逃す
  }
}

// OK: assertNever で網羅性を保証
function describe(shape: Shape): string {
  switch (shape.type) {
    case "circle":
      return "A circle";
    case "rect":
      return "A rectangle";
    case "triangle":
      return "A triangle";
    default:
      return assertNever(shape); // 新しい型追加時にコンパイルエラー
  }
}
```

---

## FAQ

### Q1: 判別子は必ず `type` という名前にすべきですか？

いいえ。`type`, `kind`, `tag`, `status`, `_tag` など、プロジェクトで統一されていれば何でも構いません。ただし、同じコードベース内では判別子の名前を統一することを強く推奨します。Redux 系では `type` が慣例です。

### Q2: 判別共用体は何個のメンバーまでスケールしますか？

TypeScript コンパイラは数百個のメンバーでも問題なく動作します。ただし、開発者の認知負荷を考えると、20〜30 個を超える場合はカテゴリごとにネストした判別共用体に分割することを検討してください。

### Q3: enum と判別共用体のどちらを使うべきですか？

判別共用体を推奨します。enum は Tree-shaking の問題があり、`const enum` は `isolatedModules` と相性が悪いです。判別共用体はリテラル型のみで構成され、ランタイムコストが最小で、TypeScript の型推論と最も相性が良い設計です。

---

## まとめ表

| 概念 | 要点 |
|------|------|
| 判別共用体 | 共通のリテラル型フィールドで型を判別するユニオン |
| 判別子 | `type`, `kind` などの共通フィールド |
| 型の絞り込み | switch/if でリテラル値をチェックすると型が絞られる |
| 網羅性チェック | `default: assertNever(x)` で全ケース処理を保証 |
| satisfies | オブジェクトマップで網羅性を型チェック |
| ネスト | 判別共用体は入れ子にして複雑な分岐を表現可能 |

---

## 次に読むべきガイド

- [エラーハンドリング](./00-error-handling.md) -- 判別共用体を使った Result 型の実装
- [ブランド型](./03-branded-types.md) -- 判別共用体とブランド型の組み合わせ
- [tRPC](../04-ecosystem/02-trpc.md) -- 判別共用体を活用した型安全 API

---

## 参考文献

1. **TypeScript Handbook - Narrowing**
   https://www.typescriptlang.org/docs/handbook/2/narrowing.html

2. **Discriminated Unions in TypeScript** -- Matt Pocock
   https://www.totaltypescript.com/discriminated-unions-are-a-typescript-essential

3. **Redux Toolkit - Using TypeScript**
   https://redux-toolkit.js.org/usage/usage-with-typescript
