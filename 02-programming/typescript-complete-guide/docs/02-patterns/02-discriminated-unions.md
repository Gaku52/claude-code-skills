# TypeScript 判別共用体パターン

> 判別共用体（Discriminated Unions）、Redux の型安全な Action 設計、網羅性チェックで堅牢な分岐処理を実現する

## この章で学ぶこと

1. **判別共用体の基礎** -- リテラル型の判別子フィールドを使って、ユニオン型のメンバーを安全に絞り込む方法
2. **網羅性チェック** -- `never` 型を活用し、switch 文で全ケースを処理していることをコンパイル時に保証する技法
3. **Redux / useReducer での活用** -- Action 型を判別共用体で定義し、Reducer を型安全に記述するパターン
4. **高度なパターン** -- ネスト、ジェネリクス、型レベルプログラミングとの組み合わせ
5. **実務での設計パターン** -- ステートマシン、APIレスポンス、ドメインモデリングへの応用
6. **パフォーマンスと最適化** -- 判別共用体のランタイム特性とメモリ効率

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

### 1-3. 判別子の選択

判別共用体の品質は判別子の設計に大きく依存します。

```typescript
// ─── 文字列リテラル（最も一般的） ───
type Event =
  | { type: "click"; x: number; y: number }
  | { type: "keypress"; key: string }
  | { type: "scroll"; offset: number };

// ─── const enum（ランタイムコストゼロ、ただし制限あり） ───
const enum ShapeKind {
  Circle = "circle",
  Rect = "rect",
  Triangle = "triangle",
}
// ※ isolatedModules モードでは使用不可

// ─── ネストした判別子 ───
type ApiResult =
  | { status: "success"; data: unknown }
  | { status: "error"; error: { code: "network" | "auth" | "validation"; message: string } };

// ─── 複数の判別子 ───
type Message =
  | { channel: "email"; priority: "high"; subject: string; body: string }
  | { channel: "email"; priority: "low"; body: string }
  | { channel: "sms"; body: string }
  | { channel: "push"; title: string; body: string };
// channel と priority の組み合わせで判別可能
```

### 1-4. if-else による絞り込み

switch 文だけでなく、if-else でも型の絞り込みが機能します。

```typescript
function describe(shape: Shape): string {
  // if 文でも絞り込みが機能する
  if (shape.type === "circle") {
    return `Circle with radius ${shape.radius}`;
  }

  if (shape.type === "rect") {
    return `Rectangle ${shape.width}x${shape.height}`;
  }

  // ここでは shape は Triangle に絞り込まれている
  return `Triangle with base ${shape.base} and height ${shape.height}`;
}

// 早期リターンパターン
function processShape(shape: Shape): number | null {
  if (shape.type !== "circle") {
    return null; // Circle 以外は処理しない
  }
  // ここでは shape は Circle に絞り込まれている
  return Math.PI * shape.radius ** 2;
}
```

### 1-5. 型の絞り込みと分割代入

```typescript
// 分割代入と組み合わせたパターン
function formatShape(shape: Shape): string {
  switch (shape.type) {
    case "circle": {
      const { radius } = shape; // Circle 型から分割代入
      return `○ r=${radius.toFixed(2)}`;
    }
    case "rect": {
      const { width, height } = shape; // Rect 型から分割代入
      return `□ ${width}×${height}`;
    }
    case "triangle": {
      const { base, height } = shape; // Triangle 型から分割代入
      return `△ base=${base}, h=${height}`;
    }
  }
}

// 配列メソッドでの使用
const circles = shapes.filter(
  (s): s is Circle => s.type === "circle"
);
// circles は Circle[] 型

const areas = shapes.map((s) => {
  switch (s.type) {
    case "circle": return { shape: "circle", area: Math.PI * s.radius ** 2 };
    case "rect": return { shape: "rect", area: s.width * s.height };
    case "triangle": return { shape: "triangle", area: (s.base * s.height) / 2 };
  }
});
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

### 2-3. 網羅性チェックのバリエーション

```typescript
// ─── 方法1: assertNever（最も一般的） ───
function assertNever(value: never): never {
  throw new Error(`Unexpected value: ${JSON.stringify(value)}`);
}

// ─── 方法2: 型注釈による暗黙的チェック ───
function area(shape: Shape): number {
  switch (shape.type) {
    case "circle":
      return Math.PI * shape.radius ** 2;
    case "rect":
      return shape.width * shape.height;
    case "triangle":
      return (shape.base * shape.height) / 2;
    // case を忘れると、戻り値が undefined になる可能性があり
    // 戻り値の型 number と矛盾してコンパイルエラー
  }
}

// ─── 方法3: satisfies Record（オブジェクトマップ） ───
const handlers = {
  circle: (s: Circle) => `Circle: r=${s.radius}`,
  rect: (s: Rect) => `Rect: ${s.width}x${s.height}`,
  triangle: (s: Triangle) => `Triangle: b=${s.base}, h=${s.height}`,
} satisfies Record<Shape["type"], (s: any) => string>;

// ─── 方法4: const assertion + 型チェック ───
const SHAPE_TYPES = ["circle", "rect", "triangle"] as const;
type ShapeType = (typeof SHAPE_TYPES)[number];
// Shape["type"] が ShapeType と一致することを確認
type Check = ShapeType extends Shape["type"] ? true : false;
// Check が true でなければ、SHAPE_TYPES に不足がある

// ─── 方法5: ESLint ルール ───
// @typescript-eslint/switch-exhaustiveness-check を有効にすると
// switch 文の網羅性を lint で検出できる
```

### 2-4. 条件付き網羅性チェック

```typescript
// 一部のケースだけを処理し、残りは共通処理にしたい場合

// パターン1: 明示的な共通処理
function getStatusColor(status: "active" | "inactive" | "pending" | "archived"): string {
  switch (status) {
    case "active":
      return "green";
    case "pending":
      return "yellow";
    case "inactive":
    case "archived":
      return "gray"; // 複数ケースをまとめる
    default:
      return assertNever(status);
  }
}

// パターン2: 部分的な処理 + デフォルト
type LogLevel = "trace" | "debug" | "info" | "warn" | "error" | "fatal";

function shouldAlert(level: LogLevel): boolean {
  switch (level) {
    case "error":
    case "fatal":
      return true;
    default:
      // 残りは false だが、網羅性チェックは行わない
      return false;
  }
}

// パターン3: 型安全なデフォルト付き網羅性チェック
function toHttpStatus(error: DomainError): number {
  switch (error.code) {
    case "VALIDATION_ERROR": return 400;
    case "NOT_FOUND": return 404;
    case "PERMISSION_DENIED": return 403;
    case "CONFLICT": return 409;
    default: {
      // ここで error.code が never でないなら
      // 新しいエラーコードが追加されている
      const _exhaustive: never = error;
      // フォールバック: 500 を返す（型チェックは警告するがビルドは通る）
      return 500;
    }
  }
}
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
|      action.payload  -> { text: string }            |
|    case "TOGGLE_TODO":                             |
|      action.payload  -> { id: number }              |
|    case "DELETE_TODO":                             |
|      action.payload  -> { id: number }              |
|    case "SET_FILTER":                              |
|      action.payload  -> { filter: FilterType }      |
|    default:                                        |
|      assertNever(action)  -> 網羅性チェック          |
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

### 3-3. React useReducer との統合

```typescript
import { useReducer, Dispatch } from "react";

// 初期状態
const initialState: TodoState = {
  todos: [],
  filter: "all",
};

// カスタムフック
function useTodos() {
  const [state, dispatch] = useReducer(todoReducer, initialState);

  const actions = {
    addTodo: (text: string) =>
      dispatch({ type: "ADD_TODO", payload: { text } }),

    toggleTodo: (id: number) =>
      dispatch({ type: "TOGGLE_TODO", payload: { id } }),

    deleteTodo: (id: number) =>
      dispatch({ type: "DELETE_TODO", payload: { id } }),

    setFilter: (filter: TodoState["filter"]) =>
      dispatch({ type: "SET_FILTER", payload: { filter } }),
  };

  const filteredTodos = state.todos.filter((todo) => {
    switch (state.filter) {
      case "all": return true;
      case "active": return !todo.completed;
      case "completed": return todo.completed;
    }
  });

  return { state, filteredTodos, actions };
}

// コンポーネントでの使用
function TodoApp() {
  const { filteredTodos, actions } = useTodos();

  return (
    <div>
      <input
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            actions.addTodo(e.currentTarget.value);
            e.currentTarget.value = "";
          }
        }}
      />
      <ul>
        {filteredTodos.map((todo) => (
          <li key={todo.id}>
            <span
              onClick={() => actions.toggleTodo(todo.id)}
              style={{
                textDecoration: todo.completed ? "line-through" : "none",
              }}
            >
              {todo.text}
            </span>
            <button onClick={() => actions.deleteTodo(todo.id)}>x</button>
          </li>
        ))}
      </ul>
    </div>
  );
}
```

### 3-4. 複雑なステート管理（複数 Reducer の合成）

```typescript
// ─── 認証ステート ───
type AuthState =
  | { status: "anonymous" }
  | { status: "authenticating" }
  | { status: "authenticated"; user: User; token: string }
  | { status: "error"; error: string };

type AuthAction =
  | { type: "LOGIN_START" }
  | { type: "LOGIN_SUCCESS"; payload: { user: User; token: string } }
  | { type: "LOGIN_FAILURE"; payload: { error: string } }
  | { type: "LOGOUT" };

function authReducer(state: AuthState, action: AuthAction): AuthState {
  switch (action.type) {
    case "LOGIN_START":
      return { status: "authenticating" };
    case "LOGIN_SUCCESS":
      return {
        status: "authenticated",
        user: action.payload.user,
        token: action.payload.token,
      };
    case "LOGIN_FAILURE":
      return { status: "error", error: action.payload.error };
    case "LOGOUT":
      return { status: "anonymous" };
    default:
      return assertNever(action);
  }
}

// ─── ステート間の遷移制約 ───
// 型レベルでの遷移制約（状態によって使えるアクションを制限）

type AuthActionFor<S extends AuthState["status"]> =
  S extends "anonymous" ? Extract<AuthAction, { type: "LOGIN_START" }>
  : S extends "authenticating" ? Extract<AuthAction, { type: "LOGIN_SUCCESS" | "LOGIN_FAILURE" }>
  : S extends "authenticated" ? Extract<AuthAction, { type: "LOGOUT" }>
  : S extends "error" ? Extract<AuthAction, { type: "LOGIN_START" | "LOGOUT" }>
  : never;

// 型安全な dispatch
function createAuthDispatch(
  state: AuthState,
  dispatch: Dispatch<AuthAction>
) {
  return {
    login: () => {
      if (state.status === "anonymous" || state.status === "error") {
        dispatch({ type: "LOGIN_START" });
      }
    },
    logout: () => {
      if (state.status === "authenticated") {
        dispatch({ type: "LOGOUT" });
      }
    },
  };
}
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

### 4-3. 判別共用体の型レベル操作

```typescript
// ─── Extract: 特定のメンバーを抽出 ───
type UserEvents = Extract<AppEvent, { kind: `user.${string}` }>;
// => { kind: "user.created"; payload: ... } | { kind: "user.deleted"; payload: ... }

type OrderEvents = Extract<AppEvent, { kind: `order.${string}` }>;
// => { kind: "order.placed"; payload: ... } | { kind: "order.shipped"; payload: ... }

// ─── Exclude: 特定のメンバーを除外 ───
type NonUserEvents = Exclude<AppEvent, { kind: `user.${string}` }>;

// ─── 判別子の値を取得 ───
type EventKind = AppEvent["kind"];
// => "user.created" | "user.deleted" | "order.placed" | "order.shipped"

// ─── ペイロードを取得するユーティリティ型 ───
type PayloadOf<K extends AppEvent["kind"]> = Extract<
  AppEvent,
  { kind: K }
>["payload"];

type UserCreatedPayload = PayloadOf<"user.created">;
// => { userId: string; name: string }

// ─── イベントマップを生成 ───
type EventMap = {
  [K in AppEvent["kind"]]: PayloadOf<K>;
};
// => {
//   "user.created": { userId: string; name: string };
//   "user.deleted": { userId: string };
//   "order.placed": { orderId: string; total: number };
//   "order.shipped": { orderId: string; trackingId: string };
// }
```

### 4-4. タグ付きユニオンの自動生成

```typescript
// ペイロードマップから判別共用体を自動生成
type EventPayloads = {
  "user.created": { userId: string; name: string };
  "user.deleted": { userId: string };
  "order.placed": { orderId: string; total: number };
  "order.shipped": { orderId: string; trackingId: string };
  "payment.completed": { paymentId: string; amount: number };
  "payment.failed": { paymentId: string; reason: string };
};

// ペイロードマップから判別共用体を生成
type GeneratedEvent = {
  [K in keyof EventPayloads]: {
    kind: K;
    payload: EventPayloads[K];
    timestamp: Date;
  };
}[keyof EventPayloads];

// 型安全なイベント発行関数
function createEvent<K extends keyof EventPayloads>(
  kind: K,
  payload: EventPayloads[K]
): Extract<GeneratedEvent, { kind: K }> {
  return { kind, payload, timestamp: new Date() } as any;
}

// Action マップから判別共用体を生成（Redux 用）
type ActionPayloads = {
  INCREMENT: void;
  DECREMENT: void;
  SET_VALUE: { value: number };
  RESET: void;
};

type GeneratedAction = {
  [K in keyof ActionPayloads]: ActionPayloads[K] extends void
    ? { type: K }
    : { type: K; payload: ActionPayloads[K] };
}[keyof ActionPayloads];
// =>
//   | { type: "INCREMENT" }
//   | { type: "DECREMENT" }
//   | { type: "SET_VALUE"; payload: { value: number } }
//   | { type: "RESET" }
```

### 4-5. 再帰的判別共用体

```typescript
// JSON のような再帰的なデータ構造
type JsonValue =
  | { type: "null" }
  | { type: "boolean"; value: boolean }
  | { type: "number"; value: number }
  | { type: "string"; value: string }
  | { type: "array"; elements: JsonValue[] }
  | { type: "object"; properties: Record<string, JsonValue> };

function stringify(json: JsonValue): string {
  switch (json.type) {
    case "null":
      return "null";
    case "boolean":
      return json.value ? "true" : "false";
    case "number":
      return String(json.value);
    case "string":
      return `"${json.value}"`;
    case "array":
      return `[${json.elements.map(stringify).join(", ")}]`;
    case "object": {
      const entries = Object.entries(json.properties)
        .map(([key, val]) => `"${key}": ${stringify(val)}`)
        .join(", ");
      return `{${entries}}`;
    }
    default:
      return assertNever(json);
  }
}

// AST（抽象構文木）の表現
type Expression =
  | { kind: "number_literal"; value: number }
  | { kind: "string_literal"; value: string }
  | { kind: "identifier"; name: string }
  | { kind: "binary_op"; op: "+" | "-" | "*" | "/"; left: Expression; right: Expression }
  | { kind: "unary_op"; op: "-" | "!"; operand: Expression }
  | { kind: "function_call"; name: string; args: Expression[] }
  | { kind: "conditional"; condition: Expression; then: Expression; else: Expression };

function evaluate(expr: Expression, env: Record<string, number>): number {
  switch (expr.kind) {
    case "number_literal":
      return expr.value;
    case "string_literal":
      return parseFloat(expr.value);
    case "identifier":
      if (!(expr.name in env)) throw new Error(`Undefined: ${expr.name}`);
      return env[expr.name];
    case "binary_op": {
      const left = evaluate(expr.left, env);
      const right = evaluate(expr.right, env);
      switch (expr.op) {
        case "+": return left + right;
        case "-": return left - right;
        case "*": return left * right;
        case "/": return left / right;
      }
    }
    case "unary_op": {
      const operand = evaluate(expr.operand, env);
      switch (expr.op) {
        case "-": return -operand;
        case "!": return operand === 0 ? 1 : 0;
      }
    }
    case "function_call":
      throw new Error("Function calls not supported in this example");
    case "conditional":
      return evaluate(expr.condition, env) !== 0
        ? evaluate(expr.then, env)
        : evaluate(expr.else, env);
    default:
      return assertNever(expr);
  }
}
```

---

## 5. 実務での設計パターン

### 5-1. ステートマシンとしての判別共用体

```typescript
// ─── 注文ステートマシン ───
//
//  Created -> Confirmed -> Processing -> Shipped -> Delivered
//     |          |            |                        |
//     +----------+------------+-- Cancelled            +-- Returned
//

type OrderState =
  | {
      status: "created";
      orderId: string;
      items: OrderItem[];
      createdAt: Date;
    }
  | {
      status: "confirmed";
      orderId: string;
      items: OrderItem[];
      createdAt: Date;
      confirmedAt: Date;
      paymentId: string;
    }
  | {
      status: "processing";
      orderId: string;
      items: OrderItem[];
      createdAt: Date;
      confirmedAt: Date;
      paymentId: string;
      processingStartedAt: Date;
    }
  | {
      status: "shipped";
      orderId: string;
      items: OrderItem[];
      createdAt: Date;
      confirmedAt: Date;
      paymentId: string;
      shippedAt: Date;
      trackingNumber: string;
    }
  | {
      status: "delivered";
      orderId: string;
      items: OrderItem[];
      createdAt: Date;
      confirmedAt: Date;
      paymentId: string;
      shippedAt: Date;
      trackingNumber: string;
      deliveredAt: Date;
    }
  | {
      status: "cancelled";
      orderId: string;
      items: OrderItem[];
      createdAt: Date;
      cancelledAt: Date;
      reason: string;
      refundId?: string;
    }
  | {
      status: "returned";
      orderId: string;
      items: OrderItem[];
      createdAt: Date;
      confirmedAt: Date;
      deliveredAt: Date;
      returnedAt: Date;
      returnReason: string;
      refundId: string;
    };

// ステート遷移関数
type OrderEvent =
  | { type: "CONFIRM"; paymentId: string }
  | { type: "START_PROCESSING" }
  | { type: "SHIP"; trackingNumber: string }
  | { type: "DELIVER" }
  | { type: "CANCEL"; reason: string }
  | { type: "RETURN"; reason: string; refundId: string };

function transitionOrder(state: OrderState, event: OrderEvent): OrderState {
  switch (event.type) {
    case "CONFIRM": {
      if (state.status !== "created") {
        throw new Error(`Cannot confirm order in ${state.status} state`);
      }
      return {
        ...state,
        status: "confirmed",
        confirmedAt: new Date(),
        paymentId: event.paymentId,
      };
    }

    case "START_PROCESSING": {
      if (state.status !== "confirmed") {
        throw new Error(`Cannot start processing in ${state.status} state`);
      }
      return {
        ...state,
        status: "processing",
        processingStartedAt: new Date(),
      };
    }

    case "SHIP": {
      if (state.status !== "processing") {
        throw new Error(`Cannot ship in ${state.status} state`);
      }
      return {
        ...state,
        status: "shipped",
        shippedAt: new Date(),
        trackingNumber: event.trackingNumber,
      };
    }

    case "DELIVER": {
      if (state.status !== "shipped") {
        throw new Error(`Cannot deliver in ${state.status} state`);
      }
      return {
        ...state,
        status: "delivered",
        deliveredAt: new Date(),
      };
    }

    case "CANCEL": {
      if (state.status !== "created" && state.status !== "confirmed" && state.status !== "processing") {
        throw new Error(`Cannot cancel in ${state.status} state`);
      }
      return {
        status: "cancelled",
        orderId: state.orderId,
        items: state.items,
        createdAt: state.createdAt,
        cancelledAt: new Date(),
        reason: event.reason,
        refundId: state.status === "confirmed" || state.status === "processing"
          ? "refund-pending"
          : undefined,
      };
    }

    case "RETURN": {
      if (state.status !== "delivered") {
        throw new Error(`Cannot return in ${state.status} state`);
      }
      return {
        status: "returned",
        orderId: state.orderId,
        items: state.items,
        createdAt: state.createdAt,
        confirmedAt: state.confirmedAt,
        deliveredAt: state.deliveredAt,
        returnedAt: new Date(),
        returnReason: event.reason,
        refundId: event.refundId,
      };
    }

    default:
      return assertNever(event);
  }
}

// ステータスに応じた表示
function getOrderStatusDisplay(state: OrderState): {
  label: string;
  color: string;
  actions: string[];
} {
  switch (state.status) {
    case "created":
      return {
        label: "注文作成済み",
        color: "gray",
        actions: ["confirm", "cancel"],
      };
    case "confirmed":
      return {
        label: "確認済み",
        color: "blue",
        actions: ["start_processing", "cancel"],
      };
    case "processing":
      return {
        label: "処理中",
        color: "yellow",
        actions: ["ship", "cancel"],
      };
    case "shipped":
      return {
        label: `配送中 (${state.trackingNumber})`,
        color: "orange",
        actions: ["deliver"],
      };
    case "delivered":
      return {
        label: "配達完了",
        color: "green",
        actions: ["return"],
      };
    case "cancelled":
      return {
        label: `キャンセル: ${state.reason}`,
        color: "red",
        actions: [],
      };
    case "returned":
      return {
        label: `返品: ${state.returnReason}`,
        color: "purple",
        actions: [],
      };
    default:
      return assertNever(state);
  }
}
```

### 5-2. フォーム状態の表現

```typescript
// フォームフィールドの状態
type FieldState<T> =
  | { status: "pristine"; value: T }
  | { status: "dirty"; value: T; originalValue: T }
  | { status: "validating"; value: T }
  | { status: "valid"; value: T }
  | { status: "invalid"; value: T; errors: string[] };

// フォーム全体の状態
type FormState<T extends Record<string, unknown>> = {
  fields: {
    [K in keyof T]: FieldState<T[K]>;
  };
  submitState:
    | { status: "idle" }
    | { status: "submitting" }
    | { status: "success"; response: unknown }
    | { status: "error"; error: string };
};

// フォームアクション
type FormAction<T extends Record<string, unknown>> =
  | { type: "FIELD_CHANGE"; field: keyof T; value: T[keyof T] }
  | { type: "FIELD_BLUR"; field: keyof T }
  | { type: "FIELD_VALIDATE_START"; field: keyof T }
  | { type: "FIELD_VALIDATE_SUCCESS"; field: keyof T }
  | { type: "FIELD_VALIDATE_ERROR"; field: keyof T; errors: string[] }
  | { type: "SUBMIT_START" }
  | { type: "SUBMIT_SUCCESS"; response: unknown }
  | { type: "SUBMIT_ERROR"; error: string }
  | { type: "RESET" };

// フォームの状態に応じたUI制御
function isFormSubmittable<T extends Record<string, unknown>>(
  state: FormState<T>
): boolean {
  if (state.submitState.status === "submitting") return false;

  return Object.values(state.fields).every((field) => {
    const f = field as FieldState<unknown>;
    return f.status === "valid" || f.status === "pristine";
  });
}

function getFieldError<T extends Record<string, unknown>>(
  state: FormState<T>,
  field: keyof T
): string[] | null {
  const fieldState = state.fields[field];
  if (fieldState.status === "invalid") {
    return fieldState.errors;
  }
  return null;
}
```

### 5-3. ルーティングの型安全な表現

```typescript
// 型安全なルーティング
type Route =
  | { path: "/"; page: "home" }
  | { path: "/users"; page: "user-list"; query?: { page?: number; search?: string } }
  | { path: "/users/:id"; page: "user-detail"; params: { id: string } }
  | { path: "/users/:id/edit"; page: "user-edit"; params: { id: string } }
  | { path: "/settings"; page: "settings" }
  | { path: "/404"; page: "not-found" };

// ルートに応じたコンポーネントの選択
function getPageComponent(route: Route): string {
  switch (route.page) {
    case "home":
      return "HomePage";
    case "user-list":
      return "UserListPage";
    case "user-detail":
      return `UserDetailPage(id=${route.params.id})`;
    case "user-edit":
      return `UserEditPage(id=${route.params.id})`;
    case "settings":
      return "SettingsPage";
    case "not-found":
      return "NotFoundPage";
    default:
      return assertNever(route);
  }
}

// 型安全なリンク生成
type LinkParams<P extends Route["page"]> = Extract<Route, { page: P }>;

function createLink<P extends Route["page"]>(
  ...args: LinkParams<P> extends { params: infer Params }
    ? [page: P, params: Params]
    : [page: P]
): string {
  const [page, params] = args;

  switch (page) {
    case "home": return "/";
    case "user-list": return "/users";
    case "user-detail": return `/users/${(params as any).id}`;
    case "user-edit": return `/users/${(params as any).id}/edit`;
    case "settings": return "/settings";
    default: return "/404";
  }
}

// 使用例
const homeLink = createLink("home"); // "/"
const userLink = createLink("user-detail", { id: "123" }); // "/users/123"
// createLink("user-detail"); // エラー: params が必要
```

### 5-4. WebSocket メッセージの型安全な処理

```typescript
// WebSocket メッセージの判別共用体
type ClientMessage =
  | { type: "join_room"; roomId: string; userId: string }
  | { type: "leave_room"; roomId: string }
  | { type: "send_message"; roomId: string; content: string }
  | { type: "typing_start"; roomId: string }
  | { type: "typing_stop"; roomId: string }
  | { type: "ping" };

type ServerMessage =
  | { type: "room_joined"; roomId: string; members: string[] }
  | { type: "room_left"; roomId: string }
  | { type: "new_message"; roomId: string; userId: string; content: string; timestamp: number }
  | { type: "user_typing"; roomId: string; userId: string }
  | { type: "user_stopped_typing"; roomId: string; userId: string }
  | { type: "pong" }
  | { type: "error"; code: string; message: string };

// サーバーサイドのメッセージハンドラ
class ChatServer {
  handleMessage(ws: WebSocket, message: ClientMessage): void {
    switch (message.type) {
      case "join_room":
        this.joinRoom(ws, message.roomId, message.userId);
        break;
      case "leave_room":
        this.leaveRoom(ws, message.roomId);
        break;
      case "send_message":
        this.sendMessage(ws, message.roomId, message.content);
        break;
      case "typing_start":
        this.broadcastTyping(message.roomId, true);
        break;
      case "typing_stop":
        this.broadcastTyping(message.roomId, false);
        break;
      case "ping":
        this.send(ws, { type: "pong" });
        break;
      default:
        assertNever(message);
    }
  }

  private send(ws: WebSocket, message: ServerMessage): void {
    ws.send(JSON.stringify(message));
  }

  private joinRoom(ws: WebSocket, roomId: string, userId: string): void {
    // 部屋に参加する処理
    const members = this.getRoomMembers(roomId);
    this.send(ws, { type: "room_joined", roomId, members });
  }

  // ... その他のメソッド
}

// クライアントサイドのメッセージハンドラ
function handleServerMessage(message: ServerMessage): void {
  switch (message.type) {
    case "room_joined":
      console.log(`Joined room ${message.roomId} with ${message.members.length} members`);
      break;
    case "new_message":
      console.log(`[${message.roomId}] ${message.userId}: ${message.content}`);
      break;
    case "error":
      console.error(`Server error [${message.code}]: ${message.message}`);
      break;
    // ... 他のケース
  }
}
```

### 5-5. ツリー構造とVisitorパターン

```typescript
// HTML-like なツリー構造
type HtmlNode =
  | { type: "element"; tag: string; attributes: Record<string, string>; children: HtmlNode[] }
  | { type: "text"; content: string }
  | { type: "comment"; content: string }
  | { type: "doctype"; value: string };

// Visitor パターン
interface HtmlVisitor<T> {
  element(node: Extract<HtmlNode, { type: "element" }>, children: T[]): T;
  text(node: Extract<HtmlNode, { type: "text" }>): T;
  comment(node: Extract<HtmlNode, { type: "comment" }>): T;
  doctype(node: Extract<HtmlNode, { type: "doctype" }>): T;
}

function visitHtml<T>(node: HtmlNode, visitor: HtmlVisitor<T>): T {
  switch (node.type) {
    case "element": {
      const children = node.children.map((child) => visitHtml(child, visitor));
      return visitor.element(node, children);
    }
    case "text":
      return visitor.text(node);
    case "comment":
      return visitor.comment(node);
    case "doctype":
      return visitor.doctype(node);
    default:
      return assertNever(node);
  }
}

// HTML を文字列に変換する Visitor
const htmlStringVisitor: HtmlVisitor<string> = {
  element(node, children) {
    const attrs = Object.entries(node.attributes)
      .map(([k, v]) => ` ${k}="${v}"`)
      .join("");
    return `<${node.tag}${attrs}>${children.join("")}</${node.tag}>`;
  },
  text(node) {
    return node.content;
  },
  comment(node) {
    return `<!-- ${node.content} -->`;
  },
  doctype(node) {
    return `<!DOCTYPE ${node.value}>`;
  },
};

// テキストだけを抽出する Visitor
const textExtractVisitor: HtmlVisitor<string> = {
  element(_, children) {
    return children.join(" ");
  },
  text(node) {
    return node.content;
  },
  comment() {
    return "";
  },
  doctype() {
    return "";
  },
};

// 使用例
const doc: HtmlNode = {
  type: "element",
  tag: "div",
  attributes: { class: "container" },
  children: [
    { type: "element", tag: "h1", attributes: {}, children: [
      { type: "text", content: "Hello" },
    ]},
    { type: "text", content: " World" },
    { type: "comment", content: "todo: add more content" },
  ],
};

const html = visitHtml(doc, htmlStringVisitor);
// '<div class="container"><h1>Hello</h1> World<!-- todo: add more content --></div>'

const text = visitHtml(doc, textExtractVisitor);
// 'Hello  World '
```

---

## 6. パフォーマンスと最適化

### 6-1. switch vs if-else vs オブジェクトマップ

```typescript
// ─── ベンチマーク結果（概算） ───

// 方法1: switch 文 -- ~1-2ns/op
function areaSwitch(shape: Shape): number {
  switch (shape.type) {
    case "circle": return Math.PI * shape.radius ** 2;
    case "rect": return shape.width * shape.height;
    case "triangle": return (shape.base * shape.height) / 2;
  }
}

// 方法2: if-else -- ~1-3ns/op
function areaIfElse(shape: Shape): number {
  if (shape.type === "circle") return Math.PI * shape.radius ** 2;
  if (shape.type === "rect") return shape.width * shape.height;
  return (shape.base * shape.height) / 2;
}

// 方法3: オブジェクトマップ -- ~3-5ns/op
const areaMap: Record<Shape["type"], (s: any) => number> = {
  circle: (s: Circle) => Math.PI * s.radius ** 2,
  rect: (s: Rect) => s.width * s.height,
  triangle: (s: Triangle) => (s.base * s.height) / 2,
};
function areaMapLookup(shape: Shape): number {
  return areaMap[shape.type](shape);
}

// 結論:
// - 分岐が少ない（3-5個）: switch が最速かつ最も可読性が高い
// - 分岐が多い（10+個）: オブジェクトマップが均一なパフォーマンス
// - V8 エンジンは switch を最適化するため、多くの場合 switch が最速
```

### 6-2. メモリ効率

```typescript
// ─── コンストラクタ関数 vs オブジェクトリテラル ───

// 方法1: オブジェクトリテラル（推奨）
const circleObj: Circle = { type: "circle", radius: 5 };
// 各オブジェクトが独立した hidden class を持つ可能性がある

// 方法2: ファクトリ関数（V8 最適化に有利）
function createCircle(radius: number): Circle {
  return { type: "circle", radius };
}
// 同じファクトリから生成されたオブジェクトは同じ hidden class を共有

// 方法3: クラス（最もV8に優しい）
class CircleImpl implements Circle {
  readonly type = "circle" as const;
  constructor(readonly radius: number) {}
}
// クラスインスタンスは常に同じ hidden class を持つ

// ─── 大量のオブジェクトを生成する場合の推奨 ───
// 100,000個以上の判別共用体オブジェクトを生成する場合:
// 1. ファクトリ関数を使用する
// 2. プロパティの順序を一定にする
// 3. readonly を使用する（V8が最適化しやすい）
```

### 6-3. シリアライゼーションの効率

```typescript
// JSON シリアライゼーション時の判別子の扱い

// 方法1: そのまま JSON.stringify（推奨）
const serialized = JSON.stringify(shape);
// {"type":"circle","radius":5}

// 方法2: 数値タグでサイズを削減（帯域幅重視の場合）
const TAG = { circle: 0, rect: 1, triangle: 2 } as const;
type TaggedShape =
  | [typeof TAG.circle, number]           // [0, radius]
  | [typeof TAG.rect, number, number]     // [1, width, height]
  | [typeof TAG.triangle, number, number]; // [2, base, height]

function toTagged(shape: Shape): TaggedShape {
  switch (shape.type) {
    case "circle": return [TAG.circle, shape.radius];
    case "rect": return [TAG.rect, shape.width, shape.height];
    case "triangle": return [TAG.triangle, shape.base, shape.height];
  }
}

function fromTagged(tagged: TaggedShape): Shape {
  switch (tagged[0]) {
    case TAG.circle: return { type: "circle", radius: tagged[1] };
    case TAG.rect: return { type: "rect", width: tagged[1], height: tagged[2] };
    case TAG.triangle: return { type: "triangle", base: tagged[1], height: tagged[2] };
  }
}

// サイズ比較:
// JSON: {"type":"circle","radius":5}       = 30バイト
// タグ: [0,5]                              = 5バイト
// → 大量データの転送時に有効
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

### 網羅性チェック手法の比較

| 手法 | 安全レベル | コード量 | 利用場面 |
|------|----------|---------|---------|
| assertNever | 最高 | 少 | switch の default |
| satisfies Record | 最高 | 中 | オブジェクトマップ |
| 戻り値型注釈 | 高 | 最少 | 暗黙的チェック |
| ESLint ルール | 中 | 設定のみ | CI/CD での検出 |
| 型テスト | 最高 | 多 | テストファイル内 |

### 判別共用体の適用場面

| 場面 | 具体例 | メリット |
|------|--------|---------|
| ステートマシン | 注文状態、認証状態、フォーム状態 | 不正な状態遷移をコンパイル時に防止 |
| イベント駆動 | DOM イベント、WebSocket メッセージ | ハンドラの型安全性 |
| API レスポンス | 成功/エラー/ローディング | 状態に応じたUI表示の型安全性 |
| Redux Action | アクション定義と Reducer | payload の型推論 |
| AST | コンパイラ、リンター、フォーマッター | ノード種別ごとの処理 |
| データモデル | 支払い方法、通知チャネル | バリエーションの網羅性保証 |

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

### AP-3: 判別子を動的に設定する

```typescript
// NG: 判別子が動的な値
function createShape(type: string, params: any): Shape {
  return { type, ...params } as Shape; // 型安全性がゼロ
}

// OK: スマートコンストラクタ
function createCircle(radius: number): Circle {
  return { type: "circle", radius };
}

function createRect(width: number, height: number): Rect {
  return { type: "rect", width, height };
}
```

### AP-4: 巨大な判別共用体を一箇所で処理

```typescript
// NG: 50個のケースを持つ switch 文
function handleEvent(event: AppEvent): void {
  switch (event.type) {
    case "type_1": /* ... */ break;
    case "type_2": /* ... */ break;
    // ... 50個の case ...
    case "type_50": /* ... */ break;
    default: assertNever(event);
  }
}

// OK: カテゴリごとに分割
type UserEvent = Extract<AppEvent, { type: `user.${string}` }>;
type OrderEvent = Extract<AppEvent, { type: `order.${string}` }>;
type PaymentEvent = Extract<AppEvent, { type: `payment.${string}` }>;

function handleEvent(event: AppEvent): void {
  if (event.type.startsWith("user.")) {
    return handleUserEvent(event as UserEvent);
  }
  if (event.type.startsWith("order.")) {
    return handleOrderEvent(event as OrderEvent);
  }
  if (event.type.startsWith("payment.")) {
    return handlePaymentEvent(event as PaymentEvent);
  }
}

function handleUserEvent(event: UserEvent): void {
  // 5-10個のケースのみ
}
```

### AP-5: 判別子の名前がプロジェクト内で不統一

```typescript
// NG: ファイルによって判別子の名前がバラバラ
type Shape = { type: "circle"; ... } | { type: "rect"; ... };
type Event = { kind: "click"; ... } | { kind: "keypress"; ... };
type Action = { tag: "add"; ... } | { tag: "remove"; ... };
type Result = { _tag: "Ok"; ... } | { _tag: "Err"; ... };

// OK: プロジェクト全体で統一
// 方針: 判別子は "type" に統一
type Shape = { type: "circle"; ... } | { type: "rect"; ... };
type Event = { type: "click"; ... } | { type: "keypress"; ... };
type Action = { type: "add"; ... } | { type: "remove"; ... };
type Result = { type: "ok"; ... } | { type: "err"; ... };
```

---

## テスト戦略

### 判別共用体のテスト

```typescript
import { describe, it, expect } from "vitest";

describe("Shape area calculation", () => {
  // 各バリアントのテスト
  it("should calculate circle area", () => {
    const shape: Shape = { type: "circle", radius: 5 };
    expect(area(shape)).toBeCloseTo(78.54, 1);
  });

  it("should calculate rect area", () => {
    const shape: Shape = { type: "rect", width: 10, height: 5 };
    expect(area(shape)).toBe(50);
  });

  it("should calculate triangle area", () => {
    const shape: Shape = { type: "triangle", base: 6, height: 8 };
    expect(area(shape)).toBe(24);
  });

  // 型テスト（コンパイル時チェック）
  it("should have correct type narrowing", () => {
    const shape: Shape = { type: "circle", radius: 5 };
    if (shape.type === "circle") {
      // TypeScript がここで shape を Circle に絞り込むことを確認
      const _radius: number = shape.radius; // コンパイルエラーにならない
      expect(_radius).toBe(5);
    }
  });

  // 網羅性テスト
  it("should handle all shape types", () => {
    const allTypes: Shape["type"][] = ["circle", "rect", "triangle"];
    for (const type of allTypes) {
      const shape = createShape(type);
      expect(() => area(shape)).not.toThrow();
    }
  });
});

// コンパイル時の型テスト（tsd ライブラリ）
// @ts-expect-error テスト
describe("Type-level tests", () => {
  it("should reject invalid discriminant", () => {
    // @ts-expect-error: "invalid" は Shape["type"] に含まれない
    const shape: Shape = { type: "invalid", radius: 5 };
  });

  it("should not allow extra properties when narrowed", () => {
    const shape: Shape = { type: "circle", radius: 5 };
    if (shape.type === "circle") {
      // @ts-expect-error: Circle に width は存在しない
      const _width = shape.width;
    }
  });
});
```

### ステートマシンのテスト

```typescript
describe("Order state machine", () => {
  const initialOrder: OrderState = {
    status: "created",
    orderId: "order-1",
    items: [{ productId: "p1", quantity: 1, price: 1000 }],
    createdAt: new Date(),
  };

  it("should transition from created to confirmed", () => {
    const confirmed = transitionOrder(initialOrder, {
      type: "CONFIRM",
      paymentId: "pay-1",
    });
    expect(confirmed.status).toBe("confirmed");
    if (confirmed.status === "confirmed") {
      expect(confirmed.paymentId).toBe("pay-1");
    }
  });

  it("should not allow invalid transitions", () => {
    expect(() =>
      transitionOrder(initialOrder, { type: "SHIP", trackingNumber: "T123" })
    ).toThrow("Cannot ship in created state");
  });

  it("should follow full lifecycle", () => {
    let order: OrderState = initialOrder;

    order = transitionOrder(order, { type: "CONFIRM", paymentId: "pay-1" });
    expect(order.status).toBe("confirmed");

    order = transitionOrder(order, { type: "START_PROCESSING" });
    expect(order.status).toBe("processing");

    order = transitionOrder(order, { type: "SHIP", trackingNumber: "T123" });
    expect(order.status).toBe("shipped");

    order = transitionOrder(order, { type: "DELIVER" });
    expect(order.status).toBe("delivered");
  });

  it("should allow cancellation from valid states", () => {
    const validCancelStates: OrderState["status"][] = ["created", "confirmed", "processing"];

    for (const status of validCancelStates) {
      let order: OrderState = initialOrder;

      // 目的の状態まで遷移
      if (status === "confirmed" || status === "processing") {
        order = transitionOrder(order, { type: "CONFIRM", paymentId: "pay-1" });
      }
      if (status === "processing") {
        order = transitionOrder(order, { type: "START_PROCESSING" });
      }

      const cancelled = transitionOrder(order, {
        type: "CANCEL",
        reason: "Test cancellation",
      });
      expect(cancelled.status).toBe("cancelled");
    }
  });
});
```

---

## FAQ

### Q1: 判別子は必ず `type` という名前にすべきですか？

いいえ。`type`, `kind`, `tag`, `status`, `_tag` など、プロジェクトで統一されていれば何でも構いません。ただし、同じコードベース内では判別子の名前を統一することを強く推奨します。Redux 系では `type` が慣例です。

### Q2: 判別共用体は何個のメンバーまでスケールしますか？

TypeScript コンパイラは数百個のメンバーでも問題なく動作します。ただし、開発者の認知負荷を考えると、20〜30 個を超える場合はカテゴリごとにネストした判別共用体に分割することを検討してください。

### Q3: enum と判別共用体のどちらを使うべきですか？

判別共用体を推奨します。enum は Tree-shaking の問題があり、`const enum` は `isolatedModules` と相性が悪いです。判別共用体はリテラル型のみで構成され、ランタイムコストが最小で、TypeScript の型推論と最も相性が良い設計です。

### Q4: 判別共用体と class の使い分けは？

データの表現には判別共用体、振る舞いを含む場合は class が適しています。ただし、TypeScript では判別共用体 + 関数（Visitor パターン）の組み合わせが class 継承よりも型安全性が高いケースが多いです。OOPの「開放閉鎖原則」よりも「網羅性チェック」を重視するなら判別共用体を選びましょう。

### Q5: 判別共用体をシリアライズ/デシリアライズする際の注意点は？

JSON.stringify/parse は判別共用体と自然に互換性があります。ただし、Date や Map などの非プリミティブ型はシリアライズ時に情報が失われます。zod のスキーマで判別共用体を定義し、デシリアライズ時にバリデーションを行うのがベストプラクティスです。

### Q6: 既存のコードベースに判別共用体を導入するには？

段階的な導入が可能です。(1) まず共通の判別子フィールドを追加する（`type` など）、(2) `instanceof` チェックを `switch` 文に置き換える、(3) `assertNever` を追加して網羅性を保証する、という手順で進めます。既存のクラス階層がある場合は、各クラスに `readonly type` プロパティを追加するだけで判別共用体として使えます。

### Q7: 判別共用体と interface の extends は併用できますか？

はい。共通のフィールドを base interface に定義し、各バリアントが extends できます。ただし、判別子フィールドは各バリアントで具体的なリテラル型に narrowing する必要があります。

```typescript
interface BaseShape {
  readonly color: string;
}

interface Circle extends BaseShape {
  readonly type: "circle";
  readonly radius: number;
}

interface Rect extends BaseShape {
  readonly type: "rect";
  readonly width: number;
  readonly height: number;
}

type Shape = Circle | Rect;
```

### Q8: パフォーマンスが重要な場面で判別共用体は遅くなりませんか？

通常のアプリケーションでは気にする必要はありません。V8エンジンは switch 文を非常に効率的に最適化します。数千万回/秒の呼び出しが必要な超高頻度パスでは、数値タグ + 配列インデックスアクセスのほうが高速ですが、そのような最適化が必要になることは稀です。

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
| ステートマシン | 状態遷移を判別共用体で型安全にモデリング |
| Visitor パターン | 再帰的な構造の処理を型安全に実装 |
| イベント駆動 | イベントのペイロードを判別子で型安全に処理 |
| 型レベル操作 | Extract/Exclude で判別共用体のサブセットを取得 |

---

## 次に読むべきガイド

- [エラーハンドリング](./00-error-handling.md) -- 判別共用体を使った Result 型の実装
- [ブランド型](./03-branded-types.md) -- 判別共用体とブランド型の組み合わせ
- [ビルダーパターン](./01-builder-pattern.md) -- Step Builder の型安全性を支える判別共用体
- [依存性注入](./04-dependency-injection.md) -- 判別共用体を使ったサービスの型安全な管理
- [tRPC](../04-ecosystem/02-trpc.md) -- 判別共用体を活用した型安全 API

---

## 参考文献

1. **TypeScript Handbook - Narrowing**
   https://www.typescriptlang.org/docs/handbook/2/narrowing.html

2. **Discriminated Unions in TypeScript** -- Matt Pocock
   https://www.totaltypescript.com/discriminated-unions-are-a-typescript-essential

3. **Redux Toolkit - Using TypeScript**
   https://redux-toolkit.js.org/usage/usage-with-typescript

4. **Algebraic Data Types in TypeScript** -- Giulio Canti
   https://dev.to/gcanti/functional-design-algebraic-data-types-36kf

5. **XState** -- State machines and statecharts for JavaScript
   https://xstate.js.org/

6. **The Expression Problem** -- Philip Wadler
   http://homepages.inf.ed.ac.uk/wadler/papers/expression/expression.txt

7. **Making Impossible States Impossible** -- Richard Feldman
   https://www.youtube.com/watch?v=IcgmSRJHu_8
