# State パターン

> オブジェクトの内部状態に応じて振る舞いを動的に切り替え、有限状態マシン (FSM) を型安全に実装する行動パターン

---

## この章で学ぶこと

1. **State パターンの基本構造と GoF の意図** -- 状態ごとにクラスを分離し、条件分岐の爆発を防ぐ設計手法の原理
2. **有限状態マシン (FSM) の型安全な実装** -- 状態遷移図をコードに落とし込み、不正な遷移をコンパイル時に防止する
3. **宣言的 FSM と階層状態マシン** -- XState 風の設定ベース FSM、親子状態による複雑さの管理
4. **実プロダクトへの適用** -- EC注文管理、UIコンポーネント、フォームバリデーション、ゲーム AI での実践例
5. **State パターンと他パターンの連携** -- Strategy、Command、Observer との組合せと使い分け

---

## 前提知識

| トピック | 必要な理解 | 参照リンク |
|---------|-----------|-----------|
| TypeScript の interface と class | インターフェース実装、ジェネリクス、型ガード | [02-programming](../../02-programming/) |
| SOLID 原則（特に OCP・SRP） | 開放閉鎖原則、単一責任原則の理解 | [clean-code-principles](../../03-software-design/clean-code-principles/) |
| Strategy パターン | アルゴリズムの切り替えの基本概念 | [01-strategy.md](./01-strategy.md) |
| Observer パターン | 状態変化の通知 | [00-observer.md](./00-observer.md) |
| Command パターン | 操作のカプセル化、Undo/Redo | [02-command.md](./02-command.md) |

---

## なぜ State パターンが必要なのか

### if/else の爆発問題

注文管理システムで「注文のステータスに応じて異なる処理を行う」要件を考えてみましょう。

```
if/else アプローチの問題:

class Order {
  status: string = 'pending';

  pay(): void {
    if (this.status === 'pending') {
      // 決済処理...
      this.status = 'paid';
    } else if (this.status === 'paid') {
      throw new Error('Already paid');
    } else if (this.status === 'shipped') {
      throw new Error('Cannot pay after shipping');
    } else if (this.status === 'delivered') {
      throw new Error('Cannot pay after delivery');
    } else if (this.status === 'cancelled') {
      throw new Error('Order is cancelled');
    }
  }

  ship(): void {
    if (this.status === 'pending') {
      throw new Error('Must pay first');
    } else if (this.status === 'paid') {
      // 発送処理...
      this.status = 'shipped';
    } else if (...) { ... }
    // 同じパターンの繰り返し...
  }

  deliver(): void { /* 同じパターン */ }
  cancel(): void { /* 同じパターン */ }
  refund(): void { /* 同じパターン */ }
}

問題:
  ┌──────────────────────────────────────────────────┐
  │ 状態 5 × メソッド 5 = 25 個の分岐               │
  │                                                  │
  │ 新しい状態 "returned" を追加すると:               │
  │   → 全 5 メソッドを修正（OCP 違反）              │
  │   → 修正漏れ → ランタイムエラー                  │
  │   → テストケースが指数的に増加                    │
  └──────────────────────────────────────────────────┘
```

### State パターンによる解決

```
State パターンの解決:

  ┌──────────────┐         ┌──────────────────┐
  │   Context    │────────►│   State (抽象)    │
  │ OrderContext │         │                  │
  │              │         │ + pay()          │
  │ - state ─────┤         │ + ship()         │
  │ + pay()      │         │ + deliver()      │
  │ + ship()     │         │ + cancel()       │
  │ + deliver()  │         └────────┬─────────┘
  │ + cancel()   │                  │
  └──────────────┘            ┌─────┴──────┐
                              │            │
                     ┌────────┴──┐  ┌──────┴────────┐
                     │ Pending   │  │ Paid          │
                     │ State     │  │ State         │
                     │           │  │               │
                     │ pay() →   │  │ ship() →      │
                     │  PaidState│  │  ShippedState  │
                     │ cancel()→ │  │ cancel() →    │
                     │  Cancelled│  │  Cancelled     │
                     └───────────┘  └───────────────┘
                     ...他の状態も同様

  利点:
  ✓ 各状態の振る舞いが1つのクラスにまとまる（SRP）
  ✓ 新しい状態の追加は新クラスの追加のみ（OCP）
  ✓ 不正な遷移はその状態クラス内で例外を投げるだけ
  ✓ 各状態クラスを独立してテスト可能
```

GoF の定義:

> "Allow an object to alter its behavior when its internal state changes. The object will appear to change its class."
>
> -- Design Patterns: Elements of Reusable Object-Oriented Software (1994)

State パターンの本質は **「条件分岐をポリモーフィズムに変換する」** ことです。`if (status === 'pending')` という条件分岐を、`PendingState` クラスの存在そのもので表現します。これにより、各状態の振る舞いが凝集し、状態の追加が既存コードに影響を与えなくなります。

---

## 1. State パターンの構造

```
State パターンの構成要素（GoF）:

  ┌──────────────────┐         ┌──────────────────┐
  │    Context        │────────►│   State (抽象)    │
  │                   │         │                  │
  │ - currentState    │         │ + handle(ctx)    │
  │ + request()       │         │                  │
  │ + setState(s)     │         └────────┬─────────┘
  └──────────────────┘                   │
                                    ┌────┴──────┐
                               ┌────┴───┐  ┌───┴────────┐
                               │State A │  │ State B    │
                               │        │  │            │
                               │handle()│  │ handle()   │
                               │→ Aの   │  │ → Bの      │
                               │ 振る舞い│  │  振る舞い   │
                               │→ Bに   │  │ → Cに      │
                               │ 遷移   │  │  遷移      │
                               └────────┘  └────────────┘

  Context.request() は currentState.handle(this) に委譲
  handle() の中で context.setState(new NextState()) を呼ぶ

  ★ Strategy パターンとの違い:
    Strategy: クライアントが外部から戦略を切り替える
    State:    状態オブジェクト自身が次の状態に遷移する

  遷移の方向:
    Strategy: Client → Context.setStrategy(new X())
    State:    StateA.handle() → context.setState(new StateB())
              （状態が自分で次の状態を決める）
```

---

## 2. 基本実装 -- 注文ステータス管理

### コード例 1: GoF スタイルの State パターン

```typescript
// order-state.ts -- 注文の状態管理

// ============================
// State インターフェース
// ============================
interface OrderState {
  readonly name: string;
  pay(context: OrderContext): void;
  ship(context: OrderContext): void;
  deliver(context: OrderContext): void;
  cancel(context: OrderContext): void;
}

// ============================
// Context: 注文
// ============================
class OrderContext {
  private state: OrderState;
  private history: { state: string; timestamp: Date; action: string }[] = [];

  constructor(
    public readonly orderId: string,
    initialState: OrderState = new PendingState()
  ) {
    this.state = initialState;
    this.recordTransition('init');
  }

  setState(state: OrderState, action: string = 'transition'): void {
    const from = this.state.name;
    this.state = state;
    this.recordTransition(action);
    console.log(`[${this.orderId}] ${from} → ${state.name} (${action})`);
  }

  getStateName(): string {
    return this.state.name;
  }

  getHistory(): Array<{ state: string; timestamp: Date; action: string }> {
    return [...this.history];
  }

  private recordTransition(action: string): void {
    this.history.push({
      state: this.state.name,
      timestamp: new Date(),
      action,
    });
  }

  // 操作を状態に委譲
  pay(): void { this.state.pay(this); }
  ship(): void { this.state.ship(this); }
  deliver(): void { this.state.deliver(this); }
  cancel(): void { this.state.cancel(this); }
}

// ============================
// Concrete State: Pending（未決済）
// ============================
class PendingState implements OrderState {
  readonly name = 'pending';

  pay(context: OrderContext): void {
    console.log('決済処理を実行...');
    context.setState(new PaidState(), 'pay');
  }

  ship(_context: OrderContext): void {
    throw new Error('未決済の注文は発送できません');
  }

  deliver(_context: OrderContext): void {
    throw new Error('未決済の注文は配達完了にできません');
  }

  cancel(context: OrderContext): void {
    console.log('注文をキャンセルしました');
    context.setState(new CancelledState(), 'cancel');
  }
}

// ============================
// Concrete State: Paid（決済済み）
// ============================
class PaidState implements OrderState {
  readonly name = 'paid';

  pay(_context: OrderContext): void {
    throw new Error('すでに決済済みです');
  }

  ship(context: OrderContext): void {
    console.log('発送処理を実行...');
    context.setState(new ShippedState(), 'ship');
  }

  deliver(_context: OrderContext): void {
    throw new Error('発送前に配達完了にはできません');
  }

  cancel(context: OrderContext): void {
    console.log('返金処理を実行...');
    context.setState(new CancelledState(), 'cancel');
  }
}

// ============================
// Concrete State: Shipped（発送済み）
// ============================
class ShippedState implements OrderState {
  readonly name = 'shipped';

  pay(_context: OrderContext): void {
    throw new Error('発送済みの注文に決済はできません');
  }

  ship(_context: OrderContext): void {
    throw new Error('すでに発送済みです');
  }

  deliver(context: OrderContext): void {
    console.log('配達完了を記録...');
    context.setState(new DeliveredState(), 'deliver');
  }

  cancel(_context: OrderContext): void {
    throw new Error('発送済みの注文はキャンセルできません（返品手続きをご利用ください）');
  }
}

// ============================
// Concrete State: Delivered（配達完了） -- 終端状態
// ============================
class DeliveredState implements OrderState {
  readonly name = 'delivered';

  pay(): void { throw new Error('配達済みの注文に決済はできません'); }
  ship(): void { throw new Error('配達済みの注文は発送できません'); }
  deliver(): void { throw new Error('すでに配達済みです'); }
  cancel(): void { throw new Error('配達済みの注文はキャンセルできません'); }
}

// ============================
// Concrete State: Cancelled（キャンセル済み） -- 終端状態
// ============================
class CancelledState implements OrderState {
  readonly name = 'cancelled';

  pay(): void { throw new Error('キャンセル済みの注文に決済はできません'); }
  ship(): void { throw new Error('キャンセル済みの注文は発送できません'); }
  deliver(): void { throw new Error('キャンセル済みの注文は配達完了にできません'); }
  cancel(): void { throw new Error('すでにキャンセル済みです'); }
}

// ============================
// 使用例
// ============================
const order = new OrderContext('ORD-001');
console.log(order.getStateName()); // "pending"

order.pay();
// 決済処理を実行...
// [ORD-001] pending → paid (pay)

order.ship();
// 発送処理を実行...
// [ORD-001] paid → shipped (ship)

order.deliver();
// 配達完了を記録...
// [ORD-001] shipped → delivered (deliver)

try {
  order.cancel(); // 配達済みなのでエラー
} catch (e) {
  console.log(e.message); // "配達済みの注文はキャンセルできません"
}

console.log(order.getHistory());
// [
//   { state: 'pending', action: 'init', ... },
//   { state: 'paid', action: 'pay', ... },
//   { state: 'shipped', action: 'ship', ... },
//   { state: 'delivered', action: 'deliver', ... },
// ]
```

```
注文状態の遷移図:

  ┌─────────┐   PAY    ┌─────────┐   SHIP   ┌──────────┐
  │ pending │────────►│  paid   │────────►│ shipped  │
  └────┬────┘         └────┬────┘         └─────┬────┘
       │                   │                     │
       │ CANCEL            │ CANCEL              │ DELIVER
       │                   │                     │
       ▼                   ▼                     ▼
  ┌──────────┐       ┌──────────┐        ┌───────────┐
  │cancelled │       │cancelled │        │ delivered │
  └──────────┘       └──────────┘        └───────────┘

  ※ delivered, cancelled は終端状態（遷移先なし）
  ※ shipped → cancel は不可（返品手続きが別途必要）
```

---

## 3. 型安全な FSM（有限状態マシン）

### コード例 2: TypeScript の型システムで遷移を制約する

```typescript
// typed-fsm.ts -- 型で遷移を制約する FSM

// ============================
// 状態とイベントの型定義
// ============================
type OrderStatus = 'pending' | 'paid' | 'shipped' | 'delivered' | 'cancelled';
type OrderEvent = 'PAY' | 'SHIP' | 'DELIVER' | 'CANCEL';

// ============================
// 許可される遷移を型レベルで定義
// ============================
type TransitionMap = {
  pending:   { PAY: 'paid'; CANCEL: 'cancelled' };
  paid:      { SHIP: 'shipped'; CANCEL: 'cancelled' };
  shipped:   { DELIVER: 'delivered' };
  delivered: {};  // 終端状態: 遷移先なし
  cancelled: {};  // 終端状態: 遷移先なし
};

// ============================
// 型安全な遷移関数の型
// ============================
// これにより、存在しない遷移はコンパイル時にエラーになる
type ValidEvent<S extends OrderStatus> = keyof TransitionMap[S];
type NextState<S extends OrderStatus, E extends ValidEvent<S>> =
  TransitionMap[S][E];

// ============================
// FSM クラス（型安全版）
// ============================
const STATE_TRANSITIONS: Record<
  OrderStatus,
  Partial<Record<OrderEvent, OrderStatus>>
> = {
  pending:   { PAY: 'paid', CANCEL: 'cancelled' },
  paid:      { SHIP: 'shipped', CANCEL: 'cancelled' },
  shipped:   { DELIVER: 'delivered' },
  delivered: {},
  cancelled: {},
};

class TypedStateMachine<S extends OrderStatus> {
  constructor(private currentState: S) {}

  getState(): S {
    return this.currentState;
  }

  /**
   * 型安全な遷移
   * - 許可されたイベントのみ引数に取れる
   * - 戻り値の型が次の状態に正しく推論される
   */
  transition<E extends ValidEvent<S>>(
    event: E
  ): TypedStateMachine<NextState<S, E> & OrderStatus> {
    const transitions = STATE_TRANSITIONS[this.currentState];
    const nextState = transitions[event as string as OrderEvent];

    if (!nextState) {
      throw new Error(
        `Invalid transition: ${this.currentState} + ${String(event)}`
      );
    }

    return new TypedStateMachine(
      nextState as NextState<S, E> & OrderStatus
    );
  }
}

// ============================
// 使用例: コンパイル時に不正な遷移を検出
// ============================
const machine = new TypedStateMachine('pending' as const);

const paid = machine.transition('PAY');       // OK: pending → paid
const shipped = paid.transition('SHIP');      // OK: paid → shipped
const delivered = shipped.transition('DELIVER'); // OK: shipped → delivered

// 以下はコンパイルエラー!
// machine.transition('SHIP');    // Error: 'SHIP' は pending で許可されていない
// machine.transition('DELIVER'); // Error: 'DELIVER' は pending で許可されていない
// paid.transition('DELIVER');    // Error: 'DELIVER' は paid で許可されていない
// delivered.transition('PAY');   // Error: delivered は終端状態（遷移先なし）

// 型推論の確認
type PaidMachine = typeof paid;    // TypedStateMachine<'paid'>
type ShippedMachine = typeof shipped; // TypedStateMachine<'shipped'>
```

この実装のポイントは、**遷移テーブルを型レベルで定義する**ことです。`TransitionMap` 型により、各状態から発行できるイベントとその遷移先が型として宣言されます。存在しない遷移（例: `pending` から `DELIVER`）は型エラーとしてコンパイル時に検出されます。

---

## 4. 宣言的 FSM（XState 風）

### コード例 3: 設定オブジェクトで定義する FSM

```typescript
// declarative-fsm.ts -- 設定ベースの FSM

// ============================
// FSM の設定型
// ============================
interface TransitionConfig<TContext> {
  target: string;
  guard?: (context: TContext) => boolean;
  action?: (context: TContext) => void;
}

interface StateConfig<TContext> {
  on?: Record<string, TransitionConfig<TContext>>;
  entry?: (context: TContext) => void;
  exit?: (context: TContext) => void;
}

interface MachineConfig<TContext> {
  id: string;
  initial: string;
  context: TContext;
  states: Record<string, StateConfig<TContext>>;
}

// ============================
// FSM エンジン
// ============================
type FSMEvent =
  | { type: 'transition'; from: string; to: string; event: string }
  | { type: 'guard-blocked'; from: string; event: string };

class FSM<TContext> {
  private currentState: string;
  private context: TContext;
  private config: MachineConfig<TContext>;
  private listeners: Array<(event: FSMEvent) => void> = [];

  constructor(config: MachineConfig<TContext>) {
    this.config = config;
    this.currentState = config.initial;
    this.context = { ...config.context };

    // 初期状態の entry アクションを実行
    this.config.states[this.currentState]?.entry?.(this.context);
  }

  /** イベントを送信して遷移を試みる */
  send(event: string): boolean {
    const stateConfig = this.config.states[this.currentState];
    const transition = stateConfig?.on?.[event];

    if (!transition) {
      console.warn(
        `No transition for event "${event}" in state "${this.currentState}"`
      );
      return false;
    }

    // ガード条件のチェック
    if (transition.guard && !transition.guard(this.context)) {
      this.notify({
        type: 'guard-blocked',
        from: this.currentState,
        event,
      });
      return false;
    }

    const from = this.currentState;

    // exit → action → entry の順序で実行
    stateConfig?.exit?.(this.context);
    transition.action?.(this.context);

    this.currentState = transition.target;
    this.config.states[this.currentState]?.entry?.(this.context);

    this.notify({ type: 'transition', from, to: this.currentState, event });
    return true;
  }

  getState(): string {
    return this.currentState;
  }

  getContext(): TContext {
    return { ...this.context };
  }

  /** 現在の状態で許可されるイベントの一覧 */
  allowedEvents(): string[] {
    const stateConfig = this.config.states[this.currentState];
    return stateConfig?.on ? Object.keys(stateConfig.on) : [];
  }

  subscribe(listener: (event: FSMEvent) => void): () => void {
    this.listeners.push(listener);
    return () => {
      this.listeners = this.listeners.filter(l => l !== listener);
    };
  }

  private notify(event: FSMEvent): void {
    for (const listener of this.listeners) {
      listener(event);
    }
  }
}

// ============================
// 使用例: 信号機
// ============================
const trafficLight = new FSM({
  id: 'traffic-light',
  initial: 'red',
  context: { cycleCount: 0 },
  states: {
    red: {
      entry: (ctx) => console.log(`赤信号 (サイクル: ${ctx.cycleCount})`),
      on: {
        TIMER: {
          target: 'green',
          action: (ctx) => { ctx.cycleCount++; },
        },
      },
    },
    green: {
      entry: () => console.log('青信号'),
      on: {
        TIMER: { target: 'yellow' },
      },
    },
    yellow: {
      entry: () => console.log('黄信号'),
      on: {
        TIMER: { target: 'red' },
      },
    },
  },
});

trafficLight.send('TIMER'); // 赤信号 (サイクル: 0) → 青信号
trafficLight.send('TIMER'); // 青信号 → 黄信号
trafficLight.send('TIMER'); // 黄信号 → 赤信号 (サイクル: 1)
console.log(trafficLight.getContext()); // { cycleCount: 1 }
```

---

## 5. 実用例: フォームバリデーション状態

### コード例 4: 複雑なフォームの状態管理

```typescript
// form-state.ts -- フォームの状態管理

// ============================
// フォームコンテキスト
// ============================
interface FormContext {
  data: Record<string, string>;
  errors: Record<string, string>;
  submitCount: number;
  lastSubmitAt: Date | null;
}

// ============================
// フォーム FSM
// ============================
const formMachine = new FSM<FormContext>({
  id: 'form',
  initial: 'idle',
  context: {
    data: {},
    errors: {},
    submitCount: 0,
    lastSubmitAt: null,
  },
  states: {
    idle: {
      entry: () => console.log('[Form] Idle - 入力待ち'),
      on: {
        CHANGE: {
          target: 'editing',
          action: () => console.log('[Form] 入力開始'),
        },
      },
    },
    editing: {
      on: {
        CHANGE: {
          target: 'editing',
        },
        SUBMIT: {
          target: 'validating',
          action: (ctx) => { ctx.submitCount++; },
        },
        RESET: {
          target: 'idle',
          action: (ctx) => {
            ctx.data = {};
            ctx.errors = {};
          },
        },
      },
    },
    validating: {
      entry: (ctx) => {
        console.log('[Form] バリデーション実行中...');
        ctx.errors = {};
        // バリデーションルールの適用
        if (!ctx.data.email?.includes('@')) {
          ctx.errors.email = 'メールアドレスの形式が不正です';
        }
        if (!ctx.data.name || ctx.data.name.length < 2) {
          ctx.errors.name = '名前は2文字以上で入力してください';
        }
      },
      on: {
        VALID: {
          target: 'submitting',
          guard: (ctx) => Object.keys(ctx.errors).length === 0,
        },
        INVALID: {
          target: 'editing',
        },
      },
    },
    submitting: {
      entry: () => console.log('[Form] 送信中...'),
      on: {
        SUCCESS: {
          target: 'success',
          action: (ctx) => { ctx.lastSubmitAt = new Date(); },
        },
        FAILURE: {
          target: 'error',
        },
      },
    },
    success: {
      entry: () => console.log('[Form] 送信成功!'),
      on: {
        RESET: {
          target: 'idle',
          action: (ctx) => {
            ctx.data = {};
            ctx.errors = {};
          },
        },
      },
    },
    error: {
      entry: () => console.log('[Form] 送信エラー'),
      on: {
        RETRY: {
          target: 'submitting',
          guard: (ctx) => ctx.submitCount < 3,
        },
        RESET: {
          target: 'idle',
          action: (ctx) => {
            ctx.data = {};
            ctx.errors = {};
          },
        },
      },
    },
  },
});

// 使用例
formMachine.send('CHANGE');      // idle → editing
formMachine.send('SUBMIT');      // editing → validating
// バリデーションエラーがある場合:
formMachine.send('INVALID');     // validating → editing
```

```
フォーム状態の遷移図:

  ┌──────┐  CHANGE  ┌─────────┐  SUBMIT  ┌────────────┐
  │ idle │────────►│ editing │────────►│ validating │
  └──────┘         └────┬────┘         └──┬─────┬───┘
       ▲                │                  │     │
       │ RESET          │ RESET    INVALID │     │ VALID
       │                │                  │     │  (guard: errors=0)
       ├────────────────┘◄─────────────────┘     │
       │                                         ▼
       │            ┌─────────┐ FAILURE ┌────────────┐
       │ RESET      │  error  │◄────────│ submitting │
       ├────────────┤         │         └─────┬──────┘
       │            │ RETRY → │               │
       │            │(guard:  │               │ SUCCESS
       │            │ cnt<3)  │               │
       │            └─────────┘               ▼
       │                              ┌───────────┐
       └──────────────────────────────│  success  │
                      RESET           └───────────┘
```

---

## 6. Python での State パターン

### コード例 5: Python の ABC とデコレータによる State

```python
# state_python.py -- Python での State パターン実装
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable


# ============================
# 自動販売機の State パターン
# ============================

class VendingState(ABC):
    """自動販売機の状態インターフェース"""

    @abstractmethod
    def insert_coin(self, machine: VendingMachine, amount: int) -> None: ...

    @abstractmethod
    def select_product(self, machine: VendingMachine, product: str) -> None: ...

    @abstractmethod
    def dispense(self, machine: VendingMachine) -> None: ...

    @abstractmethod
    def cancel(self, machine: VendingMachine) -> None: ...

    @property
    @abstractmethod
    def name(self) -> str: ...


class IdleState(VendingState):
    """待機状態: コイン投入待ち"""

    @property
    def name(self) -> str:
        return "idle"

    def insert_coin(self, machine: VendingMachine, amount: int) -> None:
        machine.balance += amount
        print(f"投入: {amount}円 (残高: {machine.balance}円)")
        machine.set_state(HasMoneyState())

    def select_product(self, machine: VendingMachine, product: str) -> None:
        print("先にコインを投入してください")

    def dispense(self, machine: VendingMachine) -> None:
        print("先にコインを投入し、商品を選択してください")

    def cancel(self, machine: VendingMachine) -> None:
        print("キャンセルする操作がありません")


class HasMoneyState(VendingState):
    """コイン投入済み: 商品選択待ち"""

    @property
    def name(self) -> str:
        return "has_money"

    def insert_coin(self, machine: VendingMachine, amount: int) -> None:
        machine.balance += amount
        print(f"追加投入: {amount}円 (残高: {machine.balance}円)")

    def select_product(self, machine: VendingMachine, product: str) -> None:
        price = machine.get_price(product)
        if price is None:
            print(f"商品 '{product}' は存在しません")
            return
        if machine.balance < price:
            print(f"残高不足: {machine.balance}円 < {price}円")
            return
        if not machine.has_stock(product):
            print(f"'{product}' は在庫切れです")
            return

        machine.selected_product = product
        machine.balance -= price
        print(f"'{product}' を選択 (価格: {price}円, 残高: {machine.balance}円)")
        machine.set_state(DispensingState())

    def dispense(self, machine: VendingMachine) -> None:
        print("先に商品を選択してください")

    def cancel(self, machine: VendingMachine) -> None:
        print(f"返金: {machine.balance}円")
        machine.balance = 0
        machine.set_state(IdleState())


class DispensingState(VendingState):
    """商品排出中"""

    @property
    def name(self) -> str:
        return "dispensing"

    def insert_coin(self, machine: VendingMachine, amount: int) -> None:
        print("商品排出中です。しばらくお待ちください")

    def select_product(self, machine: VendingMachine, product: str) -> None:
        print("商品排出中です。しばらくお待ちください")

    def dispense(self, machine: VendingMachine) -> None:
        product = machine.selected_product
        machine.reduce_stock(product)
        print(f"'{product}' を排出しました")
        machine.selected_product = None

        if machine.balance > 0:
            machine.set_state(HasMoneyState())
        else:
            machine.set_state(IdleState())

    def cancel(self, machine: VendingMachine) -> None:
        print("商品排出中はキャンセルできません")


# ============================
# Context: 自動販売機
# ============================
@dataclass
class VendingMachine:
    products: dict[str, dict] = field(default_factory=lambda: {
        "コーラ": {"price": 120, "stock": 5},
        "お茶": {"price": 100, "stock": 3},
        "コーヒー": {"price": 150, "stock": 0},  # 在庫切れ
    })
    balance: int = 0
    selected_product: str | None = None
    _state: VendingState = field(default_factory=IdleState)
    _history: list[str] = field(default_factory=list)

    def set_state(self, state: VendingState) -> None:
        old = self._state.name
        self._state = state
        self._history.append(f"{old} → {state.name}")

    def get_price(self, product: str) -> int | None:
        info = self.products.get(product)
        return info["price"] if info else None

    def has_stock(self, product: str) -> bool:
        info = self.products.get(product)
        return info is not None and info["stock"] > 0

    def reduce_stock(self, product: str) -> None:
        if product in self.products:
            self.products[product]["stock"] -= 1

    # 操作を状態に委譲
    def insert_coin(self, amount: int) -> None:
        self._state.insert_coin(self, amount)

    def select_product(self, product: str) -> None:
        self._state.select_product(self, product)

    def dispense(self) -> None:
        self._state.dispense(self)

    def cancel(self) -> None:
        self._state.cancel(self)


# ============================
# 使用例
# ============================
if __name__ == "__main__":
    vm = VendingMachine()

    vm.insert_coin(100)    # 投入: 100円
    vm.insert_coin(50)     # 追加投入: 50円 (残高: 150円)
    vm.select_product("コーラ")  # 'コーラ' を選択 (価格: 120円, 残高: 30円)
    vm.dispense()          # 'コーラ' を排出しました

    print(f"State: {vm._state.name}")  # idle (残高0) or has_money (残高あり)
    print(f"残高: {vm.balance}円")     # 30円

    vm.select_product("お茶")  # 先にコインを投入してください... wait
    # 実は has_money 状態のはず (残高30円)

    print(f"遷移履歴: {vm._history}")
```

---

## 7. 階層状態マシン（HSM: Hierarchical State Machine）

### コード例 6: 親子状態による複雑さの管理

```typescript
// hierarchical-state.ts -- 階層状態マシン

// ============================
// HSM の構造
// ============================
// 階層状態マシンでは、状態を親子関係で整理できる
// 子状態で処理できないイベントは親状態に委譲される

interface HierarchicalState {
  readonly name: string;
  readonly parent?: HierarchicalState;
  handle(event: string, context: any): HierarchicalState | null;
  entry?(context: any): void;
  exit?(context: any): void;
}

class HSMEngine {
  private currentState: HierarchicalState;

  constructor(
    private initialState: HierarchicalState,
    private context: any
  ) {
    this.currentState = initialState;
    this.enterState(initialState);
  }

  send(event: string): void {
    let state: HierarchicalState | undefined = this.currentState;

    // 現在の状態から親に向かって、ハンドラを探す
    while (state) {
      const nextState = state.handle(event, this.context);
      if (nextState !== null) {
        this.transitionTo(nextState);
        return;
      }
      state = state.parent;
    }

    console.warn(`Unhandled event "${event}" in state "${this.currentState.name}"`);
  }

  private transitionTo(target: HierarchicalState): void {
    // 共通祖先を見つけて、exit/entry を正しい順序で実行
    const exitStates = this.getAncestors(this.currentState);
    const enterStates = this.getAncestors(target);

    // 共通祖先より上はスキップ
    const common = this.findCommonAncestor(exitStates, enterStates);

    // exit: 現在の状態から共通祖先まで
    for (const s of exitStates) {
      if (s === common) break;
      s.exit?.(this.context);
    }

    // entry: 共通祖先からターゲットまで
    const toEnter = [];
    for (const s of enterStates) {
      if (s === common) break;
      toEnter.unshift(s);
    }
    for (const s of toEnter) {
      this.enterState(s);
    }

    this.currentState = target;
  }

  private enterState(state: HierarchicalState): void {
    state.entry?.(this.context);
  }

  private getAncestors(state: HierarchicalState): HierarchicalState[] {
    const ancestors: HierarchicalState[] = [state];
    let current = state.parent;
    while (current) {
      ancestors.push(current);
      current = current.parent;
    }
    return ancestors;
  }

  private findCommonAncestor(
    a: HierarchicalState[],
    b: HierarchicalState[]
  ): HierarchicalState | undefined {
    const setB = new Set(b);
    return a.find(s => setB.has(s));
  }

  getState(): string {
    return this.currentState.name;
  }
}

// ============================
// 使用例: メディアプレーヤー
// ============================
// 階層構造:
//   Root
//   ├── Stopped
//   └── Playing (親状態)
//       ├── NormalSpeed
//       └── FastForward

const stoppedState: HierarchicalState = {
  name: 'stopped',
  handle(event) {
    if (event === 'PLAY') return normalSpeedState;
    return null;
  },
  entry() { console.log('[Stopped] 停止中'); },
};

const playingState: HierarchicalState = {
  name: 'playing',
  handle(event) {
    // 子状態で処理されないイベントをここで処理
    if (event === 'STOP') return stoppedState;
    return null;
  },
  entry() { console.log('[Playing] 再生開始'); },
  exit() { console.log('[Playing] 再生終了'); },
};

const normalSpeedState: HierarchicalState = {
  name: 'playing.normal',
  parent: playingState,
  handle(event) {
    if (event === 'FAST_FORWARD') return fastForwardState;
    return null; // 親 (playing) に委譲
  },
  entry() { console.log('[Normal] 通常速度'); },
};

const fastForwardState: HierarchicalState = {
  name: 'playing.fastForward',
  parent: playingState,
  handle(event) {
    if (event === 'NORMAL') return normalSpeedState;
    return null; // 親 (playing) に委譲
  },
  entry() { console.log('[FastForward] 早送り'); },
};

// 使用例
const player = new HSMEngine(stoppedState, {});
// [Stopped] 停止中

player.send('PLAY');
// [Playing] 再生開始
// [Normal] 通常速度

player.send('FAST_FORWARD');
// [FastForward] 早送り

player.send('STOP');
// ★ fastForward 自体は STOP を処理できないので、
//    親の playing に委譲 → stoppedState に遷移
// [Playing] 再生終了
// [Stopped] 停止中
```

```
階層状態マシンの構造:

  ┌─────────────────────────────────────────┐
  │                Root                      │
  │                                          │
  │  ┌──────────┐    ┌─────────────────────┐ │
  │  │ Stopped  │    │     Playing         │ │
  │  │          │◄───┤                     │ │
  │  │          │STOP│  ┌───────┐ ┌──────┐ │ │
  │  │          │    │  │Normal │ │Fast  │ │ │
  │  │          │────►  │ Speed │→│Fwd   │ │ │
  │  │          │PLAY│  │       │←│      │ │ │
  │  │          │    │  └───────┘ └──────┘ │ │
  │  └──────────┘    └─────────────────────┘ │
  │                                          │
  └──────────────────────────────────────────┘

  イベント委譲の流れ:
    FastForward で STOP を受け取った場合:
    1. FastForward.handle('STOP') → null（処理できない）
    2. Playing.handle('STOP') → stoppedState（親が処理）
```

---

## 8. React での State パターン

### コード例 7: useReducer + State パターンでUIの状態管理

```typescript
// react-state-pattern.tsx -- React での State パターン活用

// ============================
// UI の状態定義
// ============================
type ModalState =
  | { status: 'closed' }
  | { status: 'loading' }
  | { status: 'open'; data: any }
  | { status: 'error'; message: string; retryCount: number }
  | { status: 'confirming'; data: any; message: string };

type ModalAction =
  | { type: 'OPEN' }
  | { type: 'LOADED'; data: any }
  | { type: 'ERROR'; message: string }
  | { type: 'CONFIRM'; message: string }
  | { type: 'CONFIRM_YES' }
  | { type: 'CONFIRM_NO' }
  | { type: 'CLOSE' }
  | { type: 'RETRY' };

// ============================
// State パターンを Reducer で表現
// ============================
function modalReducer(state: ModalState, action: ModalAction): ModalState {
  switch (state.status) {
    case 'closed':
      switch (action.type) {
        case 'OPEN': return { status: 'loading' };
        default: return state;
      }

    case 'loading':
      switch (action.type) {
        case 'LOADED': return { status: 'open', data: action.data };
        case 'ERROR': return { status: 'error', message: action.message, retryCount: 0 };
        case 'CLOSE': return { status: 'closed' };
        default: return state;
      }

    case 'open':
      switch (action.type) {
        case 'CLOSE': return { status: 'closed' };
        case 'CONFIRM': return {
          status: 'confirming',
          data: state.data,
          message: action.message,
        };
        default: return state;
      }

    case 'error':
      switch (action.type) {
        case 'RETRY':
          if (state.retryCount >= 3) return state; // ガード条件
          return { status: 'loading' };
        case 'CLOSE': return { status: 'closed' };
        default: return state;
      }

    case 'confirming':
      switch (action.type) {
        case 'CONFIRM_YES': return { status: 'closed' }; // 確認後に閉じる
        case 'CONFIRM_NO': return { status: 'open', data: state.data };
        default: return state;
      }

    default:
      return state;
  }
}

// ============================
// React コンポーネントでの使用
// ============================
/*
function ModalComponent() {
  const [state, dispatch] = useReducer(modalReducer, { status: 'closed' });

  // 状態に応じた UI レンダリング
  switch (state.status) {
    case 'closed':
      return <button onClick={() => dispatch({ type: 'OPEN' })}>開く</button>;

    case 'loading':
      return <div>読み込み中...</div>;

    case 'open':
      return (
        <div>
          <pre>{JSON.stringify(state.data)}</pre>
          <button onClick={() => dispatch({ type: 'CLOSE' })}>閉じる</button>
          <button onClick={() => dispatch({
            type: 'CONFIRM',
            message: '本当に削除しますか？',
          })}>
            削除
          </button>
        </div>
      );

    case 'error':
      return (
        <div>
          <p>エラー: {state.message}</p>
          {state.retryCount < 3 && (
            <button onClick={() => dispatch({ type: 'RETRY' })}>
              リトライ ({state.retryCount}/3)
            </button>
          )}
          <button onClick={() => dispatch({ type: 'CLOSE' })}>閉じる</button>
        </div>
      );

    case 'confirming':
      return (
        <div>
          <p>{state.message}</p>
          <button onClick={() => dispatch({ type: 'CONFIRM_YES' })}>はい</button>
          <button onClick={() => dispatch({ type: 'CONFIRM_NO' })}>いいえ</button>
        </div>
      );
  }
}
*/
```

React の `useReducer` は State パターンそのものです。state の `status` フィールドが「どの State クラスか」に対応し、`switch (state.status)` が「状態ごとの振る舞い分岐」に対応します。discriminated union（判別可能共用型）により、各状態で利用可能なプロパティが型安全に制限されます。

---

## 9. 深掘り: State パターンの設計判断

### 遷移の責任をどこに持たせるか

```
方式1: State 内部で遷移（GoF の推奨）
  各 State クラスが次の状態を直接知っている
  PaidState.ship() → context.setState(new ShippedState())

  利点: 各状態が自律的、分散した制御
  欠点: 状態間の結合、遷移の全体像が見づらい

方式2: Context / 遷移テーブルで一元管理
  外部テーブルで全遷移を定義
  transitions['paid']['SHIP'] = 'shipped'

  利点: 遷移の全体像が明確、変更しやすい
  欠点: テーブルが大きくなる、ガード条件の表現が冗長

方式3: ハイブリッド
  単純な遷移はテーブル、複雑な遷移はメソッド内で
  guard 条件付き遷移はメソッドで、単純遷移はテーブルで
```

### State オブジェクトの生成戦略

```
方式1: 毎回 new で生成
  context.setState(new PaidState());
  利点: 状態固有のデータを保持可能
  欠点: GC 負荷

方式2: Singleton / 共有インスタンス
  context.setState(PaidState.INSTANCE);
  利点: メモリ効率が良い
  欠点: 状態にデータを持てない

方式3: Flyweight + 外部データ
  状態は共有、データは Context に保持
  利点: メモリ効率 + データ保持
  欠点: 実装がやや複雑

  判断基準: 状態固有データがあるか？
  ある → 方式1
  ない → 方式2（推奨）
```

---

## 10. 比較表

### State vs 他のパターン

| 特性 | State パターン | Strategy パターン | if/else 分岐 |
|------|--------------|-----------------|-------------|
| 振る舞いの切替 | 内部状態に応じて自動 | 外部から明示的に注入 | 条件分岐で決定 |
| 遷移の管理 | 状態クラスが遷移先を知る | 遷移の概念なし | コード内に散在 |
| 遷移の主体 | State 自身が遷移を決定 | Client が切り替え | なし |
| OCP (新しい状態の追加) | 新クラスを追加するだけ | 新戦略を追加するだけ | 全条件を修正 |
| テスタビリティ | 状態ごとに独立テスト | 戦略ごとに独立テスト | 全パス網羅が困難 |
| 複雑さ | 中（状態数に比例） | 低い | 状態数の二乗に比例 |

### FSM ライブラリの比較

| 特性 | XState | Robot | Zustand + 自前FSM | 自前実装 |
|------|--------|-------|--------------------|---------|
| 型安全性 | 高い（v5で大幅改善） | 高い | 中 | 実装次第 |
| 可視化 | Inspector / Visualizer | なし | なし | なし |
| 階層状態 | 対応 | 非対応 | 非対応 | 要実装 |
| 並行状態 | 対応 | 非対応 | 非対応 | 要実装 |
| バンドルサイズ | ~40KB | ~1KB | ~3KB + 自前 | 0 |
| 学習コスト | 高い | 低い | 中 | 低い |
| 実績 | 大規模プロダクト多数 | 中規模 | 多数 | -- |

### State パターンの導入判断

| 判断基準 | State パターンが有効 | if/else で十分 |
|---------|---------------------|---------------|
| 状態数 | 3つ以上 | 2つ以下 |
| 状態依存メソッド数 | 2つ以上 | 1つ |
| 新しい状態の追加 | 頻繁にありえる | ほぼ固定 |
| 遷移ルール | 複雑（ガード条件あり） | 単純 |
| テスト要件 | 状態ごとの独立テストが必要 | 網羅テストで十分 |

---

## 11. アンチパターン

### アンチパターン 1: 巨大な switch/if-else チェーン

```typescript
// ============================
// [NG] 状態ごとの分岐が全メソッドに散在
// ============================
class Order {
  status: string = 'pending';

  pay(): void {
    if (this.status === 'pending') {
      this.status = 'paid';
    } else if (this.status === 'paid') {
      throw new Error('Already paid');
    } else if (this.status === 'shipped') {
      throw new Error('Cannot pay after shipping');
    } else if (this.status === 'delivered') {
      throw new Error('Cannot pay after delivery');
    } else if (this.status === 'cancelled') {
      throw new Error('Order is cancelled');
    }
    // 新しい状態 "returned" を追加 → ここに分岐を追加
  }

  ship(): void {
    if (this.status === 'paid') { /* ... */ }
    else if (this.status === 'pending') { throw new Error('Must pay first'); }
    else if (this.status === 'shipped') { throw new Error('Already shipped'); }
    // ... 同じパターンの繰り返し
    // 新しい状態 "returned" を追加 → ここにも分岐を追加
  }

  // deliver(), cancel() も同様...
  // 5状態 x 4メソッド = 20箇所の条件分岐!!
}

// ============================
// [OK] State パターンで状態ごとのクラスに分離
// ============================
// → 各状態の振る舞いが1クラスにまとまり、保守性が向上
// → 新しい状態の追加は新クラスの追加のみ
// → 既存コードの修正不要（OCP 準拠）
// (実装は上記コード例 1 を参照)
```

### アンチパターン 2: 遷移の暗黙的な副作用

```typescript
// ============================
// [NG] 遷移の副作用が State 内に隠れている
// ============================
class PaidStateNG implements OrderState {
  readonly name = 'paid';

  ship(context: OrderContext): void {
    // ★ 遷移のたびに大量の副作用が暗黙的に実行される
    sendEmail(context.orderId, 'Your order has been shipped!');
    updateInventory(context.orderId);
    notifyWarehouse(context.orderId);
    logToAnalytics('order_shipped', context.orderId);
    context.setState(new ShippedState(), 'ship');
    // どの副作用が実行されるか、外から全く見えない
  }
  // ...
}

// ============================
// [OK] 副作用を遷移アクション/ミドルウェアとして明示
// ============================
// 方法1: FSM の action として宣言的に定義
const orderFSM = new FSM<OrderContext>({
  id: 'order',
  initial: 'pending',
  context: { orderId: '', items: [] },
  states: {
    paid: {
      on: {
        SHIP: {
          target: 'shipped',
          action: (ctx) => {
            // 副作用が明示的に定義されている
            emailService.send(ctx.orderId, 'shipped');
            inventoryService.update(ctx.orderId);
            warehouseService.notify(ctx.orderId);
          },
        },
      },
    },
    // ...
  },
});

// 方法2: Observer パターンと組合せ
// State の遷移をイベントとして通知し、
// 副作用はリスナー側で処理する
class OrderContextWithEvents extends OrderContext {
  private listeners: Array<(event: { from: string; to: string }) => void> = [];

  override setState(state: OrderState, action: string): void {
    const from = this.getStateName();
    super.setState(state, action);
    // 遷移後にイベント通知
    for (const listener of this.listeners) {
      listener({ from, to: state.name });
    }
  }

  onTransition(listener: (event: { from: string; to: string }) => void): void {
    this.listeners.push(listener);
  }
}

// 副作用をリスナーとして登録（テスト時はモックに差し替え可能）
const order2 = new OrderContextWithEvents('ORD-002');
order2.onTransition(({ from, to }) => {
  if (from === 'paid' && to === 'shipped') {
    emailService.send(order2.orderId, 'shipped');
    inventoryService.update(order2.orderId);
  }
});
```

### アンチパターン 3: 状態とデータの混同

```typescript
// ============================
// [NG] フラグの組み合わせで「状態」を表現
// ============================
class FormNG {
  isSubmitting: boolean = false;
  isValidating: boolean = false;
  hasError: boolean = false;
  isSuccess: boolean = false;

  submit(): void {
    if (this.isSubmitting) return;
    if (this.isValidating) return;
    // isSubmitting && hasError はどういう状態？
    // フラグの組み合わせ: 2^4 = 16 通り
    // 実際に有効な状態はその一部だが、不正な組み合わせを防げない
  }
}

// ============================
// [OK] Discriminated Union で有効な状態のみを表現
// ============================
type FormState =
  | { status: 'idle' }
  | { status: 'editing'; data: Record<string, string> }
  | { status: 'validating'; data: Record<string, string> }
  | { status: 'submitting'; data: Record<string, string> }
  | { status: 'success'; submittedAt: Date }
  | { status: 'error'; message: string; retryCount: number };

// 不正な状態の組み合わせが型レベルで存在しない
// status === 'idle' のときに data にアクセス → 型エラー
function handleForm(state: FormState): void {
  switch (state.status) {
    case 'error':
      console.log(state.message);      // OK: error 状態にのみ message がある
      console.log(state.retryCount);   // OK
      break;
    case 'idle':
      // console.log(state.message);   // 型エラー! idle に message はない
      break;
  }
}
```

---

## 12. 演習問題

### 演習 1（基礎）: ATM の状態管理

以下の仕様を満たす ATM の State パターンを実装してください。

**仕様:**
- 状態: `idle`（待機） → `cardInserted`（カード挿入済み） → `pinVerified`（暗証番号確認済み） → `transacting`（取引中） → `idle`
- 操作: `insertCard()`, `enterPin(pin)`, `selectTransaction(type)`, `ejectCard()`
- 暗証番号は3回間違えるとカードをロック（`locked` 状態）

**期待される出力:**
```
atm.insertCard('1234-5678')
→ [ATM] idle → cardInserted
atm.enterPin('0000')  // 間違い
→ "暗証番号が違います (残り2回)"
atm.enterPin('1234')  // 正解
→ [ATM] cardInserted → pinVerified
atm.selectTransaction('withdraw')
→ [ATM] pinVerified → transacting
atm.ejectCard()
→ [ATM] transacting → idle
```

---

### 演習 2（応用）: WebSocket 接続の FSM

以下の仕様を満たす WebSocket 接続状態の FSM を実装してください。

**仕様:**
- 状態: `disconnected`, `connecting`, `connected`, `reconnecting`, `error`
- イベント: `CONNECT`, `CONNECTED`, `DISCONNECT`, `ERROR`, `RETRY`
- ガード条件: `reconnecting` → `connecting` はリトライ回数が5回以下の場合のみ
- 自動リコネクト: `error` 状態で3秒後に自動 RETRY

**期待される出力:**
```
ws.send('CONNECT')    → disconnected → connecting
ws.send('CONNECTED')  → connecting → connected
ws.send('ERROR')      → connected → error → (3秒後) → reconnecting → connecting
ws.send('CONNECTED')  → connecting → connected
// 5回以上リトライ失敗:
ws.send('ERROR')      → "最大リトライ回数に達しました"
```

---

### 演習 3（発展）: ゲーム AI の行動状態マシン

以下の仕様を満たす敵キャラクター AI の階層状態マシンを実装してください。

**仕様:**
- 親状態: `alive`, `dead`
- `alive` の子状態: `idle`, `patrol`, `chase`, `attack`
- 遷移条件:
  - `idle` → `patrol`: 一定時間経過
  - `patrol` → `chase`: プレイヤーを検知（距離 < 100）
  - `chase` → `attack`: 攻撃範囲内（距離 < 20）
  - `chase` → `patrol`: プレイヤーを見失う（距離 > 150）
  - `attack` → `chase`: プレイヤーが攻撃範囲外に逃げる
  - `alive.*` → `dead`: HP <= 0

**期待される出力:**
```
enemy.update({ playerDistance: 200, hp: 100 })
→ [idle] 待機中...
enemy.update({ playerDistance: 80, hp: 100 })
→ [idle → chase] プレイヤー検知! 追跡開始
enemy.update({ playerDistance: 15, hp: 100 })
→ [chase → attack] 攻撃範囲内! 攻撃開始
enemy.update({ playerDistance: 15, hp: 0 })
→ [attack → dead] HP が 0 になりました（alive の子状態で共通ハンドリング）
```

---

## 13. FAQ

### Q1: State パターンと Strategy パターンの違いは何ですか？

構造（UML 図）はほぼ同じですが、**意図と遷移の有無**が本質的に異なります。

| 比較項目 | State | Strategy |
|---------|-------|----------|
| 意図 | 内部状態に応じて振る舞いが変わる | アルゴリズムを外部から注入 |
| 遷移 | State 自身が次の状態に遷移する | 遷移の概念なし |
| 切替の主体 | State オブジェクト自身 | Client（外部） |
| 典型例 | 注文のライフサイクル管理 | ソートアルゴリズムの選択 |

判断基準: 「振る舞いの切り替えが自動か手動か」。ユーザーの操作とは無関係に、内部の条件によって振る舞いが変わるなら State、外部から明示的にアルゴリズムを選択するなら Strategy です。

### Q2: 状態の数が多い場合はどうすべきですか？

状態が 10 以上になる場合は以下を検討してください。

1. **階層状態マシン（HSM）**: 共通の振る舞いを親状態にまとめる。XState はネイティブサポートあり。
2. **並行状態マシン**: 独立した関心事を別の FSM に分離し並行で動かす。例: 「接続状態」と「認証状態」は別 FSM。
3. **状態の分解**: 1つの「状態」が実は2つの独立した概念の組み合わせではないか検討する。例: `loadingAuthenticated` は `loading` x `authenticated` に分解。

### Q3: State パターンをどの程度の規模から導入すべきですか？

**状態が3つ以上**かつ、**状態によって振る舞いが異なるメソッドが2つ以上**ある場合に検討する価値があります。状態が2つで分岐も1〜2箇所なら、シンプルな if 文の方が可読性が高いです。

判断の公式:
```
修正コスト = 新しい状態の追加時に修正するファイル/メソッドの数
  if/else 方式: 修正コスト = メソッド数 × 1
  State 方式:   修正コスト = 1（新クラスの追加のみ）

修正コスト > 3 なら State パターンの導入効果あり
```

### Q4: XState を使うべきか、自前実装すべきか？

判断基準:

| 要件 | XState | 自前実装 |
|------|--------|---------|
| 階層状態が必要 | XState | 実装が大変 |
| 可視化が必要 | XState（Inspector） | 別途開発が必要 |
| バンドルサイズ制限 | 自前（~0KB） | -- |
| 並行状態が必要 | XState | 実装が非常に大変 |
| 単純な FSM（状態5個以下） | 自前 | -- |
| チーム全員が XState を知っている | XState | -- |

### Q5: State パターンと状態管理ライブラリ（Redux, Zustand）の関係は？

Redux や Zustand は「アプリケーション全体の状態」を管理するライブラリであり、State パターンは「特定のオブジェクトの振る舞いの切り替え」を管理するパターンです。

両者は排他ではなく、**併用**するのが一般的です。例: Redux の store に注文のステータス（`pending`, `paid`, ...）を保持しつつ、各ステータスの振る舞い（許可される操作、UI の表示内容）は State パターンの reducer で管理する。React の `useReducer` は State パターンの典型的な実装例です。

---

## まとめ

| 項目 | 要点 |
|------|------|
| State パターンの本質 | 条件分岐をポリモーフィズムに変換。if/else の爆発を防止 |
| Context | 現在の状態を保持し、操作を状態オブジェクトに委譲 |
| 遷移の責任 | State 自身が次の状態を決定する（Strategy との決定的な違い） |
| 型安全 FSM | TypeScript の型システムで遷移を制約。不正な遷移をコンパイル時に検出 |
| 宣言的 FSM | 設定オブジェクトで状態遷移を定義。可視化・テストが容易 |
| 階層状態マシン (HSM) | 親子関係で状態を整理。イベント委譲による共通処理の集約 |
| React との連携 | useReducer + Discriminated Union が State パターンの実装 |
| 導入判断 | 状態3以上 & 状態依存メソッド2以上で検討 |

---

## 次に読むべきガイド

- [02-command.md](./02-command.md) -- Command パターンと Undo/Redo（Command + State で状態付き操作管理）
- [01-strategy.md](./01-strategy.md) -- Strategy パターン（State との構造的類似点と意図の違い）
- [04-iterator.md](./04-iterator.md) -- Iterator パターンとジェネレータ
- [00-observer.md](./00-observer.md) -- Observer パターン（State 遷移の通知に活用）
- [Event Sourcing / CQRS](../../03-software-design/system-design-guide/) -- State の変化をイベントとして記録

---

## 参考文献

1. **Design Patterns: Elements of Reusable Object-Oriented Software** -- Gamma, Helm, Johnson, Vlissides (GoF, 1994) -- State パターンの原典。Chapter 5, pp.305-313
2. **XState Documentation** -- https://xstate.js.org/docs/ -- 宣言的状態マシンライブラリの公式ドキュメント
3. **Statecharts: A Visual Formalism for Complex Systems** -- David Harel (1987) -- 階層状態マシンの理論的基盤。State パターンの学術的ルーツ
4. **Refactoring.Guru - State** -- https://refactoring.guru/design-patterns/state -- 図解と多言語実装例
5. **Ian Horrocks - Constructing the User Interface with Statecharts** -- Addison-Wesley (1999) -- UI 設計における状態マシンの実践的ガイド
