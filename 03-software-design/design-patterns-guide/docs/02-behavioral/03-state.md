# State パターン

> オブジェクトの内部状態に応じて振る舞いを動的に切り替え、有限状態マシン (FSM) を型安全に実装する

## この章で学ぶこと

1. **State パターンの基本構造** — 状態ごとにクラスを分離し、条件分岐の爆発を防ぐ設計手法
2. **有限状態マシン (FSM) の実装** — 状態遷移図をコードに落とし込む体系的アプローチ
3. **状態遷移の検証と可視化** — 不正な遷移を型システムで防止し、状態遷移図を自動生成する

---

## 1. State パターンの構造

```
State パターンの構成要素:

  ┌──────────────┐         ┌──────────────────┐
  │   Context    │────────►│   State (抽象)    │
  │ (コンテキスト) │         │                  │
  │              │         │ + handle()        │
  │ - state ─────┤         └────────┬─────────┘
  │ + request()  │                  │
  └──────────────┘            ┌─────┴──────┐
                              │            │
                     ┌────────┴──┐  ┌──────┴────────┐
                     │ ConcreteA │  │ ConcreteB     │
                     │ State     │  │ State         │
                     │           │  │               │
                     │ handle()  │  │ handle()      │
                     │ → 状態Aの │  │ → 状態Bの     │
                     │   振る舞い │  │   振る舞い     │
                     └───────────┘  └───────────────┘

  Context.request() は state.handle() に委譲
  handle() の中で Context の状態を遷移させる
```

---

## 2. 基本実装 — 注文ステータス

```typescript
// order-state.ts — 注文の状態管理
interface OrderState {
  readonly name: string;
  pay(context: OrderContext): void;
  ship(context: OrderContext): void;
  deliver(context: OrderContext): void;
  cancel(context: OrderContext): void;
}

class OrderContext {
  private state: OrderState;
  private history: { state: string; timestamp: Date }[] = [];

  constructor(
    public readonly orderId: string,
    initialState: OrderState = new PendingState()
  ) {
    this.state = initialState;
    this.recordTransition();
  }

  setState(state: OrderState): void {
    console.log(`[${this.orderId}] ${this.state.name} → ${state.name}`);
    this.state = state;
    this.recordTransition();
  }

  getStateName(): string {
    return this.state.name;
  }

  private recordTransition(): void {
    this.history.push({ state: this.state.name, timestamp: new Date() });
  }

  // 状態に委譲
  pay(): void { this.state.pay(this); }
  ship(): void { this.state.ship(this); }
  deliver(): void { this.state.deliver(this); }
  cancel(): void { this.state.cancel(this); }
}

// 具体的な状態クラス
class PendingState implements OrderState {
  readonly name = 'pending';

  pay(context: OrderContext): void {
    console.log('決済処理を実行...');
    context.setState(new PaidState());
  }

  ship(_context: OrderContext): void {
    throw new Error('未決済の注文は発送できません');
  }

  deliver(_context: OrderContext): void {
    throw new Error('未決済の注文は配達完了にできません');
  }

  cancel(context: OrderContext): void {
    console.log('注文をキャンセルしました');
    context.setState(new CancelledState());
  }
}

class PaidState implements OrderState {
  readonly name = 'paid';

  pay(_context: OrderContext): void {
    throw new Error('すでに決済済みです');
  }

  ship(context: OrderContext): void {
    console.log('発送処理を実行...');
    context.setState(new ShippedState());
  }

  deliver(_context: OrderContext): void {
    throw new Error('発送前に配達完了にはできません');
  }

  cancel(context: OrderContext): void {
    console.log('返金処理を実行...');
    context.setState(new CancelledState());
  }
}

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
    context.setState(new DeliveredState());
  }

  cancel(_context: OrderContext): void {
    throw new Error('発送済みの注文はキャンセルできません');
  }
}

class DeliveredState implements OrderState {
  readonly name = 'delivered';

  pay(): void { throw new Error('配達済みの注文に決済はできません'); }
  ship(): void { throw new Error('配達済みの注文は発送できません'); }
  deliver(): void { throw new Error('すでに配達済みです'); }
  cancel(): void { throw new Error('配達済みの注文はキャンセルできません'); }
}

class CancelledState implements OrderState {
  readonly name = 'cancelled';

  pay(): void { throw new Error('キャンセル済みの注文に決済はできません'); }
  ship(): void { throw new Error('キャンセル済みの注文は発送できません'); }
  deliver(): void { throw new Error('キャンセル済みの注文は配達完了にできません'); }
  cancel(): void { throw new Error('すでにキャンセル済みです'); }
}
```

---

## 3. 型安全な FSM (有限状態マシン)

```typescript
// typed-fsm.ts — 型で遷移を制約する FSM
type OrderStatus = 'pending' | 'paid' | 'shipped' | 'delivered' | 'cancelled';
type OrderEvent = 'PAY' | 'SHIP' | 'DELIVER' | 'CANCEL';

// 許可される遷移を型で定義
type TransitionMap = {
  pending:   { PAY: 'paid'; CANCEL: 'cancelled' };
  paid:      { SHIP: 'shipped'; CANCEL: 'cancelled' };
  shipped:   { DELIVER: 'delivered' };
  delivered: {};
  cancelled: {};
};

// 型安全な遷移関数
type Transition<
  S extends OrderStatus,
  E extends keyof TransitionMap[S]
> = TransitionMap[S][E];

// FSM クラス
class StateMachine<S extends OrderStatus> {
  constructor(private currentState: S) {}

  getState(): S {
    return this.currentState;
  }

  transition<E extends keyof TransitionMap[S]>(
    event: E
  ): StateMachine<TransitionMap[S][E] & OrderStatus> {
    const transitions = STATE_TRANSITIONS[this.currentState] as Record<string, OrderStatus>;
    const nextState = transitions[event as string];

    if (!nextState) {
      throw new Error(
        `Invalid transition: ${this.currentState} + ${String(event)}`
      );
    }

    return new StateMachine(nextState as TransitionMap[S][E] & OrderStatus);
  }
}

const STATE_TRANSITIONS: Record<OrderStatus, Partial<Record<OrderEvent, OrderStatus>>> = {
  pending:   { PAY: 'paid', CANCEL: 'cancelled' },
  paid:      { SHIP: 'shipped', CANCEL: 'cancelled' },
  shipped:   { DELIVER: 'delivered' },
  delivered: {},
  cancelled: {},
};

// 使用例 — コンパイル時に不正な遷移を検出
const machine = new StateMachine('pending' as const);
const paid = machine.transition('PAY');      // OK: pending → paid
const shipped = paid.transition('SHIP');     // OK: paid → shipped
// paid.transition('DELIVER');               // 型エラー! paid から DELIVER は不可
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

  ※ delivered, cancelled は終端状態 (遷移先なし)
```

---

## 4. XState 風の宣言的 FSM

```typescript
// declarative-fsm.ts — 設定ベースの FSM
interface StateConfig<TContext> {
  on?: Record<string, {
    target: string;
    guard?: (context: TContext) => boolean;
    action?: (context: TContext) => void;
  }>;
  entry?: (context: TContext) => void;
  exit?: (context: TContext) => void;
}

interface MachineConfig<TContext> {
  id: string;
  initial: string;
  context: TContext;
  states: Record<string, StateConfig<TContext>>;
}

class FSM<TContext> {
  private currentState: string;
  private context: TContext;
  private config: MachineConfig<TContext>;

  constructor(config: MachineConfig<TContext>) {
    this.config = config;
    this.currentState = config.initial;
    this.context = { ...config.context };

    // 初期状態の entry を実行
    this.config.states[this.currentState]?.entry?.(this.context);
  }

  send(event: string): void {
    const stateConfig = this.config.states[this.currentState];
    const transition = stateConfig?.on?.[event];

    if (!transition) {
      console.warn(`No transition for event "${event}" in state "${this.currentState}"`);
      return;
    }

    // ガード条件のチェック
    if (transition.guard && !transition.guard(this.context)) {
      console.warn(`Guard prevented transition: ${this.currentState} → ${transition.target}`);
      return;
    }

    // exit → action → entry の順序で実行
    stateConfig?.exit?.(this.context);
    transition.action?.(this.context);

    this.currentState = transition.target;
    this.config.states[this.currentState]?.entry?.(this.context);
  }

  getState(): string { return this.currentState; }
  getContext(): TContext { return { ...this.context }; }
}

// 使用例: 信号機
const trafficLight = new FSM({
  id: 'traffic-light',
  initial: 'red',
  context: { cycleCount: 0 },
  states: {
    red: {
      entry: (ctx) => console.log(`赤信号 (サイクル: ${ctx.cycleCount})`),
      on: {
        TIMER: { target: 'green', action: (ctx) => { ctx.cycleCount++; } },
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

trafficLight.send('TIMER'); // 赤 → 青
trafficLight.send('TIMER'); // 青 → 黄
trafficLight.send('TIMER'); // 黄 → 赤
```

---

## 5. 実用例 — フォームバリデーション状態

```typescript
// form-state.ts — フォームの状態管理
type FormStatus = 'idle' | 'editing' | 'validating' | 'submitting' | 'success' | 'error';

interface FormContext {
  data: Record<string, string>;
  errors: Record<string, string>;
  submitCount: number;
}

const formMachine = new FSM<FormContext>({
  id: 'form',
  initial: 'idle',
  context: { data: {}, errors: {}, submitCount: 0 },
  states: {
    idle: {
      on: {
        CHANGE: { target: 'editing' },
      },
    },
    editing: {
      on: {
        CHANGE: { target: 'editing' },
        SUBMIT: {
          target: 'validating',
          action: (ctx) => { ctx.submitCount++; },
        },
      },
    },
    validating: {
      entry: (ctx) => {
        // バリデーション実行
        ctx.errors = {};
        if (!ctx.data.email?.includes('@')) {
          ctx.errors.email = 'メールアドレスの形式が不正です';
        }
      },
      on: {
        VALID: { target: 'submitting' },
        INVALID: { target: 'editing' },
      },
    },
    submitting: {
      entry: () => console.log('送信中...'),
      on: {
        SUCCESS: { target: 'success' },
        FAILURE: { target: 'error' },
      },
    },
    success: {
      entry: () => console.log('送信成功'),
      on: {
        RESET: { target: 'idle' },
      },
    },
    error: {
      on: {
        RETRY: {
          target: 'submitting',
          guard: (ctx) => ctx.submitCount < 3,
        },
        RESET: { target: 'idle' },
      },
    },
  },
});
```

---

## 6. 比較表

| 特性 | State パターン | Strategy パターン | if/else 分岐 |
|------|--------------|-----------------|-------------|
| 振る舞いの切替 | 内部状態に応じて自動 | 外部から明示的に注入 | 条件分岐 |
| 遷移の管理 | 状態クラスが遷移先を知る | 遷移の概念なし | コード内に散在 |
| Open/Closed | 新しい状態を追加しやすい | 新しい戦略を追加しやすい | 全条件を修正 |
| 複雑さ | 中 | 低い | 状態数に比例して増大 |
| テスタビリティ | 状態ごとに独立テスト可能 | 戦略ごとに独立テスト可能 | 全パス網羅が困難 |

| FSM ライブラリ | XState | Robot | 自前実装 |
|---------------|--------|-------|---------|
| 型安全性 | 高い | 高い | 実装次第 |
| 可視化 | 対応 (Inspector) | なし | なし |
| 階層状態 | 対応 | 非対応 | 要実装 |
| 並行状態 | 対応 | 非対応 | 要実装 |
| バンドルサイズ | 大きい (~40KB) | 小さい (~1KB) | 0 |
| 学習コスト | 高い | 低い | 中 |

---

## 7. アンチパターン

### アンチパターン 1: 巨大な switch/if-else チェーン

```typescript
// 悪い例: 状態ごとの分岐が散在
class Order {
  status: string = 'pending';

  pay(): void {
    if (this.status === 'pending') {
      this.status = 'paid';
    } else if (this.status === 'paid') {
      throw new Error('Already paid');
    } else if (this.status === 'shipped') {
      throw new Error('Cannot pay after shipping');
    }
    // ... 状態が増えるたびに全メソッドに分岐を追加
  }

  ship(): void {
    if (this.status === 'paid') { /* ... */ }
    else if (this.status === 'pending') { /* ... */ }
    // ... 同じ問題の繰り返し
  }
}

// 良い例: State パターンで状態ごとのクラスに分離
// → 各状態の振る舞いが1箇所にまとまり、保守性が向上
```

### アンチパターン 2: 遷移の暗黙的な副作用

```typescript
// 悪い例: 遷移の副作用がState内に隠れている
class PaidState implements OrderState {
  ship(context: OrderContext): void {
    sendEmail(context.orderId);       // メール送信
    updateInventory(context.orderId); // 在庫更新
    notifyWarehouse(context.orderId); // 倉庫通知
    context.setState(new ShippedState());
  }
}

// 良い例: 副作用を遷移アクションとして明示
// FSM の設定で entry/exit/action を宣言的に定義
// テスト時にモックしやすく、副作用の全体像が把握しやすい
```

---

## 8. FAQ

### Q1: State パターンと Strategy パターンの違いは何ですか？

構造はほぼ同じですが意図が異なります。**State** は「内部状態に応じて振る舞いが変わる」ことがポイントで、状態遷移の概念があります。**Strategy** は「アルゴリズムを外部から注入して切り替える」ことがポイントで、遷移の概念はありません。注文のライフサイクル管理は State、ソートアルゴリズムの選択は Strategy が適切です。

### Q2: 状態の数が多い場合はどうすべきですか？

状態が 10 以上になる場合は**階層状態マシン（HSM: Hierarchical State Machine）**を検討してください。共通の振る舞いを親状態にまとめ、差分だけを子状態で定義します。XState はこの HSM をネイティブにサポートしています。また、状態の組み合わせ爆発を避けるために、独立した関心事は別の状態マシンに分離し、並行で動かすことも有効です。

### Q3: State パターンをどの程度の規模から導入すべきですか？

状態が 3 つ以上かつ、状態によって振る舞いが異なるメソッドが 2 つ以上ある場合に検討する価値があります。状態が 2 つで分岐も 1〜2 箇所程度なら、シンプルな if 文の方が可読性が高いです。判断基準は「新しい状態を追加するとき、既存コードの何箇所を修正する必要があるか」です。修正箇所が多いなら State パターンの導入効果があります。

---

## まとめ

| 項目 | 要点 |
|------|------|
| State パターン | 状態ごとのクラスで振る舞いを分離。条件分岐の爆発を防止 |
| Context | 現在の状態を保持し、操作を状態オブジェクトに委譲 |
| FSM | 状態と遷移を宣言的に定義。不正な遷移を防止 |
| 型安全 FSM | TypeScript の型システムで遷移を制約 |
| 宣言的 FSM | 設定オブジェクトで状態遷移を定義。可視化・テストが容易 |
| 階層状態マシン | 状態が多い場合に親子関係で整理 |

---

## 次に読むべきガイド

- [02-command.md](./02-command.md) — Command パターンと Undo/Redo
- [04-iterator.md](./04-iterator.md) — Iterator パターンとジェネレータ
- [../04-architectural/02-event-sourcing-cqrs.md](../04-architectural/02-event-sourcing-cqrs.md) — Event Sourcing/CQRS

---

## 参考文献

1. **Design Patterns** — Gamma, Helm, Johnson, Vlissides (GoF, 1994) — State パターンの原典
2. **XState Documentation** — https://xstate.js.org/docs/ — 宣言的状態マシンライブラリ
3. **Statecharts: A Visual Formalism for Complex Systems** — David Harel (1987) — 階層状態マシンの理論的基盤
