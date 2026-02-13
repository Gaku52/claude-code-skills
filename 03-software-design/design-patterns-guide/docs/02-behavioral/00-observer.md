# Observer パターン

> オブジェクト間に **1対多** の依存関係を定義し、あるオブジェクトの状態変化を依存するすべてのオブジェクトに自動通知する振る舞いパターン。イベント駆動設計の基盤であり、疎結合なシステムを構築するための最重要パターンの1つである。

---

## この章で学ぶこと

1. Observer パターン（Pub/Sub）の構造とイベント駆動設計の基礎を理解し、GoF の設計意図と現代の適用を把握する
2. Push 型と Pull 型の通知モデルの違いと使い分け、型安全な EventEmitter の設計方法を習得する
3. メモリリーク防止、イベントの順序保証、非同期通知、バックプレッシャーなど実運用上の課題と対策を身につける

---

## 前提知識

このガイドを読む前に、以下の概念を理解しておくことを推奨します。

| 前提知識 | 説明 | 参照リンク |
|---------|------|-----------|
| インタフェースとポリモーフィズム | 共通の契約を通じて異なる型を統一的に扱う概念 | [SOLID 原則](../../../clean-code-principles/docs/00-principles/01-solid.md) |
| コールバック関数 | 他の関数に渡されて後から呼び出される関数 | [関数型パターン](../03-functional/02-fp-patterns.md) |
| 依存性の方向と結合度 | モジュール間の依存関係の管理 | [クリーンコード原則](../../../clean-code-principles/docs/00-principles/) |
| Promise/async-await | 非同期処理の基礎（非同期 Observer の理解に必要） | [モナドパターン](../03-functional/00-monad.md) |

---

## 1. Observer パターンとは何か

### 1.1 解決する問題

ソフトウェアシステムでは「ある部分の状態が変わったとき、それに依存する他の部分を更新したい」という要求が頻繁にある。

- ユーザーが商品を購入したら、メール送信・在庫更新・分析データ記録を行いたい
- データが変更されたら、関連するUIを全て再描画したい
- センサーの値が変わったら、モニター・アラーム・ログの全てに反映したい

これらを直接的な関数呼び出しで実装すると、呼び出し元が全ての依存先を知る必要があり、**密結合**になる。新しい依存先を追加するたびに呼び出し元のコードを変更しなければならず、Open/Closed Principle に違反する。

### 1.2 パターンの意図

GoF の定義:

> Define a one-to-many dependency between objects so that when one object changes state, all its dependents are notified and updated automatically.

日本語訳:

> オブジェクト間に1対多の依存関係を定義し、あるオブジェクトの状態が変化したとき、依存する全てのオブジェクトに自動的に通知され更新されるようにする。

### 1.3 WHY: なぜ Observer パターンが必要なのか

根本的な理由は **依存関係の方向を逆転させる** ことにある。

```
直接呼び出し（密結合）:
  OrderService ----> EmailService
              |----> InventoryService
              |----> AnalyticsService
  問題: OrderService が全ての後続処理を知っている

Observer パターン（疎結合）:
  OrderService --emit("ordered")--> EventBus
                                      |
  EmailService      <--- subscribe ---+
  InventoryService  <--- subscribe ---+
  AnalyticsService  <--- subscribe ---+
  利点: OrderService は後続処理を知らない
```

1. **疎結合**: Subject は Observer の具体的な型を知らない。インタフェースのみに依存する
2. **Open/Closed Principle**: 新しい Observer を追加しても Subject のコードは変更不要
3. **実行時の動的構成**: Observer の登録・解除を実行時に自由に行える

---

## 2. Observer の構造

### 2.1 クラス図

```
+------------------+          +------------------+
|    Subject       |  1    *  |    Observer      |
|    (Publisher)   |--------->|   (Subscriber)   |
+------------------+          +------------------+
| - observers[]   |          | + update(data)   |
| + subscribe()   |          +------------------+
| + unsubscribe() |                  ^
| + notify()      |           _______|_______
+------------------+          |              |
                        +----------+  +----------+
                        |ObserverA |  |ObserverB |
                        +----------+  +----------+
```

### 2.2 構成要素の役割

| 構成要素 | 役割 | 責務 |
|---------|------|------|
| Subject (Publisher) | 状態を保持し、変更時に通知を発行 | Observer の登録/解除/通知の管理 |
| Observer (Subscriber) | Subject の変更に反応 | update() で通知を受け取り処理 |
| ConcreteSubject | 具体的な状態を持つ Subject | 状態変更時に notify() を呼ぶ |
| ConcreteObserver | 具体的な反応ロジック | update() に応じた処理を実行 |

### 2.3 処理シーケンス

```
時系列の処理フロー:

Client          Subject              ObserverA       ObserverB
  |                |                    |               |
  |-- subscribe(A) -->|                 |               |
  |                |-- 登録 ----------->|               |
  |-- subscribe(B) -->|                 |               |
  |                |-- 登録 --------------------------->|
  |                |                    |               |
  |-- setState() --->|                  |               |
  |                |-- notify() ------->|               |
  |                |   update(data)     |               |
  |                |-- notify() --------------------------->|
  |                |                    |   update(data)|
  |                |                    |               |
  |-- unsubscribe(A)->|                |               |
  |                |-- 解除 ----------->|               |
  |                |                    |               |
  |-- setState() --->|                  |               |
  |                |-- notify() --------------------------->|
  |                |                    |   update(data)|
  |                |   (A には通知されない)              |
```

---

## 3. Push 型 vs Pull 型

Observer パターンの通知モデルには Push 型と Pull 型の2種類がある。

```
Push型: Subject が変更データを直接渡す
Subject --notify(data)--> Observer
  利点: Observer は必要なデータをすぐ取得できる
  欠点: 不要なデータも送られる、データ量が大きいと非効率

Pull型: Subject は通知のみ、Observer が取りに行く
Subject --notify()--> Observer --getState()--> Subject
  利点: Observer が必要なデータだけ取得できる
  欠点: Subject への追加アクセスが必要、Subject の公開インタフェースが増える

ハイブリッド型: イベント種別を通知し、Observer が詳細を取得
Subject --notify(eventType)--> Observer --getRelevantData()--> Subject
  利点: Push と Pull の良いとこ取り
  欠点: 設計がやや複雑
```

### Push 型 vs Pull 型の判断基準

| 判断基準 | Push 型 | Pull 型 |
|---------|---------|---------|
| データが小さく固定的 | 適切 | 過剰 |
| Observer ごとに必要なデータが異なる | 非効率 | 適切 |
| Subject の状態が頻繁に変わる | 変更データのみ送信で効率的 | 毎回取得で非効率 |
| Observer 数が多い | 各 Observer に同じデータを送信 | 各 Observer が個別に取得 |
| リアルタイム性が重要 | 適切（遅延なし） | 追加通信による遅延 |

---

## 4. コード例

### コード例 1: 型安全な EventEmitter（TypeScript）

```typescript
// typed-event-emitter.ts — TypeScript で型安全な Observer パターン

// イベントマップ: イベント名とそのデータ型を定義
type EventMap = {
  userCreated: { id: string; name: string; email: string };
  userDeleted: { id: string; reason: string };
  userUpdated: { id: string; changes: Partial<{ name: string; email: string }> };
  orderPlaced: { orderId: string; userId: string; total: number };
};

class TypedEventEmitter<T extends Record<string, any>> {
  private listeners = new Map<keyof T, Set<Function>>();
  private onceListeners = new Map<keyof T, Set<Function>>();

  /**
   * イベントを購読する
   * @returns unsubscribe 関数
   */
  on<K extends keyof T>(event: K, handler: (data: T[K]) => void): () => void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(handler);

    // unsubscribe 関数を返す（クリーンアップ用）
    return () => {
      this.listeners.get(event)?.delete(handler);
    };
  }

  /**
   * 一度だけ購読する
   */
  once<K extends keyof T>(event: K, handler: (data: T[K]) => void): () => void {
    if (!this.onceListeners.has(event)) {
      this.onceListeners.set(event, new Set());
    }
    this.onceListeners.get(event)!.add(handler);

    return () => {
      this.onceListeners.get(event)?.delete(handler);
    };
  }

  /**
   * イベントを発行する
   */
  emit<K extends keyof T>(event: K, data: T[K]): void {
    // 通常のリスナー
    this.listeners.get(event)?.forEach(fn => fn(data));

    // once リスナー（実行後に削除）
    const onceHandlers = this.onceListeners.get(event);
    if (onceHandlers) {
      onceHandlers.forEach(fn => fn(data));
      onceHandlers.clear();
    }
  }

  /**
   * 特定イベントの全リスナーを解除
   */
  removeAllListeners<K extends keyof T>(event?: K): void {
    if (event) {
      this.listeners.delete(event);
      this.onceListeners.delete(event);
    } else {
      this.listeners.clear();
      this.onceListeners.clear();
    }
  }

  /**
   * リスナー数を取得（デバッグ・監視用）
   */
  listenerCount<K extends keyof T>(event: K): number {
    return (this.listeners.get(event)?.size ?? 0) +
           (this.onceListeners.get(event)?.size ?? 0);
  }
}

// --- 使用例 ---
const bus = new TypedEventEmitter<EventMap>();

// 型安全: handler の引数は自動推論される
const unsubUser = bus.on("userCreated", (user) => {
  console.log(`Welcome, ${user.name}! (${user.email})`);
  // user.id, user.name, user.email が型安全にアクセス可能
});

bus.on("orderPlaced", (order) => {
  console.log(`Order ${order.orderId}: $${order.total}`);
});

// once: 最初の1回だけ
bus.once("userCreated", (user) => {
  console.log(`First user bonus for ${user.name}!`);
});

bus.emit("userCreated", { id: "1", name: "Taro", email: "taro@example.com" });
// "Welcome, Taro! (taro@example.com)"
// "First user bonus for Taro!"

bus.emit("userCreated", { id: "2", name: "Hanako", email: "hanako@example.com" });
// "Welcome, Hanako! (hanako@example.com)"
// (once は実行されない)

unsubUser(); // 購読解除

bus.emit("userCreated", { id: "3", name: "Jiro", email: "jiro@example.com" });
// (何も出力されない — 解除済み)
```

### コード例 2: React ── カスタム Observable Hook

```typescript
// use-observable.ts — React でリアクティブデータを扱う Hook
import { useState, useEffect, useRef, useCallback } from 'react';

// Observable インタフェース
interface Observable<T> {
  subscribe(observer: (value: T) => void): { unsubscribe: () => void };
  getValue(): T;
}

// SimpleObservable 実装
class SimpleObservable<T> implements Observable<T> {
  private observers = new Set<(value: T) => void>();
  private currentValue: T;

  constructor(initialValue: T) {
    this.currentValue = initialValue;
  }

  getValue(): T {
    return this.currentValue;
  }

  next(value: T): void {
    this.currentValue = value;
    this.observers.forEach(observer => observer(value));
  }

  subscribe(observer: (value: T) => void): { unsubscribe: () => void } {
    this.observers.add(observer);
    return {
      unsubscribe: () => this.observers.delete(observer),
    };
  }
}

// React Hook
function useObservable<T>(observable$: Observable<T>): T {
  const [value, setValue] = useState<T>(() => observable$.getValue());

  useEffect(() => {
    // 値が変わっている可能性があるので同期
    setValue(observable$.getValue());

    const subscription = observable$.subscribe(setValue);
    return () => subscription.unsubscribe(); // クリーンアップ
  }, [observable$]);

  return value;
}

// 複数の Observable を組み合わせる Hook
function useCombinedObservable<T extends Record<string, Observable<any>>>(
  observables: T
): { [K in keyof T]: T[K] extends Observable<infer U> ? U : never } {
  const keys = Object.keys(observables) as (keyof T)[];
  const [values, setValues] = useState(() => {
    const initial: any = {};
    keys.forEach(key => {
      initial[key] = observables[key].getValue();
    });
    return initial;
  });

  useEffect(() => {
    const subscriptions = keys.map(key =>
      observables[key].subscribe((val: any) => {
        setValues((prev: any) => ({ ...prev, [key]: val }));
      })
    );
    return () => subscriptions.forEach(sub => sub.unsubscribe());
  }, []);

  return values;
}

// --- 使用例 ---
// グローバルな Observable ストア
const priceStore = new SimpleObservable<number>(100);
const statusStore = new SimpleObservable<string>("idle");

function StockPrice({ symbol }: { symbol: string }) {
  const price = useObservable(priceStore);
  const status = useObservable(statusStore);

  return (
    <div>
      <span>{symbol}: ${price}</span>
      <span>Status: {status}</span>
    </div>
  );
}

// 外部から値を更新
priceStore.next(105); // 全ての購読コンポーネントが自動再描画
```

### コード例 3: Node.js EventEmitter ── ドメインイベント

```typescript
// order-service.ts — Node.js EventEmitter を使ったドメインイベント
import { EventEmitter } from "events";

interface Order {
  id: string;
  userId: string;
  items: { productId: string; quantity: number; price: number }[];
  total: number;
  status: string;
}

// ドメインサービス: 注文処理
class OrderService extends EventEmitter {
  private orders = new Map<string, Order>();

  async placeOrder(userId: string, items: Order["items"]): Promise<Order> {
    const order: Order = {
      id: `ORD-${Date.now()}`,
      userId,
      items,
      total: items.reduce((sum, i) => sum + i.price * i.quantity, 0),
      status: "confirmed",
    };

    this.orders.set(order.id, order);

    // ドメインイベントを発行（OrderService は後続処理を知らない）
    this.emit("orderPlaced", order);
    return order;
  }

  async cancelOrder(orderId: string): Promise<void> {
    const order = this.orders.get(orderId);
    if (!order) throw new Error(`Order not found: ${orderId}`);

    order.status = "cancelled";
    this.emit("orderCancelled", order);
  }
}

// --- Observer の登録（アプリケーション起動時） ---
const orderService = new OrderService();

// メール送信
orderService.on("orderPlaced", (order: Order) => {
  console.log(`[Email] Sending confirmation for order ${order.id}`);
  // emailService.sendConfirmation(order);
});

// 在庫管理
orderService.on("orderPlaced", (order: Order) => {
  console.log(`[Inventory] Decrementing stock for ${order.items.length} items`);
  // order.items.forEach(item => inventoryService.decrement(item.productId, item.quantity));
});

// 分析
orderService.on("orderPlaced", (order: Order) => {
  console.log(`[Analytics] Recording purchase: $${order.total}`);
  // analyticsService.trackPurchase(order);
});

// キャンセル時のハンドラ
orderService.on("orderCancelled", (order: Order) => {
  console.log(`[Email] Sending cancellation notice for ${order.id}`);
  console.log(`[Inventory] Restoring stock for ${order.items.length} items`);
});

// --- 使用例 ---
orderService.placeOrder("user-1", [
  { productId: "p-1", quantity: 2, price: 1000 },
  { productId: "p-2", quantity: 1, price: 3000 },
]);
// [Email] Sending confirmation for order ORD-...
// [Inventory] Decrementing stock for 2 items
// [Analytics] Recording purchase: $5000
```

### コード例 4: Python ── Observer with WeakRef

```python
# event_bus.py — WeakRef でメモリリークを防ぐ Observer パターン
from __future__ import annotations
import weakref
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Event:
    """イベントの基底クラス"""
    timestamp: datetime
    source: str


@dataclass
class UserCreatedEvent(Event):
    user_id: str
    name: str
    email: str


@dataclass
class OrderPlacedEvent(Event):
    order_id: str
    user_id: str
    total: float


class EventBus:
    """WeakRef 対応の EventBus"""

    def __init__(self) -> None:
        self._subscribers: Dict[str, List[weakref.ref]] = {}
        self._function_subscribers: Dict[str, List[Callable]] = {}

    def subscribe(self, event_type: str, handler: Callable) -> Callable[[], None]:
        """
        イベントを購読する。
        handler がメソッドの場合は WeakRef で保持し、
        GC 時に自動的に解除される。
        """
        if hasattr(handler, '__self__'):
            # バウンドメソッドの場合: WeakRef で保持
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            ref = weakref.ref(handler.__self__)
            method_name = handler.__func__.__name__
            self._subscribers[event_type].append(ref)
        else:
            # 通常の関数の場合
            if event_type not in self._function_subscribers:
                self._function_subscribers[event_type] = []
            self._function_subscribers[event_type].append(handler)

        def unsubscribe() -> None:
            if event_type in self._function_subscribers:
                try:
                    self._function_subscribers[event_type].remove(handler)
                except ValueError:
                    pass

        return unsubscribe

    def publish(self, event_type: str, data: Any = None) -> None:
        """イベントを発行する"""
        # 通常の関数ハンドラ
        for handler in self._function_subscribers.get(event_type, []):
            handler(data)

        # WeakRef ハンドラ（GC 済みのものを除去）
        if event_type in self._subscribers:
            alive_refs = []
            for ref in self._subscribers[event_type]:
                obj = ref()
                if obj is not None:
                    alive_refs.append(ref)
                    # handle メソッドを呼び出す
                    if hasattr(obj, 'handle_event'):
                        obj.handle_event(event_type, data)
            self._subscribers[event_type] = alive_refs


# --- 使用例 ---
bus = EventBus()


def on_user_created(event: UserCreatedEvent) -> None:
    print(f"[Handler] Welcome email sent to {event.email}")


unsub = bus.subscribe("user.created", on_user_created)

bus.publish("user.created", UserCreatedEvent(
    timestamp=datetime.now(),
    source="user-service",
    user_id="u-1",
    name="Taro",
    email="taro@example.com",
))
# [Handler] Welcome email sent to taro@example.com

unsub()  # 購読解除

bus.publish("user.created", UserCreatedEvent(
    timestamp=datetime.now(),
    source="user-service",
    user_id="u-2",
    name="Hanako",
    email="hanako@example.com",
))
# (何も出力されない)
```

### コード例 5: 非同期 Observer（Promise ベース）

```typescript
// async-event-emitter.ts — 非同期 Observer の実装
type AsyncHandler<T> = (data: T) => Promise<void> | void;

interface EmitOptions {
  /** 並列実行か順次実行か */
  mode: 'parallel' | 'sequential';
  /** タイムアウト（ms） */
  timeout?: number;
  /** エラー時に他のハンドラを継続するか */
  continueOnError?: boolean;
}

class AsyncEventEmitter<T extends Record<string, any>> {
  private listeners = new Map<keyof T, Set<AsyncHandler<any>>>();

  on<K extends keyof T>(event: K, handler: AsyncHandler<T[K]>): () => void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(handler);
    return () => this.listeners.get(event)?.delete(handler);
  }

  /**
   * 並列実行: 全ハンドラを同時に実行
   * 順序保証は不要だがスループットを最大化したい場合に使用
   */
  async emitParallel<K extends keyof T>(
    event: K,
    data: T[K],
    options?: { timeout?: number; continueOnError?: boolean }
  ): Promise<{ successes: number; errors: Error[] }> {
    const handlers = this.listeners.get(event);
    if (!handlers) return { successes: 0, errors: [] };

    const errors: Error[] = [];
    let successes = 0;

    const promises = [...handlers].map(async (fn) => {
      try {
        const promise = fn(data);
        if (options?.timeout && promise instanceof Promise) {
          await Promise.race([
            promise,
            new Promise((_, reject) =>
              setTimeout(() => reject(new Error('Handler timeout')), options.timeout)
            ),
          ]);
        } else {
          await promise;
        }
        successes++;
      } catch (error) {
        errors.push(error as Error);
        if (!options?.continueOnError) throw error;
      }
    });

    if (options?.continueOnError) {
      await Promise.allSettled(promises);
    } else {
      await Promise.all(promises);
    }

    return { successes, errors };
  }

  /**
   * 順次実行: ハンドラを登録順に1つずつ実行
   * 順序保証が必要な場合に使用
   */
  async emitSequential<K extends keyof T>(
    event: K,
    data: T[K]
  ): Promise<void> {
    const handlers = this.listeners.get(event);
    if (!handlers) return;

    for (const fn of handlers) {
      await fn(data);
    }
  }
}

// --- 使用例 ---
type AppEvents = {
  orderPlaced: { orderId: string; total: number };
  paymentProcessed: { orderId: string; amount: number };
};

const emitter = new AsyncEventEmitter<AppEvents>();

emitter.on("orderPlaced", async (order) => {
  await new Promise(resolve => setTimeout(resolve, 100));
  console.log(`[Email] Sent for ${order.orderId}`);
});

emitter.on("orderPlaced", async (order) => {
  await new Promise(resolve => setTimeout(resolve, 50));
  console.log(`[Inventory] Updated for ${order.orderId}`);
});

// 並列実行: 全ハンドラが同時に開始、最も遅いもので完了
const result = await emitter.emitParallel(
  "orderPlaced",
  { orderId: "ORD-1", total: 5000 },
  { timeout: 3000, continueOnError: true }
);
console.log(`Success: ${result.successes}, Errors: ${result.errors.length}`);

// 順次実行: Email -> Inventory の順番で実行
await emitter.emitSequential("orderPlaced", { orderId: "ORD-2", total: 3000 });
```

### コード例 6: AbortController 統合 ── 安全な購読管理

```typescript
// abort-event-emitter.ts — AbortController で一括解除
class ManagedEventEmitter<T extends Record<string, any>> {
  private listeners = new Map<keyof T, Set<Function>>();

  on<K extends keyof T>(
    event: K,
    handler: (data: T[K]) => void,
    signal?: AbortSignal
  ): () => void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(handler);

    const unsubscribe = () => {
      this.listeners.get(event)?.delete(handler);
    };

    // AbortSignal と連携: signal が abort されたら自動解除
    if (signal) {
      signal.addEventListener('abort', unsubscribe, { once: true });
    }

    return unsubscribe;
  }

  emit<K extends keyof T>(event: K, data: T[K]): void {
    this.listeners.get(event)?.forEach(fn => fn(data));
  }
}

// --- 使用例: コンポーネントのライフサイクルで一括管理 ---
class DashboardComponent {
  private abortController = new AbortController();

  constructor(private emitter: ManagedEventEmitter<AppEvents>) {
    // AbortController の signal を渡す
    const signal = this.abortController.signal;

    emitter.on("orderPlaced", (order) => {
      console.log(`Dashboard: New order ${order.orderId}`);
    }, signal);

    emitter.on("paymentProcessed", (payment) => {
      console.log(`Dashboard: Payment ${payment.amount}`);
    }, signal);
  }

  destroy(): void {
    // 全ての購読を一括解除
    this.abortController.abort();
    console.log("Dashboard: All subscriptions removed");
  }
}

const emitter = new ManagedEventEmitter<AppEvents>();
const dashboard = new DashboardComponent(emitter);

emitter.emit("orderPlaced", { orderId: "1", total: 100 });
// Dashboard: New order 1

dashboard.destroy();
// Dashboard: All subscriptions removed

emitter.emit("orderPlaced", { orderId: "2", total: 200 });
// (何も出力されない — 全て解除済み)
```

### コード例 7: Reactive Store（Redux 風 Observer）

```typescript
// reactive-store.ts — Observer パターンで状態管理
type Reducer<S, A> = (state: S, action: A) => S;
type Listener = () => void;
type Middleware<S, A> = (store: Store<S, A>) =>
  (next: (action: A) => void) => (action: A) => void;

class Store<S, A extends { type: string }> {
  private state: S;
  private listeners = new Set<Listener>();
  private reducer: Reducer<S, A>;
  private dispatch: (action: A) => void;

  constructor(
    reducer: Reducer<S, A>,
    initialState: S,
    middlewares: Middleware<S, A>[] = []
  ) {
    this.reducer = reducer;
    this.state = initialState;

    // ミドルウェアチェーンの構築
    let dispatch = (action: A) => {
      this.state = this.reducer(this.state, action);
      this.listeners.forEach(listener => listener()); // 全 Observer に通知
    };

    // ミドルウェアを逆順に適用
    for (let i = middlewares.length - 1; i >= 0; i--) {
      dispatch = middlewares[i](this)(dispatch);
    }

    this.dispatch = dispatch;
  }

  getState(): S {
    return this.state;
  }

  subscribe(listener: Listener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  send(action: A): void {
    this.dispatch(action);
  }
}

// --- 使用例: カウンターストア ---
type CounterState = { count: number; history: number[] };
type CounterAction =
  | { type: 'INCREMENT'; amount: number }
  | { type: 'DECREMENT'; amount: number }
  | { type: 'RESET' };

const counterReducer: Reducer<CounterState, CounterAction> = (state, action) => {
  switch (action.type) {
    case 'INCREMENT':
      return {
        count: state.count + action.amount,
        history: [...state.history, state.count + action.amount]
      };
    case 'DECREMENT':
      return {
        count: state.count - action.amount,
        history: [...state.history, state.count - action.amount]
      };
    case 'RESET':
      return { count: 0, history: [0] };
    default:
      return state;
  }
};

// ロギングミドルウェア
const logger: Middleware<CounterState, CounterAction> =
  (store) => (next) => (action) => {
    console.log(`[Logger] Action: ${action.type}, Before: ${store.getState().count}`);
    next(action);
    console.log(`[Logger] After: ${store.getState().count}`);
  };

const store = new Store(counterReducer, { count: 0, history: [0] }, [logger]);

// Observer (購読者) の登録
const unsub = store.subscribe(() => {
  console.log(`[UI] Count changed to: ${store.getState().count}`);
});

store.send({ type: 'INCREMENT', amount: 5 });
// [Logger] Action: INCREMENT, Before: 0
// [UI] Count changed to: 5
// [Logger] After: 5

store.send({ type: 'DECREMENT', amount: 2 });
// [Logger] Action: DECREMENT, Before: 5
// [UI] Count changed to: 3
// [Logger] After: 3

unsub(); // UI の購読解除

store.send({ type: 'RESET' });
// [Logger] Action: RESET, Before: 3
// [Logger] After: 0
// (UI には通知されない)
```

---

## 5. Observer vs Pub/Sub vs Reactive Streams

```
Observer パターン（直接参照）
  Subject <--------- Observer
    |  notify()  -->  |
    +---------------->+
  特徴: Subject と Observer が互いを知る
  用途: 単純な通知、UIバインディング

Pub/Sub パターン（仲介者あり）
  Publisher --> EventBus/Broker --> Subscriber
    publish()       |           subscribe()
                    |
  特徴: 完全に疎結合（互いを知らない）
  用途: マイクロサービス間通信、分散システム

Reactive Streams（ストリーム処理）
  Observable --pipe(operators)--> Observer
    |                                |
    + map, filter, debounce,         + subscribe
      merge, switchMap etc.
  特徴: オペレータによるデータ変換パイプライン
  用途: 複雑なイベント処理、リアルタイムストリーム
```

---

## 6. 比較表

### 比較表 1: Observer vs Pub/Sub vs Reactive Streams

| 観点 | Observer | Pub/Sub | Reactive (RxJS) |
|------|:---:|:---:|:---:|
| 結合度 | 中（Subject を知る） | 低（Bus 経由） | 低（ストリーム） |
| 非同期対応 | 手動 | 手動/組み込み | 組み込み |
| バックプレッシャー | なし | なし | あり |
| オペレータ | なし | なし | 豊富（200+） |
| エラーハンドリング | 手動 | 手動 | 組み込み |
| メモリ管理 | 手動解除 | 手動解除 | 自動（complete） |
| 適用規模 | 小〜中 | 中〜大 | 中〜大 |
| 使用場面 | シンプルな通知 | マイクロサービス | ストリーム処理 |

### 比較表 2: 同期 vs 非同期通知

| 観点 | 同期通知 | 非同期通知（並列） | 非同期通知（順次） |
|------|:---:|:---:|:---:|
| 実装難易度 | 低い | 中 | 中 |
| エラーハンドリング | 容易（try/catch） | 要設計（Promise.allSettled） | 容易（for await） |
| パフォーマンス | ブロッキング | 高スループット | 中 |
| 順序保証 | 自然に保証 | なし | あり |
| デバッグ | 容易 | 困難 | 中 |
| タイムアウト | 不要 | 推奨 | 推奨 |

### 比較表 3: フレームワークの Observer 実装

| フレームワーク | メカニズム | 購読解除 | 型安全性 |
|--------------|----------|---------|---------|
| Node.js EventEmitter | on/emit | removeListener | 低い |
| DOM EventTarget | addEventListener | removeEventListener | 中 |
| React (useState) | setState + 再描画 | 自動 | 高い |
| Vue (Reactive) | Proxy ベース | 自動 | 高い |
| RxJS | Observable.subscribe | unsubscribe | 高い |
| Redux | store.subscribe | 返り値の関数 | 高い |
| Angular (Signals) | signal/effect | 自動 | 高い |

---

## 7. アンチパターン

### アンチパターン 1: メモリリーク（購読解除忘れ）

```typescript
// NG: コンポーネント破棄後もリスナーが残る
class BadComponent {
  constructor(private emitter: EventEmitter) {
    // 登録はするが、解除しない！
    emitter.on("data", this.handleData);
  }

  handleData = (data: any) => {
    this.element.textContent = data; // 破棄済みの要素にアクセス → エラー
  };

  destroy(): void {
    // handleData の解除を忘れている
    this.element.remove();
  }
}

// OK: 確実に購読解除する（複数の方法）
class GoodComponent {
  private unsubscribers: (() => void)[] = [];

  constructor(private emitter: TypedEventEmitter<AppEvents>) {
    // 方法1: unsubscribe 関数を保持
    this.unsubscribers.push(
      emitter.on("orderPlaced", this.handleOrder)
    );
    this.unsubscribers.push(
      emitter.on("paymentProcessed", this.handlePayment)
    );
  }

  handleOrder = (order: any) => { /* ... */ };
  handlePayment = (payment: any) => { /* ... */ };

  destroy(): void {
    // 全ての購読を一括解除
    this.unsubscribers.forEach(unsub => unsub());
    this.unsubscribers = [];
    this.element.remove();
  }
}

// OK: React での正しいクリーンアップ
function GoodReactComponent() {
  useEffect(() => {
    const unsub = emitter.on("data", handleData);
    return () => unsub(); // useEffect のクリーンアップで確実に解除
  }, []);
}

// OK: AbortController による一括管理
class BetterComponent {
  private controller = new AbortController();

  constructor(private emitter: ManagedEventEmitter<AppEvents>) {
    const signal = this.controller.signal;
    emitter.on("orderPlaced", this.handleOrder, signal);
    emitter.on("paymentProcessed", this.handlePayment, signal);
  }

  handleOrder = (order: any) => { /* ... */ };
  handlePayment = (payment: any) => { /* ... */ };

  destroy(): void {
    this.controller.abort(); // 全購読を一括解除
  }
}
```

### アンチパターン 2: イベントの連鎖による無限ループ

```typescript
// NG: A の変更が B に通知 -> B の変更が A に通知 -> ...
const emitter = new TypedEventEmitter<{
  priceChanged: { price: number };
  taxChanged: { tax: number };
}>();

emitter.on("priceChanged", ({ price }) => {
  const newTax = price * 0.1;
  emitter.emit("taxChanged", { tax: newTax }); // taxChanged を発行
});

emitter.on("taxChanged", ({ tax }) => {
  const newPrice = tax / 0.1;
  emitter.emit("priceChanged", { price: newPrice }); // priceChanged を再発行！
  // -> 無限ループ
});

// OK: 循環検出ガード付き EventEmitter
class SafeEventEmitter<T extends Record<string, any>> {
  private listeners = new Map<keyof T, Set<Function>>();
  private emitting = new Set<keyof T>(); // 現在発行中のイベント

  on<K extends keyof T>(event: K, handler: (data: T[K]) => void): () => void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(handler);
    return () => this.listeners.get(event)?.delete(handler);
  }

  emit<K extends keyof T>(event: K, data: T[K]): void {
    if (this.emitting.has(event)) {
      console.warn(`[SafeEmitter] Circular emit detected for "${String(event)}". Skipping.`);
      return; // 循環を防止
    }

    this.emitting.add(event);
    try {
      this.listeners.get(event)?.forEach(fn => fn(data));
    } finally {
      this.emitting.delete(event);
    }
  }
}
```

### アンチパターン 3: God Observer（1つの Observer が全てを処理）

```typescript
// NG: 1つの巨大な handler が全イベントを処理
class GodObserver {
  handle(eventType: string, data: any): void {
    switch (eventType) {
      case "userCreated":
        this.sendWelcomeEmail(data);
        this.createDefaultSettings(data);
        this.notifyAdmin(data);
        break;
      case "orderPlaced":
        this.sendConfirmation(data);
        this.updateInventory(data);
        this.processPayment(data);
        break;
      // ... 数十のイベントタイプ
    }
  }
}

// OK: 責務ごとに Observer を分離
class EmailObserver {
  constructor(private emitter: EventEmitter) {
    emitter.on("userCreated", this.sendWelcome);
    emitter.on("orderPlaced", this.sendConfirmation);
    emitter.on("orderCancelled", this.sendCancellation);
  }

  private sendWelcome = (data: any) => { /* メール送信のみ */ };
  private sendConfirmation = (data: any) => { /* メール送信のみ */ };
  private sendCancellation = (data: any) => { /* メール送信のみ */ };
}

class InventoryObserver {
  constructor(private emitter: EventEmitter) {
    emitter.on("orderPlaced", this.decrementStock);
    emitter.on("orderCancelled", this.restoreStock);
  }

  private decrementStock = (data: any) => { /* 在庫管理のみ */ };
  private restoreStock = (data: any) => { /* 在庫管理のみ */ };
}
```

---

## 8. 実世界での Observer パターン

### 8.1 ブラウザ DOM イベント

```
DOM のイベント伝搬（Observer パターンの実装）:

     [window]          Capture Phase（上→下）
        |
     [document]
        |
     [body]
        |
     [div.parent]
        |
     [button]    <---- Target Phase
        |
     [div.parent]
        |
     [body]            Bubble Phase（下→上）
        |
     [document]
        |
     [window]

addEventListener(event, handler, { capture: true/false })
  capture: true  → Capture Phase で実行
  capture: false → Bubble Phase で実行（デフォルト）
```

### 8.2 React のリアクティブシステム

```
React の状態更新フロー:

  setState(newValue)
      |
      v
  [Reconciler] -- 差分計算 (Virtual DOM diff)
      |
      v
  [Commit Phase] -- DOM 更新
      |
      v
  useEffect cleanup  → useEffect callback
  (前の副作用のクリーン)   (新しい副作用の実行)

  本質: useState は Observer パターン
  - setState = Subject.notify()
  - コンポーネントの再描画 = Observer.update()
```

### 8.3 マイクロサービスの Event-Driven Architecture

```
Event-Driven Architecture:

  Order Service --publish("order.created")--> Message Broker (Kafka/RabbitMQ)
                                                    |
  Email Service       <--- subscribe("order.*") ----+
  Inventory Service   <--- subscribe("order.*") ----+
  Analytics Service   <--- subscribe("order.*") ----+
  Payment Service     <--- subscribe("order.created")+

  メリット:
  - サービス間の完全な疎結合
  - 独立したデプロイとスケーリング
  - 障害の伝搬を防止
  - イベントの永続化と再生が可能
```

---

## 9. 実践演習

### 演習 1: 基礎 ── 型安全な EventEmitter の実装

**課題**: 以下の要件を満たす型安全な EventEmitter を実装せよ。

1. `on(event, handler)`: イベントを購読し、購読解除関数を返す
2. `once(event, handler)`: 1回だけ購読する
3. `emit(event, data)`: イベントを発行する
4. `listenerCount(event)`: リスナー数を返す
5. イベント名とデータ型がジェネリクスで型安全に連携する

**テストケース**:

```typescript
type Events = {
  message: { text: string; from: string };
  error: { code: number; message: string };
};

const emitter = new TypedEventEmitter<Events>();

const unsub = emitter.on("message", (msg) => {
  console.log(`${msg.from}: ${msg.text}`);
});

emitter.once("error", (err) => {
  console.log(`Error ${err.code}: ${err.message}`);
});

emitter.emit("message", { text: "Hello", from: "Alice" });
// "Alice: Hello"

console.log(emitter.listenerCount("message")); // 1
console.log(emitter.listenerCount("error"));   // 1

emitter.emit("error", { code: 404, message: "Not Found" });
// "Error 404: Not Found"

console.log(emitter.listenerCount("error"));   // 0 (once は消費済み)

unsub();
console.log(emitter.listenerCount("message")); // 0
```

**期待される出力**: 上記コメントの通り。

<details>
<summary>模範解答（クリックで展開）</summary>

```typescript
class TypedEventEmitter<T extends Record<string, any>> {
  private listeners = new Map<keyof T, Set<Function>>();
  private onceListeners = new Map<keyof T, Set<Function>>();

  on<K extends keyof T>(event: K, handler: (data: T[K]) => void): () => void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(handler);
    return () => {
      this.listeners.get(event)?.delete(handler);
    };
  }

  once<K extends keyof T>(event: K, handler: (data: T[K]) => void): () => void {
    if (!this.onceListeners.has(event)) {
      this.onceListeners.set(event, new Set());
    }
    this.onceListeners.get(event)!.add(handler);
    return () => {
      this.onceListeners.get(event)?.delete(handler);
    };
  }

  emit<K extends keyof T>(event: K, data: T[K]): void {
    this.listeners.get(event)?.forEach(fn => fn(data));

    const onceHandlers = this.onceListeners.get(event);
    if (onceHandlers) {
      onceHandlers.forEach(fn => fn(data));
      onceHandlers.clear();
    }
  }

  listenerCount<K extends keyof T>(event: K): number {
    return (this.listeners.get(event)?.size ?? 0) +
           (this.onceListeners.get(event)?.size ?? 0);
  }

  removeAllListeners<K extends keyof T>(event?: K): void {
    if (event) {
      this.listeners.delete(event);
      this.onceListeners.delete(event);
    } else {
      this.listeners.clear();
      this.onceListeners.clear();
    }
  }
}
```

**設計ポイント:**
- `on` と `once` を別の Map で管理し、`once` は emit 時に clear する
- 各メソッドが unsubscribe 関数を返すことで、クリーンアップを容易にする
- `listenerCount` は両方の Map のサイズを合算する

</details>

---

### 演習 2: 応用 ── リアクティブ Store の実装

**課題**: Redux 風のリアクティブ Store を Observer パターンで実装せよ。

要件:
1. `Store<S, A>` クラス: Reducer で状態を管理
2. `getState()`: 現在の状態を取得
3. `dispatch(action)`: アクションを発行し、全 Observer に通知
4. `subscribe(listener)`: 状態変更を購読
5. `select(selector)`: 状態の一部だけを監視し、変更時のみ通知

**テストケース**:

```typescript
type State = { count: number; name: string };
type Action =
  | { type: 'INCREMENT' }
  | { type: 'SET_NAME'; name: string };

const store = new Store<State, Action>(
  (state, action) => {
    switch (action.type) {
      case 'INCREMENT': return { ...state, count: state.count + 1 };
      case 'SET_NAME': return { ...state, name: action.name };
      default: return state;
    }
  },
  { count: 0, name: "initial" }
);

// count だけを監視
const unsubCount = store.select(
  s => s.count,
  (count) => console.log(`Count: ${count}`)
);

store.dispatch({ type: 'SET_NAME', name: 'Taro' });
// (count は変わっていないので何も出力されない)

store.dispatch({ type: 'INCREMENT' });
// "Count: 1"

unsubCount();
```

**期待される出力**: 上記コメントの通り。

<details>
<summary>模範解答（クリックで展開）</summary>

```typescript
type Reducer<S, A> = (state: S, action: A) => S;
type Listener = () => void;
type Selector<S, R> = (state: S) => R;

class Store<S, A extends { type: string }> {
  private state: S;
  private listeners = new Set<Listener>();
  private reducer: Reducer<S, A>;

  constructor(reducer: Reducer<S, A>, initialState: S) {
    this.reducer = reducer;
    this.state = initialState;
  }

  getState(): S {
    return this.state;
  }

  dispatch(action: A): void {
    this.state = this.reducer(this.state, action);
    this.listeners.forEach(listener => listener());
  }

  subscribe(listener: Listener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /**
   * 状態の一部を監視し、変更時のみコールバックを呼ぶ
   * 前回の選択結果と比較して変更があった場合のみ通知する
   */
  select<R>(selector: Selector<S, R>, callback: (value: R) => void): () => void {
    let previousValue = selector(this.state);

    return this.subscribe(() => {
      const currentValue = selector(this.state);
      if (currentValue !== previousValue) {
        previousValue = currentValue;
        callback(currentValue);
      }
    });
  }
}
```

**設計ポイント:**
- `select` は内部で `subscribe` を利用し、セレクタの結果が変わった場合のみコールバックを呼ぶ
- 前回値との比較には `!==`（参照等価性）を使い、プリミティブ値とオブジェクト参照の両方に対応
- `dispatch` は Reducer で新しい状態を生成してから全 Observer に通知する

</details>

---

### 演習 3: 発展 ── 非同期 Event Bus with Retry

**課題**: 非同期ハンドラをサポートし、失敗時にリトライ機能を持つ EventBus を実装せよ。

要件:
1. `on(event, handler)`: 非同期ハンドラを登録
2. `emit(event, data, options)`: イベント発行（並列/順次を選択可能）
3. リトライ: 失敗したハンドラを指数バックオフで最大3回リトライ
4. Dead Letter Queue: 全リトライが失敗したイベントを記録
5. タイムアウト: 各ハンドラに制限時間を設定

**テストケース**:

```typescript
const bus = new ResilientEventBus();

let callCount = 0;
bus.on("process", async (data: { id: string }) => {
  callCount++;
  if (callCount < 3) {
    throw new Error("Temporary failure");
  }
  console.log(`Processed: ${data.id}`);
});

await bus.emit("process", { id: "item-1" }, {
  mode: 'sequential',
  retry: { maxAttempts: 3, backoffMs: 100 },
  timeoutMs: 5000,
});
// 1回目: 失敗 (100ms 待機)
// 2回目: 失敗 (200ms 待機)
// 3回目: "Processed: item-1"

console.log(bus.getDeadLetterQueue().length); // 0 (成功したため)
```

**期待される出力**: 上記コメントの通り。

<details>
<summary>模範解答（クリックで展開）</summary>

```typescript
type AsyncHandler<T> = (data: T) => Promise<void> | void;

interface RetryOptions {
  maxAttempts: number;
  backoffMs: number;
}

interface EmitOptions {
  mode: 'parallel' | 'sequential';
  retry?: RetryOptions;
  timeoutMs?: number;
}

interface DeadLetterEntry {
  event: string;
  data: any;
  error: Error;
  timestamp: Date;
  attempts: number;
}

class ResilientEventBus {
  private listeners = new Map<string, Set<AsyncHandler<any>>>();
  private deadLetterQueue: DeadLetterEntry[] = [];

  on<T>(event: string, handler: AsyncHandler<T>): () => void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(handler);
    return () => this.listeners.get(event)?.delete(handler);
  }

  async emit<T>(event: string, data: T, options: EmitOptions): Promise<void> {
    const handlers = this.listeners.get(event);
    if (!handlers) return;

    const wrappedHandlers = [...handlers].map(fn =>
      () => this.executeWithRetry(fn, data, event, options)
    );

    if (options.mode === 'parallel') {
      await Promise.allSettled(wrappedHandlers.map(fn => fn()));
    } else {
      for (const fn of wrappedHandlers) {
        await fn();
      }
    }
  }

  private async executeWithRetry<T>(
    handler: AsyncHandler<T>,
    data: T,
    event: string,
    options: EmitOptions,
  ): Promise<void> {
    const maxAttempts = options.retry?.maxAttempts ?? 1;
    const backoffMs = options.retry?.backoffMs ?? 100;

    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        const promise = handler(data);
        if (options.timeoutMs && promise instanceof Promise) {
          await Promise.race([
            promise,
            new Promise<never>((_, reject) =>
              setTimeout(() => reject(new Error('Handler timeout')), options.timeoutMs)
            ),
          ]);
        } else {
          await promise;
        }
        return; // 成功
      } catch (error) {
        if (attempt < maxAttempts) {
          // 指数バックオフで待機
          const delay = backoffMs * Math.pow(2, attempt - 1);
          await new Promise(resolve => setTimeout(resolve, delay));
        } else {
          // 全リトライ失敗 → Dead Letter Queue に記録
          this.deadLetterQueue.push({
            event,
            data,
            error: error as Error,
            timestamp: new Date(),
            attempts: maxAttempts,
          });
        }
      }
    }
  }

  getDeadLetterQueue(): DeadLetterEntry[] {
    return [...this.deadLetterQueue];
  }

  clearDeadLetterQueue(): void {
    this.deadLetterQueue = [];
  }
}
```

**設計ポイント:**
- `executeWithRetry` で指数バックオフ（`backoffMs * 2^(attempt-1)`）を実装
- タイムアウトは `Promise.race` でハンドラの Promise と競合させる
- 全リトライが失敗したイベントは Dead Letter Queue に記録し、後から調査可能にする
- `parallel` モードでは `Promise.allSettled` を使い、1つの失敗が他のハンドラに影響しないようにする

</details>

---

## 10. FAQ

### Q1: Observer パターンはどの言語/フレームワークで使われていますか？

DOM の EventListener、Node.js の EventEmitter、Vue.js のリアクティブシステム、RxJS の Observable、Android の LiveData/Flow、React の useState/useEffect、Redux の store.subscribe、Angular の Signals、Swift の Combine フレームワークなど、ほぼ全てのUI/イベント駆動フレームワークで使われています。Observer パターンを知らずにモダンなフロントエンド/バックエンド開発を行うことは不可能です。

### Q2: Observer が多すぎるとパフォーマンスに影響しますか？

はい。通知が同期的な場合、Observer の数に比例してブロッキング時間が増えます。対策として: (1) 非同期通知に切り替える、(2) バッチ処理（React の自動バッチングのように複数の更新を1回にまとめる）、(3) デバウンス/スロットル（高頻度の通知を間引く）、(4) セレクタベースの購読（変更された部分のみ通知する）を検討してください。

### Q3: Redux と Observer パターンの関係は？

Redux の `store.subscribe()` は Observer パターンそのものです。Action の dispatch で状態が変更され、購読しているコンポーネントに通知されます。React-Redux の `useSelector` は、セレクタの結果が変わった場合のみ再描画する最適化された Observer です。

### Q4: EventEmitter と Promise/async-await の使い分けは？

一回限りの非同期操作（API呼び出し、ファイル読み込み）は Promise が適切です。繰り返し発生するイベント（クリック、メッセージ受信、状態変更）は EventEmitter が適切です。両方の特性が必要な場合は AsyncIterator や RxJS の Observable を検討してください。

### Q5: WeakRef を使った Observer はいつ有効ですか？

WeakRef は Observer のライフサイクルが不明確な場合に有効です。例えば、プラグインシステムでプラグインが動的にロード/アンロードされる場合、WeakRef を使えばプラグインが GC された時点で自動的に購読が解除されます。ただし、GC のタイミングは不確定なため、明示的な購読解除が可能な場合はそちらを優先してください。

---

## 11. まとめ

| 項目 | ポイント |
|------|---------|
| 目的 | 1対多の状態変化通知。疎結合なイベント駆動設計 |
| Push 型 | データを直接渡す（シンプル、データが小さい場合に最適） |
| Pull 型 | Observer が取りに行く（柔軟、Observer ごとに必要なデータが異なる場合） |
| 購読解除 | メモリリーク防止のため**必須**。AbortController で一括管理が便利 |
| 非同期通知 | 並列（高スループット）と順次（順序保証）を使い分ける |
| 循環防止 | emitting ガードで同一イベントの再帰発行を防止 |
| 進化系 | Pub/Sub（完全疎結合）、Reactive Streams（オペレータ付き） |

---

## 次に読むべきガイド

- [Strategy パターン](./01-strategy.md) -- アルゴリズムの交換
- [Command パターン](./02-command.md) -- 操作のカプセル化と Undo/Redo
- [State パターン](./03-state.md) -- 状態遷移の管理
- [イベント駆動アーキテクチャ](../../../system-design-guide/docs/02-architecture/03-event-driven.md) -- マイクロサービスでの Observer
- [モナドパターン](../03-functional/00-monad.md) -- Promise/async-await の理論的基盤

---

## 参考文献

1. Gamma, E., Helm, R., Johnson, R., Vlissides, J. (1994). *Design Patterns: Elements of Reusable Object-Oriented Software*. Addison-Wesley. -- Observer パターンの原典。
2. ReactiveX Documentation. https://reactivex.io/ -- リアクティブプログラミングの包括的リファレンス。
3. Node.js Events Documentation. https://nodejs.org/api/events.html -- Node.js の EventEmitter の公式ドキュメント。
4. Redux Documentation. https://redux.js.org/ -- Observer パターンに基づく状態管理ライブラリ。
5. MDN Web Docs -- EventTarget. https://developer.mozilla.org/en-US/docs/Web/API/EventTarget -- ブラウザのイベントシステム。
