# Observer パターン

> オブジェクト間に **1対多** の依存関係を定義し、あるオブジェクトの状態変化を依存するすべてのオブジェクトに自動通知する振る舞いパターン。

---

## この章で学ぶこと

1. Observer パターン（Pub/Sub）の構造とイベント駆動設計の基礎
2. Push 型と Pull 型の通知モデルの違いと使い分け
3. メモリリーク防止、イベントの順序保証、非同期通知の注意点

---

## 1. Observer の構造

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

---

## 2. Push 型 vs Pull 型

```
Push型: Subject が変更データを直接渡す
Subject ──notify(data)──> Observer
  利点: Observer は必要なデータをすぐ取得
  欠点: 不要なデータも送られる

Pull型: Subject は通知のみ、Observer が取りに行く
Subject ──notify()──> Observer ──getState()──> Subject
  利点: Observer が必要なデータだけ取得
  欠点: Subject への追加アクセスが必要
```

---

## 3. コード例

### コード例 1: 型安全な EventEmitter

```typescript
type EventMap = {
  userCreated: { id: string; name: string };
  userDeleted: { id: string };
  orderPlaced: { orderId: string; total: number };
};

class TypedEventEmitter<T extends Record<string, any>> {
  private listeners = new Map<keyof T, Set<Function>>();

  on<K extends keyof T>(event: K, handler: (data: T[K]) => void): () => void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(handler);

    // unsubscribe 関数を返す
    return () => this.listeners.get(event)?.delete(handler);
  }

  emit<K extends keyof T>(event: K, data: T[K]): void {
    this.listeners.get(event)?.forEach(fn => fn(data));
  }
}

// 使用
const bus = new TypedEventEmitter<EventMap>();
const unsub = bus.on("userCreated", (user) => {
  console.log(`Welcome, ${user.name}!`); // 型安全
});
bus.emit("userCreated", { id: "1", name: "Taro" });
unsub(); // 購読解除
```

### コード例 2: React — カスタム Observable Hook

```typescript
function useObservable<T>(observable$: Observable<T>, initial: T): T {
  const [value, setValue] = useState<T>(initial);

  useEffect(() => {
    const subscription = observable$.subscribe(setValue);
    return () => subscription.unsubscribe(); // クリーンアップ
  }, [observable$]);

  return value;
}

// 使用
function StockPrice({ symbol }: { symbol: string }) {
  const price = useObservable(stockService.getPrice$(symbol), 0);
  return <div>{symbol}: {price}</div>;
}
```

### コード例 3: Node.js EventEmitter

```typescript
import { EventEmitter } from "events";

class OrderService extends EventEmitter {
  placeOrder(order: Order): void {
    // 注文処理
    this.emit("orderPlaced", order);
  }
}

// リスナー登録
const orderService = new OrderService();

orderService.on("orderPlaced", (order: Order) => {
  emailService.sendConfirmation(order);
});

orderService.on("orderPlaced", (order: Order) => {
  inventoryService.decrementStock(order.items);
});

orderService.on("orderPlaced", (order: Order) => {
  analyticsService.trackPurchase(order);
});
```

### コード例 4: Python — Observer

```python
from abc import ABC, abstractmethod
from typing import Any

class EventBus:
    def __init__(self):
        self._subscribers: dict[str, list[callable]] = {}

    def subscribe(self, event: str, handler: callable) -> callable:
        if event not in self._subscribers:
            self._subscribers[event] = []
        self._subscribers[event].append(handler)

        def unsubscribe():
            self._subscribers[event].remove(handler)
        return unsubscribe

    def publish(self, event: str, data: Any = None) -> None:
        for handler in self._subscribers.get(event, []):
            handler(data)

# 使用
bus = EventBus()

def on_user_created(user: dict):
    print(f"Sending welcome email to {user['email']}")

unsub = bus.subscribe("user.created", on_user_created)
bus.publish("user.created", {"email": "taro@example.com"})
unsub()  # 購読解除
```

### コード例 5: 非同期 Observer（Promise ベース）

```typescript
class AsyncEventEmitter<T extends Record<string, any>> {
  private listeners = new Map<keyof T, Set<Function>>();

  on<K extends keyof T>(event: K, handler: (data: T[K]) => Promise<void> | void): () => void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(handler);
    return () => this.listeners.get(event)?.delete(handler);
  }

  async emit<K extends keyof T>(event: K, data: T[K]): Promise<void> {
    const handlers = this.listeners.get(event);
    if (!handlers) return;

    // 並列実行（順序保証不要の場合）
    await Promise.all([...handlers].map(fn => fn(data)));
  }

  async emitSequential<K extends keyof T>(event: K, data: T[K]): Promise<void> {
    const handlers = this.listeners.get(event);
    if (!handlers) return;

    // 順次実行（順序保証が必要な場合）
    for (const fn of handlers) {
      await fn(data);
    }
  }
}
```

---

## 4. Observer vs Pub/Sub の関係

```
Observer パターン（直接参照）
Subject ←──────── Observer
  │  notify()  ──>  │
  └────────────────>─┘

Pub/Sub パターン（仲介者あり）
Publisher ──> EventBus ──> Subscriber
  publish()     │         subscribe()
                │
     完全に疎結合（互いを知らない）
```

---

## 5. 比較表

### 比較表 1: Observer vs Pub/Sub vs Reactive Streams

| 観点 | Observer | Pub/Sub | Reactive (RxJS) |
|------|:---:|:---:|:---:|
| 結合度 | 中（Subject を知る） | 低（Bus 経由） | 低 |
| 非同期対応 | 手動 | 手動 | 組み込み |
| バックプレッシャー | なし | なし | あり |
| オペレータ | なし | なし | 豊富 |
| 使用場面 | シンプルな通知 | マイクロサービス | ストリーム処理 |

### 比較表 2: 同期 vs 非同期通知

| 観点 | 同期通知 | 非同期通知 |
|------|:---:|:---:|
| 実装難易度 | 低い | 中〜高 |
| エラーハンドリング | 容易 | 要設計 |
| パフォーマンス | ブロッキング | ノンブロッキング |
| 順序保証 | 自然に保証 | 明示的に設計 |
| デバッグ | 容易 | 困難 |

---

## 6. アンチパターン

### アンチパターン 1: メモリリーク（購読解除忘れ）

```typescript
// BAD: コンポーネント破棄後もリスナーが残る
class BadComponent {
  constructor(private emitter: EventEmitter) {
    emitter.on("data", this.handleData); // 解除されない！
  }

  handleData = (data: any) => {
    this.element.textContent = data; // 破棄済みの要素にアクセス
  };
}
```

**改善**: `useEffect` のクリーンアップ、`AbortController`、WeakRef 等で確実に購読解除する。

### アンチパターン 2: イベントの連鎖による無限ループ

```typescript
// BAD: A の変更が B に通知 → B の変更が A に通知 → ...
emitter.on("priceChanged", (price) => {
  const newTax = price * 0.1;
  emitter.emit("taxChanged", newTax); // これが priceChanged を再トリガー
});
```

**改善**: イベントのサイクルを検出する仕組みを入れるか、設計で循環を防ぐ。

---

## 7. FAQ

### Q1: Observer パターンはどの言語/フレームワークで使われていますか？

DOM EventListener、Node.js EventEmitter、Vue.js のリアクティブシステム、RxJS、Android の LiveData/Flow、React の useState/useEffect など、ほぼすべてのUI/イベント駆動フレームワークで使われています。

### Q2: Observer が多すぎるとパフォーマンスに影響しますか？

通知が同期的な場合、Observer の数に比例してブロッキング時間が増えます。数千の Observer がある場合は非同期通知やバッチ処理を検討してください。

### Q3: Redux と Observer パターンの関係は？

Redux の `store.subscribe()` は Observer パターンそのものです。Action の dispatch によって状態が変更され、購読しているコンポーネントに通知されます。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| 目的 | 1対多の状態変化通知 |
| Push 型 | データを直接渡す（シンプル） |
| Pull 型 | Observer が取りに行く（柔軟） |
| 購読解除 | メモリリーク防止のため必須 |
| 進化系 | Pub/Sub、Reactive Streams |

---

## 次に読むべきガイド

- [Strategy パターン](./01-strategy.md) — アルゴリズムの交換
- [Command パターン](./02-command.md) — 操作のカプセル化
- [イベント駆動アーキテクチャ](../../../system-design-guide/docs/02-architecture/03-event-driven.md)

---

## 参考文献

1. Gamma, E. et al. (1994). *Design Patterns: Elements of Reusable Object-Oriented Software*. Addison-Wesley.
2. ReactiveX Documentation. https://reactivex.io/
3. Node.js Events Documentation. https://nodejs.org/api/events.html
