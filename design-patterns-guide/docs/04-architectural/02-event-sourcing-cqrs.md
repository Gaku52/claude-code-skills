# Event Sourcing / CQRS — イベント駆動設計

> Event Sourcing で状態の変化を不変のイベントログとして記録し、CQRS で読み取りと書き込みを分離することで、監査可能性・スケーラビリティ・ドメインの表現力を最大化するための実践ガイド。

---

## この章で学ぶこと

1. **Event Sourcing** — 状態をイベントの履歴として保存するパターン
2. **CQRS（Command Query Responsibility Segregation）** — 読み取りと書き込みの責務分離
3. **Event Sourcing + CQRS** の組み合わせとプロジェクション設計

---

## 1. 従来の CRUD vs Event Sourcing

### 1.1 根本的な違い

```
┌──────────────────────────────────────────────────────┐
│  従来の CRUD（状態保存）                              │
│                                                      │
│  orders テーブル:                                    │
│  ┌──────┬────────┬────────┬───────────┐              │
│  │ id   │ status │ total  │ updated_at│              │
│  ├──────┼────────┼────────┼───────────┤              │
│  │ O-01 │ shipped│ 15,000 │ 02-10     │ ← 現在の状態│
│  └──────┴────────┴────────┴───────────┘   のみ保存  │
│                                                      │
│  問題: 「いつ」「なぜ」注文が変更されたか分からない   │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│  Event Sourcing（イベント保存）                       │
│                                                      │
│  events テーブル:                                    │
│  ┌────┬──────┬────────────────┬──────────┬───────┐   │
│  │ #  │ 集約 │ イベント型      │ データ    │ 日時  │   │
│  ├────┼──────┼────────────────┼──────────┼───────┤   │
│  │ 1  │ O-01 │ OrderCreated   │ {items}  │ 02-08 │   │
│  │ 2  │ O-01 │ ItemAdded      │ {item}   │ 02-08 │   │
│  │ 3  │ O-01 │ PaymentReceived│ {amount} │ 02-09 │   │
│  │ 4  │ O-01 │ OrderShipped   │ {carrier}│ 02-10 │   │
│  └────┴──────┴────────────────┴──────────┴───────┘   │
│                                                      │
│  利点: 全ての変更履歴が残る、任意時点の状態を再構築   │
└──────────────────────────────────────────────────────┘
```

### 1.2 Event Sourcing の状態復元

```
イベントの再生 (Replay) による状態復元:

  Event 1: OrderCreated      → Order { status: "created", items: [], total: 0 }
      │
      ▼
  Event 2: ItemAdded          → Order { status: "created", items: [A], total: 5000 }
      │
      ▼
  Event 3: ItemAdded          → Order { status: "created", items: [A,B], total: 12000 }
      │
      ▼
  Event 4: PaymentReceived    → Order { status: "paid", items: [A,B], total: 12000 }
      │
      ▼
  Event 5: OrderShipped       → Order { status: "shipped", items: [A,B], total: 12000 }

  ※ 2月9日時点の状態が必要 → Event 1-4 を再生すれば OK
```

---

## 2. Event Sourcing の実装

### 2.1 イベント定義

```typescript
// === Domain Events ===

// 基底イベント型
interface DomainEvent {
  eventId: string;
  aggregateId: string;
  aggregateType: string;
  eventType: string;
  data: Record<string, unknown>;
  metadata: {
    timestamp: Date;
    version: number;
    userId?: string;
    correlationId?: string;
  };
}

// 注文ドメインのイベント
interface OrderCreated extends DomainEvent {
  eventType: "OrderCreated";
  data: {
    customerId: string;
    items: Array<{ productId: string; quantity: number; price: number }>;
  };
}

interface ItemAdded extends DomainEvent {
  eventType: "ItemAdded";
  data: {
    productId: string;
    quantity: number;
    price: number;
  };
}

interface PaymentReceived extends DomainEvent {
  eventType: "PaymentReceived";
  data: {
    amount: number;
    paymentMethod: string;
    transactionId: string;
  };
}

interface OrderShipped extends DomainEvent {
  eventType: "OrderShipped";
  data: {
    carrier: string;
    trackingNumber: string;
  };
}

type OrderEvent = OrderCreated | ItemAdded | PaymentReceived | OrderShipped;
```

### 2.2 集約（Aggregate）の実装

```typescript
// === Order Aggregate ===

class Order {
  private id: string;
  private status: "created" | "paid" | "shipped" | "delivered" | "cancelled";
  private items: OrderItem[] = [];
  private totalAmount: number = 0;
  private version: number = 0;
  private uncommittedEvents: OrderEvent[] = [];

  // イベントから状態を復元（ファクトリメソッド）
  static fromEvents(events: OrderEvent[]): Order {
    const order = new Order();
    for (const event of events) {
      order.apply(event, false);  // 既存イベントは uncommitted に追加しない
    }
    return order;
  }

  // コマンド: 注文作成
  static create(orderId: string, customerId: string, items: OrderItem[]): Order {
    const order = new Order();
    order.applyNew({
      eventType: "OrderCreated",
      aggregateId: orderId,
      data: { customerId, items },
    });
    return order;
  }

  // コマンド: 商品追加
  addItem(productId: string, quantity: number, price: number): void {
    if (this.status !== "created") {
      throw new Error("支払い済みの注文に商品を追加できません");
    }
    this.applyNew({
      eventType: "ItemAdded",
      aggregateId: this.id,
      data: { productId, quantity, price },
    });
  }

  // コマンド: 支払い受領
  receivePayment(amount: number, method: string, txId: string): void {
    if (this.status !== "created") {
      throw new Error("この注文はすでに支払い済みです");
    }
    if (amount < this.totalAmount) {
      throw new Error("支払い金額が不足しています");
    }
    this.applyNew({
      eventType: "PaymentReceived",
      aggregateId: this.id,
      data: { amount, paymentMethod: method, transactionId: txId },
    });
  }

  // イベント適用（状態変更ロジック）
  private apply(event: OrderEvent, isNew: boolean = true): void {
    switch (event.eventType) {
      case "OrderCreated":
        this.id = event.aggregateId;
        this.status = "created";
        this.items = event.data.items;
        this.totalAmount = this.items.reduce((sum, i) => sum + i.price * i.quantity, 0);
        break;

      case "ItemAdded":
        this.items.push({
          productId: event.data.productId,
          quantity: event.data.quantity,
          price: event.data.price,
        });
        this.totalAmount += event.data.price * event.data.quantity;
        break;

      case "PaymentReceived":
        this.status = "paid";
        break;

      case "OrderShipped":
        this.status = "shipped";
        break;
    }

    this.version++;
    if (isNew) {
      this.uncommittedEvents.push(event);
    }
  }

  private applyNew(eventData: Partial<OrderEvent>): void {
    const event = {
      ...eventData,
      eventId: crypto.randomUUID(),
      aggregateType: "Order",
      metadata: {
        timestamp: new Date(),
        version: this.version + 1,
      },
    } as OrderEvent;
    this.apply(event, true);
  }

  getUncommittedEvents(): OrderEvent[] {
    return [...this.uncommittedEvents];
  }

  clearUncommittedEvents(): void {
    this.uncommittedEvents = [];
  }
}
```

### 2.3 Event Store の実装

```typescript
// === Event Store ===

interface EventStore {
  append(aggregateId: string, events: DomainEvent[], expectedVersion: number): Promise<void>;
  getEvents(aggregateId: string): Promise<DomainEvent[]>;
  getEventsAfterVersion(aggregateId: string, version: number): Promise<DomainEvent[]>;
}

class PostgresEventStore implements EventStore {
  constructor(private pool: Pool) {}

  async append(
    aggregateId: string,
    events: DomainEvent[],
    expectedVersion: number
  ): Promise<void> {
    const client = await this.pool.connect();
    try {
      await client.query("BEGIN");

      // 楽観的ロック: 現在のバージョンを確認
      const { rows } = await client.query(
        "SELECT MAX(version) as current_version FROM events WHERE aggregate_id = $1",
        [aggregateId]
      );
      const currentVersion = rows[0]?.current_version ?? 0;

      if (currentVersion !== expectedVersion) {
        throw new ConcurrencyError(
          `Expected version ${expectedVersion}, but found ${currentVersion}`
        );
      }

      // イベントを追記
      for (const event of events) {
        await client.query(
          `INSERT INTO events (event_id, aggregate_id, aggregate_type, event_type, data, metadata, version)
           VALUES ($1, $2, $3, $4, $5, $6, $7)`,
          [
            event.eventId,
            event.aggregateId,
            event.aggregateType,
            event.eventType,
            JSON.stringify(event.data),
            JSON.stringify(event.metadata),
            event.metadata.version,
          ]
        );
      }

      await client.query("COMMIT");
    } catch (error) {
      await client.query("ROLLBACK");
      throw error;
    } finally {
      client.release();
    }
  }

  async getEvents(aggregateId: string): Promise<DomainEvent[]> {
    const { rows } = await this.pool.query(
      "SELECT * FROM events WHERE aggregate_id = $1 ORDER BY version ASC",
      [aggregateId]
    );
    return rows.map(this.toDomainEvent);
  }

  async getEventsAfterVersion(aggregateId: string, version: number): Promise<DomainEvent[]> {
    const { rows } = await this.pool.query(
      "SELECT * FROM events WHERE aggregate_id = $1 AND version > $2 ORDER BY version ASC",
      [aggregateId, version]
    );
    return rows.map(this.toDomainEvent);
  }

  private toDomainEvent(row: any): DomainEvent {
    return {
      eventId: row.event_id,
      aggregateId: row.aggregate_id,
      aggregateType: row.aggregate_type,
      eventType: row.event_type,
      data: row.data,
      metadata: row.metadata,
    };
  }
}
```

---

## 3. CQRS の設計

### 3.1 CQRS の全体構造

```
┌──────────────────────────────────────────────────────────┐
│                     CQRS アーキテクチャ                    │
│                                                          │
│  Command Side (書き込み)        Query Side (読み取り)     │
│  ┌──────────────────┐          ┌──────────────────┐      │
│  │ Command Handler  │          │ Query Handler    │      │
│  │ (CreateOrder等)  │          │ (GetOrderDetail) │      │
│  └────────┬─────────┘          └────────┬─────────┘      │
│           │                             │                │
│           ▼                             ▼                │
│  ┌──────────────────┐          ┌──────────────────┐      │
│  │ Domain Model     │          │ Read Model       │      │
│  │ (Aggregate)      │          │ (Projection)     │      │
│  └────────┬─────────┘          └────────▲─────────┘      │
│           │                             │                │
│           ▼                             │ Projection     │
│  ┌──────────────────┐          ┌───────┴──────────┐      │
│  │ Event Store      │ ──────→ │ Event Handler    │      │
│  │ (Write DB)       │  Event  │ (Projector)      │      │
│  └──────────────────┘  Stream └──────────────────┘      │
│                                         │                │
│                                         ▼                │
│                                ┌──────────────────┐      │
│                                │ Read DB           │      │
│                                │ (最適化されたビュー)│     │
│                                └──────────────────┘      │
└──────────────────────────────────────────────────────────┘
```

### 3.2 Command と Query の分離

```typescript
// === Command Side ===

// Command 定義
interface CreateOrderCommand {
  type: "CreateOrder";
  customerId: string;
  items: Array<{ productId: string; quantity: number; price: number }>;
}

interface ShipOrderCommand {
  type: "ShipOrder";
  orderId: string;
  carrier: string;
  trackingNumber: string;
}

// Command Handler
class OrderCommandHandler {
  constructor(
    private eventStore: EventStore,
    private eventBus: EventBus,
  ) {}

  async handle(command: CreateOrderCommand | ShipOrderCommand): Promise<void> {
    switch (command.type) {
      case "CreateOrder":
        return this.handleCreateOrder(command);
      case "ShipOrder":
        return this.handleShipOrder(command);
    }
  }

  private async handleCreateOrder(cmd: CreateOrderCommand): Promise<void> {
    const orderId = crypto.randomUUID();
    const order = Order.create(orderId, cmd.customerId, cmd.items);

    const events = order.getUncommittedEvents();
    await this.eventStore.append(orderId, events, 0);
    order.clearUncommittedEvents();

    // イベントを発行（プロジェクション更新のため）
    for (const event of events) {
      await this.eventBus.publish(event);
    }
  }

  private async handleShipOrder(cmd: ShipOrderCommand): Promise<void> {
    const events = await this.eventStore.getEvents(cmd.orderId);
    const order = Order.fromEvents(events);

    order.ship(cmd.carrier, cmd.trackingNumber);

    const newEvents = order.getUncommittedEvents();
    await this.eventStore.append(cmd.orderId, newEvents, events.length);
    order.clearUncommittedEvents();

    for (const event of newEvents) {
      await this.eventBus.publish(event);
    }
  }
}
```

```typescript
// === Query Side ===

// Read Model (Projection)
interface OrderSummaryView {
  orderId: string;
  customerName: string;
  status: string;
  itemCount: number;
  totalAmount: number;
  lastUpdated: Date;
}

// Projector: イベントから Read Model を構築
class OrderSummaryProjector {
  constructor(private readDb: ReadDatabase) {}

  async on(event: DomainEvent): Promise<void> {
    switch (event.eventType) {
      case "OrderCreated":
        await this.readDb.upsert("order_summaries", {
          orderId: event.aggregateId,
          customerName: event.data.customerName,
          status: "created",
          itemCount: event.data.items.length,
          totalAmount: event.data.items.reduce(
            (sum: number, i: any) => sum + i.price * i.quantity, 0
          ),
          lastUpdated: event.metadata.timestamp,
        });
        break;

      case "OrderShipped":
        await this.readDb.update("order_summaries", event.aggregateId, {
          status: "shipped",
          lastUpdated: event.metadata.timestamp,
        });
        break;
    }
  }
}

// Query Handler
class OrderQueryHandler {
  constructor(private readDb: ReadDatabase) {}

  async getOrderSummary(orderId: string): Promise<OrderSummaryView | null> {
    return this.readDb.findOne("order_summaries", { orderId });
  }

  async getOrdersByCustomer(
    customerId: string,
    page: number = 1
  ): Promise<OrderSummaryView[]> {
    return this.readDb.find("order_summaries", {
      where: { customerId },
      orderBy: { lastUpdated: "desc" },
      limit: 20,
      offset: (page - 1) * 20,
    });
  }
}
```

---

## 4. スナップショット最適化

### 4.1 スナップショットの仕組み

```
イベント数が増大すると再生コストが増加:

  Without Snapshot:
  Event 1 → Event 2 → ... → Event 999 → Event 1000
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  全1000イベントを再生（遅い）

  With Snapshot (100イベントごと):
  [Snapshot @ Event 900] → Event 901 → ... → Event 1000
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                           100イベントのみ再生（速い）

  ┌──────────────────────────────────────────────────┐
  │  スナップショット戦略                             │
  │                                                  │
  │  保存タイミング:                                  │
  │  - N イベントごと (例: 100件ごと)                 │
  │  - 時間ベース (例: 1時間ごと)                     │
  │  - 手動 (負荷が高い集約のみ)                      │
  │                                                  │
  │  保存先:                                          │
  │  - 同じ Event Store の別テーブル                   │
  │  - Redis / Memcached (高速アクセス)               │
  └──────────────────────────────────────────────────┘
```

### 4.2 スナップショット実装

```typescript
interface Snapshot {
  aggregateId: string;
  version: number;
  state: Record<string, unknown>;
  createdAt: Date;
}

class SnapshotRepository {
  constructor(private pool: Pool) {}

  async save(snapshot: Snapshot): Promise<void> {
    await this.pool.query(
      `INSERT INTO snapshots (aggregate_id, version, state, created_at)
       VALUES ($1, $2, $3, $4)
       ON CONFLICT (aggregate_id) DO UPDATE
       SET version = $2, state = $3, created_at = $4`,
      [snapshot.aggregateId, snapshot.version, JSON.stringify(snapshot.state), snapshot.createdAt]
    );
  }

  async load(aggregateId: string): Promise<Snapshot | null> {
    const { rows } = await this.pool.query(
      "SELECT * FROM snapshots WHERE aggregate_id = $1",
      [aggregateId]
    );
    return rows[0] ?? null;
  }
}

// スナップショットを活用した集約の復元
class OrderRepository {
  constructor(
    private eventStore: EventStore,
    private snapshotRepo: SnapshotRepository,
    private snapshotInterval: number = 100,
  ) {}

  async load(orderId: string): Promise<Order> {
    // 1. スナップショットを取得
    const snapshot = await this.snapshotRepo.load(orderId);

    let order: Order;
    if (snapshot) {
      // 2a. スナップショットから復元 + 差分イベントのみ再生
      order = Order.fromSnapshot(snapshot.state);
      const newEvents = await this.eventStore.getEventsAfterVersion(orderId, snapshot.version);
      order.replayEvents(newEvents);
    } else {
      // 2b. 全イベントから復元
      const events = await this.eventStore.getEvents(orderId);
      order = Order.fromEvents(events);
    }

    return order;
  }

  async save(order: Order): Promise<void> {
    const events = order.getUncommittedEvents();
    await this.eventStore.append(order.id, events, order.version - events.length);
    order.clearUncommittedEvents();

    // スナップショット保存判定
    if (order.version % this.snapshotInterval === 0) {
      await this.snapshotRepo.save({
        aggregateId: order.id,
        version: order.version,
        state: order.toSnapshot(),
        createdAt: new Date(),
      });
    }
  }
}
```

---

## 5. 比較表

### 5.1 CRUD vs Event Sourcing

| 観点 | 従来の CRUD | Event Sourcing |
|------|-----------|----------------|
| **データモデル** | 現在の状態のみ | イベントの履歴 |
| **変更履歴** | 手動で監査テーブル作成 | 自動的に全履歴保持 |
| **任意時点の復元** | 不可能（別途実装必要） | イベント再生で可能 |
| **データ容量** | 少ない（現在の状態のみ） | 大きい（全イベント保存） |
| **読み取り性能** | 高い（直接クエリ） | 低い（再生が必要、プロジェクションで解決） |
| **書き込み性能** | 中（UPDATE） | 高い（追記のみ、ロック少ない） |
| **複雑性** | 低い | 高い |
| **デバッグ** | 現在の状態のみ確認可能 | イベントログで原因追跡容易 |

### 5.2 CQRS 導入パターン比較

| パターン | 説明 | 整合性 | 複雑度 | ユースケース |
|---------|------|--------|--------|------------|
| **単一 DB + ビュー** | 同一 DB に Read Model をビューとして作成 | 即時整合 | 低 | 小規模 |
| **単一 DB + テーブル分離** | Write 用テーブルと Read 用テーブルを分離 | ほぼ即時 | 中 | 中規模 |
| **DB 分離 + 同期更新** | Write DB と Read DB を同期的に更新 | 即時整合 | 中〜高 | 整合性重視 |
| **DB 分離 + 非同期更新** | イベントで非同期にプロジェクション更新 | 結果整合 | 高 | 高スケーラビリティ |

---

## 6. アンチパターン

### 6.1 全てのドメインに Event Sourcing を適用

```
NG: マスタデータ（商品カテゴリ、設定値）にも Event Sourcing
    → 過度な複雑性、CRUD で十分な領域

OK: Event Sourcing の適用基準
    ┌───────────────────────────────────────┐
    │ Event Sourcing が適する:              │
    │ - 注文処理、金融取引                  │
    │ - 監査証跡が法的に必要               │
    │ - ビジネスイベントの分析が重要       │
    │ - 複雑な状態遷移がある               │
    │                                       │
    │ CRUD で十分:                          │
    │ - ユーザープロフィール               │
    │ - マスタデータ管理                   │
    │ - 設定値管理                         │
    │ - CRUD が中心の管理画面              │
    └───────────────────────────────────────┘
```

### 6.2 イベントスキーマを頻繁に変更する

```typescript
// NG: 既存イベントの構造を変更
// v1: { amount: 1000 }
// v2: { amount: 1000, currency: "JPY" }  ← 既存イベントが壊れる

// OK: イベントのバージョニング
interface OrderCreatedV1 {
  eventType: "OrderCreated";
  version: 1;
  data: { amount: number };
}

interface OrderCreatedV2 {
  eventType: "OrderCreated";
  version: 2;
  data: { amount: number; currency: string };
}

// Upcaster: 古いバージョンを新しいバージョンに変換
function upcastOrderCreated(event: OrderCreatedV1): OrderCreatedV2 {
  return {
    ...event,
    version: 2,
    data: {
      ...event.data,
      currency: "JPY",  // デフォルト値を付与
    },
  };
}
```

---

## 7. FAQ

### Q1. Event Sourcing のイベントが増え続けてストレージが心配

**A.** 3 つの対策がある:
1. **スナップショット** — N イベントごとに状態を保存し、古いイベントの再生を省略
2. **アーカイブ** — 一定期間経過したイベントを S3/Glacier に移動
3. **コンパクション** — 完了した集約のイベントを圧縮（監査要件を確認の上）

実務では、数千万イベントでも PostgreSQL で問題なく運用できる。パーティショニング（月別等）を適用すれば検索性能も維持可能。

### Q2. CQRS で結果整合性になると UI はどう対応する？

**A.** 3 つのパターンがある:
1. **楽観的 UI** — コマンド成功後に UI をローカルで即座に更新（Read Model の更新を待たない）
2. **ポーリング** — コマンド後に短い間隔で Read Model を確認
3. **WebSocket / SSE** — プロジェクション更新完了をリアルタイムに通知

```typescript
// 楽観的 UI の例
async function submitOrder(orderData) {
  // 1. コマンド送信
  await api.post("/commands/create-order", orderData);

  // 2. UI をローカルで即座に更新（Read Model 更新を待たない）
  setOrders(prev => [...prev, { ...orderData, status: "processing" }]);

  // 3. バックグラウンドで Read Model の反映を確認
  await pollUntilConsistent(`/orders/${orderData.id}`);
}
```

### Q3. Event Sourcing を既存の CRUD アプリに段階的に導入できる？

**A.** 可能。推奨される段階的アプローチ:
1. **Phase 1**: 重要なドメインイベントを CRUD と並行してイベントテーブルに記録（Dual Write）
2. **Phase 2**: Read Model（プロジェクション）を構築し、読み取りクエリを移行
3. **Phase 3**: 書き込みを Event Sourcing に移行（集約単位で段階的に）
4. **Phase 4**: 旧 CRUD テーブルを廃止

Strangler Fig パターンと組み合わせて、既存システムを段階的に置き換えるのが安全。

---

## 8. まとめ

| 項目 | ポイント |
|------|---------|
| **Event Sourcing** | 状態の変化をイベントとして追記保存、完全な監査証跡 |
| **CQRS** | 読み取りと書き込みを分離、それぞれを独立にスケール |
| **プロジェクション** | イベントから最適化された Read Model を構築 |
| **スナップショット** | イベント再生コストを削減する最適化手法 |
| **適用基準** | 監査要件、複雑な状態遷移、高スケーラビリティが必要な場合に適用 |

---

## 次に読むべきガイド

- [00-mvc-mvvm.md](./00-mvc-mvvm.md) — UI レイヤーのアーキテクチャパターン
- [01-repository-pattern.md](./01-repository-pattern.md) — データアクセスの抽象化
- メッセージキューガイド — イベントバスの実装（Kafka, RabbitMQ）

---

## 参考文献

1. **Martin Fowler** — "Event Sourcing" — https://martinfowler.com/eaaDev/EventSourcing.html
2. **Martin Fowler** — "CQRS" — https://martinfowler.com/bliki/CQRS.html
3. **Greg Young** — "CQRS and Event Sourcing" — https://cqrs.files.wordpress.com/2010/11/cqrs_documents.pdf
4. **Microsoft** — "CQRS pattern" — https://learn.microsoft.com/en-us/azure/architecture/patterns/cqrs
