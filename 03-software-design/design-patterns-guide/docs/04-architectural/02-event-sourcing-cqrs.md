# Event Sourcing / CQRS — イベント駆動設計

> Event Sourcing で状態の変化を不変のイベントログとして記録し、CQRS で読み取りと書き込みを分離することで、監査可能性・スケーラビリティ・ドメインの表現力を最大化するための実践ガイド。イベントストア設計、プロジェクション、スナップショット最適化、段階的導入戦略まで網羅する。

---

## 前提知識

| トピック | 必要レベル | 参照先 |
|---------|-----------|--------|
| TypeScript / JavaScript | 中級（ジェネリクス、async/await、discriminated union） | [02-programming](../../../../02-programming/) |
| リレーショナルDB の基礎 | 基礎（INSERT, SELECT, トランザクション） | [06-data-and-security](../../../../06-data-and-security/) |
| Repository パターン | 基礎 | [01-repository-pattern.md](./01-repository-pattern.md) |
| DDD の基礎概念 | 基礎（集約、ドメインイベント） | [../../clean-code-principles/](../../clean-code-principles/) |
| メッセージキューの基礎 | 基礎（Pub/Sub モデル） | [../../system-design-guide/](../../system-design-guide/) |

---

## この章で学ぶこと

1. **Event Sourcing** — 状態をイベントの履歴として保存するパターンの設計と実装
2. **CQRS（Command Query Responsibility Segregation）** — 読み取りと書き込みの責務分離の4段階
3. **Event Sourcing + CQRS** の組み合わせ — プロジェクション設計と結果整合性の管理
4. **スナップショット最適化** — イベント再生コストを削減する戦略
5. **段階的導入** — 既存 CRUD アプリからの移行パスとイベントスキーマのバージョニング

---

## 1. 従来の CRUD vs Event Sourcing

### WHY: なぜ Event Sourcing が必要か

従来の CRUD はデータの「現在の状態」だけを保存する。これにより以下の問題が発生する:

1. **監査証跡の欠如** — 「いつ」「誰が」「なぜ」変更したかが分からない
2. **時系列分析不可** — 過去の任意時点の状態を再構築できない
3. **ドメイン知識の損失** — 「価格を変更した」のか「割引を適用した」のか区別できない
4. **同時更新の競合** — UPDATE は上書きのため、最後の書き込みが勝つ（Lost Update）

Event Sourcing はこれらを根本的に解決する。

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
│        注文が 12,000円 → 15,000円 に変わった理由は？  │
│        ・商品追加？ ・価格変更？ ・送料追加？          │
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
│        「なぜ 15,000円か」= Event 1+2 の合計           │
└──────────────────────────────────────────────────────┘
```

### 1.2 Event Sourcing の状態復元

```
イベントの再生 (Replay) による状態復元:

  Event 1: OrderCreated      → Order { status: "created", items: [], total: 0 }
      │
      ▼
  Event 2: ItemAdded(A,5000)  → Order { status: "created", items: [A], total: 5000 }
      │
      ▼
  Event 3: ItemAdded(B,7000)  → Order { status: "created", items: [A,B], total: 12000 }
      │
      ▼
  Event 4: ShippingAdded(3000)→ Order { status: "created", items: [A,B], total: 15000 }
      │
      ▼
  Event 5: PaymentReceived    → Order { status: "paid", items: [A,B], total: 15000 }
      │
      ▼
  Event 6: OrderShipped       → Order { status: "shipped", items: [A,B], total: 15000 }

  ※ 2月9日時点の状態が必要 → Event 1-5 を再生すれば OK
  ※ 「なぜ15,000円？」→ Event 2(5000) + Event 3(7000) + Event 4(3000)
```

### 1.3 Event Sourcing の核心概念

```
┌───────────────────────────────────────────────────────────┐
│  Event Sourcing の3つの核心                                │
│                                                           │
│  1. イベントは不変 (Immutable)                             │
│     ────────────────────────                               │
│     一度記録されたイベントは決して変更・削除しない          │
│     → 監査証跡の信頼性を保証                               │
│                                                           │
│  2. イベントは追記のみ (Append-Only)                       │
│     ────────────────────────────                           │
│     新しいイベントを末尾に追加するだけ                     │
│     → ロック競合が少なく、書き込み性能が高い               │
│                                                           │
│  3. 現在の状態はイベントの導出値 (Derived State)           │
│     ──────────────────────────────────                     │
│     状態 = fold(初期状態, 全イベント, apply関数)           │
│     → 同じイベント列からは常に同じ状態が再構築される       │
│     → 関数型プログラミングの reduce と同じ概念             │
└───────────────────────────────────────────────────────────┘

数学的表現:
  state(t) = reduce(apply, initialState, events[0..t])

TypeScript 風:
  const currentState = events.reduce(
    (state, event) => applyEvent(state, event),
    initialState
  );
```

---

## 2. Event Sourcing の実装

### 2.1 イベント定義（TypeScript）

```typescript
// ============================================================
// Domain Events — 型安全なイベント定義
// ============================================================

// 基底イベント型
interface DomainEvent {
  readonly eventId: string;
  readonly aggregateId: string;
  readonly aggregateType: string;
  readonly eventType: string;
  readonly data: Readonly<Record<string, unknown>>;
  readonly metadata: Readonly<{
    timestamp: Date;
    version: number;
    userId?: string;          // 誰がこの変更を行ったか
    correlationId?: string;   // 関連する操作のID（分散トレーシング）
    causationId?: string;     // このイベントの原因となったイベントID
  }>;
}

// 注文ドメインのイベント（Discriminated Union で型安全に）
type OrderEvent =
  | OrderCreated
  | ItemAdded
  | ItemRemoved
  | DiscountApplied
  | PaymentReceived
  | OrderShipped
  | OrderCancelled;

interface OrderCreated extends DomainEvent {
  readonly eventType: "OrderCreated";
  readonly data: {
    readonly customerId: string;
    readonly items: ReadonlyArray<{
      productId: string;
      quantity: number;
      unitPrice: number;
    }>;
    readonly shippingAddress: {
      street: string;
      city: string;
      postalCode: string;
    };
  };
}

interface ItemAdded extends DomainEvent {
  readonly eventType: "ItemAdded";
  readonly data: {
    readonly productId: string;
    readonly productName: string;
    readonly quantity: number;
    readonly unitPrice: number;
  };
}

interface ItemRemoved extends DomainEvent {
  readonly eventType: "ItemRemoved";
  readonly data: {
    readonly productId: string;
    readonly quantity: number;
    readonly reason: string;
  };
}

interface DiscountApplied extends DomainEvent {
  readonly eventType: "DiscountApplied";
  readonly data: {
    readonly discountType: "percentage" | "fixed";
    readonly value: number;
    readonly couponCode?: string;
    readonly reason: string;
  };
}

interface PaymentReceived extends DomainEvent {
  readonly eventType: "PaymentReceived";
  readonly data: {
    readonly amount: number;
    readonly paymentMethod: "credit_card" | "bank_transfer" | "wallet";
    readonly transactionId: string;
  };
}

interface OrderShipped extends DomainEvent {
  readonly eventType: "OrderShipped";
  readonly data: {
    readonly carrier: string;
    readonly trackingNumber: string;
    readonly estimatedDelivery: string;
  };
}

interface OrderCancelled extends DomainEvent {
  readonly eventType: "OrderCancelled";
  readonly data: {
    readonly reason: string;
    readonly cancelledBy: string;
    readonly refundAmount: number;
  };
}
```

### 2.2 集約（Aggregate）の実装

```typescript
// ============================================================
// Order Aggregate — イベントソーシング対応
// ============================================================

interface OrderItem {
  productId: string;
  productName: string;
  quantity: number;
  unitPrice: number;
}

type OrderStatus = "created" | "paid" | "shipped" | "delivered" | "cancelled";

class Order {
  private _id: string = "";
  private _customerId: string = "";
  private _status: OrderStatus = "created";
  private _items: OrderItem[] = [];
  private _totalAmount: number = 0;
  private _discountAmount: number = 0;
  private _version: number = 0;
  private _uncommittedEvents: OrderEvent[] = [];

  // ============================================
  // クエリ（読み取り）
  // ============================================
  get id(): string { return this._id; }
  get status(): OrderStatus { return this._status; }
  get items(): ReadonlyArray<OrderItem> { return this._items; }
  get totalAmount(): number { return this._totalAmount; }
  get netAmount(): number { return this._totalAmount - this._discountAmount; }
  get version(): number { return this._version; }

  // ============================================
  // ファクトリ: イベントから状態を復元
  // ============================================
  static fromEvents(events: OrderEvent[]): Order {
    const order = new Order();
    for (const event of events) {
      order.applyEvent(event, false);  // 既存イベントは uncommitted に追加しない
    }
    return order;
  }

  static fromSnapshot(snapshot: OrderSnapshot, newEvents: OrderEvent[]): Order {
    const order = new Order();
    order._id = snapshot.id;
    order._customerId = snapshot.customerId;
    order._status = snapshot.status;
    order._items = [...snapshot.items];
    order._totalAmount = snapshot.totalAmount;
    order._discountAmount = snapshot.discountAmount;
    order._version = snapshot.version;

    // スナップショット以降のイベントを再生
    for (const event of newEvents) {
      order.applyEvent(event, false);
    }
    return order;
  }

  // ============================================
  // コマンド（書き込み） — ビジネスルールの検証 + イベント発行
  // ============================================
  static create(
    orderId: string,
    customerId: string,
    items: OrderItem[],
    shippingAddress: { street: string; city: string; postalCode: string },
    userId: string
  ): Order {
    // ビジネスルールの検証
    if (items.length === 0) {
      throw new DomainError("注文には1つ以上の商品が必要です");
    }
    if (items.some((i) => i.quantity <= 0)) {
      throw new DomainError("商品数量は正の整数でなければなりません");
    }

    const order = new Order();
    order.raiseEvent({
      eventType: "OrderCreated",
      aggregateId: orderId,
      data: {
        customerId,
        items: items.map((i) => ({
          productId: i.productId,
          quantity: i.quantity,
          unitPrice: i.unitPrice,
        })),
        shippingAddress,
      },
    }, userId);
    return order;
  }

  addItem(item: OrderItem, userId: string): void {
    // ビジネスルール: 支払い済みの注文には商品を追加できない
    if (this._status !== "created") {
      throw new DomainError(
        `状態 "${this._status}" の注文に商品を追加できません`
      );
    }
    if (item.quantity <= 0) {
      throw new DomainError("商品数量は正の整数でなければなりません");
    }

    this.raiseEvent({
      eventType: "ItemAdded",
      aggregateId: this._id,
      data: {
        productId: item.productId,
        productName: item.productName,
        quantity: item.quantity,
        unitPrice: item.unitPrice,
      },
    }, userId);
  }

  removeItem(productId: string, quantity: number, reason: string, userId: string): void {
    if (this._status !== "created") {
      throw new DomainError("支払い済みの注文から商品を削除できません");
    }
    const existingItem = this._items.find((i) => i.productId === productId);
    if (!existingItem) {
      throw new DomainError(`商品 ${productId} は注文に含まれていません`);
    }
    if (existingItem.quantity < quantity) {
      throw new DomainError("削除数量が注文数量を超えています");
    }

    this.raiseEvent({
      eventType: "ItemRemoved",
      aggregateId: this._id,
      data: { productId, quantity, reason },
    }, userId);
  }

  applyDiscount(type: "percentage" | "fixed", value: number, reason: string, couponCode?: string, userId?: string): void {
    if (this._status !== "created") {
      throw new DomainError("支払い済みの注文に割引を適用できません");
    }
    if (type === "percentage" && (value < 0 || value > 100)) {
      throw new DomainError("割引率は 0-100% の範囲でなければなりません");
    }

    this.raiseEvent({
      eventType: "DiscountApplied",
      aggregateId: this._id,
      data: { discountType: type, value, couponCode, reason },
    }, userId ?? "system");
  }

  receivePayment(amount: number, method: PaymentReceived["data"]["paymentMethod"], txId: string, userId: string): void {
    if (this._status !== "created") {
      throw new DomainError("この注文はすでに支払い済みです");
    }
    if (amount < this.netAmount) {
      throw new DomainError(
        `支払い金額 ${amount} が不足しています（必要額: ${this.netAmount}）`
      );
    }

    this.raiseEvent({
      eventType: "PaymentReceived",
      aggregateId: this._id,
      data: { amount, paymentMethod: method, transactionId: txId },
    }, userId);
  }

  ship(carrier: string, trackingNumber: string, estimatedDelivery: string, userId: string): void {
    if (this._status !== "paid") {
      throw new DomainError("支払い済みの注文のみ発送できます");
    }

    this.raiseEvent({
      eventType: "OrderShipped",
      aggregateId: this._id,
      data: { carrier, trackingNumber, estimatedDelivery },
    }, userId);
  }

  cancel(reason: string, userId: string): void {
    if (this._status === "shipped" || this._status === "delivered") {
      throw new DomainError("発送済みの注文はキャンセルできません");
    }
    if (this._status === "cancelled") {
      throw new DomainError("既にキャンセル済みです");
    }

    const refundAmount = this._status === "paid" ? this.netAmount : 0;
    this.raiseEvent({
      eventType: "OrderCancelled",
      aggregateId: this._id,
      data: { reason, cancelledBy: userId, refundAmount },
    }, userId);
  }

  // ============================================
  // イベント適用（状態変更ロジック）— 純粋関数
  // ============================================
  private applyEvent(event: OrderEvent, isNew: boolean = true): void {
    switch (event.eventType) {
      case "OrderCreated":
        this._id = event.aggregateId;
        this._customerId = event.data.customerId;
        this._status = "created";
        this._items = event.data.items.map((i) => ({
          productId: i.productId,
          productName: "",
          quantity: i.quantity,
          unitPrice: i.unitPrice,
        }));
        this._totalAmount = this._items.reduce(
          (sum, i) => sum + i.unitPrice * i.quantity, 0
        );
        break;

      case "ItemAdded":
        this._items.push({
          productId: event.data.productId,
          productName: event.data.productName,
          quantity: event.data.quantity,
          unitPrice: event.data.unitPrice,
        });
        this._totalAmount += event.data.unitPrice * event.data.quantity;
        break;

      case "ItemRemoved": {
        const idx = this._items.findIndex(
          (i) => i.productId === event.data.productId
        );
        if (idx >= 0) {
          this._totalAmount -= this._items[idx].unitPrice * event.data.quantity;
          this._items[idx].quantity -= event.data.quantity;
          if (this._items[idx].quantity <= 0) {
            this._items.splice(idx, 1);
          }
        }
        break;
      }

      case "DiscountApplied":
        if (event.data.discountType === "percentage") {
          this._discountAmount = this._totalAmount * (event.data.value / 100);
        } else {
          this._discountAmount = event.data.value;
        }
        break;

      case "PaymentReceived":
        this._status = "paid";
        break;

      case "OrderShipped":
        this._status = "shipped";
        break;

      case "OrderCancelled":
        this._status = "cancelled";
        break;
    }

    this._version++;
    if (isNew) {
      this._uncommittedEvents.push(event as OrderEvent);
    }
  }

  // ============================================
  // イベント発行ヘルパー
  // ============================================
  private raiseEvent(
    eventData: Pick<OrderEvent, "eventType" | "aggregateId" | "data">,
    userId: string
  ): void {
    const event = {
      ...eventData,
      eventId: crypto.randomUUID(),
      aggregateType: "Order",
      metadata: {
        timestamp: new Date(),
        version: this._version + 1,
        userId,
      },
    } as OrderEvent;
    this.applyEvent(event, true);
  }

  getUncommittedEvents(): ReadonlyArray<OrderEvent> {
    return [...this._uncommittedEvents];
  }

  clearUncommittedEvents(): void {
    this._uncommittedEvents = [];
  }

  toSnapshot(): OrderSnapshot {
    return {
      id: this._id,
      customerId: this._customerId,
      status: this._status,
      items: [...this._items],
      totalAmount: this._totalAmount,
      discountAmount: this._discountAmount,
      version: this._version,
    };
  }
}

interface OrderSnapshot {
  id: string;
  customerId: string;
  status: OrderStatus;
  items: OrderItem[];
  totalAmount: number;
  discountAmount: number;
  version: number;
}

class DomainError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "DomainError";
  }
}
```

### 2.3 Event Store の実装

```typescript
// ============================================================
// Event Store Interface（ドメイン層）
// ============================================================
interface EventStore {
  append(
    aggregateId: string,
    events: DomainEvent[],
    expectedVersion: number
  ): Promise<void>;
  getEvents(aggregateId: string): Promise<DomainEvent[]>;
  getEventsAfterVersion(
    aggregateId: string,
    version: number
  ): Promise<DomainEvent[]>;
  getAllEvents(options?: {
    eventTypes?: string[];
    since?: Date;
    limit?: number;
  }): AsyncIterable<DomainEvent>;
}

// ============================================================
// PostgreSQL 実装
// ============================================================
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
        `SELECT COALESCE(MAX(version), 0) as current_version
         FROM events WHERE aggregate_id = $1
         FOR UPDATE`,  // SELECT FOR UPDATE で排他ロック
        [aggregateId]
      );
      const currentVersion = rows[0].current_version;

      if (currentVersion !== expectedVersion) {
        throw new ConcurrencyError(
          `Expected version ${expectedVersion}, but found ${currentVersion}. ` +
          `Another process may have modified this aggregate.`
        );
      }

      // イベントを追記（バッチ INSERT）
      const values: unknown[] = [];
      const placeholders: string[] = [];
      let paramIndex = 1;

      for (const event of events) {
        placeholders.push(
          `($${paramIndex}, $${paramIndex + 1}, $${paramIndex + 2}, ` +
          `$${paramIndex + 3}, $${paramIndex + 4}, $${paramIndex + 5}, $${paramIndex + 6})`
        );
        values.push(
          event.eventId,
          event.aggregateId,
          event.aggregateType,
          event.eventType,
          JSON.stringify(event.data),
          JSON.stringify(event.metadata),
          event.metadata.version
        );
        paramIndex += 7;
      }

      await client.query(
        `INSERT INTO events
         (event_id, aggregate_id, aggregate_type, event_type, data, metadata, version)
         VALUES ${placeholders.join(", ")}`,
        values
      );

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

  async getEventsAfterVersion(
    aggregateId: string,
    version: number
  ): Promise<DomainEvent[]> {
    const { rows } = await this.pool.query(
      `SELECT * FROM events
       WHERE aggregate_id = $1 AND version > $2
       ORDER BY version ASC`,
      [aggregateId, version]
    );
    return rows.map(this.toDomainEvent);
  }

  async *getAllEvents(options?: {
    eventTypes?: string[];
    since?: Date;
    limit?: number;
  }): AsyncIterable<DomainEvent> {
    let query = "SELECT * FROM events WHERE 1=1";
    const params: unknown[] = [];
    let paramIndex = 1;

    if (options?.eventTypes?.length) {
      query += ` AND event_type = ANY($${paramIndex})`;
      params.push(options.eventTypes);
      paramIndex++;
    }
    if (options?.since) {
      query += ` AND (metadata->>'timestamp')::timestamptz >= $${paramIndex}`;
      params.push(options.since);
      paramIndex++;
    }
    query += " ORDER BY (metadata->>'timestamp')::timestamptz ASC";
    if (options?.limit) {
      query += ` LIMIT $${paramIndex}`;
      params.push(options.limit);
    }

    const { rows } = await this.pool.query(query, params);
    for (const row of rows) {
      yield this.toDomainEvent(row);
    }
  }

  private toDomainEvent(row: any): DomainEvent {
    return {
      eventId: row.event_id,
      aggregateId: row.aggregate_id,
      aggregateType: row.aggregate_type,
      eventType: row.event_type,
      data: row.data,
      metadata: {
        ...row.metadata,
        timestamp: new Date(row.metadata.timestamp),
      },
    };
  }
}

class ConcurrencyError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ConcurrencyError";
  }
}
```

### 2.4 Event Store のスキーマ

```sql
-- ============================================================
-- PostgreSQL: Event Store テーブル定義
-- ============================================================

CREATE TABLE events (
    -- 主キー
    id              BIGSERIAL PRIMARY KEY,

    -- イベント識別
    event_id        UUID NOT NULL UNIQUE,
    aggregate_id    UUID NOT NULL,
    aggregate_type  VARCHAR(100) NOT NULL,
    event_type      VARCHAR(100) NOT NULL,

    -- イベントデータ
    data            JSONB NOT NULL,
    metadata        JSONB NOT NULL,

    -- バージョニング（楽観的ロック用）
    version         INTEGER NOT NULL,

    -- タイムスタンプ
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- 集約ごとのバージョンはユニーク
    CONSTRAINT unique_aggregate_version
        UNIQUE (aggregate_id, version)
);

-- インデックス
CREATE INDEX idx_events_aggregate_id ON events (aggregate_id, version);
CREATE INDEX idx_events_event_type ON events (event_type);
CREATE INDEX idx_events_created_at ON events (created_at);

-- スナップショットテーブル
CREATE TABLE snapshots (
    aggregate_id    UUID PRIMARY KEY,
    aggregate_type  VARCHAR(100) NOT NULL,
    version         INTEGER NOT NULL,
    state           JSONB NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

---

## 3. CQRS の設計

### WHY: なぜ CQRS が必要か

読み取りと書き込みは根本的に異なる要件を持つ:

```
┌───────────────────────────────────────────────────────┐
│  読み取りと書き込みの非対称性                           │
│                                                       │
│  書き込み (Command):                                  │
│  ・ドメインモデルの一貫性が重要                        │
│  ・ビジネスルールの検証が必要                          │
│  ・トランザクション整合性                              │
│  ・通常は1つの集約を操作                              │
│  ・トラフィックの 10-20%                              │
│                                                       │
│  読み取り (Query):                                    │
│  ・表示に最適化されたデータ形式が重要                  │
│  ・複数の集約をJOINした情報が必要                      │
│  ・キャッシュが効果的                                  │
│  ・結果整合性で十分なことが多い                        │
│  ・トラフィックの 80-90%                              │
│                                                       │
│  → 同じモデルで両方を最適化するのは困難                │
│  → 分離すればそれぞれに最適な設計ができる              │
└───────────────────────────────────────────────────────┘
```

### 3.1 CQRS の4段階

```
┌───────────────────────────────────────────────────────────┐
│  CQRS の段階的導入                                         │
│                                                           │
│  Level 0: 従来型（分離なし）                               │
│  ┌──────────────┐                                         │
│  │ 同一 Model   │ ← 読み書き両方に同じモデル               │
│  │ (User Entity)│                                         │
│  └──────┬───────┘                                         │
│         │                                                 │
│  ┌──────▼───────┐                                         │
│  │   同一 DB    │                                         │
│  └──────────────┘                                         │
│                                                           │
│  Level 1: モデル分離（同一 DB）                             │
│  ┌──────────┐  ┌──────────┐                               │
│  │ Write    │  │ Read     │ ← 異なるモデル                 │
│  │ Model    │  │ Model    │    (DTO / View Model)          │
│  └────┬─────┘  └────┬─────┘                               │
│       └──────┬──────┘                                     │
│       ┌──────▼───────┐                                    │
│       │   同一 DB    │                                    │
│       └──────────────┘                                    │
│                                                           │
│  Level 2: DB 分離（同期更新）                              │
│  ┌──────────┐           ┌──────────┐                      │
│  │ Write    │           │ Read     │                      │
│  │ Model    │           │ Model    │                      │
│  └────┬─────┘           └────┬─────┘                      │
│  ┌────▼─────┐  同期    ┌────▼─────┐                      │
│  │ Write DB │ ───────→ │ Read DB  │                      │
│  └──────────┘          └──────────┘                      │
│                                                           │
│  Level 3: Event Sourcing + 非同期プロジェクション          │
│  ┌──────────┐           ┌──────────┐                      │
│  │ Command  │           │ Query    │                      │
│  │ Handler  │           │ Handler  │                      │
│  └────┬─────┘           └────┬─────┘                      │
│  ┌────▼──────┐  Event  ┌────▼─────┐                      │
│  │ Event     │ ──────→ │ Read DB  │ ← 非同期で最適化      │
│  │ Store     │ Stream  │(Projection)│  された形に投影      │
│  └───────────┘         └──────────┘                      │
└───────────────────────────────────────────────────────────┘
```

### 3.2 Command と Query の分離

```typescript
// ============================================================
// Command Side（書き込み）
// ============================================================

// Command 定義
type OrderCommand =
  | CreateOrderCommand
  | AddItemCommand
  | ApplyDiscountCommand
  | ReceivePaymentCommand
  | ShipOrderCommand
  | CancelOrderCommand;

interface CreateOrderCommand {
  type: "CreateOrder";
  orderId: string;
  customerId: string;
  items: Array<{ productId: string; quantity: number; unitPrice: number }>;
  shippingAddress: { street: string; city: string; postalCode: string };
  userId: string;
}

interface ShipOrderCommand {
  type: "ShipOrder";
  orderId: string;
  carrier: string;
  trackingNumber: string;
  estimatedDelivery: string;
  userId: string;
}

// Command Handler
class OrderCommandHandler {
  constructor(
    private eventStore: EventStore,
    private snapshotRepo: SnapshotRepository,
    private eventBus: EventBus,
  ) {}

  async handle(command: OrderCommand): Promise<void> {
    switch (command.type) {
      case "CreateOrder":
        return this.handleCreateOrder(command);
      case "ShipOrder":
        return this.handleShipOrder(command);
      // ... 他のコマンド
    }
  }

  private async handleCreateOrder(cmd: CreateOrderCommand): Promise<void> {
    // 新しい集約を作成
    const order = Order.create(
      cmd.orderId,
      cmd.customerId,
      cmd.items.map((i) => ({
        productId: i.productId,
        productName: "",
        quantity: i.quantity,
        unitPrice: i.unitPrice,
      })),
      cmd.shippingAddress,
      cmd.userId
    );

    // イベントを永続化
    const events = order.getUncommittedEvents();
    await this.eventStore.append(cmd.orderId, [...events], 0);
    order.clearUncommittedEvents();

    // イベントを発行（プロジェクション更新のため）
    for (const event of events) {
      await this.eventBus.publish(event);
    }
  }

  private async handleShipOrder(cmd: ShipOrderCommand): Promise<void> {
    // 集約を復元
    const order = await this.loadOrder(cmd.orderId);

    // コマンド実行（ビジネスルール検証 + イベント発行）
    order.ship(
      cmd.carrier,
      cmd.trackingNumber,
      cmd.estimatedDelivery,
      cmd.userId
    );

    // イベントを永続化
    const newEvents = order.getUncommittedEvents();
    await this.eventStore.append(
      cmd.orderId,
      [...newEvents],
      order.version - newEvents.length
    );
    order.clearUncommittedEvents();

    // イベントを発行
    for (const event of newEvents) {
      await this.eventBus.publish(event);
    }
  }

  private async loadOrder(orderId: string): Promise<Order> {
    // スナップショットがあれば使用
    const snapshot = await this.snapshotRepo.load(orderId);
    if (snapshot) {
      const newEvents = await this.eventStore.getEventsAfterVersion(
        orderId,
        snapshot.version
      );
      return Order.fromSnapshot(snapshot.state as OrderSnapshot, newEvents as OrderEvent[]);
    }

    // なければ全イベントから復元
    const events = await this.eventStore.getEvents(orderId);
    if (events.length === 0) {
      throw new Error(`Order ${orderId} not found`);
    }
    return Order.fromEvents(events as OrderEvent[]);
  }
}
```

```typescript
// ============================================================
// Query Side（読み取り）
// ============================================================

// Read Model (Projection) — 表示に最適化されたデータ構造
interface OrderSummaryView {
  orderId: string;
  customerName: string;
  customerEmail: string;
  status: string;
  statusLabel: string;  // 表示用ラベル（「発送済み」等）
  itemCount: number;
  totalAmount: number;
  discountAmount: number;
  netAmount: number;
  lastUpdated: Date;
  trackingUrl?: string;
}

interface OrderDetailView extends OrderSummaryView {
  items: Array<{
    productName: string;
    quantity: number;
    unitPrice: number;
    subtotal: number;
  }>;
  shippingAddress: {
    street: string;
    city: string;
    postalCode: string;
  };
  timeline: Array<{
    event: string;
    timestamp: Date;
    description: string;
  }>;
}

// Projector: イベントから Read Model を構築
class OrderProjector {
  constructor(
    private readDb: ReadDatabase,
    private customerService: CustomerService,
  ) {}

  async handle(event: DomainEvent): Promise<void> {
    switch (event.eventType) {
      case "OrderCreated":
        await this.onOrderCreated(event as OrderCreated);
        break;
      case "ItemAdded":
        await this.onItemAdded(event as ItemAdded);
        break;
      case "PaymentReceived":
        await this.onPaymentReceived(event as PaymentReceived);
        break;
      case "OrderShipped":
        await this.onOrderShipped(event as OrderShipped);
        break;
      case "OrderCancelled":
        await this.onOrderCancelled(event as OrderCancelled);
        break;
    }
  }

  private async onOrderCreated(event: OrderCreated): Promise<void> {
    // 顧客情報を取得（Read Model は非正規化してOK）
    const customer = await this.customerService.getById(event.data.customerId);
    const items = event.data.items;
    const totalAmount = items.reduce(
      (sum, i) => sum + i.unitPrice * i.quantity, 0
    );

    await this.readDb.upsert("order_summaries", event.aggregateId, {
      orderId: event.aggregateId,
      customerName: customer.name,
      customerEmail: customer.email,
      status: "created",
      statusLabel: "注文受付",
      itemCount: items.length,
      totalAmount,
      discountAmount: 0,
      netAmount: totalAmount,
      lastUpdated: event.metadata.timestamp,
    });
  }

  private async onItemAdded(event: ItemAdded): Promise<void> {
    const current = await this.readDb.findOne(
      "order_summaries",
      event.aggregateId
    );
    if (!current) return;

    const addedAmount = event.data.unitPrice * event.data.quantity;
    await this.readDb.update("order_summaries", event.aggregateId, {
      itemCount: current.itemCount + 1,
      totalAmount: current.totalAmount + addedAmount,
      netAmount: current.netAmount + addedAmount,
      lastUpdated: event.metadata.timestamp,
    });
  }

  private async onPaymentReceived(event: PaymentReceived): Promise<void> {
    await this.readDb.update("order_summaries", event.aggregateId, {
      status: "paid",
      statusLabel: "支払い完了",
      lastUpdated: event.metadata.timestamp,
    });
  }

  private async onOrderShipped(event: OrderShipped): Promise<void> {
    await this.readDb.update("order_summaries", event.aggregateId, {
      status: "shipped",
      statusLabel: "発送済み",
      trackingUrl: `https://tracking.example.com/${event.data.trackingNumber}`,
      lastUpdated: event.metadata.timestamp,
    });
  }

  private async onOrderCancelled(event: OrderCancelled): Promise<void> {
    await this.readDb.update("order_summaries", event.aggregateId, {
      status: "cancelled",
      statusLabel: "キャンセル",
      lastUpdated: event.metadata.timestamp,
    });
  }
}

// Query Handler — Read Model に対する検索
class OrderQueryHandler {
  constructor(private readDb: ReadDatabase) {}

  async getOrderSummary(orderId: string): Promise<OrderSummaryView | null> {
    return this.readDb.findOne("order_summaries", orderId);
  }

  async getOrdersByCustomer(
    customerId: string,
    page: number = 1,
    perPage: number = 20
  ): Promise<{ data: OrderSummaryView[]; total: number }> {
    return this.readDb.findPaginated("order_summaries", {
      where: { customerId },
      orderBy: { lastUpdated: "desc" },
      page,
      perPage,
    });
  }

  async getRecentOrders(limit: number = 50): Promise<OrderSummaryView[]> {
    return this.readDb.find("order_summaries", {
      orderBy: { lastUpdated: "desc" },
      limit,
    });
  }

  async getOrderStats(): Promise<{
    totalOrders: number;
    pendingOrders: number;
    shippedToday: number;
    totalRevenue: number;
  }> {
    return this.readDb.aggregate("order_summaries", {
      totalOrders: { count: "*" },
      pendingOrders: { count: { where: { status: "created" } } },
      shippedToday: {
        count: {
          where: { status: "shipped", lastUpdated: { gte: today() } },
        },
      },
      totalRevenue: { sum: "netAmount", where: { status: { ne: "cancelled" } } },
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
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  全1000イベントを再生（遅い）

  With Snapshot (100イベントごと):
  [Snapshot @ Event 900] → Event 901 → ... → Event 1000
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                           100イベントのみ再生（速い）

  ┌──────────────────────────────────────────────────┐
  │  スナップショット戦略                             │
  │                                                  │
  │  保存タイミング:                                  │
  │  - N イベントごと (例: 100件ごと)                 │
  │  - 時間ベース (例: 1時間ごと)                     │
  │  - 手動 (負荷が高い集約のみ)                      │
  │  - イベント再生時間が閾値超過時                    │
  │                                                  │
  │  保存先:                                          │
  │  - 同じ Event Store の別テーブル                   │
  │  - Redis / Memcached (高速アクセス)               │
  │  - S3 (大量データ、低頻度アクセス)                │
  └──────────────────────────────────────────────────┘
```

### 4.2 スナップショット実装

```typescript
// ============================================================
// Snapshot Repository
// ============================================================
interface Snapshot {
  aggregateId: string;
  version: number;
  state: Record<string, unknown>;
  createdAt: Date;
}

interface SnapshotRepository {
  save(snapshot: Snapshot): Promise<void>;
  load(aggregateId: string): Promise<Snapshot | null>;
}

class PostgresSnapshotRepository implements SnapshotRepository {
  constructor(private pool: Pool) {}

  async save(snapshot: Snapshot): Promise<void> {
    await this.pool.query(
      `INSERT INTO snapshots (aggregate_id, aggregate_type, version, state, created_at)
       VALUES ($1, $2, $3, $4, $5)
       ON CONFLICT (aggregate_id) DO UPDATE
       SET version = $3, state = $4, created_at = $5`,
      ["Order", snapshot.aggregateId, snapshot.version,
       JSON.stringify(snapshot.state), snapshot.createdAt]
    );
  }

  async load(aggregateId: string): Promise<Snapshot | null> {
    const { rows } = await this.pool.query(
      "SELECT * FROM snapshots WHERE aggregate_id = $1",
      [aggregateId]
    );
    if (rows.length === 0) return null;
    return {
      aggregateId: rows[0].aggregate_id,
      version: rows[0].version,
      state: rows[0].state,
      createdAt: rows[0].created_at,
    };
  }
}

// ============================================================
// スナップショット対応の Order Repository
// ============================================================
class EventSourcedOrderRepository {
  constructor(
    private eventStore: EventStore,
    private snapshotRepo: SnapshotRepository,
    private snapshotInterval: number = 100,
  ) {}

  async load(orderId: string): Promise<Order> {
    const snapshot = await this.snapshotRepo.load(orderId);

    if (snapshot) {
      const newEvents = await this.eventStore.getEventsAfterVersion(
        orderId,
        snapshot.version
      );
      return Order.fromSnapshot(
        snapshot.state as OrderSnapshot,
        newEvents as OrderEvent[]
      );
    }

    const events = await this.eventStore.getEvents(orderId);
    if (events.length === 0) throw new Error(`Order ${orderId} not found`);
    return Order.fromEvents(events as OrderEvent[]);
  }

  async save(order: Order): Promise<void> {
    const events = order.getUncommittedEvents();
    const baseVersion = order.version - events.length;

    await this.eventStore.append(order.id, [...events], baseVersion);
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

## 5. イベントスキーマのバージョニング

### WHY: なぜバージョニングが必要か

イベントは不変（immutable）であるため、スキーマ変更時に既存イベントを書き換えることはできない。代わりに **Upcaster** パターンで古いバージョンのイベントを読み取り時に変換する。

```typescript
// ============================================================
// イベントバージョニングと Upcaster
// ============================================================

// v1: 初期バージョン
interface OrderCreatedV1 {
  eventType: "OrderCreated";
  schemaVersion: 1;
  data: {
    customerId: string;
    items: Array<{ productId: string; quantity: number; price: number }>;
  };
}

// v2: 通貨フィールドを追加
interface OrderCreatedV2 {
  eventType: "OrderCreated";
  schemaVersion: 2;
  data: {
    customerId: string;
    items: Array<{
      productId: string;
      quantity: number;
      price: number;
      currency: string;  // 新規追加
    }>;
    currency: string;  // 新規追加
  };
}

// v3: 配送先を追加
interface OrderCreatedV3 {
  eventType: "OrderCreated";
  schemaVersion: 3;
  data: {
    customerId: string;
    items: Array<{
      productId: string;
      quantity: number;
      price: number;
      currency: string;
    }>;
    currency: string;
    shippingAddress: {  // 新規追加
      street: string;
      city: string;
      postalCode: string;
      country: string;
    };
  };
}

// Upcaster Chain
class EventUpcaster {
  private upcasters: Map<string, Map<number, (event: any) => any>> = new Map();

  register(eventType: string, fromVersion: number, transform: (event: any) => any): void {
    if (!this.upcasters.has(eventType)) {
      this.upcasters.set(eventType, new Map());
    }
    this.upcasters.get(eventType)!.set(fromVersion, transform);
  }

  upcast(event: DomainEvent): DomainEvent {
    const version = (event as any).schemaVersion ?? 1;
    const eventUpcasters = this.upcasters.get(event.eventType);
    if (!eventUpcasters) return event;

    let current = event;
    let currentVersion = version;

    while (eventUpcasters.has(currentVersion)) {
      current = eventUpcasters.get(currentVersion)!(current);
      currentVersion++;
    }

    return current;
  }
}

// Upcaster の登録
const upcaster = new EventUpcaster();

// v1 → v2: currency フィールドを追加
upcaster.register("OrderCreated", 1, (event: OrderCreatedV1): OrderCreatedV2 => ({
  ...event,
  schemaVersion: 2,
  data: {
    ...event.data,
    currency: "JPY",  // デフォルト値
    items: event.data.items.map((i) => ({
      ...i,
      currency: "JPY",
    })),
  },
}));

// v2 → v3: shippingAddress を追加
upcaster.register("OrderCreated", 2, (event: OrderCreatedV2): OrderCreatedV3 => ({
  ...event,
  schemaVersion: 3,
  data: {
    ...event.data,
    shippingAddress: {
      street: "不明",
      city: "不明",
      postalCode: "000-0000",
      country: "JP",
    },
  },
}));
```

---

## 6. 比較表

### 6.1 CRUD vs Event Sourcing

| 観点 | 従来の CRUD | Event Sourcing |
|------|-----------|----------------|
| **データモデル** | 現在の状態のみ | イベントの履歴（不変） |
| **変更履歴** | 手動で監査テーブル作成 | 自動的に全履歴保持 |
| **任意時点の復元** | 不可能（別途実装必要） | イベント再生で可能 |
| **データ容量** | 少ない（現在の状態のみ） | 大きい（全イベント保存） |
| **読み取り性能** | 高い（直接クエリ） | 低い（再生必要 → プロジェクションで解決） |
| **書き込み性能** | 中（UPDATE + ロック） | 高い（追記のみ、ロック少ない） |
| **スキーマ変更** | マイグレーションで対応 | Upcaster で旧バージョンを変換 |
| **デバッグ** | 現在の状態のみ確認可能 | イベントログで原因追跡容易 |
| **複雑性** | 低い | 高い |
| **テスト** | DB モック必要 | イベントの入出力で検証可能 |

### 6.2 CQRS 導入パターン比較

| パターン | 説明 | 整合性 | 複雑度 | スケーラビリティ | ユースケース |
|---------|------|--------|--------|----------------|------------|
| **Level 0: 分離なし** | 同一モデル、同一 DB | 即時整合 | 低 | 低 | 小規模 CRUD |
| **Level 1: モデル分離** | 異なるモデル、同一 DB | 即時整合 | 低〜中 | 中 | 読み書きの最適化 |
| **Level 2: DB 分離 + 同期** | Write DB → Read DB を同期更新 | ほぼ即時 | 中 | 中〜高 | 整合性重視 |
| **Level 3: ES + 非同期** | Event Store + 非同期プロジェクション | 結果整合 | 高 | 最高 | 高スケーラビリティ |

### 6.3 Event Store 実装の選択肢

| 実装 | 特徴 | 適する規模 | 運用コスト |
|------|------|-----------|----------|
| **PostgreSQL + JSONB** | 汎用的、既存インフラ活用 | 小〜中（〜数千万イベント） | 低 |
| **EventStoreDB** | ES 専用、プロジェクションエンジン内蔵 | 中〜大 | 中 |
| **Apache Kafka** | ストリーム処理、高スループット | 大（数十億イベント） | 高 |
| **DynamoDB Streams** | AWS ネイティブ、サーバーレス | 中〜大 | 中（従量課金） |
| **MongoDB + Change Streams** | ドキュメント指向、柔軟なスキーマ | 中 | 中 |

---

## 7. アンチパターン

### 7.1 全てのドメインに Event Sourcing を適用

```
NG: マスタデータ（商品カテゴリ、設定値）にも Event Sourcing
    → 過度な複雑性、CRUD で十分な領域

OK: Event Sourcing の適用基準
    ┌───────────────────────────────────────┐
    │ Event Sourcing が適する:              │
    │ ✓ 注文処理、金融取引                  │
    │ ✓ 監査証跡が法的に必要               │
    │ ✓ ビジネスイベントの分析が重要       │
    │ ✓ 複雑な状態遷移がある               │
    │ ✓ 取り消し/補償トランザクションが必要 │
    │                                       │
    │ CRUD で十分:                          │
    │ ✗ ユーザープロフィール               │
    │ ✗ マスタデータ管理                   │
    │ ✗ 設定値管理                         │
    │ ✗ CRUD が中心の管理画面              │
    │ ✗ 単純な参照データ                   │
    └───────────────────────────────────────┘
```

**なぜ NG か**: Event Sourcing はイベント設計、Projector 実装、結果整合性の管理など大きなコストがかかる。ビジネス上の価値がないドメインに適用すると、複雑さだけが増す。

### 7.2 イベントスキーマを頻繁に変更する

```typescript
// NG: 既存イベントの構造を直接変更
// v1 のデータ: { amount: 1000 }
// v2 のデータ: { amount: 1000, currency: "JPY" }
// → 既存の v1 イベントを読み込むとクラッシュ

// OK: Upcaster でバージョン移行
function upcastOrderCreatedV1toV2(event: OrderCreatedV1): OrderCreatedV2 {
  return {
    ...event,
    schemaVersion: 2,
    data: {
      ...event.data,
      currency: "JPY",  // デフォルト値を付与
    },
  };
}
```

**なぜ NG か**: イベントは不変であり、既に保存されたデータの構造を変えることはできない。Upcaster を使って読み取り時に変換するのが正しいアプローチ。

### 7.3 イベントにドメインの意図を含めない

```typescript
// NG: CRUD 風の汎用イベント
interface OrderUpdated {
  eventType: "OrderUpdated";
  data: {
    fields: Record<string, unknown>;  // 何が変わったか分からない
  };
}

// OK: ドメインの意図を表現するイベント
interface DiscountApplied {
  eventType: "DiscountApplied";
  data: {
    discountType: "percentage" | "fixed";
    value: number;
    reason: "loyalty_program" | "coupon" | "manual";
    couponCode?: string;
  };
}
// 「なぜ金額が変わったか」が明確
```

**なぜ NG か**: Event Sourcing の価値は「ドメインの歴史を記録すること」。汎用的な "Updated" イベントでは、CRUD と同じく「何が起こったか」しか分からず、「なぜ起こったか」が失われる。

### 7.4 プロジェクションで副作用を実行する

```typescript
// NG: Projector 内で外部 API を呼ぶ
class OrderProjector {
  async handle(event: OrderShipped) {
    // Read Model の更新
    await this.readDb.update(/* ... */);

    // 副作用（NG！）
    await this.emailService.sendShipmentNotification(event);
    await this.smsService.sendTrackingLink(event);
    // → プロジェクションを再構築（Rebuild）するたびに
    //   メールとSMSが再送信されてしまう！
  }
}

// OK: 副作用は別のイベントハンドラで
class OrderProjector {
  async handle(event: OrderShipped) {
    // Read Model の更新のみ
    await this.readDb.update(/* ... */);
  }
}

// 通知は専用のハンドラで（べき等性を保証）
class ShipmentNotificationHandler {
  async handle(event: OrderShipped) {
    // べき等性チェック
    const alreadySent = await this.notificationLog.exists(event.eventId);
    if (alreadySent) return;

    await this.emailService.sendShipmentNotification(event);
    await this.notificationLog.record(event.eventId);
  }
}
```

**なぜ NG か**: プロジェクションは再構築（Rebuild）される可能性がある。副作用がプロジェクション内にあると、Rebuild のたびに副作用が再実行される。通知やメール送信は別のイベントハンドラに分離し、べき等性を保証する。

---

## 8. 実践演習

### 演習 1（基礎）: イベント定義と集約

銀行口座（BankAccount）の集約を Event Sourcing で設計せよ。

**イベント**:
- AccountOpened（口座開設）
- MoneyDeposited（入金）
- MoneyWithdrawn（出金）
- AccountFrozen（口座凍結）
- AccountClosed（口座解約）

**ビジネスルール**:
- 残高がマイナスになる出金は不可
- 凍結中の口座は入出金不可
- 閉鎖済み口座は操作不可

**期待する出力**: `BankAccount` 集約クラス（TypeScript）とテストコード

---

### 演習 2（応用）: プロジェクションの設計

演習 1 の銀行口座イベントから、以下の Read Model を構築する Projector を実装せよ。

1. **AccountBalanceView** — 口座残高一覧（口座ID、名前、残高、状態）
2. **TransactionHistoryView** — 取引履歴（日時、種別、金額、残高）
3. **DailyReportView** — 日次レポート（日ごとの入金合計、出金合計、純増減）

**期待する出力**: 3つの Projector クラスと Read Model 定義

---

### 演習 3（発展）: 段階的 CRUD → ES 移行

既存の CRUD ベースの注文システムを、Strangler Fig パターンで段階的に Event Sourcing に移行する計画を立てよ。

**現行システム**: orders テーブル（id, status, total, user_id, created_at, updated_at）

**移行計画**:
1. Phase 1: Dual Write（CRUD + イベント記録を並行）
2. Phase 2: Read Model 構築（イベントからプロジェクション作成）
3. Phase 3: 読み取りをプロジェクションに移行
4. Phase 4: 書き込みを Event Sourcing に移行
5. Phase 5: 旧テーブルの廃止

**期待する出力**: 各フェーズの具体的なコード例（TypeScript）とリスク分析

---

## 9. FAQ

### Q1. Event Sourcing のイベントが増え続けてストレージが心配

**A.** 3 つの対策がある:
1. **スナップショット** — N イベントごとに状態を保存し、古いイベントの再生を省略
2. **アーカイブ** — 一定期間経過したイベントを S3/Glacier に移動（法的要件に注意）
3. **パーティショニング** — 月別 / 年別でテーブルを分割し、検索性能を維持

実務では、数千万イベントでも PostgreSQL で問題なく運用できる。パーティショニングを適用すれば数億イベントも対応可能。1イベント平均 1KB として、1億イベント = 約 100GB であり、現代のストレージコストでは許容範囲。

### Q2. CQRS で結果整合性になると UI はどう対応する？

**A.** 3 つのパターンがある:

1. **楽観的 UI** — コマンド成功後に UI をローカルで即座に更新
2. **ポーリング** — コマンド後に短い間隔で Read Model を確認
3. **WebSocket / SSE** — プロジェクション更新完了をリアルタイムに通知

```typescript
// 楽観的 UI の例（React）
async function submitOrder(orderData: CreateOrderInput) {
  // 1. コマンド送信
  await api.post("/commands/create-order", orderData);

  // 2. UI をローカルで即座に更新（Read Model 更新を待たない）
  setOrders((prev) => [
    { ...orderData, status: "processing", id: tempId },
    ...prev,
  ]);

  // 3. バックグラウンドで Read Model の反映を確認
  const confirmed = await pollUntil(
    () => api.get(`/orders/${orderData.id}`),
    (res) => res.status !== 404,
    { maxAttempts: 10, interval: 500 }
  );

  if (confirmed) {
    // 楽観的更新を確定データで置換
    setOrders((prev) =>
      prev.map((o) => (o.id === tempId ? confirmed : o))
    );
  }
}
```

### Q3. Event Sourcing を既存の CRUD アプリに段階的に導入できる？

**A.** 可能。Strangler Fig パターンと組み合わせた段階的アプローチが推奨される:

1. **Phase 1: Dual Write** — CRUD と並行してイベントテーブルに記録
2. **Phase 2: Read Model 構築** — イベントからプロジェクションを作成し、読み取りクエリの一部を移行
3. **Phase 3: 読み取り移行完了** — 全読み取りをプロジェクションに切り替え
4. **Phase 4: 書き込み移行** — 集約単位で書き込みを Event Sourcing に移行
5. **Phase 5: 旧テーブル廃止** — CRUD テーブルを削除

各フェーズ間でシステムが正常動作することを確認し、問題があればロールバック可能な状態を維持する。全フェーズの完了には通常 3-6 ヶ月かかる。

### Q4. Event Sourcing のテストはどう書く？

**A.** 「Given-When-Then」パターンが最も自然:

```typescript
describe("Order Aggregate", () => {
  test("支払い済み注文に商品を追加するとエラー", () => {
    // Given: 支払い済みの注文
    const events: OrderEvent[] = [
      orderCreatedEvent({ items: [item1] }),
      paymentReceivedEvent({ amount: 5000 }),
    ];
    const order = Order.fromEvents(events);

    // When + Then: 商品追加するとエラー
    expect(() => {
      order.addItem(item2, "user-1");
    }).toThrow('状態 "paid" の注文に商品を追加できません');
  });

  test("注文作成で OrderCreated イベントが発行される", () => {
    // When: 注文を作成
    const order = Order.create("order-1", "customer-1", [item1], address, "user-1");

    // Then: 発行されたイベントを検証
    const events = order.getUncommittedEvents();
    expect(events).toHaveLength(1);
    expect(events[0].eventType).toBe("OrderCreated");
    expect(events[0].data.customerId).toBe("customer-1");
  });
});
```

### Q5. プロジェクションが壊れた場合の復旧方法は？

**A.** イベントは不変であるため、プロジェクションはいつでも再構築（Rebuild）可能:

1. Read Model テーブルを TRUNCATE
2. 全イベントを最初から再生してプロジェクションを再構築
3. Read Model のスキーマを変更した場合も、同じ手順で新しい形式のデータを作成可能

これが Event Sourcing の最大の利点の一つ。「データの変換ミス」があっても、Projector を修正して Rebuild すれば正しい状態に復旧できる。

---

## 10. まとめ

| 項目 | ポイント |
|------|---------|
| **Event Sourcing** | 状態の変化をイベントとして追記保存。完全な監査証跡、任意時点の復元が可能 |
| **イベントの3原則** | 不変（Immutable）、追記のみ（Append-Only）、導出値（Derived State） |
| **CQRS** | 読み取りと書き込みを分離。それぞれを独立にスケール・最適化 |
| **プロジェクション** | イベントから最適化された Read Model を構築。壊れても再構築可能 |
| **スナップショット** | イベント再生コストを削減する最適化手法。N イベントごとに状態保存 |
| **バージョニング** | Upcaster パターンでイベントスキーマの進化に対応。既存イベントは不変 |
| **適用基準** | 監査要件、複雑な状態遷移、高スケーラビリティが必要な場合に適用。CRUD で十分な領域には不要 |
| **テスト** | Given-When-Then パターン。イベントの入出力で検証可能。DB 不要 |

---

## 次に読むべきガイド

- [00-mvc-mvvm.md](./00-mvc-mvvm.md) — UI レイヤーのアーキテクチャパターン
- [01-repository-pattern.md](./01-repository-pattern.md) — データアクセスの抽象化
- [../../system-design-guide/](../../system-design-guide/) — メッセージキュー、分散システム設計
- [../../clean-code-principles/](../../clean-code-principles/) — DDD、SOLID 原則
- [../02-behavioral/](../02-behavioral/) — Observer パターン（イベント駆動の基盤）

---

## 参考文献

1. **Martin Fowler** — "Event Sourcing" — https://martinfowler.com/eaaDev/EventSourcing.html
2. **Martin Fowler** — "CQRS" — https://martinfowler.com/bliki/CQRS.html
3. **Greg Young** — "CQRS and Event Sourcing" — https://cqrs.files.wordpress.com/2010/11/cqrs_documents.pdf
4. **Microsoft** — "CQRS pattern" — https://learn.microsoft.com/en-us/azure/architecture/patterns/cqrs
5. **Vaughn Vernon** — "Implementing Domain-Driven Design" (2013) — Event Sourcing と DDD の統合
6. **EventStoreDB Documentation** — https://www.eventstore.com/docs/
7. **Adam Dymitruk** — "Event Modeling" — https://eventmodeling.org/
