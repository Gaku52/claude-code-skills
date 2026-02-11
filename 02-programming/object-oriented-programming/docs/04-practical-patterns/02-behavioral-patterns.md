# 振る舞いパターン（Behavioral Patterns）

> オブジェクト間の責任の分配とアルゴリズムのカプセル化に関するパターン。Strategy、Observer、Command、State、Iterator の5つを実践的に解説。

## この章で学ぶこと

- [ ] 各振る舞いパターンの目的と適用場面を理解する
- [ ] 各パターンの実装方法を把握する
- [ ] 現代のフレームワークでの応用を学ぶ

---

## 1. Strategy パターン

```
目的: アルゴリズムをカプセル化し、実行時に切り替え可能にする
現代の応用: React のレンダー戦略、ソートアルゴリズムの選択
```

```typescript
// 圧縮戦略
interface CompressionStrategy {
  compress(data: Buffer): Buffer;
  decompress(data: Buffer): Buffer;
  readonly name: string;
}

class GzipCompression implements CompressionStrategy {
  name = "gzip";
  compress(data: Buffer): Buffer { /* gzip圧縮 */ return data; }
  decompress(data: Buffer): Buffer { /* gzip展開 */ return data; }
}

class BrotliCompression implements CompressionStrategy {
  name = "brotli";
  compress(data: Buffer): Buffer { /* brotli圧縮 */ return data; }
  decompress(data: Buffer): Buffer { /* brotli展開 */ return data; }
}

class FileProcessor {
  constructor(private compression: CompressionStrategy) {}

  setCompression(strategy: CompressionStrategy): void {
    this.compression = strategy;
  }

  async processFile(path: string): Promise<Buffer> {
    const data = await fs.readFile(path);
    return this.compression.compress(data);
  }
}
```

---

## 2. Observer パターン

```
目的: オブジェクトの状態変化を他のオブジェクトに自動通知する
現代の応用: イベントシステム、React の状態管理、RxJS
```

```typescript
// 型安全な EventEmitter
type EventMap = {
  userCreated: { userId: string; email: string };
  userDeleted: { userId: string };
  orderPlaced: { orderId: string; total: number };
};

class TypedEventEmitter<T extends Record<string, any>> {
  private listeners = new Map<keyof T, Set<Function>>();

  on<K extends keyof T>(event: K, listener: (data: T[K]) => void): () => void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(listener);

    // unsubscribe 関数を返す
    return () => this.listeners.get(event)?.delete(listener);
  }

  emit<K extends keyof T>(event: K, data: T[K]): void {
    this.listeners.get(event)?.forEach(listener => listener(data));
  }
}

// 使用例
const events = new TypedEventEmitter<EventMap>();

// Observer（購読者）を登録
const unsubscribe = events.on("userCreated", (data) => {
  console.log(`Welcome email to ${data.email}`); // 型安全
});

events.on("orderPlaced", (data) => {
  console.log(`Order ${data.orderId}: ¥${data.total}`);
});

// Subject（発行者）がイベントを発行
events.emit("userCreated", { userId: "1", email: "tanaka@example.com" });
events.emit("orderPlaced", { orderId: "O-001", total: 5000 });

unsubscribe(); // 購読解除
```

---

## 3. Command パターン

```
目的: リクエストをオブジェクトとしてカプセル化する
利点: 取り消し（Undo）、キュー、ログ、トランザクション
現代の応用: Redux のアクション、エディタのUndo/Redo
```

```typescript
// Command インターフェース
interface Command {
  execute(): void;
  undo(): void;
  describe(): string;
}

// テキストエディタのコマンド
class InsertTextCommand implements Command {
  constructor(
    private editor: TextEditor,
    private position: number,
    private text: string,
  ) {}

  execute(): void {
    this.editor.insertAt(this.position, this.text);
  }

  undo(): void {
    this.editor.deleteRange(this.position, this.position + this.text.length);
  }

  describe(): string {
    return `Insert "${this.text}" at ${this.position}`;
  }
}

class DeleteTextCommand implements Command {
  private deletedText: string = "";

  constructor(
    private editor: TextEditor,
    private start: number,
    private end: number,
  ) {}

  execute(): void {
    this.deletedText = this.editor.getRange(this.start, this.end);
    this.editor.deleteRange(this.start, this.end);
  }

  undo(): void {
    this.editor.insertAt(this.start, this.deletedText);
  }

  describe(): string {
    return `Delete [${this.start}:${this.end}]`;
  }
}

// コマンド履歴管理（Undo/Redo）
class CommandHistory {
  private undoStack: Command[] = [];
  private redoStack: Command[] = [];

  execute(command: Command): void {
    command.execute();
    this.undoStack.push(command);
    this.redoStack = []; // Redo履歴をクリア
  }

  undo(): void {
    const command = this.undoStack.pop();
    if (command) {
      command.undo();
      this.redoStack.push(command);
    }
  }

  redo(): void {
    const command = this.redoStack.pop();
    if (command) {
      command.execute();
      this.undoStack.push(command);
    }
  }
}
```

---

## 4. State パターン

```
目的: オブジェクトの状態に応じて振る舞いを変える
現代の応用: ステートマシン、UIコンポーネントの状態管理
```

```typescript
// 状態インターフェース
interface OrderState {
  confirm(order: Order): void;
  ship(order: Order): void;
  deliver(order: Order): void;
  cancel(order: Order): void;
  toString(): string;
}

class PendingState implements OrderState {
  confirm(order: Order) { order.setState(new ConfirmedState()); }
  ship() { throw new Error("未確認の注文は発送できません"); }
  deliver() { throw new Error("未確認の注文は配達できません"); }
  cancel(order: Order) { order.setState(new CancelledState()); }
  toString() { return "保留中"; }
}

class ConfirmedState implements OrderState {
  confirm() { throw new Error("既に確認済みです"); }
  ship(order: Order) { order.setState(new ShippedState()); }
  deliver() { throw new Error("未発送の注文は配達できません"); }
  cancel(order: Order) { order.setState(new CancelledState()); }
  toString() { return "確認済み"; }
}

class ShippedState implements OrderState {
  confirm() { throw new Error("発送済みです"); }
  ship() { throw new Error("既に発送済みです"); }
  deliver(order: Order) { order.setState(new DeliveredState()); }
  cancel() { throw new Error("発送済みの注文はキャンセルできません"); }
  toString() { return "発送済み"; }
}

class DeliveredState implements OrderState {
  confirm() { throw new Error("配達済みです"); }
  ship() { throw new Error("配達済みです"); }
  deliver() { throw new Error("既に配達済みです"); }
  cancel() { throw new Error("配達済みの注文はキャンセルできません"); }
  toString() { return "配達済み"; }
}

class CancelledState implements OrderState {
  confirm() { throw new Error("キャンセル済みです"); }
  ship() { throw new Error("キャンセル済みです"); }
  deliver() { throw new Error("キャンセル済みです"); }
  cancel() { throw new Error("既にキャンセル済みです"); }
  toString() { return "キャンセル済み"; }
}

// Context
class Order {
  private state: OrderState = new PendingState();

  setState(state: OrderState): void { this.state = state; }
  confirm(): void { this.state.confirm(this); }
  ship(): void { this.state.ship(this); }
  deliver(): void { this.state.deliver(this); }
  cancel(): void { this.state.cancel(this); }
  getStatus(): string { return this.state.toString(); }
}

const order = new Order();
order.confirm();  // 保留中 → 確認済み
order.ship();     // 確認済み → 発送済み
order.deliver();  // 発送済み → 配達済み
// order.cancel(); // Error: 配達済みの注文はキャンセルできません
```

---

## 5. Iterator パターン

```
目的: コレクションの内部構造を公開せずに要素を順番にアクセスする
現代の応用: for...of, Python の __iter__, Rust の Iterator トレイト
```

```typescript
// TypeScript: カスタムイテレータ
class Range implements Iterable<number> {
  constructor(
    private start: number,
    private end: number,
    private step: number = 1,
  ) {}

  [Symbol.iterator](): Iterator<number> {
    let current = this.start;
    const end = this.end;
    const step = this.step;

    return {
      next(): IteratorResult<number> {
        if (current < end) {
          const value = current;
          current += step;
          return { value, done: false };
        }
        return { value: undefined, done: true };
      },
    };
  }
}

// for...of で使える
for (const n of new Range(0, 10, 2)) {
  console.log(n); // 0, 2, 4, 6, 8
}

// スプレッド演算子も使える
const numbers = [...new Range(1, 6)]; // [1, 2, 3, 4, 5]
```

---

## まとめ

| パターン | 目的 | 現代の応用 |
|---------|------|-----------|
| Strategy | アルゴリズムの切り替え | DI、ポリシー |
| Observer | 状態変化の通知 | イベント、Reactive |
| Command | 操作のオブジェクト化 | Undo/Redo、Redux |
| State | 状態による振る舞い変更 | ステートマシン |
| Iterator | 順次アクセス | for...of、ジェネレータ |

---

## 次に読むべきガイド
→ [[03-anti-patterns.md]] — アンチパターン

---

## 参考文献
1. Gamma, E. et al. "Design Patterns." Addison-Wesley, 1994.
