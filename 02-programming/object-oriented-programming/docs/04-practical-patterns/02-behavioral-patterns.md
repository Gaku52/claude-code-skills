# 振る舞いパターン（Behavioral Patterns）

> オブジェクト間の責任の分配とアルゴリズムのカプセル化に関するパターン。Strategy、Observer、Command、State、Iterator、Chain of Responsibility、Template Method、Mediator、Visitor の9つを実践的に解説。

## この章で学ぶこと

- [ ] 各振る舞いパターンの目的と適用場面を理解する
- [ ] 各パターンの実装方法を複数言語で把握する
- [ ] 現代のフレームワークでの応用を学ぶ
- [ ] パターン間の違いと組み合わせを理解する
- [ ] テスタビリティを考慮した振る舞い設計ができるようになる

---

## 1. Strategy パターン

### 1.1 概要と目的

```
目的: アルゴリズムをカプセル化し、実行時に切り替え可能にする

構造:
  ┌─────────┐      ┌────────────────┐
  │ Context │─────→│   Strategy     │
  │         │      │ (インターフェース) │
  └─────────┘      └───────┬────────┘
                           │
              ┌────────────┼────────────┐
         ┌────┴────┐  ┌───┴───┐  ┌────┴────┐
         │StratA   │  │StratB │  │StratC   │
         └─────────┘  └───────┘  └─────────┘

現代の応用: React のレンダー戦略、ソートアルゴリズムの選択、DI

いつ使うか:
  → 同じ処理に複数のアルゴリズムがある
  → 実行時にアルゴリズムを切り替えたい
  → if-else/switch の分岐が増えてきた
  → アルゴリズムの詳細をクライアントから隠したい
```

### 1.2 圧縮戦略

```typescript
// 圧縮戦略
interface CompressionStrategy {
  compress(data: Buffer): Promise<Buffer>;
  decompress(data: Buffer): Promise<Buffer>;
  readonly name: string;
  readonly extension: string;
}

class GzipCompression implements CompressionStrategy {
  name = "gzip";
  extension = ".gz";

  async compress(data: Buffer): Promise<Buffer> {
    const { promisify } = require("util");
    const { gzip } = require("zlib");
    return promisify(gzip)(data);
  }

  async decompress(data: Buffer): Promise<Buffer> {
    const { promisify } = require("util");
    const { gunzip } = require("zlib");
    return promisify(gunzip)(data);
  }
}

class BrotliCompression implements CompressionStrategy {
  name = "brotli";
  extension = ".br";

  async compress(data: Buffer): Promise<Buffer> {
    const { promisify } = require("util");
    const { brotliCompress } = require("zlib");
    return promisify(brotliCompress)(data);
  }

  async decompress(data: Buffer): Promise<Buffer> {
    const { promisify } = require("util");
    const { brotliDecompress } = require("zlib");
    return promisify(brotliDecompress)(data);
  }
}

class NoCompression implements CompressionStrategy {
  name = "none";
  extension = "";

  async compress(data: Buffer): Promise<Buffer> { return data; }
  async decompress(data: Buffer): Promise<Buffer> { return data; }
}

// Context: 圧縮戦略を使うファイルプロセッサ
class FileProcessor {
  constructor(private compression: CompressionStrategy) {}

  setCompression(strategy: CompressionStrategy): void {
    this.compression = strategy;
  }

  async processFile(inputPath: string, outputPath: string): Promise<{
    originalSize: number;
    compressedSize: number;
    ratio: number;
  }> {
    const data = await fs.readFile(inputPath);
    const compressed = await this.compression.compress(data);
    const finalPath = outputPath + this.compression.extension;
    await fs.writeFile(finalPath, compressed);

    return {
      originalSize: data.length,
      compressedSize: compressed.length,
      ratio: compressed.length / data.length,
    };
  }

  async restoreFile(inputPath: string, outputPath: string): Promise<void> {
    const data = await fs.readFile(inputPath);
    const decompressed = await this.compression.decompress(data);
    await fs.writeFile(outputPath, decompressed);
  }
}

// 使用例: ファイルサイズに応じて戦略を自動選択
function selectCompression(fileSize: number): CompressionStrategy {
  if (fileSize < 1024) return new NoCompression();           // 1KB未満: 圧縮不要
  if (fileSize < 1024 * 1024) return new GzipCompression();  // 1MB未満: gzip
  return new BrotliCompression();                             // 1MB以上: brotli
}
```

### 1.3 バリデーション戦略

```typescript
// バリデーション戦略パターン
interface ValidationStrategy<T> {
  validate(data: T): ValidationResult;
  readonly name: string;
}

interface ValidationResult {
  valid: boolean;
  errors: Array<{ field: string; message: string }>;
}

class RequiredFieldsValidator implements ValidationStrategy<Record<string, any>> {
  name = "required-fields";

  constructor(private requiredFields: string[]) {}

  validate(data: Record<string, any>): ValidationResult {
    const errors = this.requiredFields
      .filter(field => !data[field] || (typeof data[field] === "string" && data[field].trim() === ""))
      .map(field => ({ field, message: `${field}は必須です` }));

    return { valid: errors.length === 0, errors };
  }
}

class EmailFormatValidator implements ValidationStrategy<Record<string, any>> {
  name = "email-format";
  private emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

  constructor(private emailFields: string[]) {}

  validate(data: Record<string, any>): ValidationResult {
    const errors = this.emailFields
      .filter(field => data[field] && !this.emailRegex.test(data[field]))
      .map(field => ({ field, message: `${data[field]}は有効なメールアドレスではありません` }));

    return { valid: errors.length === 0, errors };
  }
}

class RangeValidator implements ValidationStrategy<Record<string, any>> {
  name = "range";

  constructor(
    private rules: Array<{ field: string; min?: number; max?: number }>,
  ) {}

  validate(data: Record<string, any>): ValidationResult {
    const errors: Array<{ field: string; message: string }> = [];

    for (const rule of this.rules) {
      const value = data[rule.field];
      if (value === undefined || value === null) continue;

      if (rule.min !== undefined && value < rule.min) {
        errors.push({ field: rule.field, message: `${rule.field}は${rule.min}以上である必要があります` });
      }
      if (rule.max !== undefined && value > rule.max) {
        errors.push({ field: rule.field, message: `${rule.field}は${rule.max}以下である必要があります` });
      }
    }

    return { valid: errors.length === 0, errors };
  }
}

// 複数の戦略を組み合わせる Composite Strategy
class CompositeValidator implements ValidationStrategy<Record<string, any>> {
  name = "composite";
  private strategies: ValidationStrategy<Record<string, any>>[] = [];

  add(strategy: ValidationStrategy<Record<string, any>>): this {
    this.strategies.push(strategy);
    return this;
  }

  validate(data: Record<string, any>): ValidationResult {
    const allErrors: Array<{ field: string; message: string }> = [];

    for (const strategy of this.strategies) {
      const result = strategy.validate(data);
      allErrors.push(...result.errors);
    }

    return { valid: allErrors.length === 0, errors: allErrors };
  }
}

// 使用例
const userValidator = new CompositeValidator()
  .add(new RequiredFieldsValidator(["name", "email", "age"]))
  .add(new EmailFormatValidator(["email"]))
  .add(new RangeValidator([{ field: "age", min: 0, max: 150 }]));

const result = userValidator.validate({
  name: "太郎",
  email: "invalid-email",
  age: 200,
});
// { valid: false, errors: [
//   { field: "email", message: "invalid-emailは有効なメールアドレスではありません" },
//   { field: "age", message: "ageは150以下である必要があります" }
// ]}
```

### 1.4 Python での Strategy パターン

```python
# Python: Strategy パターン（関数ベースとクラスベース）
from typing import Protocol, Callable
from abc import abstractmethod

# Protocol ベースの Strategy
class SortStrategy(Protocol):
    def sort(self, data: list) -> list: ...

class QuickSort:
    def sort(self, data: list) -> list:
        if len(data) <= 1:
            return data
        pivot = data[len(data) // 2]
        left = [x for x in data if x < pivot]
        middle = [x for x in data if x == pivot]
        right = [x for x in data if x > pivot]
        return self.sort(left) + middle + self.sort(right)

class MergeSort:
    def sort(self, data: list) -> list:
        if len(data) <= 1:
            return data
        mid = len(data) // 2
        left = self.sort(data[:mid])
        right = self.sort(data[mid:])
        return self._merge(left, right)

    def _merge(self, left: list, right: list) -> list:
        result = []
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        result.extend(left[i:])
        result.extend(right[j:])
        return result

# Context
class DataProcessor:
    def __init__(self, sort_strategy: SortStrategy):
        self._sort_strategy = sort_strategy

    def set_strategy(self, strategy: SortStrategy) -> None:
        self._sort_strategy = strategy

    def process(self, data: list) -> list:
        return self._sort_strategy.sort(data)

# 関数ベースの Strategy（Python らしいアプローチ）
def process_data(data: list, sort_fn: Callable[[list], list] = sorted) -> list:
    return sort_fn(data)

# 使用例
processor = DataProcessor(QuickSort())
result = processor.process([3, 1, 4, 1, 5, 9, 2, 6])
```

---

## 2. Observer パターン

### 2.1 概要と目的

```
目的: オブジェクトの状態変化を他のオブジェクトに自動通知する

構造:
  ┌──────────┐      ┌──────────────┐
  │ Subject  │─────→│   Observer   │ ×N
  │ (発行者) │      │  (購読者)    │
  └──────────┘      └──────────────┘

現代の応用: イベントシステム、React の状態管理、RxJS、Pub/Sub

いつ使うか:
  → オブジェクトの状態変化を複数のオブジェクトに通知したい
  → 疎結合な通信メカニズムが必要
  → 購読者の数や種類が動的に変わる
```

### 2.2 型安全な EventEmitter

```typescript
// 型安全な EventEmitter
type EventMap = {
  userCreated: { userId: string; email: string };
  userDeleted: { userId: string };
  userUpdated: { userId: string; changes: Record<string, any> };
  orderPlaced: { orderId: string; total: number };
  orderCancelled: { orderId: string; reason: string };
};

class TypedEventEmitter<T extends Record<string, any>> {
  private listeners = new Map<keyof T, Set<Function>>();
  private onceListeners = new Map<keyof T, Set<Function>>();

  on<K extends keyof T>(event: K, listener: (data: T[K]) => void): () => void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(listener);

    // unsubscribe 関数を返す
    return () => this.listeners.get(event)?.delete(listener);
  }

  once<K extends keyof T>(event: K, listener: (data: T[K]) => void): () => void {
    if (!this.onceListeners.has(event)) {
      this.onceListeners.set(event, new Set());
    }
    this.onceListeners.get(event)!.add(listener);
    return () => this.onceListeners.get(event)?.delete(listener);
  }

  emit<K extends keyof T>(event: K, data: T[K]): void {
    // 通常のリスナー
    this.listeners.get(event)?.forEach(listener => listener(data));

    // 一度だけのリスナー
    this.onceListeners.get(event)?.forEach(listener => listener(data));
    this.onceListeners.delete(event);
  }

  off<K extends keyof T>(event: K, listener?: Function): void {
    if (listener) {
      this.listeners.get(event)?.delete(listener);
    } else {
      this.listeners.delete(event);
    }
  }

  listenerCount<K extends keyof T>(event: K): number {
    return (this.listeners.get(event)?.size ?? 0) +
           (this.onceListeners.get(event)?.size ?? 0);
  }

  removeAllListeners(): void {
    this.listeners.clear();
    this.onceListeners.clear();
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

// 1回だけ購読
events.once("userCreated", (data) => {
  console.log(`First user bonus for ${data.userId}`);
});

// Subject（発行者）がイベントを発行
events.emit("userCreated", { userId: "1", email: "tanaka@example.com" });
events.emit("orderPlaced", { orderId: "O-001", total: 5000 });

unsubscribe(); // 購読解除
```

### 2.3 Reactive Store（状態管理）

```typescript
// Reactive Store: React/Vue風の状態管理
interface StoreOptions<T> {
  initialState: T;
  middleware?: Array<(prev: T, next: T, action: string) => T>;
}

class ReactiveStore<T extends Record<string, any>> {
  private state: T;
  private listeners = new Set<(state: T) => void>();
  private selectorListeners = new Map<string, Set<(value: any) => void>>();
  private middleware: Array<(prev: T, next: T, action: string) => T>;

  constructor(options: StoreOptions<T>) {
    this.state = { ...options.initialState };
    this.middleware = options.middleware ?? [];
  }

  getState(): Readonly<T> {
    return this.state;
  }

  // 状態全体を購読
  subscribe(listener: (state: T) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  // 特定のプロパティを購読（セレクタ）
  select<K extends keyof T>(key: K, listener: (value: T[K]) => void): () => void {
    const keyStr = String(key);
    if (!this.selectorListeners.has(keyStr)) {
      this.selectorListeners.set(keyStr, new Set());
    }
    this.selectorListeners.get(keyStr)!.add(listener);
    return () => this.selectorListeners.get(keyStr)?.delete(listener);
  }

  // 状態を更新
  setState(updater: Partial<T> | ((prev: T) => Partial<T>), action = "setState"): void {
    const updates = typeof updater === "function" ? updater(this.state) : updater;
    const prev = { ...this.state };
    let next = { ...this.state, ...updates };

    // ミドルウェアを適用
    for (const mw of this.middleware) {
      next = mw(prev, next, action);
    }

    this.state = next;

    // 全体リスナーに通知
    this.listeners.forEach(listener => listener(this.state));

    // 変更されたプロパティのセレクタリスナーに通知
    for (const key of Object.keys(updates)) {
      if (prev[key] !== this.state[key as keyof T]) {
        this.selectorListeners.get(key)?.forEach(listener =>
          listener(this.state[key as keyof T])
        );
      }
    }
  }
}

// 使用例: ログインミドルウェア
const loggingMiddleware = <T>(prev: T, next: T, action: string): T => {
  console.log(`[${action}]`, { prev, next });
  return next;
};

interface AppState {
  user: { name: string; email: string } | null;
  theme: "light" | "dark";
  notifications: number;
  isLoading: boolean;
}

const store = new ReactiveStore<AppState>({
  initialState: {
    user: null,
    theme: "light",
    notifications: 0,
    isLoading: false,
  },
  middleware: [loggingMiddleware],
});

// テーマの変更だけを監視
store.select("theme", (theme) => {
  document.body.className = theme;
});

// 通知数の変更を監視
store.select("notifications", (count) => {
  console.log(`未読通知: ${count}`);
});

// 状態を更新
store.setState({ user: { name: "太郎", email: "taro@example.com" } }, "LOGIN");
store.setState(prev => ({ notifications: prev.notifications + 1 }), "NEW_NOTIFICATION");
store.setState({ theme: "dark" }, "TOGGLE_THEME");
```

---

## 3. Command パターン

### 3.1 概要と目的

```
目的: リクエストをオブジェクトとしてカプセル化する

利点: 取り消し（Undo）、キュー、ログ、トランザクション
現代の応用: Redux のアクション、エディタのUndo/Redo、CQRSパターン

構造:
  ┌──────────┐    ┌─────────┐    ┌──────────┐
  │ Invoker  │───→│ Command │───→│ Receiver │
  │ (実行者) │    │ (命令)  │    │ (受信者) │
  └──────────┘    └─────────┘    └──────────┘
```

### 3.2 テキストエディタのUndo/Redo

```typescript
// Command インターフェース
interface Command {
  execute(): void;
  undo(): void;
  describe(): string;
}

// テキストエディタ（Receiver）
class TextEditor {
  private content = "";
  private selectionStart = 0;
  private selectionEnd = 0;

  getContent(): string { return this.content; }

  insertAt(position: number, text: string): void {
    this.content = this.content.slice(0, position) + text + this.content.slice(position);
  }

  deleteRange(start: number, end: number): string {
    const deleted = this.content.slice(start, end);
    this.content = this.content.slice(0, start) + this.content.slice(end);
    return deleted;
  }

  getRange(start: number, end: number): string {
    return this.content.slice(start, end);
  }

  replaceRange(start: number, end: number, text: string): string {
    const replaced = this.content.slice(start, end);
    this.content = this.content.slice(0, start) + text + this.content.slice(end);
    return replaced;
  }

  setSelection(start: number, end: number): void {
    this.selectionStart = start;
    this.selectionEnd = end;
  }

  getSelection(): { start: number; end: number } {
    return { start: this.selectionStart, end: this.selectionEnd };
  }
}

// 具体的なコマンド群
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
    return `Insert "${this.text.slice(0, 20)}${this.text.length > 20 ? "..." : ""}" at ${this.position}`;
  }
}

class DeleteTextCommand implements Command {
  private deletedText = "";

  constructor(
    private editor: TextEditor,
    private start: number,
    private end: number,
  ) {}

  execute(): void {
    this.deletedText = this.editor.deleteRange(this.start, this.end);
  }

  undo(): void {
    this.editor.insertAt(this.start, this.deletedText);
  }

  describe(): string {
    return `Delete [${this.start}:${this.end}] "${this.deletedText.slice(0, 20)}"`;
  }
}

class ReplaceTextCommand implements Command {
  private replacedText = "";

  constructor(
    private editor: TextEditor,
    private start: number,
    private end: number,
    private newText: string,
  ) {}

  execute(): void {
    this.replacedText = this.editor.replaceRange(this.start, this.end, this.newText);
  }

  undo(): void {
    this.editor.replaceRange(this.start, this.start + this.newText.length, this.replacedText);
  }

  describe(): string {
    return `Replace [${this.start}:${this.end}] with "${this.newText.slice(0, 20)}"`;
  }
}

// マクロコマンド: 複数のコマンドを1つにまとめる
class MacroCommand implements Command {
  private commands: Command[];

  constructor(commands: Command[]) {
    this.commands = [...commands];
  }

  execute(): void {
    for (const cmd of this.commands) {
      cmd.execute();
    }
  }

  undo(): void {
    // 逆順でundo
    for (let i = this.commands.length - 1; i >= 0; i--) {
      this.commands[i].undo();
    }
  }

  describe(): string {
    return `Macro [${this.commands.map(c => c.describe()).join(", ")}]`;
  }
}

// コマンド履歴管理（Undo/Redo）
class CommandHistory {
  private undoStack: Command[] = [];
  private redoStack: Command[] = [];
  private maxHistory: number;

  constructor(maxHistory = 100) {
    this.maxHistory = maxHistory;
  }

  execute(command: Command): void {
    command.execute();
    this.undoStack.push(command);
    this.redoStack = []; // Redo履歴をクリア

    // 履歴の上限チェック
    if (this.undoStack.length > this.maxHistory) {
      this.undoStack.shift();
    }
  }

  undo(): Command | undefined {
    const command = this.undoStack.pop();
    if (command) {
      command.undo();
      this.redoStack.push(command);
    }
    return command;
  }

  redo(): Command | undefined {
    const command = this.redoStack.pop();
    if (command) {
      command.execute();
      this.undoStack.push(command);
    }
    return command;
  }

  canUndo(): boolean { return this.undoStack.length > 0; }
  canRedo(): boolean { return this.redoStack.length > 0; }

  getHistory(): string[] {
    return this.undoStack.map(c => c.describe());
  }

  clear(): void {
    this.undoStack = [];
    this.redoStack = [];
  }
}

// 使用例
const editor = new TextEditor();
const history = new CommandHistory();

history.execute(new InsertTextCommand(editor, 0, "Hello, "));
history.execute(new InsertTextCommand(editor, 7, "World!"));
console.log(editor.getContent()); // "Hello, World!"

history.undo();
console.log(editor.getContent()); // "Hello, "

history.redo();
console.log(editor.getContent()); // "Hello, World!"

// 検索と置換（マクロコマンド）
const findAndReplace = new MacroCommand([
  new DeleteTextCommand(editor, 7, 13),
  new InsertTextCommand(editor, 7, "TypeScript!"),
]);
history.execute(findAndReplace);
console.log(editor.getContent()); // "Hello, TypeScript!"
```

### 3.3 タスクキュー

```typescript
// Command パターンによるタスクキュー
interface AsyncCommand {
  execute(): Promise<void>;
  describe(): string;
  readonly priority: number;
}

class TaskQueue {
  private queue: AsyncCommand[] = [];
  private running = false;
  private concurrency: number;
  private activeCount = 0;

  constructor(concurrency = 1) {
    this.concurrency = concurrency;
  }

  enqueue(command: AsyncCommand): void {
    this.queue.push(command);
    // 優先度でソート（高い優先度が先）
    this.queue.sort((a, b) => b.priority - a.priority);
    this.processNext();
  }

  private async processNext(): Promise<void> {
    while (this.activeCount < this.concurrency && this.queue.length > 0) {
      const command = this.queue.shift()!;
      this.activeCount++;

      command.execute()
        .then(() => {
          console.log(`Completed: ${command.describe()}`);
        })
        .catch(err => {
          console.error(`Failed: ${command.describe()}`, err);
        })
        .finally(() => {
          this.activeCount--;
          this.processNext();
        });
    }
  }

  get pendingCount(): number { return this.queue.length; }
  get runningCount(): number { return this.activeCount; }
}
```

---

## 4. State パターン

### 4.1 概要と目的

```
目的: オブジェクトの状態に応じて振る舞いを変える

現代の応用: ステートマシン、UIコンポーネントの状態管理、ワークフロー

構造:
  ┌─────────┐      ┌─────────────┐
  │ Context │─────→│   State     │
  │         │      │ (状態)      │
  └─────────┘      └──────┬──────┘
                          │
              ┌───────────┼───────────┐
         ┌────┴────┐ ┌───┴───┐ ┌────┴────┐
         │ StateA  │ │StateB │ │ StateC  │
         └─────────┘ └───────┘ └─────────┘
```

### 4.2 注文ステートマシン

```typescript
// 状態インターフェース
interface OrderState {
  confirm(order: Order): void;
  ship(order: Order): void;
  deliver(order: Order): void;
  cancel(order: Order): void;
  refund(order: Order): void;
  toString(): string;
  allowedTransitions(): string[];
}

class PendingState implements OrderState {
  confirm(order: Order) {
    if (order.getItems().length === 0) {
      throw new Error("商品がない注文は確認できません");
    }
    order.setState(new ConfirmedState());
    order.addLog("注文確認済み");
  }
  ship() { throw new Error("未確認の注文は発送できません"); }
  deliver() { throw new Error("未確認の注文は配達できません"); }
  cancel(order: Order) {
    order.setState(new CancelledState("顧客によるキャンセル"));
    order.addLog("注文キャンセル");
  }
  refund() { throw new Error("保留中の注文は返金できません"); }
  toString() { return "保留中"; }
  allowedTransitions() { return ["confirm", "cancel"]; }
}

class ConfirmedState implements OrderState {
  confirm() { throw new Error("既に確認済みです"); }
  ship(order: Order) {
    order.setState(new ShippedState());
    order.addLog("発送済み");
  }
  deliver() { throw new Error("未発送の注文は配達できません"); }
  cancel(order: Order) {
    order.setState(new CancelledState("確認後キャンセル"));
    order.addLog("注文キャンセル（確認後）");
  }
  refund() { throw new Error("未発送の注文は返金できません"); }
  toString() { return "確認済み"; }
  allowedTransitions() { return ["ship", "cancel"]; }
}

class ShippedState implements OrderState {
  confirm() { throw new Error("発送済みです"); }
  ship() { throw new Error("既に発送済みです"); }
  deliver(order: Order) {
    order.setState(new DeliveredState());
    order.addLog("配達完了");
  }
  cancel() { throw new Error("発送済みの注文はキャンセルできません"); }
  refund() { throw new Error("配達前の返金はサポートに連絡してください"); }
  toString() { return "発送済み"; }
  allowedTransitions() { return ["deliver"]; }
}

class DeliveredState implements OrderState {
  confirm() { throw new Error("配達済みです"); }
  ship() { throw new Error("配達済みです"); }
  deliver() { throw new Error("既に配達済みです"); }
  cancel() { throw new Error("配達済みの注文はキャンセルできません"); }
  refund(order: Order) {
    const deliveredAt = order.getLastLogTime();
    const now = new Date();
    const daysSinceDelivery = (now.getTime() - deliveredAt.getTime()) / (1000 * 60 * 60 * 24);
    if (daysSinceDelivery > 30) {
      throw new Error("配達から30日以上経過した注文は返金できません");
    }
    order.setState(new RefundedState());
    order.addLog("返金処理開始");
  }
  toString() { return "配達済み"; }
  allowedTransitions() { return ["refund"]; }
}

class CancelledState implements OrderState {
  constructor(private reason: string) {}
  confirm() { throw new Error("キャンセル済みです"); }
  ship() { throw new Error("キャンセル済みです"); }
  deliver() { throw new Error("キャンセル済みです"); }
  cancel() { throw new Error("既にキャンセル済みです"); }
  refund() { throw new Error("キャンセル済みの注文は返金対象外です"); }
  toString() { return `キャンセル済み（${this.reason}）`; }
  allowedTransitions() { return []; }
}

class RefundedState implements OrderState {
  confirm() { throw new Error("返金済みです"); }
  ship() { throw new Error("返金済みです"); }
  deliver() { throw new Error("返金済みです"); }
  cancel() { throw new Error("返金済みです"); }
  refund() { throw new Error("既に返金済みです"); }
  toString() { return "返金済み"; }
  allowedTransitions() { return []; }
}

// Context
class Order {
  private state: OrderState = new PendingState();
  private items: OrderItem[] = [];
  private logs: Array<{ message: string; timestamp: Date }> = [];

  setState(state: OrderState): void { this.state = state; }
  confirm(): void { this.state.confirm(this); }
  ship(): void { this.state.ship(this); }
  deliver(): void { this.state.deliver(this); }
  cancel(): void { this.state.cancel(this); }
  refund(): void { this.state.refund(this); }
  getStatus(): string { return this.state.toString(); }
  getAllowedActions(): string[] { return this.state.allowedTransitions(); }
  getItems(): OrderItem[] { return this.items; }

  addItem(item: OrderItem): void {
    this.items.push(item);
  }

  addLog(message: string): void {
    this.logs.push({ message, timestamp: new Date() });
  }

  getLastLogTime(): Date {
    return this.logs[this.logs.length - 1]?.timestamp ?? new Date();
  }

  getHistory(): Array<{ message: string; timestamp: Date }> {
    return [...this.logs];
  }
}

// 使用例
const order = new Order();
order.addItem({ productId: "P-001", quantity: 1, price: 1000 });

console.log(order.getStatus());          // "保留中"
console.log(order.getAllowedActions());   // ["confirm", "cancel"]

order.confirm();  // 保留中 → 確認済み
order.ship();     // 確認済み → 発送済み
order.deliver();  // 発送済み → 配達済み

console.log(order.getStatus());          // "配達済み"
console.log(order.getAllowedActions());   // ["refund"]

// order.cancel(); // Error: 配達済みの注文はキャンセルできません
```

### 4.3 汎用的なステートマシン

```typescript
// 汎用ステートマシンの実装
interface StateMachineConfig<S extends string, E extends string> {
  initial: S;
  states: Record<S, {
    on?: Partial<Record<E, S | { target: S; guard?: () => boolean; action?: () => void }>>;
    onEnter?: () => void;
    onExit?: () => void;
  }>;
}

class StateMachine<S extends string, E extends string> {
  private current: S;
  private config: StateMachineConfig<S, E>;
  private listeners = new Set<(state: S, event: E) => void>();

  constructor(config: StateMachineConfig<S, E>) {
    this.config = config;
    this.current = config.initial;
    // 初期状態の onEnter を実行
    config.states[config.initial]?.onEnter?.();
  }

  getState(): S { return this.current; }

  send(event: E): S {
    const stateConfig = this.config.states[this.current];
    const transition = stateConfig?.on?.[event];

    if (!transition) {
      throw new Error(`No transition for event "${event}" from state "${this.current}"`);
    }

    let targetState: S;
    if (typeof transition === "string") {
      targetState = transition;
    } else {
      if (transition.guard && !transition.guard()) {
        throw new Error(`Guard failed for event "${event}" from state "${this.current}"`);
      }
      targetState = transition.target;
      transition.action?.();
    }

    // onExit → 遷移 → onEnter
    stateConfig?.onExit?.();
    this.current = targetState;
    this.config.states[targetState]?.onEnter?.();

    // リスナーに通知
    this.listeners.forEach(listener => listener(this.current, event));

    return this.current;
  }

  canSend(event: E): boolean {
    const stateConfig = this.config.states[this.current];
    return !!stateConfig?.on?.[event];
  }

  subscribe(listener: (state: S, event: E) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }
}

// 使用例: 信号機
type TrafficState = "red" | "yellow" | "green";
type TrafficEvent = "timer" | "emergency";

const trafficLight = new StateMachine<TrafficState, TrafficEvent>({
  initial: "red",
  states: {
    red: {
      on: {
        timer: "green",
        emergency: "red",  // 赤のまま
      },
      onEnter: () => console.log("赤信号: 停止"),
    },
    green: {
      on: {
        timer: "yellow",
        emergency: "red",
      },
      onEnter: () => console.log("青信号: 進行"),
    },
    yellow: {
      on: {
        timer: "red",
        emergency: "red",
      },
      onEnter: () => console.log("黄信号: 注意"),
    },
  },
});

trafficLight.send("timer");  // red → green
trafficLight.send("timer");  // green → yellow
trafficLight.send("timer");  // yellow → red
```

---

## 5. Iterator パターン

### 5.1 概要と目的

```
目的: コレクションの内部構造を公開せずに要素を順番にアクセスする
現代の応用: for...of, Python の __iter__, Rust の Iterator トレイト

いつ使うか:
  → コレクションの実装を隠蔽したい
  → 複数の走査方法を提供したい
  → 遅延評価でメモリ効率を上げたい
```

### 5.2 カスタムイテレータ

```typescript
// TypeScript: カスタムイテレータ
class Range implements Iterable<number> {
  constructor(
    private start: number,
    private end: number,
    private step: number = 1,
  ) {
    if (step === 0) throw new Error("Step cannot be zero");
  }

  [Symbol.iterator](): Iterator<number> {
    let current = this.start;
    const end = this.end;
    const step = this.step;

    return {
      next(): IteratorResult<number> {
        if ((step > 0 && current < end) || (step < 0 && current > end)) {
          const value = current;
          current += step;
          return { value, done: false };
        }
        return { value: undefined, done: true };
      },
    };
  }

  // ユーティリティメソッド
  toArray(): number[] { return [...this]; }
  map<T>(fn: (n: number) => T): T[] { return [...this].map(fn); }
  filter(fn: (n: number) => boolean): number[] { return [...this].filter(fn); }
  reduce<T>(fn: (acc: T, n: number) => T, initial: T): T {
    return [...this].reduce(fn, initial);
  }
}

// for...of で使える
for (const n of new Range(0, 10, 2)) {
  console.log(n); // 0, 2, 4, 6, 8
}

// スプレッド演算子も使える
const numbers = [...new Range(1, 6)]; // [1, 2, 3, 4, 5]

// 逆順
const reversed = [...new Range(10, 0, -1)]; // [10, 9, 8, ..., 1]
```

### 5.3 ジェネレータベースのイテレータ

```typescript
// ジェネレータ関数によるイテレータ
function* fibonacci(limit?: number): Generator<number> {
  let a = 0, b = 1;
  let count = 0;
  while (!limit || count < limit) {
    yield a;
    [a, b] = [b, a + b];
    count++;
  }
}

// 最初の10個のフィボナッチ数
for (const n of fibonacci(10)) {
  console.log(n); // 0, 1, 1, 2, 3, 5, 8, 13, 21, 34
}

// ツリーの深さ優先走査
interface TreeNode<T> {
  value: T;
  children: TreeNode<T>[];
}

function* depthFirst<T>(node: TreeNode<T>): Generator<T> {
  yield node.value;
  for (const child of node.children) {
    yield* depthFirst(child);
  }
}

function* breadthFirst<T>(root: TreeNode<T>): Generator<T> {
  const queue: TreeNode<T>[] = [root];
  while (queue.length > 0) {
    const node = queue.shift()!;
    yield node.value;
    queue.push(...node.children);
  }
}

// ページネーションイテレータ
async function* paginatedFetch<T>(
  fetchPage: (page: number) => Promise<{ data: T[]; hasMore: boolean }>,
): AsyncGenerator<T> {
  let page = 1;
  let hasMore = true;

  while (hasMore) {
    const result = await fetchPage(page);
    for (const item of result.data) {
      yield item;
    }
    hasMore = result.hasMore;
    page++;
  }
}

// 使用例: 全ユーザーを遅延取得
const allUsers = paginatedFetch(async (page) => {
  const response = await fetch(`/api/users?page=${page}&limit=100`);
  const data = await response.json();
  return { data: data.users, hasMore: data.hasMore };
});

// 必要な分だけ取得（全件メモリに載せない）
for await (const user of allUsers) {
  console.log(user.name);
  if (someCondition) break; // 途中で止められる
}
```

### 5.4 Python でのイテレータ

```python
# Python: イテレータプロトコル
class FileLineIterator:
    """大きなファイルを1行ずつ遅延読み込み"""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file = None

    def __iter__(self):
        self.file = open(self.filepath, 'r', encoding='utf-8')
        return self

    def __next__(self) -> str:
        line = self.file.readline()
        if line:
            return line.rstrip('\n')
        self.file.close()
        raise StopIteration

# ジェネレータ関数
def chunked(iterable, size: int):
    """イテラブルをchunkに分割"""
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

# 使用例
for chunk in chunked(range(100), 10):
    print(f"Processing chunk of {len(chunk)} items")
```

---

## 6. Chain of Responsibility パターン

### 6.1 概要と目的

```
目的: リクエストを処理できるオブジェクトのチェーンを通じて、
      適切なハンドラに到達させる

現代の応用: Express/Koa のミドルウェア、DOM イベントバブリング

構造:
  ┌───────────┐    ┌───────────┐    ┌───────────┐
  │ Handler1  │───→│ Handler2  │───→│ Handler3  │
  └───────────┘    └───────────┘    └───────────┘
```

### 6.2 HTTPミドルウェアチェーン

```typescript
// Express風のミドルウェアチェーン
type Middleware = (
  req: Request,
  res: Response,
  next: () => Promise<void>
) => Promise<void>;

class MiddlewarePipeline {
  private middlewares: Middleware[] = [];

  use(middleware: Middleware): this {
    this.middlewares.push(middleware);
    return this;
  }

  async execute(req: Request, res: Response): Promise<void> {
    let index = 0;

    const next = async (): Promise<void> => {
      if (index >= this.middlewares.length) return;
      const middleware = this.middlewares[index++];
      await middleware(req, res, next);
    };

    await next();
  }
}

// ミドルウェアの例
const loggingMiddleware: Middleware = async (req, res, next) => {
  const start = Date.now();
  console.log(`→ ${req.method} ${req.url}`);
  await next();
  console.log(`← ${res.statusCode} (${Date.now() - start}ms)`);
};

const authMiddleware: Middleware = async (req, res, next) => {
  const token = req.headers.get("Authorization")?.replace("Bearer ", "");
  if (!token) {
    res.statusCode = 401;
    return; // チェーンを中断
  }
  (req as any).userId = "decoded-user-id";
  await next();
};

const rateLimitMiddleware: Middleware = async (req, res, next) => {
  const clientIp = req.headers.get("X-Forwarded-For") ?? "unknown";
  // レートリミットチェック...
  await next();
};

const corsMiddleware: Middleware = async (req, res, next) => {
  res.headers.set("Access-Control-Allow-Origin", "*");
  res.headers.set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE");
  if (req.method === "OPTIONS") {
    res.statusCode = 204;
    return;
  }
  await next();
};

// パイプラインの構築
const pipeline = new MiddlewarePipeline()
  .use(loggingMiddleware)
  .use(corsMiddleware)
  .use(rateLimitMiddleware)
  .use(authMiddleware);
```

### 6.3 バリデーションチェーン

```typescript
// バリデーションチェーン
abstract class ValidationHandler {
  private next?: ValidationHandler;

  setNext(handler: ValidationHandler): ValidationHandler {
    this.next = handler;
    return handler; // チェーンの最後のハンドラを返す
  }

  async handle(data: any): Promise<ValidationResult> {
    const result = await this.validate(data);
    if (!result.valid) return result;
    if (this.next) return this.next.handle(data);
    return { valid: true, errors: [] };
  }

  protected abstract validate(data: any): Promise<ValidationResult>;
}

class SyntaxValidator extends ValidationHandler {
  protected async validate(data: any): Promise<ValidationResult> {
    if (typeof data !== "object" || data === null) {
      return { valid: false, errors: [{ field: "root", message: "データはオブジェクトである必要があります" }] };
    }
    return { valid: true, errors: [] };
  }
}

class SchemaValidator extends ValidationHandler {
  constructor(private schema: Record<string, string>) { super(); }

  protected async validate(data: any): Promise<ValidationResult> {
    const errors: Array<{ field: string; message: string }> = [];
    for (const [field, type] of Object.entries(this.schema)) {
      if (typeof data[field] !== type) {
        errors.push({ field, message: `${field}は${type}型である必要があります` });
      }
    }
    return { valid: errors.length === 0, errors };
  }
}

class BusinessRuleValidator extends ValidationHandler {
  protected async validate(data: any): Promise<ValidationResult> {
    const errors: Array<{ field: string; message: string }> = [];
    if (data.age && data.age < 18) {
      errors.push({ field: "age", message: "18歳未満は登録できません" });
    }
    return { valid: errors.length === 0, errors };
  }
}

// チェーンの構築
const syntaxValidator = new SyntaxValidator();
const schemaValidator = new SchemaValidator({ name: "string", age: "number" });
const businessValidator = new BusinessRuleValidator();

syntaxValidator.setNext(schemaValidator).setNext(businessValidator);

// 使用
const result = await syntaxValidator.handle({ name: "太郎", age: 15 });
// { valid: false, errors: [{ field: "age", message: "18歳未満は登録できません" }] }
```

---

## 7. Template Method パターン

### 7.1 概要と目的

```
目的: アルゴリズムの骨格を基底クラスで定義し、
      具体的なステップをサブクラスで実装する

構造:
  ┌────────────────────┐
  │  AbstractClass     │
  │  templateMethod()  │ ← 骨格（変更不可）
  │  step1()           │ ← 抽象（サブクラスが実装）
  │  step2()           │
  │  hook()            │ ← フック（オプション）
  └─────────┬──────────┘
            │
  ┌─────────┴──────────┐
  │  ConcreteClass     │
  │  step1() { ... }   │
  │  step2() { ... }   │
  └────────────────────┘
```

### 7.2 データエクスポートのテンプレート

```typescript
// Template Method: データエクスポートの共通フロー
abstract class DataExporter<T> {
  // テンプレートメソッド: 全体のフロー（変更不可）
  async export(query: ExportQuery): Promise<ExportResult> {
    console.log(`Starting export: ${this.getFormatName()}`);

    // 1. データ取得
    const rawData = await this.fetchData(query);

    // 2. フィルタリング（フック: オプション）
    const filtered = this.filterData(rawData, query);

    // 3. データ変換
    const transformed = await this.transformData(filtered);

    // 4. フォーマット（サブクラスが実装）
    const formatted = await this.formatOutput(transformed);

    // 5. 出力
    const result = await this.writeOutput(formatted, query.outputPath);

    // 6. 後処理（フック: オプション）
    await this.postProcess(result);

    return result;
  }

  // 抽象メソッド: サブクラスが実装
  protected abstract getFormatName(): string;
  protected abstract formatOutput(data: T[]): Promise<string | Buffer>;

  // 共通実装（必要に応じてオーバーライド可能）
  protected async fetchData(query: ExportQuery): Promise<T[]> {
    // デフォルトのデータ取得ロジック
    return [];
  }

  // フック: オプションのステップ
  protected filterData(data: T[], query: ExportQuery): T[] {
    return data; // デフォルトではフィルタリングなし
  }

  protected async transformData(data: T[]): Promise<T[]> {
    return data; // デフォルトでは変換なし
  }

  protected async writeOutput(
    content: string | Buffer,
    outputPath: string,
  ): Promise<ExportResult> {
    await fs.writeFile(outputPath, content);
    const stats = await fs.stat(outputPath);
    return {
      path: outputPath,
      size: stats.size,
      format: this.getFormatName(),
    };
  }

  protected async postProcess(result: ExportResult): Promise<void> {
    // デフォルトでは何もしない
  }
}

// CSV エクスポーター
class CsvExporter extends DataExporter<Record<string, any>> {
  protected getFormatName(): string { return "CSV"; }

  protected async formatOutput(data: Record<string, any>[]): Promise<string> {
    if (data.length === 0) return "";
    const headers = Object.keys(data[0]);
    const rows = data.map(row =>
      headers.map(h => `"${String(row[h]).replace(/"/g, '""')}"`).join(",")
    );
    return [headers.join(","), ...rows].join("\n");
  }

  protected filterData(data: Record<string, any>[], query: ExportQuery) {
    // CSVは全カラムをエクスポート、不要なカラムを除外
    if (!query.excludeColumns) return data;
    return data.map(row => {
      const filtered = { ...row };
      for (const col of query.excludeColumns!) {
        delete filtered[col];
      }
      return filtered;
    });
  }
}

// JSON エクスポーター
class JsonExporter extends DataExporter<Record<string, any>> {
  protected getFormatName(): string { return "JSON"; }

  protected async formatOutput(data: Record<string, any>[]): Promise<string> {
    return JSON.stringify(data, null, 2);
  }
}

// Excel エクスポーター
class ExcelExporter extends DataExporter<Record<string, any>> {
  protected getFormatName(): string { return "Excel"; }

  protected async formatOutput(data: Record<string, any>[]): Promise<Buffer> {
    // ExcelJS等を使ってExcelファイルを生成
    return Buffer.from("excel-content");
  }

  protected async postProcess(result: ExportResult): Promise<void> {
    // Excelファイルにパスワード保護を追加
    console.log(`Password protecting ${result.path}`);
  }
}
```

---

## 8. Mediator パターン

### 8.1 概要と目的

```
目的: オブジェクト間の直接的な通信を仲介者を通じて行い、疎結合にする

構造:
  ┌───────────┐
  │ Mediator  │
  │ (仲介者)  │
  └─────┬─────┘
        │
  ┌─────┼─────┐
  │     │     │
  ┌──┐ ┌──┐ ┌──┐
  │A │ │B │ │C │  ← 同僚（Colleague）
  └──┘ └──┘ └──┘
  A, B, C は互いに知らない（Mediator のみ知る）
```

### 8.2 チャットルームの Mediator

```typescript
// Mediator: チャットルーム
interface ChatMediator {
  register(participant: ChatParticipant): void;
  sendMessage(sender: ChatParticipant, message: string): void;
  sendDirectMessage(sender: ChatParticipant, recipient: string, message: string): void;
}

abstract class ChatParticipant {
  constructor(
    public readonly name: string,
    protected mediator?: ChatMediator,
  ) {}

  setMediator(mediator: ChatMediator): void {
    this.mediator = mediator;
  }

  send(message: string): void {
    this.mediator?.sendMessage(this, message);
  }

  sendTo(recipient: string, message: string): void {
    this.mediator?.sendDirectMessage(this, recipient, message);
  }

  abstract receive(sender: string, message: string): void;
}

class ChatRoom implements ChatMediator {
  private participants = new Map<string, ChatParticipant>();
  private messageLog: Array<{ from: string; to: string; message: string; timestamp: Date }> = [];

  register(participant: ChatParticipant): void {
    this.participants.set(participant.name, participant);
    participant.setMediator(this);
    this.broadcast("System", `${participant.name}が参加しました`);
  }

  sendMessage(sender: ChatParticipant, message: string): void {
    this.messageLog.push({
      from: sender.name,
      to: "all",
      message,
      timestamp: new Date(),
    });

    for (const [name, participant] of this.participants) {
      if (name !== sender.name) {
        participant.receive(sender.name, message);
      }
    }
  }

  sendDirectMessage(sender: ChatParticipant, recipientName: string, message: string): void {
    const recipient = this.participants.get(recipientName);
    if (!recipient) {
      sender.receive("System", `${recipientName}は見つかりません`);
      return;
    }

    this.messageLog.push({
      from: sender.name,
      to: recipientName,
      message,
      timestamp: new Date(),
    });

    recipient.receive(sender.name, `[DM] ${message}`);
  }

  private broadcast(from: string, message: string): void {
    for (const participant of this.participants.values()) {
      participant.receive(from, message);
    }
  }
}

class User extends ChatParticipant {
  private messages: Array<{ from: string; message: string }> = [];

  receive(sender: string, message: string): void {
    this.messages.push({ from: sender, message });
    console.log(`[${this.name}] ${sender}: ${message}`);
  }

  getMessages(): Array<{ from: string; message: string }> {
    return [...this.messages];
  }
}

// 使用例
const chatRoom = new ChatRoom();
const alice = new User("Alice");
const bob = new User("Bob");
const charlie = new User("Charlie");

chatRoom.register(alice);
chatRoom.register(bob);
chatRoom.register(charlie);

alice.send("こんにちは、みなさん！");    // Bob と Charlie に送信
bob.sendTo("Alice", "元気ですか？");     // Alice にDM
```

---

## 9. Visitor パターン

### 9.1 概要と目的

```
目的: データ構造とそれに対する操作を分離する

いつ使うか:
  → データ構造は安定しているが、操作が頻繁に追加される
  → 複数の無関係な操作をデータ構造に適用したい
  → Double Dispatch が必要な場合
```

### 9.2 AST（抽象構文木）のVisitor

```typescript
// AST ノード
interface ASTNode {
  accept<T>(visitor: ASTVisitor<T>): T;
}

class NumberLiteral implements ASTNode {
  constructor(public value: number) {}
  accept<T>(visitor: ASTVisitor<T>): T { return visitor.visitNumber(this); }
}

class StringLiteral implements ASTNode {
  constructor(public value: string) {}
  accept<T>(visitor: ASTVisitor<T>): T { return visitor.visitString(this); }
}

class BinaryExpression implements ASTNode {
  constructor(
    public left: ASTNode,
    public operator: "+" | "-" | "*" | "/",
    public right: ASTNode,
  ) {}
  accept<T>(visitor: ASTVisitor<T>): T { return visitor.visitBinary(this); }
}

class FunctionCall implements ASTNode {
  constructor(public name: string, public args: ASTNode[]) {}
  accept<T>(visitor: ASTVisitor<T>): T { return visitor.visitFunctionCall(this); }
}

// Visitor インターフェース
interface ASTVisitor<T> {
  visitNumber(node: NumberLiteral): T;
  visitString(node: StringLiteral): T;
  visitBinary(node: BinaryExpression): T;
  visitFunctionCall(node: FunctionCall): T;
}

// 評価 Visitor
class EvaluatorVisitor implements ASTVisitor<number> {
  visitNumber(node: NumberLiteral): number { return node.value; }
  visitString(node: StringLiteral): number { return parseFloat(node.value) || 0; }

  visitBinary(node: BinaryExpression): number {
    const left = node.left.accept(this);
    const right = node.right.accept(this);
    switch (node.operator) {
      case "+": return left + right;
      case "-": return left - right;
      case "*": return left * right;
      case "/": return right !== 0 ? left / right : NaN;
    }
  }

  visitFunctionCall(node: FunctionCall): number {
    const args = node.args.map(a => a.accept(this));
    switch (node.name) {
      case "max": return Math.max(...args);
      case "min": return Math.min(...args);
      case "abs": return Math.abs(args[0]);
      default: throw new Error(`Unknown function: ${node.name}`);
    }
  }
}

// 文字列化 Visitor
class PrinterVisitor implements ASTVisitor<string> {
  visitNumber(node: NumberLiteral): string { return String(node.value); }
  visitString(node: StringLiteral): string { return `"${node.value}"`; }

  visitBinary(node: BinaryExpression): string {
    return `(${node.left.accept(this)} ${node.operator} ${node.right.accept(this)})`;
  }

  visitFunctionCall(node: FunctionCall): string {
    const args = node.args.map(a => a.accept(this)).join(", ");
    return `${node.name}(${args})`;
  }
}

// 使用例: max(3 + 4, 2 * 5)
const ast = new FunctionCall("max", [
  new BinaryExpression(new NumberLiteral(3), "+", new NumberLiteral(4)),
  new BinaryExpression(new NumberLiteral(2), "*", new NumberLiteral(5)),
]);

const evaluator = new EvaluatorVisitor();
const printer = new PrinterVisitor();

console.log(printer.visitFunctionCall(ast as FunctionCall)); // "max((3 + 4), (2 * 5))"
console.log(ast.accept(evaluator));                           // 10
```

---

## まとめ

| パターン | 目的 | 現代の応用 |
|---------|------|-----------|
| Strategy | アルゴリズムの切り替え | DI、ポリシー、プラグイン |
| Observer | 状態変化の通知 | イベント、Reactive、Pub/Sub |
| Command | 操作のオブジェクト化 | Undo/Redo、Redux、CQRS |
| State | 状態による振る舞い変更 | ステートマシン、ワークフロー |
| Iterator | 順次アクセス | for...of、ジェネレータ、ストリーム |
| Chain of Responsibility | リクエストの連鎖処理 | ミドルウェア、パイプライン |
| Template Method | アルゴリズムの骨格定義 | フレームワークの拡張ポイント |
| Mediator | オブジェクト間の疎結合通信 | チャット、イベントバス |
| Visitor | データ構造と操作の分離 | AST操作、シリアライズ |

### パターン選択の指針

```
振る舞いの問題を解決したい
├── アルゴリズムを実行時に切替 → Strategy
├── 状態変化を通知 → Observer
├── 操作を記録・再実行・取消 → Command
├── 状態に応じた振る舞い → State
├── コレクションの走査 → Iterator
├── リクエストの連鎖処理 → Chain of Responsibility
├── 処理フローの骨格定義 → Template Method
├── オブジェクト間の疎結合通信 → Mediator
└── データ構造への操作追加 → Visitor
```

---

## 次に読むべきガイド
→ [[03-anti-patterns.md]] — アンチパターン

---

## 参考文献
1. Gamma, E. et al. "Design Patterns: Elements of Reusable Object-Oriented Software." Addison-Wesley, 1994.
2. Freeman, E. et al. "Head First Design Patterns." O'Reilly, 2020.
3. Nystrom, R. "Game Programming Patterns." Genever Benning, 2014.
4. Fowler, M. "Patterns of Enterprise Application Architecture." Addison-Wesley, 2002.
