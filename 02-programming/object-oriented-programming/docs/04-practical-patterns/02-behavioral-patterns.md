# Behavioral Patterns

> Patterns concerning the distribution of responsibilities between objects and the encapsulation of algorithms. A practical guide to nine patterns: Strategy, Observer, Command, State, Iterator, Chain of Responsibility, Template Method, Mediator, and Visitor.

## What You Will Learn in This Chapter

- [ ] Understand the purpose and applicable scenarios of each behavioral pattern
- [ ] Grasp the implementation of each pattern across multiple languages
- [ ] Learn how these patterns are applied in modern frameworks
- [ ] Understand the differences between patterns and how to combine them
- [ ] Be able to design behaviors with testability in mind

## Prerequisites

Your understanding will deepen if you have the following knowledge before reading this guide:

- Basic programming knowledge
- Understanding of related foundational concepts
- Understanding of the contents of [Structural Patterns](./01-structural-patterns.md)

---

## 1. Strategy Pattern

### 1.1 Overview and Purpose

```
Purpose: Encapsulate algorithms and make them interchangeable at runtime

Structure:
  ┌─────────┐      ┌────────────────┐
  │ Context │─────→│   Strategy     │
  │         │      │  (interface)   │
  └─────────┘      └───────┬────────┘
                           │
              ┌────────────┼────────────┐
         ┌────┴────┐  ┌───┴───┐  ┌────┴────┐
         │StratA   │  │StratB │  │StratC   │
         └─────────┘  └───────┘  └─────────┘

Modern applications: React render strategies, sort algorithm selection, DI

When to use:
  → Multiple algorithms exist for the same process
  → You want to switch algorithms at runtime
  → if-else/switch branches are growing out of hand
  → You want to hide algorithm details from clients
```

### 1.2 Compression Strategies

```typescript
// Compression strategy
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

// Context: file processor that uses a compression strategy
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

// Usage: automatically select strategy based on file size
function selectCompression(fileSize: number): CompressionStrategy {
  if (fileSize < 1024) return new NoCompression();           // less than 1KB: no compression
  if (fileSize < 1024 * 1024) return new GzipCompression();  // less than 1MB: gzip
  return new BrotliCompression();                             // 1MB or more: brotli
}
```

### 1.3 Validation Strategies

```typescript
// Validation strategy pattern
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
      .map(field => ({ field, message: `${field} is required` }));

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
      .map(field => ({ field, message: `${data[field]} is not a valid email address` }));

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
        errors.push({ field: rule.field, message: `${rule.field} must be at least ${rule.min}` });
      }
      if (rule.max !== undefined && value > rule.max) {
        errors.push({ field: rule.field, message: `${rule.field} must be at most ${rule.max}` });
      }
    }

    return { valid: errors.length === 0, errors };
  }
}

// Composite Strategy that combines multiple strategies
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

// Usage
const userValidator = new CompositeValidator()
  .add(new RequiredFieldsValidator(["name", "email", "age"]))
  .add(new EmailFormatValidator(["email"]))
  .add(new RangeValidator([{ field: "age", min: 0, max: 150 }]));

const result = userValidator.validate({
  name: "Taro",
  email: "invalid-email",
  age: 200,
});
// { valid: false, errors: [
//   { field: "email", message: "invalid-email is not a valid email address" },
//   { field: "age", message: "age must be at most 150" }
// ]}
```

### 1.4 Strategy Pattern in Python

```python
# Python: Strategy pattern (function-based and class-based)
from typing import Protocol, Callable
from abc import abstractmethod

# Protocol-based Strategy
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

# Function-based Strategy (the Pythonic approach)
def process_data(data: list, sort_fn: Callable[[list], list] = sorted) -> list:
    return sort_fn(data)

# Usage
processor = DataProcessor(QuickSort())
result = processor.process([3, 1, 4, 1, 5, 9, 2, 6])
```

---

## 2. Observer Pattern

### 2.1 Overview and Purpose

```
Purpose: Automatically notify other objects of state changes in an object

Structure:
  ┌──────────┐      ┌──────────────┐
  │ Subject  │─────→│   Observer   │ ×N
  │(publisher)│      │ (subscriber) │
  └──────────┘      └──────────────┘

Modern applications: event systems, React state management, RxJS, Pub/Sub

When to use:
  → You want to notify multiple objects of state changes
  → You need a loosely coupled communication mechanism
  → The number and kinds of subscribers change dynamically
```

### 2.2 Type-Safe EventEmitter

```typescript
// Type-safe EventEmitter
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

    // Return an unsubscribe function
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
    // Regular listeners
    this.listeners.get(event)?.forEach(listener => listener(data));

    // One-time listeners
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

// Usage
const events = new TypedEventEmitter<EventMap>();

// Register Observer (subscriber)
const unsubscribe = events.on("userCreated", (data) => {
  console.log(`Welcome email to ${data.email}`); // type-safe
});

events.on("orderPlaced", (data) => {
  console.log(`Order ${data.orderId}: ¥${data.total}`);
});

// Subscribe only once
events.once("userCreated", (data) => {
  console.log(`First user bonus for ${data.userId}`);
});

// Subject (publisher) emits events
events.emit("userCreated", { userId: "1", email: "tanaka@example.com" });
events.emit("orderPlaced", { orderId: "O-001", total: 5000 });

unsubscribe(); // Unsubscribe
```

### 2.3 Reactive Store (state management)

```typescript
// Reactive Store: React/Vue-style state management
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

  // Subscribe to the entire state
  subscribe(listener: (state: T) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  // Subscribe to a specific property (selector)
  select<K extends keyof T>(key: K, listener: (value: T[K]) => void): () => void {
    const keyStr = String(key);
    if (!this.selectorListeners.has(keyStr)) {
      this.selectorListeners.set(keyStr, new Set());
    }
    this.selectorListeners.get(keyStr)!.add(listener);
    return () => this.selectorListeners.get(keyStr)?.delete(listener);
  }

  // Update the state
  setState(updater: Partial<T> | ((prev: T) => Partial<T>), action = "setState"): void {
    const updates = typeof updater === "function" ? updater(this.state) : updater;
    const prev = { ...this.state };
    let next = { ...this.state, ...updates };

    // Apply middleware
    for (const mw of this.middleware) {
      next = mw(prev, next, action);
    }

    this.state = next;

    // Notify global listeners
    this.listeners.forEach(listener => listener(this.state));

    // Notify selector listeners for changed properties
    for (const key of Object.keys(updates)) {
      if (prev[key] !== this.state[key as keyof T]) {
        this.selectorListeners.get(key)?.forEach(listener =>
          listener(this.state[key as keyof T])
        );
      }
    }
  }
}

// Usage: logging middleware
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

// Watch only theme changes
store.select("theme", (theme) => {
  document.body.className = theme;
});

// Watch notification count changes
store.select("notifications", (count) => {
  console.log(`Unread notifications: ${count}`);
});

// Update state
store.setState({ user: { name: "Taro", email: "taro@example.com" } }, "LOGIN");
store.setState(prev => ({ notifications: prev.notifications + 1 }), "NEW_NOTIFICATION");
store.setState({ theme: "dark" }, "TOGGLE_THEME");
```

---

## 3. Command Pattern

### 3.1 Overview and Purpose

```
Purpose: Encapsulate a request as an object

Benefits: Undo, queuing, logging, transactions
Modern applications: Redux actions, editor Undo/Redo, CQRS pattern

Structure:
  ┌──────────┐    ┌─────────┐    ┌──────────┐
  │ Invoker  │───→│ Command │───→│ Receiver │
  │(executor)│    │(command)│    │(receiver)│
  └──────────┘    └─────────┘    └──────────┘
```

### 3.2 Text Editor Undo/Redo

```typescript
// Command interface
interface Command {
  execute(): void;
  undo(): void;
  describe(): string;
}

// Text editor (Receiver)
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

// Concrete commands
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

// Macro command: bundle multiple commands into one
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
    // Undo in reverse order
    for (let i = this.commands.length - 1; i >= 0; i--) {
      this.commands[i].undo();
    }
  }

  describe(): string {
    return `Macro [${this.commands.map(c => c.describe()).join(", ")}]`;
  }
}

// Command history management (Undo/Redo)
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
    this.redoStack = []; // Clear redo history

    // Check history limit
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

// Usage
const editor = new TextEditor();
const history = new CommandHistory();

history.execute(new InsertTextCommand(editor, 0, "Hello, "));
history.execute(new InsertTextCommand(editor, 7, "World!"));
console.log(editor.getContent()); // "Hello, World!"

history.undo();
console.log(editor.getContent()); // "Hello, "

history.redo();
console.log(editor.getContent()); // "Hello, World!"

// Find and replace (macro command)
const findAndReplace = new MacroCommand([
  new DeleteTextCommand(editor, 7, 13),
  new InsertTextCommand(editor, 7, "TypeScript!"),
]);
history.execute(findAndReplace);
console.log(editor.getContent()); // "Hello, TypeScript!"
```

### 3.3 Task Queue

```typescript
// Task queue using the Command pattern
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
    // Sort by priority (higher priority first)
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

## 4. State Pattern

### 4.1 Overview and Purpose

```
Purpose: Change an object's behavior based on its state

Modern applications: state machines, UI component state management, workflows

Structure:
  ┌─────────┐      ┌─────────────┐
  │ Context │─────→│   State     │
  │         │      │  (state)    │
  └─────────┘      └──────┬──────┘
                          │
              ┌───────────┼───────────┐
         ┌────┴────┐ ┌───┴───┐ ┌────┴────┐
         │ StateA  │ │StateB │ │ StateC  │
         └─────────┘ └───────┘ └─────────┘
```

### 4.2 Order State Machine

```typescript
// State interface
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
      throw new Error("Cannot confirm an order with no items");
    }
    order.setState(new ConfirmedState());
    order.addLog("Order confirmed");
  }
  ship() { throw new Error("Cannot ship an unconfirmed order"); }
  deliver() { throw new Error("Cannot deliver an unconfirmed order"); }
  cancel(order: Order) {
    order.setState(new CancelledState("Cancelled by customer"));
    order.addLog("Order cancelled");
  }
  refund() { throw new Error("Cannot refund a pending order"); }
  toString() { return "Pending"; }
  allowedTransitions() { return ["confirm", "cancel"]; }
}

class ConfirmedState implements OrderState {
  confirm() { throw new Error("Already confirmed"); }
  ship(order: Order) {
    order.setState(new ShippedState());
    order.addLog("Shipped");
  }
  deliver() { throw new Error("Cannot deliver an unshipped order"); }
  cancel(order: Order) {
    order.setState(new CancelledState("Cancelled after confirmation"));
    order.addLog("Order cancelled (after confirmation)");
  }
  refund() { throw new Error("Cannot refund an unshipped order"); }
  toString() { return "Confirmed"; }
  allowedTransitions() { return ["ship", "cancel"]; }
}

class ShippedState implements OrderState {
  confirm() { throw new Error("Already shipped"); }
  ship() { throw new Error("Already shipped"); }
  deliver(order: Order) {
    order.setState(new DeliveredState());
    order.addLog("Delivery completed");
  }
  cancel() { throw new Error("Cannot cancel a shipped order"); }
  refund() { throw new Error("Please contact support for refunds before delivery"); }
  toString() { return "Shipped"; }
  allowedTransitions() { return ["deliver"]; }
}

class DeliveredState implements OrderState {
  confirm() { throw new Error("Already delivered"); }
  ship() { throw new Error("Already delivered"); }
  deliver() { throw new Error("Already delivered"); }
  cancel() { throw new Error("Cannot cancel a delivered order"); }
  refund(order: Order) {
    const deliveredAt = order.getLastLogTime();
    const now = new Date();
    const daysSinceDelivery = (now.getTime() - deliveredAt.getTime()) / (1000 * 60 * 60 * 24);
    if (daysSinceDelivery > 30) {
      throw new Error("Cannot refund an order more than 30 days after delivery");
    }
    order.setState(new RefundedState());
    order.addLog("Refund process started");
  }
  toString() { return "Delivered"; }
  allowedTransitions() { return ["refund"]; }
}

class CancelledState implements OrderState {
  constructor(private reason: string) {}
  confirm() { throw new Error("Already cancelled"); }
  ship() { throw new Error("Already cancelled"); }
  deliver() { throw new Error("Already cancelled"); }
  cancel() { throw new Error("Already cancelled"); }
  refund() { throw new Error("Cancelled orders are not eligible for refund"); }
  toString() { return `Cancelled (${this.reason})`; }
  allowedTransitions() { return []; }
}

class RefundedState implements OrderState {
  confirm() { throw new Error("Already refunded"); }
  ship() { throw new Error("Already refunded"); }
  deliver() { throw new Error("Already refunded"); }
  cancel() { throw new Error("Already refunded"); }
  refund() { throw new Error("Already refunded"); }
  toString() { return "Refunded"; }
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

// Usage
const order = new Order();
order.addItem({ productId: "P-001", quantity: 1, price: 1000 });

console.log(order.getStatus());          // "Pending"
console.log(order.getAllowedActions());   // ["confirm", "cancel"]

order.confirm();  // Pending → Confirmed
order.ship();     // Confirmed → Shipped
order.deliver();  // Shipped → Delivered

console.log(order.getStatus());          // "Delivered"
console.log(order.getAllowedActions());   // ["refund"]

// order.cancel(); // Error: Cannot cancel a delivered order
```

### 4.3 Generic State Machine

```typescript
// Generic state machine implementation
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
    // Run onEnter of the initial state
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

    // onExit → transition → onEnter
    stateConfig?.onExit?.();
    this.current = targetState;
    this.config.states[targetState]?.onEnter?.();

    // Notify listeners
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

// Usage: traffic light
type TrafficState = "red" | "yellow" | "green";
type TrafficEvent = "timer" | "emergency";

const trafficLight = new StateMachine<TrafficState, TrafficEvent>({
  initial: "red",
  states: {
    red: {
      on: {
        timer: "green",
        emergency: "red",  // stays red
      },
      onEnter: () => console.log("Red light: Stop"),
    },
    green: {
      on: {
        timer: "yellow",
        emergency: "red",
      },
      onEnter: () => console.log("Green light: Go"),
    },
    yellow: {
      on: {
        timer: "red",
        emergency: "red",
      },
      onEnter: () => console.log("Yellow light: Caution"),
    },
  },
});

trafficLight.send("timer");  // red → green
trafficLight.send("timer");  // green → yellow
trafficLight.send("timer");  // yellow → red
```

---

## 5. Iterator Pattern

### 5.1 Overview and Purpose

```
Purpose: Access elements of a collection sequentially without exposing its internal structure
Modern applications: for...of, Python's __iter__, Rust's Iterator trait

When to use:
  → You want to hide the implementation of a collection
  → You want to provide multiple ways to traverse it
  → You want to improve memory efficiency with lazy evaluation
```

### 5.2 Custom Iterator

```typescript
// TypeScript: custom iterator
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

  // Utility methods
  toArray(): number[] { return [...this]; }
  map<T>(fn: (n: number) => T): T[] { return [...this].map(fn); }
  filter(fn: (n: number) => boolean): number[] { return [...this].filter(fn); }
  reduce<T>(fn: (acc: T, n: number) => T, initial: T): T {
    return [...this].reduce(fn, initial);
  }
}

// Usable with for...of
for (const n of new Range(0, 10, 2)) {
  console.log(n); // 0, 2, 4, 6, 8
}

// Also works with the spread operator
const numbers = [...new Range(1, 6)]; // [1, 2, 3, 4, 5]

// Reverse order
const reversed = [...new Range(10, 0, -1)]; // [10, 9, 8, ..., 1]
```

### 5.3 Generator-Based Iterator

```typescript
// Iterator via a generator function
function* fibonacci(limit?: number): Generator<number> {
  let a = 0, b = 1;
  let count = 0;
  while (!limit || count < limit) {
    yield a;
    [a, b] = [b, a + b];
    count++;
  }
}

// First 10 Fibonacci numbers
for (const n of fibonacci(10)) {
  console.log(n); // 0, 1, 1, 2, 3, 5, 8, 13, 21, 34
}

// Depth-first traversal of a tree
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

// Pagination iterator
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

// Usage: lazily fetch all users
const allUsers = paginatedFetch(async (page) => {
  const response = await fetch(`/api/users?page=${page}&limit=100`);
  const data = await response.json();
  return { data: data.users, hasMore: data.hasMore };
});

// Fetch only as many as needed (no need to load everything into memory)
for await (const user of allUsers) {
  console.log(user.name);
  if (someCondition) break; // Can stop midway
}
```

### 5.4 Iterators in Python

```python
# Python: iterator protocol
class FileLineIterator:
    """Lazily reads a large file one line at a time"""

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

# Generator function
def chunked(iterable, size: int):
    """Split an iterable into chunks"""
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

# Usage
for chunk in chunked(range(100), 10):
    print(f"Processing chunk of {len(chunk)} items")
```

---

## 6. Chain of Responsibility Pattern

### 6.1 Overview and Purpose

```
Purpose: Pass a request through a chain of objects that can handle it
         until it reaches an appropriate handler

Modern applications: Express/Koa middleware, DOM event bubbling

Structure:
  ┌───────────┐    ┌───────────┐    ┌───────────┐
  │ Handler1  │───→│ Handler2  │───→│ Handler3  │
  └───────────┘    └───────────┘    └───────────┘
```

### 6.2 HTTP Middleware Chain

```typescript
// Express-style middleware chain
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

// Example middlewares
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
    return; // Short-circuit the chain
  }
  (req as any).userId = "decoded-user-id";
  await next();
};

const rateLimitMiddleware: Middleware = async (req, res, next) => {
  const clientIp = req.headers.get("X-Forwarded-For") ?? "unknown";
  // Rate limit check...
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

// Build the pipeline
const pipeline = new MiddlewarePipeline()
  .use(loggingMiddleware)
  .use(corsMiddleware)
  .use(rateLimitMiddleware)
  .use(authMiddleware);
```

### 6.3 Validation Chain

```typescript
// Validation chain
abstract class ValidationHandler {
  private next?: ValidationHandler;

  setNext(handler: ValidationHandler): ValidationHandler {
    this.next = handler;
    return handler; // Return the last handler of the chain
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
      return { valid: false, errors: [{ field: "root", message: "Data must be an object" }] };
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
        errors.push({ field, message: `${field} must be of type ${type}` });
      }
    }
    return { valid: errors.length === 0, errors };
  }
}

class BusinessRuleValidator extends ValidationHandler {
  protected async validate(data: any): Promise<ValidationResult> {
    const errors: Array<{ field: string; message: string }> = [];
    if (data.age && data.age < 18) {
      errors.push({ field: "age", message: "Users under 18 cannot register" });
    }
    return { valid: errors.length === 0, errors };
  }
}

// Build the chain
const syntaxValidator = new SyntaxValidator();
const schemaValidator = new SchemaValidator({ name: "string", age: "number" });
const businessValidator = new BusinessRuleValidator();

syntaxValidator.setNext(schemaValidator).setNext(businessValidator);

// Usage
const result = await syntaxValidator.handle({ name: "Taro", age: 15 });
// { valid: false, errors: [{ field: "age", message: "Users under 18 cannot register" }] }
```

---

## 7. Template Method Pattern

### 7.1 Overview and Purpose

```
Purpose: Define the skeleton of an algorithm in a base class,
         letting subclasses implement the concrete steps

Structure:
  ┌────────────────────┐
  │  AbstractClass     │
  │  templateMethod()  │ ← skeleton (cannot be changed)
  │  step1()           │ ← abstract (implemented by subclasses)
  │  step2()           │
  │  hook()            │ ← hook (optional)
  └─────────┬──────────┘
            │
  ┌─────────┴──────────┐
  │  ConcreteClass     │
  │  step1() { ... }   │
  │  step2() { ... }   │
  └────────────────────┘
```

### 7.2 Data Export Template

```typescript
// Template Method: shared flow for data export
abstract class DataExporter<T> {
  // Template method: the overall flow (cannot be changed)
  async export(query: ExportQuery): Promise<ExportResult> {
    console.log(`Starting export: ${this.getFormatName()}`);

    // 1. Fetch data
    const rawData = await this.fetchData(query);

    // 2. Filtering (hook: optional)
    const filtered = this.filterData(rawData, query);

    // 3. Data transformation
    const transformed = await this.transformData(filtered);

    // 4. Formatting (implemented by subclass)
    const formatted = await this.formatOutput(transformed);

    // 5. Output
    const result = await this.writeOutput(formatted, query.outputPath);

    // 6. Post-processing (hook: optional)
    await this.postProcess(result);

    return result;
  }

  // Abstract methods: implemented by subclasses
  protected abstract getFormatName(): string;
  protected abstract formatOutput(data: T[]): Promise<string | Buffer>;

  // Shared implementation (can be overridden if needed)
  protected async fetchData(query: ExportQuery): Promise<T[]> {
    // Default data fetching logic
    return [];
  }

  // Hook: optional step
  protected filterData(data: T[], query: ExportQuery): T[] {
    return data; // No filtering by default
  }

  protected async transformData(data: T[]): Promise<T[]> {
    return data; // No transformation by default
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
    // Does nothing by default
  }
}

// CSV exporter
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
    // CSV exports all columns, excluding unnecessary ones
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

// JSON exporter
class JsonExporter extends DataExporter<Record<string, any>> {
  protected getFormatName(): string { return "JSON"; }

  protected async formatOutput(data: Record<string, any>[]): Promise<string> {
    return JSON.stringify(data, null, 2);
  }
}

// Excel exporter
class ExcelExporter extends DataExporter<Record<string, any>> {
  protected getFormatName(): string { return "Excel"; }

  protected async formatOutput(data: Record<string, any>[]): Promise<Buffer> {
    // Generate an Excel file using ExcelJS or similar
    return Buffer.from("excel-content");
  }

  protected async postProcess(result: ExportResult): Promise<void> {
    // Add password protection to the Excel file
    console.log(`Password protecting ${result.path}`);
  }
}
```

---

## 8. Mediator Pattern

### 8.1 Overview and Purpose

```
Purpose: Have objects communicate through a mediator rather than directly,
         promoting loose coupling

Structure:
  ┌───────────┐
  │ Mediator  │
  │(mediator) │
  └─────┬─────┘
        │
  ┌─────┼─────┐
  │     │     │
  ┌──┐ ┌──┐ ┌──┐
  │A │ │B │ │C │  ← Colleagues
  └──┘ └──┘ └──┘
  A, B, and C do not know each other (only the Mediator does)
```

### 8.2 Chat Room Mediator

```typescript
// Mediator: chat room
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
    this.broadcast("System", `${participant.name} has joined`);
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
      sender.receive("System", `${recipientName} was not found`);
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

// Usage
const chatRoom = new ChatRoom();
const alice = new User("Alice");
const bob = new User("Bob");
const charlie = new User("Charlie");

chatRoom.register(alice);
chatRoom.register(bob);
chatRoom.register(charlie);

alice.send("Hello, everyone!");          // Sent to Bob and Charlie
bob.sendTo("Alice", "How are you?");     // DM to Alice
```

---

## 9. Visitor Pattern

### 9.1 Overview and Purpose

```
Purpose: Separate a data structure from the operations performed on it

When to use:
  → The data structure is stable, but operations are frequently added
  → You want to apply multiple unrelated operations to a data structure
  → Double Dispatch is required
```

### 9.2 Visitor for an AST (Abstract Syntax Tree)

```typescript
// AST node
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

// Visitor interface
interface ASTVisitor<T> {
  visitNumber(node: NumberLiteral): T;
  visitString(node: StringLiteral): T;
  visitBinary(node: BinaryExpression): T;
  visitFunctionCall(node: FunctionCall): T;
}

// Evaluation Visitor
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

// Stringification Visitor
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

// Usage: max(3 + 4, 2 * 5)
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


## FAQ

### Q1: What is the most important point when learning this topic?

Gaining practical experience is the most important thing. Your understanding deepens not only by studying theory but also by actually writing code and verifying how it behaves.

### Q2: What mistakes do beginners commonly make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this used in practice?

Knowledge of this topic is frequently applied in day-to-day development work. It becomes especially important during code reviews and architectural design.

---

## Summary

| Pattern | Purpose | Modern Applications |
|---------|------|-----------|
| Strategy | Switching algorithms | DI, policies, plugins |
| Observer | Notifying state changes | Events, Reactive, Pub/Sub |
| Command | Turning operations into objects | Undo/Redo, Redux, CQRS |
| State | Changing behavior based on state | State machines, workflows |
| Iterator | Sequential access | for...of, generators, streams |
| Chain of Responsibility | Chained request handling | Middleware, pipelines |
| Template Method | Defining an algorithm's skeleton | Framework extension points |
| Mediator | Loosely coupled object communication | Chat, event buses |
| Visitor | Separating data structure from operations | AST manipulation, serialization |

### Guidelines for Choosing a Pattern

```
I want to solve a behavior problem
├── Switch algorithms at runtime → Strategy
├── Notify of state changes → Observer
├── Record, replay, or undo operations → Command
├── Change behavior based on state → State
├── Traverse a collection → Iterator
├── Chain request handling → Chain of Responsibility
├── Define the skeleton of a process → Template Method
├── Loosely coupled object communication → Mediator
└── Add operations to a data structure → Visitor
```

---

## Next Guides to Read

---

## References
1. Gamma, E. et al. "Design Patterns: Elements of Reusable Object-Oriented Software." Addison-Wesley, 1994.
2. Freeman, E. et al. "Head First Design Patterns." O'Reilly, 2020.
3. Nystrom, R. "Game Programming Patterns." Genever Benning, 2014.
4. Fowler, M. "Patterns of Enterprise Application Architecture." Addison-Wesley, 2002.
