# Command パターン

> 操作をオブジェクトとしてカプセル化し、Undo/Redo、キューイング、マクロ記録、トランザクション制御を実現する行動パターン

---

## この章で学ぶこと

1. **Command パターンの基本構造と GoF の意図** -- コマンドオブジェクトによる操作のカプセル化、実行の遅延・記録・再生の仕組み
2. **Undo/Redo の設計と実装** -- コマンド履歴を管理し、操作の取り消しとやり直しを実現する設計手法とその内部動作原理
3. **マクロ・キューイング・トランザクション** -- 複数コマンドの合成、非同期実行キュー、ロールバック付きトランザクション実行
4. **実プロダクトにおける Command** -- テキストエディタ、Redux、CQRS/Event Sourcing、ゲームリプレイなど現実の適用例
5. **関数型アプローチとの統合** -- クロージャベース Command、TypeScript の型安全な Command バス

---

## 前提知識

| トピック | 必要な理解 | 参照リンク |
|---------|-----------|-----------|
| TypeScript の interface と class | インターフェースの定義と実装、ジェネリクスの基本 | [02-programming](../../02-programming/) |
| SOLID 原則（特に SRP・OCP） | 単一責任と開放閉鎖原則の理解 | [clean-code-principles](../../03-software-design/clean-code-principles/) |
| Observer パターン | イベント駆動の基本概念 | [00-observer.md](./00-observer.md) |
| Strategy パターン | アルゴリズムの切り替え | [01-strategy.md](./01-strategy.md) |
| Promise / async-await | 非同期処理の基本 | [02-programming](../../02-programming/) |

---

## なぜ Command パターンが必要なのか

### 直接呼び出しの問題

ボタンのクリックで「保存」を実行するUIを考えてみましょう。

```
直接呼び出しの問題:

  ┌──────────┐     ┌──────────────┐
  │ Button   │────►│ Document     │
  │ onClick()│     │ save()       │
  └──────────┘     └──────────────┘

  問題1: ボタンが Document を直接知っている（密結合）
  問題2: キーボードショートカットでも同じ処理をしたい
  問題3: Undo したいが、どうやって？
  問題4: 操作を記録・再生したいが、仕組みがない
```

### Command パターンによる解決

```
Command パターンの解決:

  ┌──────────┐     ┌──────────────┐     ┌──────────────┐
  │ Button   │────►│  SaveCommand │────►│  Document    │
  │(Invoker) │     │  execute()   │     │  (Receiver)  │
  └──────────┘     │  undo()      │     └──────────────┘
                   │  describe()  │
  ┌──────────┐     └──────────────┘
  │Shortcut  │────►│ (同じCommand) │
  │ Ctrl+S   │
  └──────────┘

  利点:
  ✓ Invoker は Command のみ知る（疎結合）
  ✓ 同じ Command を複数の Invoker から使える
  ✓ undo() で操作を元に戻せる
  ✓ Command 履歴で操作ログを保持
  ✓ Command をシリアライズして永続化できる
```

GoF の定義:

> "Encapsulate a request as an object, thereby letting you parameterize clients with different requests, queue or log requests, and support undoable operations."
>
> -- Design Patterns: Elements of Reusable Object-Oriented Software (1994)

Command パターンの本質は **「操作を第一級オブジェクト（first-class object）にする」** ことです。操作がオブジェクトになることで、操作を変数に代入し、配列に格納し、引数として渡し、シリアライズして保存し、ネットワーク越しに送信できるようになります。これは関数型プログラミングにおける「関数が第一級市民」の概念と通じるものです。

---

## 1. Command パターンの構造

```
Command パターンの構成要素（GoF）:

  ┌──────────────┐
  │    Client    │  コマンドを生成し、Receiver を設定
  └──────┬───────┘
         │ creates
         ▼
  ┌──────────────┐     ┌──────────────────────┐
  │   Invoker    │────►│   Command (interface) │
  │              │     │                       │
  │ + setCommand │     │ + execute(): void     │
  │ + invoke()   │     │ + undo(): void        │
  │              │     │ + describe(): string  │
  └──────────────┘     └───────────┬───────────┘
         │                         │
         │                    ┌────┴─────┐
         │              ┌─────┴────┐ ┌───┴──────────┐
         │              │ConcreteA │ │ ConcreteB    │
         │              │Command   │ │ Command      │
         │              │          │ │              │
         │              │-receiver │ │ -receiver    │
         │              │+execute()│ │ +execute()   │
         │              │+undo()   │ │ +undo()      │
         │              └────┬─────┘ └──────┬───────┘
         │                   │              │
         │                   ▼              ▼
         │              ┌──────────────────────┐
         │              │     Receiver         │
         │              │ (実際のビジネスロジック)  │
         │              │                      │
         │              │ + action1()          │
         │              │ + action2()          │
         └─────────────►│                      │
           直接呼ばない    └──────────────────────┘

  ┌──────────────┐
  │   History    │  コマンド履歴の管理
  │              │  Undo/Redo スタック
  │ + push(cmd)  │
  │ + undo()     │
  │ + redo()     │
  └──────────────┘

各役割:
  Client   : ConcreteCommand を生成し Receiver を注入
  Invoker  : Command の execute() を呼ぶ（何をするかは知らない）
  Command  : execute/undo のインターフェースを定義
  Receiver : 実際のビジネスロジックを持つ
  History  : 実行済みコマンドの履歴を保持（Undo/Redo 用）
```

---

## 2. 基本実装 -- テキストエディタ

### コード例 1: 完全なテキストエディタ Command

```typescript
// command.ts -- Command パターンの基本構造

// ============================
// Command インターフェース
// ============================
interface Command {
  execute(): void;
  undo(): void;
  describe(): string;
  /** Undo 不可能な操作かどうかを示すフラグ */
  readonly isUndoable: boolean;
}

// ============================
// Receiver: テキストエディタ
// ============================
class TextEditor {
  private content: string = '';
  private cursorPosition: number = 0;
  private selectionStart: number = -1;
  private selectionEnd: number = -1;

  getContent(): string {
    return this.content;
  }

  getCursorPosition(): number {
    return this.cursorPosition;
  }

  setCursorPosition(pos: number): void {
    this.cursorPosition = Math.max(0, Math.min(pos, this.content.length));
  }

  setSelection(start: number, end: number): void {
    this.selectionStart = start;
    this.selectionEnd = end;
  }

  getSelection(): { start: number; end: number } | null {
    if (this.selectionStart < 0) return null;
    return { start: this.selectionStart, end: this.selectionEnd };
  }

  clearSelection(): void {
    this.selectionStart = -1;
    this.selectionEnd = -1;
  }

  insertAt(position: number, text: string): void {
    this.content =
      this.content.slice(0, position) + text + this.content.slice(position);
    this.cursorPosition = position + text.length;
  }

  deleteRange(start: number, end: number): string {
    const deleted = this.content.slice(start, end);
    this.content = this.content.slice(0, start) + this.content.slice(end);
    this.cursorPosition = start;
    return deleted;
  }

  replaceRange(start: number, end: number, newText: string): string {
    const deleted = this.content.slice(start, end);
    this.content =
      this.content.slice(0, start) + newText + this.content.slice(end);
    this.cursorPosition = start + newText.length;
    return deleted;
  }

  getLength(): number {
    return this.content.length;
  }
}

// ============================
// Concrete Command: テキスト挿入
// ============================
class InsertTextCommand implements Command {
  readonly isUndoable = true;

  constructor(
    private editor: TextEditor,
    private position: number,
    private text: string
  ) {}

  execute(): void {
    this.editor.insertAt(this.position, this.text);
  }

  undo(): void {
    this.editor.deleteRange(this.position, this.position + this.text.length);
  }

  describe(): string {
    return `Insert "${this.text}" at position ${this.position}`;
  }
}

// ============================
// Concrete Command: テキスト削除
// ============================
class DeleteTextCommand implements Command {
  readonly isUndoable = true;
  private deletedText: string = '';

  constructor(
    private editor: TextEditor,
    private start: number,
    private end: number
  ) {}

  execute(): void {
    this.deletedText = this.editor.deleteRange(this.start, this.end);
  }

  undo(): void {
    this.editor.insertAt(this.start, this.deletedText);
  }

  describe(): string {
    return `Delete "${this.deletedText}" from ${this.start} to ${this.end}`;
  }
}

// ============================
// Concrete Command: テキスト置換
// ============================
class ReplaceTextCommand implements Command {
  readonly isUndoable = true;
  private originalText: string = '';

  constructor(
    private editor: TextEditor,
    private start: number,
    private end: number,
    private newText: string
  ) {}

  execute(): void {
    this.originalText = this.editor.replaceRange(
      this.start, this.end, this.newText
    );
  }

  undo(): void {
    this.editor.replaceRange(
      this.start, this.start + this.newText.length, this.originalText
    );
  }

  describe(): string {
    return `Replace "${this.originalText}" with "${this.newText}"`;
  }
}

// ============================
// 使用例
// ============================
const editor = new TextEditor();

const cmd1 = new InsertTextCommand(editor, 0, 'Hello');
cmd1.execute();
console.log(editor.getContent()); // "Hello"

const cmd2 = new InsertTextCommand(editor, 5, ' World');
cmd2.execute();
console.log(editor.getContent()); // "Hello World"

const cmd3 = new ReplaceTextCommand(editor, 0, 5, 'Hi');
cmd3.execute();
console.log(editor.getContent()); // "Hi World"

cmd3.undo();
console.log(editor.getContent()); // "Hello World"

cmd2.undo();
console.log(editor.getContent()); // "Hello"
```

---

## 3. Undo/Redo マネージャー

### コード例 2: 高機能 Undo/Redo マネージャー

```typescript
// undo-redo-manager.ts -- Undo/Redo の管理

// ============================
// イベント通知付き UndoRedoManager
// ============================
type HistoryEvent =
  | { type: 'execute'; command: Command }
  | { type: 'undo'; command: Command }
  | { type: 'redo'; command: Command }
  | { type: 'clear' };

type HistoryListener = (event: HistoryEvent) => void;

class UndoRedoManager {
  private undoStack: Command[] = [];
  private redoStack: Command[] = [];
  private readonly maxHistory: number;
  private listeners: HistoryListener[] = [];
  private batchLevel: number = 0;
  private batchCommands: Command[] = [];

  constructor(maxHistory: number = 100) {
    this.maxHistory = maxHistory;
  }

  /** コマンドを実行し、Undo スタックに追加 */
  execute(command: Command): void {
    command.execute();

    if (this.batchLevel > 0) {
      // バッチモード中はバッチに蓄積
      this.batchCommands.push(command);
      return;
    }

    this.undoStack.push(command);

    // 新しいコマンド実行時は Redo スタックをクリア
    // (分岐した履歴は破棄される)
    this.redoStack = [];

    // 履歴の上限を超えたら古いものを削除
    if (this.undoStack.length > this.maxHistory) {
      this.undoStack.shift();
    }

    this.notify({ type: 'execute', command });
  }

  /** 直前のコマンドを取り消す */
  undo(): boolean {
    const command = this.undoStack.pop();
    if (!command) return false;

    command.undo();
    this.redoStack.push(command);
    this.notify({ type: 'undo', command });
    return true;
  }

  /** 取り消したコマンドをやり直す */
  redo(): boolean {
    const command = this.redoStack.pop();
    if (!command) return false;

    command.execute();
    this.undoStack.push(command);
    this.notify({ type: 'redo', command });
    return true;
  }

  /** バッチ操作の開始（複数操作を1つの Undo 単位にまとめる） */
  beginBatch(): void {
    this.batchLevel++;
    if (this.batchLevel === 1) {
      this.batchCommands = [];
    }
  }

  /** バッチ操作の終了 */
  endBatch(description?: string): void {
    this.batchLevel--;
    if (this.batchLevel === 0 && this.batchCommands.length > 0) {
      const macro = new MacroCommand(this.batchCommands, description);
      this.undoStack.push(macro);
      this.redoStack = [];

      if (this.undoStack.length > this.maxHistory) {
        this.undoStack.shift();
      }
      this.notify({ type: 'execute', command: macro });
    }
  }

  canUndo(): boolean {
    return this.undoStack.length > 0;
  }

  canRedo(): boolean {
    return this.redoStack.length > 0;
  }

  getHistory(): string[] {
    return this.undoStack.map(cmd => cmd.describe());
  }

  getUndoCount(): number {
    return this.undoStack.length;
  }

  getRedoCount(): number {
    return this.redoStack.length;
  }

  /** 全履歴をクリア */
  clear(): void {
    this.undoStack = [];
    this.redoStack = [];
    this.notify({ type: 'clear' });
  }

  /** 履歴変更のリスナーを追加 */
  subscribe(listener: HistoryListener): () => void {
    this.listeners.push(listener);
    return () => {
      this.listeners = this.listeners.filter(l => l !== listener);
    };
  }

  private notify(event: HistoryEvent): void {
    for (const listener of this.listeners) {
      listener(event);
    }
  }
}

// ============================
// 使用例
// ============================
const editor2 = new TextEditor();
const manager = new UndoRedoManager();

// イベント監視
manager.subscribe(event => {
  console.log(`[History] ${event.type}: ${
    event.type !== 'clear' ? event.command.describe() : 'all'
  }`);
});

manager.execute(new InsertTextCommand(editor2, 0, 'Hello'));
// [History] execute: Insert "Hello" at position 0
console.log(editor2.getContent()); // "Hello"

manager.execute(new InsertTextCommand(editor2, 5, ' World'));
// [History] execute: Insert " World" at position 5
console.log(editor2.getContent()); // "Hello World"

manager.undo();
// [History] undo: Insert " World" at position 5
console.log(editor2.getContent()); // "Hello"

manager.redo();
// [History] redo: Insert " World" at position 5
console.log(editor2.getContent()); // "Hello World"

// バッチ操作: 「検索と置換」を1つの Undo 単位に
manager.beginBatch();
manager.execute(new DeleteTextCommand(editor2, 0, 5));
manager.execute(new InsertTextCommand(editor2, 0, 'Hi'));
manager.endBatch('Replace "Hello" with "Hi"');

console.log(editor2.getContent()); // "Hi World"
manager.undo(); // バッチ全体が1回の Undo で元に戻る
console.log(editor2.getContent()); // "Hello World"
```

```
Undo/Redo のスタック操作の詳細:

  execute("Hello")  execute(" World")   undo()           redo()
  ┌─────────┐      ┌─────────┐       ┌─────────┐      ┌─────────┐
  │ Undo    │      │ Undo    │       │ Undo    │      │ Undo    │
  │┌───────┐│      │┌───────┐│       │┌───────┐│      │┌───────┐│
  ││"Hello"││      ││"World"││       ││"Hello"││      ││"World"││
  │└───────┘│      │┌───────┐│       │└───────┘│      │┌───────┐│
  │         │      ││"Hello"││       │         │      ││"Hello"││
  │         │      │└───────┘│       │         │      │└───────┘│
  ├─────────┤      ├─────────┤       ├─────────┤      ├─────────┤
  │ Redo    │      │ Redo    │       │ Redo    │      │ Redo    │
  │ (空)    │      │ (空)    │       │┌───────┐│      │ (空)    │
  │         │      │         │       ││"World"││      │         │
  │         │      │         │       │└───────┘│      │         │
  └─────────┘      └─────────┘       └─────────┘      └─────────┘

  ★ 重要: undo() 後に新しい execute() を行うと、
    Redo スタックは全てクリアされる（分岐履歴は失われる）

  分岐が発生する場面:
  execute(A) → execute(B) → undo() → execute(C)
                                      ↑ この時点で B は Redo 不可能に

  ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐
  │Undo: │    │Undo: │    │Undo: │    │Undo: │
  │  [A] │    │[A,B] │    │  [A] │    │[A,C] │
  │      │    │      │    │      │    │      │
  │Redo: │    │Redo: │    │Redo: │    │Redo: │
  │  []  │    │  []  │    │  [B] │    │  []  │ ← B は消えた
  └──────┘    └──────┘    └──────┘    └──────┘
```

---

## 4. マクロコマンド（Composite Command）

### コード例 3: マクロの記録と再生

```typescript
// macro-command.ts -- 複数コマンドの合成（Composite パターンとの組合せ）

// ============================
// MacroCommand: 複数コマンドを1つにまとめる
// ============================
class MacroCommand implements Command {
  readonly isUndoable = true;
  private commands: Command[];
  private label: string;

  constructor(commands: Command[] = [], label?: string) {
    this.commands = [...commands];
    this.label = label ?? 'Macro';
  }

  add(command: Command): this {
    this.commands.push(command);
    return this;
  }

  execute(): void {
    for (const command of this.commands) {
      command.execute();
    }
  }

  undo(): void {
    // 逆順で Undo（LIFO: 最後に実行したものから戻す）
    for (let i = this.commands.length - 1; i >= 0; i--) {
      this.commands[i].undo();
    }
  }

  describe(): string {
    if (this.label !== 'Macro') return this.label;
    return `Macro [${this.commands.map(c => c.describe()).join(', ')}]`;
  }
}

// ============================
// MacroRecorder: ユーザー操作の記録・再生
// ============================
class MacroRecorder {
  private recording: boolean = false;
  private commands: Command[] = [];
  private macros: Map<string, MacroCommand> = new Map();

  startRecording(): void {
    this.recording = true;
    this.commands = [];
    console.log('Recording started...');
  }

  stopRecording(name: string): MacroCommand {
    this.recording = false;
    const macro = new MacroCommand([...this.commands], name);
    this.macros.set(name, macro);
    console.log(`Macro "${name}" saved (${this.commands.length} commands)`);
    return macro;
  }

  recordCommand(command: Command): void {
    if (this.recording) {
      this.commands.push(command);
    }
  }

  isRecording(): boolean {
    return this.recording;
  }

  getMacro(name: string): MacroCommand | undefined {
    return this.macros.get(name);
  }

  listMacros(): string[] {
    return [...this.macros.keys()];
  }
}

// ============================
// 使用例: テキスト整形マクロ
// ============================
const editorMacro = new TextEditor();
const managerMacro = new UndoRedoManager();
const recorder = new MacroRecorder();

// マクロの記録開始
recorder.startRecording();

// 操作を記録しつつ実行
const m1 = new InsertTextCommand(editorMacro, 0, '# ');
recorder.recordCommand(m1);
managerMacro.execute(m1);

const m2 = new InsertTextCommand(
  editorMacro, editorMacro.getLength(), '\n---\n'
);
recorder.recordCommand(m2);
managerMacro.execute(m2);

// マクロを名前付きで保存
const formatMacro = recorder.stopRecording('heading-format');

console.log(editorMacro.getContent());
// "# \n---\n"

// マクロを別のテキストに再適用
// ※ 新しい editor でコマンドを再生成する必要がある点に注意
```

---

## 5. 非同期コマンドキュー

### コード例 4: リトライ・ロールバック付き非同期キュー

```typescript
// command-queue.ts -- 非同期コマンドの順序実行

// ============================
// AsyncCommand インターフェース
// ============================
interface AsyncCommand {
  execute(): Promise<void>;
  undo(): Promise<void>;
  describe(): string;
  /** リトライ可能かどうか */
  canRetry(): boolean;
  /** 最大リトライ回数 */
  maxRetries: number;
}

// ============================
// CommandQueue: 順序実行キュー
// ============================
type QueueEvent =
  | { type: 'enqueue'; command: AsyncCommand }
  | { type: 'execute'; command: AsyncCommand }
  | { type: 'retry'; command: AsyncCommand; attempt: number }
  | { type: 'fail'; command: AsyncCommand; error: Error }
  | { type: 'rollback'; command: AsyncCommand }
  | { type: 'complete' };

class CommandQueue {
  private queue: Array<{ command: AsyncCommand; retries: number }> = [];
  private processing: boolean = false;
  private executed: AsyncCommand[] = [];
  private listeners: Array<(event: QueueEvent) => void> = [];

  enqueue(command: AsyncCommand): void {
    this.queue.push({ command, retries: 0 });
    this.notify({ type: 'enqueue', command });
    this.processNext();
  }

  enqueueAll(commands: AsyncCommand[]): void {
    for (const command of commands) {
      this.queue.push({ command, retries: 0 });
      this.notify({ type: 'enqueue', command });
    }
    this.processNext();
  }

  private async processNext(): Promise<void> {
    if (this.processing || this.queue.length === 0) return;

    this.processing = true;
    const entry = this.queue.shift()!;

    try {
      await entry.command.execute();
      this.executed.push(entry.command);
      this.notify({ type: 'execute', command: entry.command });
    } catch (error) {
      const err = error instanceof Error ? error : new Error(String(error));
      this.notify({ type: 'fail', command: entry.command, error: err });

      if (entry.command.canRetry() && entry.retries < entry.command.maxRetries) {
        // リトライ: キューの先頭に戻す
        entry.retries++;
        this.notify({
          type: 'retry',
          command: entry.command,
          attempt: entry.retries,
        });
        this.queue.unshift(entry);
      } else {
        // 失敗時のロールバック
        await this.rollback();
      }
    } finally {
      this.processing = false;
      if (this.queue.length > 0) {
        this.processNext();
      } else {
        this.notify({ type: 'complete' });
      }
    }
  }

  private async rollback(): Promise<void> {
    console.log('Rolling back executed commands...');
    while (this.executed.length > 0) {
      const cmd = this.executed.pop()!;
      try {
        await cmd.undo();
        this.notify({ type: 'rollback', command: cmd });
      } catch (err) {
        console.error(`Rollback failed for: ${cmd.describe()}`, err);
        // 補償トランザクションのロールバック失敗は
        // 人間の介入が必要な場面 → アラート
      }
    }
    // ロールバック後はキューも全てクリア
    this.queue = [];
  }

  subscribe(listener: (event: QueueEvent) => void): () => void {
    this.listeners.push(listener);
    return () => {
      this.listeners = this.listeners.filter(l => l !== listener);
    };
  }

  private notify(event: QueueEvent): void {
    for (const listener of this.listeners) {
      listener(event);
    }
  }
}

// ============================
// 使用例: API コール のトランザクション
// ============================
class CreateUserCommand implements AsyncCommand {
  maxRetries = 2;
  private userId: string | null = null;

  constructor(private userData: { name: string; email: string }) {}

  async execute(): Promise<void> {
    // API 呼び出し (シミュレーション)
    console.log(`Creating user: ${this.userData.name}`);
    this.userId = `user_${Date.now()}`;
  }

  async undo(): Promise<void> {
    if (this.userId) {
      console.log(`Deleting user: ${this.userId}`);
      this.userId = null;
    }
  }

  canRetry(): boolean { return true; }
  describe(): string { return `CreateUser(${this.userData.name})`; }
}

class SendWelcomeEmailCommand implements AsyncCommand {
  maxRetries = 3;

  constructor(private email: string) {}

  async execute(): Promise<void> {
    console.log(`Sending welcome email to: ${this.email}`);
    // ネットワークエラーのシミュレーション
    if (Math.random() < 0.3) {
      throw new Error('SMTP connection timeout');
    }
  }

  async undo(): Promise<void> {
    console.log(`Email to ${this.email} cannot be unsent (no-op)`);
  }

  canRetry(): boolean { return true; }
  describe(): string { return `SendEmail(${this.email})`; }
}

// キューの使用
const queue = new CommandQueue();

queue.subscribe(event => {
  switch (event.type) {
    case 'execute':
      console.log(`Executed: ${event.command.describe()}`);
      break;
    case 'retry':
      console.log(`Retrying: ${event.command.describe()} (attempt ${event.attempt})`);
      break;
    case 'rollback':
      console.log(`Rolled back: ${event.command.describe()}`);
      break;
  }
});

queue.enqueueAll([
  new CreateUserCommand({ name: 'Alice', email: 'alice@example.com' }),
  new SendWelcomeEmailCommand('alice@example.com'),
]);
```

---

## 6. Python での Command パターン

### コード例 5: Python Protocol ベースの Command

```python
# command_python.py -- Python での Command パターン実装
from __future__ import annotations
from typing import Protocol, runtime_checkable
from dataclasses import dataclass, field
from datetime import datetime


# ============================
# Command プロトコル
# ============================
@runtime_checkable
class Command(Protocol):
    def execute(self) -> None: ...
    def undo(self) -> None: ...
    def describe(self) -> str: ...


# ============================
# Receiver: スプレッドシート
# ============================
class Spreadsheet:
    def __init__(self, rows: int = 100, cols: int = 26):
        self._cells: dict[tuple[int, int], str | float] = {}
        self._rows = rows
        self._cols = cols

    def get_cell(self, row: int, col: int) -> str | float | None:
        return self._cells.get((row, col))

    def set_cell(self, row: int, col: int, value: str | float) -> None:
        self._cells[(row, col)] = value

    def delete_cell(self, row: int, col: int) -> str | float | None:
        return self._cells.pop((row, col), None)

    def get_range(
        self, r1: int, c1: int, r2: int, c2: int
    ) -> dict[tuple[int, int], str | float]:
        return {
            (r, c): v
            for (r, c), v in self._cells.items()
            if r1 <= r <= r2 and c1 <= c <= c2
        }


# ============================
# Concrete Command: セル値の設定
# ============================
@dataclass
class SetCellCommand:
    sheet: Spreadsheet
    row: int
    col: int
    new_value: str | float
    _old_value: str | float | None = field(default=None, init=False)

    def execute(self) -> None:
        self._old_value = self.sheet.get_cell(self.row, self.col)
        self.sheet.set_cell(self.row, self.col, self.new_value)

    def undo(self) -> None:
        if self._old_value is None:
            self.sheet.delete_cell(self.row, self.col)
        else:
            self.sheet.set_cell(self.row, self.col, self._old_value)

    def describe(self) -> str:
        col_letter = chr(ord("A") + self.col)
        return f"Set {col_letter}{self.row + 1} = {self.new_value}"


# ============================
# Concrete Command: 範囲の一括クリア
# ============================
@dataclass
class ClearRangeCommand:
    sheet: Spreadsheet
    r1: int
    c1: int
    r2: int
    c2: int
    _saved: dict[tuple[int, int], str | float] = field(
        default_factory=dict, init=False
    )

    def execute(self) -> None:
        self._saved = self.sheet.get_range(self.r1, self.c1, self.r2, self.c2)
        for (r, c) in self._saved:
            self.sheet.delete_cell(r, c)

    def undo(self) -> None:
        for (r, c), v in self._saved.items():
            self.sheet.set_cell(r, c, v)

    def describe(self) -> str:
        c1_letter = chr(ord("A") + self.c1)
        c2_letter = chr(ord("A") + self.c2)
        return f"Clear {c1_letter}{self.r1 + 1}:{c2_letter}{self.r2 + 1}"


# ============================
# UndoRedoManager (Python 版)
# ============================
@dataclass
class UndoRedoManager:
    max_history: int = 100
    _undo_stack: list[Command] = field(default_factory=list, init=False)
    _redo_stack: list[Command] = field(default_factory=list, init=False)

    def execute(self, command: Command) -> None:
        command.execute()
        self._undo_stack.append(command)
        self._redo_stack.clear()
        if len(self._undo_stack) > self.max_history:
            self._undo_stack.pop(0)

    def undo(self) -> bool:
        if not self._undo_stack:
            return False
        cmd = self._undo_stack.pop()
        cmd.undo()
        self._redo_stack.append(cmd)
        return True

    def redo(self) -> bool:
        if not self._redo_stack:
            return False
        cmd = self._redo_stack.pop()
        cmd.execute()
        self._undo_stack.append(cmd)
        return True

    def history(self) -> list[str]:
        return [cmd.describe() for cmd in self._undo_stack]


# ============================
# 使用例
# ============================
if __name__ == "__main__":
    sheet = Spreadsheet()
    mgr = UndoRedoManager()

    mgr.execute(SetCellCommand(sheet, 0, 0, "Name"))
    mgr.execute(SetCellCommand(sheet, 0, 1, "Age"))
    mgr.execute(SetCellCommand(sheet, 1, 0, "Alice"))
    mgr.execute(SetCellCommand(sheet, 1, 1, 30))

    print(f"A1={sheet.get_cell(0, 0)}, B1={sheet.get_cell(0, 1)}")
    # A1=Name, B1=Age
    print(f"A2={sheet.get_cell(1, 0)}, B2={sheet.get_cell(1, 1)}")
    # A2=Alice, B2=30

    mgr.undo()  # B2 = 30 を取り消し
    print(f"B2 after undo: {sheet.get_cell(1, 1)}")
    # B2 after undo: None

    mgr.redo()  # B2 = 30 を復元
    print(f"B2 after redo: {sheet.get_cell(1, 1)}")
    # B2 after redo: 30

    print("History:", mgr.history())
    # History: ['Set A1 = Name', 'Set B1 = Age', 'Set A2 = Alice', 'Set B2 = 30']
```

---

## 7. 関数型 Command -- クロージャベース

### コード例 6: クロージャと高階関数による Command

```typescript
// functional-command.ts -- 関数型アプローチの Command パターン

// ============================
// 関数型 Command の型定義
// ============================
interface FunctionalCommand {
  execute: () => void;
  undo: () => void;
  describe: () => string;
}

// ============================
// Command ファクトリ
// ============================
function createInsertCommand(
  editor: TextEditor,
  position: number,
  text: string
): FunctionalCommand {
  return {
    execute: () => editor.insertAt(position, text),
    undo: () => editor.deleteRange(position, position + text.length),
    describe: () => `Insert "${text}" at ${position}`,
  };
}

function createDeleteCommand(
  editor: TextEditor,
  start: number,
  end: number
): FunctionalCommand {
  let deleted = '';
  return {
    execute: () => { deleted = editor.deleteRange(start, end); },
    undo: () => editor.insertAt(start, deleted),
    describe: () => `Delete [${start}:${end}]`,
  };
}

// ============================
// 汎用 Command ファクトリ（任意の操作をコマンドに）
// ============================
function makeCommand(
  doFn: () => void,
  undoFn: () => void,
  description: string
): FunctionalCommand {
  return {
    execute: doFn,
    undo: undoFn,
    describe: () => description,
  };
}

// ============================
// 関数型 UndoRedoManager
// ============================
function createUndoRedoManager() {
  const undoStack: FunctionalCommand[] = [];
  const redoStack: FunctionalCommand[] = [];

  return {
    execute(cmd: FunctionalCommand): void {
      cmd.execute();
      undoStack.push(cmd);
      redoStack.length = 0;
    },
    undo(): boolean {
      const cmd = undoStack.pop();
      if (!cmd) return false;
      cmd.undo();
      redoStack.push(cmd);
      return true;
    },
    redo(): boolean {
      const cmd = redoStack.pop();
      if (!cmd) return false;
      cmd.execute();
      undoStack.push(cmd);
      return true;
    },
    history(): string[] {
      return undoStack.map(c => c.describe());
    },
  };
}

// ============================
// 使用例: クラス不要の軽量 Command
// ============================
const ed = new TextEditor();
const mgr = createUndoRedoManager();

mgr.execute(createInsertCommand(ed, 0, 'Hello'));
mgr.execute(createInsertCommand(ed, 5, ' World'));
console.log(ed.getContent()); // "Hello World"

mgr.undo();
console.log(ed.getContent()); // "Hello"

// 任意の操作もコマンドに変換可能
let logBuffer: string[] = [];
mgr.execute(makeCommand(
  () => logBuffer.push('entry'),
  () => logBuffer.pop(),
  'Add log entry'
));
```

---

## 8. 型安全な Command バス

### コード例 7: TypeScript のジェネリクスを活用した Command バス

```typescript
// command-bus.ts -- 型安全な Command ディスパッチ

// ============================
// Command と Handler の型定義
// ============================
interface TypedCommand<TName extends string, TPayload, TResult> {
  readonly type: TName;
  readonly payload: TPayload;
  // TResult は型レベルの情報のみ（実行時に使用しない）
  readonly __result?: TResult;
}

type CommandHandler<C extends TypedCommand<string, unknown, unknown>> =
  (command: C) => C extends TypedCommand<string, unknown, infer R> ? R : never;

// ============================
// Command バス
// ============================
class CommandBus {
  private handlers = new Map<string, CommandHandler<any>>();
  private middleware: Array<(
    command: TypedCommand<string, unknown, unknown>,
    next: () => unknown
  ) => unknown> = [];

  /** ハンドラの登録 */
  register<C extends TypedCommand<string, unknown, unknown>>(
    type: C['type'],
    handler: CommandHandler<C>
  ): void {
    this.handlers.set(type, handler);
  }

  /** ミドルウェアの追加（ロギング、認証等） */
  use(
    middleware: (
      command: TypedCommand<string, unknown, unknown>,
      next: () => unknown
    ) => unknown
  ): void {
    this.middleware.push(middleware);
  }

  /** コマンドのディスパッチ */
  dispatch<C extends TypedCommand<string, unknown, unknown>>(
    command: C
  ): C extends TypedCommand<string, unknown, infer R> ? R : never {
    const handler = this.handlers.get(command.type);
    if (!handler) {
      throw new Error(`No handler registered for command: ${command.type}`);
    }

    // ミドルウェアチェーンの構築
    const chain = this.middleware.reduceRight(
      (next, mw) => () => mw(command, next),
      () => handler(command)
    );

    return chain() as any;
  }
}

// ============================
// 具体的なコマンド定義
// ============================
type CreateOrderCommand = TypedCommand<
  'CreateOrder',
  { userId: string; items: Array<{ sku: string; qty: number }> },
  { orderId: string }
>;

type CancelOrderCommand = TypedCommand<
  'CancelOrder',
  { orderId: string; reason: string },
  { refundAmount: number }
>;

// ============================
// 使用例
// ============================
const bus = new CommandBus();

// ロギングミドルウェア
bus.use((command, next) => {
  console.log(`[CommandBus] Dispatching: ${command.type}`, command.payload);
  const start = performance.now();
  const result = next();
  const elapsed = performance.now() - start;
  console.log(`[CommandBus] Completed: ${command.type} (${elapsed.toFixed(2)}ms)`);
  return result;
});

// ハンドラ登録
bus.register<CreateOrderCommand>('CreateOrder', (cmd) => {
  console.log(`Creating order for user: ${cmd.payload.userId}`);
  return { orderId: `ORD-${Date.now()}` };
});

bus.register<CancelOrderCommand>('CancelOrder', (cmd) => {
  console.log(`Cancelling order: ${cmd.payload.orderId}`);
  return { refundAmount: 1500 };
});

// ディスパッチ（型安全: 戻り値が自動推論される）
const result = bus.dispatch<CreateOrderCommand>({
  type: 'CreateOrder',
  payload: { userId: 'u1', items: [{ sku: 'SKU-001', qty: 2 }] },
});
console.log(result.orderId); // 型安全: string
```

---

## 9. 深掘り: Command パターンの内部設計判断

### Undo 実装の3方式比較

```
方式1: Command 履歴 (Command Pattern)
  各コマンドが undo() を持ち、逆操作を実行
  ┌─────┐  ┌─────┐  ┌─────┐
  │ C1  │→│ C2  │→│ C3  │  ← Undo スタック
  │undo │  │undo │  │undo │
  └─────┘  └─────┘  └─────┘
  メモリ: 差分のみ保持（省メモリ）
  速度:   O(1) per undo

方式2: Memento (スナップショット)
  操作前の全状態を保存
  ┌───────┐  ┌───────┐  ┌───────┐
  │State 0│  │State 1│  │State 2│  ← スナップショット
  │(全体) │  │(全体) │  │(全体) │
  └───────┘  └───────┘  └───────┘
  メモリ: 状態サイズ × 履歴数（大量消費）
  速度:   O(1) per undo（スワップのみ）

方式3: Event Sourcing
  全イベントを記録し、再生で状態を再構築
  ┌─────┐  ┌─────┐  ┌─────┐
  │ E1  │→│ E2  │→│ E3  │  ← イベントログ
  │(追記)│  │(追記)│  │(追記)│
  └─────┘  └─────┘  └─────┘
  メモリ: イベントサイズ × 件数
  速度:   O(N) per undo（先頭から再生）
  ※スナップショットで高速化可能
```

| 比較項目 | Command 履歴 | Memento | Event Sourcing |
|---------|-------------|---------|----------------|
| メモリ使用 | 低い（差分のみ） | 高い（全状態 x N） | 中（イベント列） |
| 実装の複雑さ | 中 | 低い | 高い |
| 部分的 Undo | 困難 | 不可 | 可能 |
| 永続化 | 容易 | 容易 | 容易 |
| 監査証跡 | 操作ログとして利用可 | 不向き | 最適 |
| デバッグ容易性 | 中 | 高い（状態を直接確認） | 高い（リプレイ可能） |
| 適用場面 | エディタ、操作記録 | ゲームのセーブ | ドメインイベント記録 |

### Command の粒度設計

Command の粒度（1つのコマンドがカバーする操作の範囲）は、ユーザー体験に直結する重要な設計判断です。

```
粒度が細かすぎる場合:
  1文字ずつ Command → Undo が1文字ずつ戻る（UX が悪い）
  "Hello" = 5 Commands → 5回 Undo して初めて消える

粒度が粗すぎる場合:
  1ページ全体で1 Command → 小さな変更の Undo で大量の変更が消える

理想的な粒度:
  ┌────────────────────────────────────────────┐
  │ ユーザーの「意図」に対応する単位             │
  │                                            │
  │ 例: テキストエディタ                         │
  │   - 単語の入力 → 1 Command                  │
  │   - Backspace で単語削除 → 1 Command        │
  │   - 検索と置換 → 1 Command（MacroCommand）  │
  │   - 書式変更 → 1 Command                    │
  │                                            │
  │ 例: グラフィックエディタ                      │
  │   - 図形の移動 → 1 Command                  │
  │   - 複数選択して移動 → 1 Command             │
  │   - 色変更 → 1 Command                      │
  └────────────────────────────────────────────┘

バッファリングによる粒度制御:
  入力開始 → タイマー開始
  入力中   → バッファに蓄積
  500ms 無入力 or Enter → Command 確定

  "H" "e" "l" "l" "o" [500ms] → InsertCommand("Hello")
```

---

## 10. 実世界での Command パターン

### Redux / Flux アーキテクチャ

```
Redux は Command パターンの変形:

  Action     = Command（type + payload で操作を記述）
  Reducer    = execute()（状態を更新）
  Store      = Invoker + History
  Middleware = Command の前後にフック

  dispatch({ type: 'ADD_TODO', payload: { text: '...' } })
  ↓
  [Middleware] → [Reducer] → [New State]

  Redux DevTools = Undo/Redo マネージャー
    - 全 Action の履歴を保持
    - タイムトラベルデバッグ（任意の時点に戻る）
    - Action のリプレイ
```

### Git の操作モデル

```
Git は Command パターンの具体例:

  git commit   = execute()    新しいスナップショットを記録
  git revert   = undo()       補償コミットで変更を打ち消す
  git cherry-pick = コマンドの再適用
  git reflog   = Command 履歴

  ※ git reset --hard は「Memento 方式」（状態を直接復元）
  ※ git revert は「Command 方式」（逆操作を追加）
```

### ゲームのリプレイシステム

```
ゲームリプレイ = Command のシリアライズと再生:

  記録フェーズ:
  Frame 1: [MoveCommand(player, {x:1, y:0})]
  Frame 2: [AttackCommand(player, target)]
  Frame 3: [MoveCommand(player, {x:0, y:1}), UseItemCommand(player, potion)]
  ...

  再生フェーズ:
  同じ初期状態 + 同じ Command 列 → 同じ結果（決定的実行）

  圧縮: 同種の連続コマンドをマージ
  Frame 1-10: MoveCommand(player, {x:10, y:0})  ← 10フレーム分をまとめ
```

---

## 11. 比較表

### Command vs 他のパターン

| 特性 | Command | Strategy | Observer | Memento |
|------|---------|----------|----------|---------|
| 目的 | 操作のカプセル化 | アルゴリズムの切替 | 状態変化の通知 | 状態の保存・復元 |
| Undo/Redo | 対応（逆操作） | 非対応 | 非対応 | 対応（スナップショット） |
| 履歴管理 | 可能 | 不要 | 不要 | 可能 |
| 遅延実行 | 可能 | 即座に実行 | イベント駆動 | 即座に保存 |
| キューイング | 可能 | 不要 | 不要 | 不要 |
| シリアライズ | 容易 | 困難 | 不要 | 容易 |
| メモリ効率 | 高い（差分） | -- | -- | 低い（全状態） |

### Command の実装アプローチ比較

| アプローチ | クラスベース | 関数型（クロージャ） | オブジェクトリテラル |
|-----------|-------------|-------------------|-------------------|
| Undo | execute/undo メソッド | do/undo クロージャ | execute/undo プロパティ |
| 型安全性 | インターフェースで強い | 型定義が必要 | 型定義が必要 |
| シリアライズ | 容易（クラス名 + パラメータ） | 困難（クロージャは直列化不可） | 困難 |
| テスタビリティ | モック容易 | 関数の差し替え容易 | 同左 |
| ボイラープレート | 多い | 少ない | 少ない |
| 適用場面 | 大規模、シリアライズ要 | 軽量、一時的 | 中規模 |

### Command パターンの導入判断

| 判断基準 | Command パターンが有効 | 直接関数呼び出しで十分 |
|---------|---------------------|---------------------|
| Undo/Redo | 必要 | 不要 |
| 操作のログ記録 | 必要（監査、デバッグ） | 不要 |
| 遅延実行 / キューイング | 必要 | 不要 |
| マクロ記録 | 必要 | 不要 |
| シリアライズ / ネットワーク送信 | 必要 | 不要 |
| 操作の種類 | 多い・増える予定 | 少ない・固定 |

---

## 12. アンチパターン

### アンチパターン 1: コマンドの粒度が不適切

```typescript
// ============================
// [NG] 1文字ごとにコマンドを生成
// ============================
// Undo が1文字ずつ戻る、メモリも浪費
manager.execute(new InsertTextCommand(editor, 0, 'H'));
manager.execute(new InsertTextCommand(editor, 1, 'e'));
manager.execute(new InsertTextCommand(editor, 2, 'l'));
manager.execute(new InsertTextCommand(editor, 3, 'l'));
manager.execute(new InsertTextCommand(editor, 4, 'o'));
// Undo 5回で "Hello" → "Hell" → "Hel" → "He" → "H" → ""
// ユーザーの期待: Undo 1回で "Hello" が消える

// ============================
// [OK] 意味のある単位でコマンドを生成
// ============================
// 入力をバッファリングし、一定時間の無入力でコマンドを確定
class BufferedTextInput {
  private buffer: string = '';
  private bufferStart: number = 0;
  private timer: ReturnType<typeof setTimeout> | null = null;

  constructor(
    private editor: TextEditor,
    private manager: UndoRedoManager,
    private debounceMs: number = 500
  ) {}

  /** 1文字入力のたびに呼ばれる */
  type(char: string): void {
    if (this.buffer === '') {
      this.bufferStart = this.editor.getCursorPosition();
    }
    // エディタに直接挿入（Command 経由ではない）
    this.editor.insertAt(this.editor.getCursorPosition(), char);
    this.buffer += char;

    // デバウンス: 一定時間入力がなければ Command を確定
    if (this.timer) clearTimeout(this.timer);
    this.timer = setTimeout(() => this.flush(), this.debounceMs);
  }

  /** Enter キーや操作切り替え時にも呼ぶ */
  flush(): void {
    if (this.buffer) {
      // ★ すでに editor には反映済みなので、
      //    undo 用に逆操作を記録するだけの Command を作る
      const text = this.buffer;
      const start = this.bufferStart;
      const cmd: Command = {
        isUndoable: true,
        execute: () => { /* 既に反映済み */ },
        undo: () => this.editor.deleteRange(start, start + text.length),
        describe: () => `Type "${text}"`,
      };
      // execute() は呼ばず、履歴にだけ追加
      this.manager['undoStack'].push(cmd);
      this.buffer = '';
    }
  }
}
```

### アンチパターン 2: Undo 不可能なコマンドの放置

```typescript
// ============================
// [NG] undo() が実装されていない（空のメソッド）
// ============================
class SendEmailCommand implements Command {
  readonly isUndoable = true; // 嘘の宣言!

  execute(): void {
    emailService.send(this.email);
  }

  undo(): void {
    // 何もしない...?? 送信済みメールは取り消せないのに
    // isUndoable = true なので、ユーザーは Undo できると期待する
  }

  describe(): string { return `Send email to ${this.email}`; }
}

// ============================
// [OK 方法1] 補償アクション (Compensating Action) を定義
// ============================
class SendEmailCommandV2 implements Command {
  readonly isUndoable = true;
  private sentId: string | null = null;

  constructor(private email: string) {}

  execute(): void {
    // 即座に送信せず、「送信予約」にする（5分後に実送信）
    this.sentId = emailService.schedule(this.email, { delayMinutes: 5 });
  }

  undo(): void {
    if (this.sentId) {
      // 5分以内なら予約をキャンセル
      const cancelled = emailService.cancelScheduled(this.sentId);
      if (!cancelled) {
        // 既に送信済みの場合は取り消しメールを送信
        emailService.sendCancellation(this.sentId);
      }
    }
  }

  describe(): string { return `Schedule email to ${this.email}`; }
}

// ============================
// [OK 方法2] Undo 不可であることを明示的に宣言
// ============================
class IrreversibleSendEmailCommand implements Command {
  readonly isUndoable = false; // 正直に宣言

  constructor(private email: string) {}

  execute(): void {
    emailService.send(this.email);
  }

  undo(): void {
    throw new Error('Email sending cannot be undone');
  }

  describe(): string { return `Send email to ${this.email} (irreversible)`; }
}

// UndoRedoManager 側でも対応
class SafeUndoRedoManager extends UndoRedoManager {
  override undo(): boolean {
    const lastCmd = this['undoStack'][this['undoStack'].length - 1];
    if (lastCmd && !lastCmd.isUndoable) {
      console.warn(`Cannot undo: ${lastCmd.describe()} is irreversible`);
      return false;
    }
    return super.undo();
  }
}
```

### アンチパターン 3: God Command（巨大コマンド）

```typescript
// ============================
// [NG] 1つのコマンドに複数の責任を詰め込む
// ============================
class ProcessOrderCommand implements Command {
  readonly isUndoable = true;

  execute(): void {
    // 1つの Command に全ビジネスロジックが...
    this.validateOrder();
    this.calculateTax();
    this.applyDiscount();
    this.chargePayment();
    this.updateInventory();
    this.sendConfirmation();
    this.notifyWarehouse();
  }

  undo(): void {
    // 7つの逆操作を正しい順序で...
    // テスト不可能、バグの温床
  }
  // ...
}

// ============================
// [OK] 単一責任の小さな Command に分割し、MacroCommand で合成
// ============================
const processOrder = new MacroCommand([
  new ValidateOrderCommand(order),
  new CalculateTaxCommand(order),
  new ApplyDiscountCommand(order),
  new ChargePaymentCommand(order),
  new UpdateInventoryCommand(order),
  new SendConfirmationCommand(order),
  new NotifyWarehouseCommand(order),
], 'Process Order');

// 利点:
// - 各 Command が独立してテスト可能
// - 途中で失敗した場合、実行済みの Command だけ undo
// - 新しいステップの追加・削除が容易
// - 各 Command の再利用が可能
```

---

## 13. 演習問題

### 演習 1（基礎）: DrawingCanvas のコマンド実装

以下の仕様を満たす描画キャンバスの Command を実装してください。

**仕様:**
- `DrawCircleCommand`: 円を描画（x, y, radius, color）
- `DrawRectCommand`: 四角を描画（x, y, width, height, color）
- `ClearCanvasCommand`: キャンバス全体をクリア
- `UndoRedoManager` と組み合わせて Undo/Redo を実現

```typescript
// ヒント: Canvas（Receiver）の定義
interface Shape {
  type: 'circle' | 'rect';
  id: string;
  // ...shape-specific properties
}

class DrawingCanvas {
  private shapes: Shape[] = [];

  addShape(shape: Shape): void { /* ... */ }
  removeShape(id: string): Shape | undefined { /* ... */ }
  clear(): Shape[] { /* ... */ }
  getShapes(): Shape[] { /* ... */ }
}
```

**期待される出力:**
```
canvas.getShapes() → []
execute(DrawCircleCommand(50, 50, 20, 'red'))
canvas.getShapes() → [{ type: 'circle', id: '...', x: 50, y: 50, radius: 20, color: 'red' }]
execute(DrawRectCommand(10, 10, 100, 50, 'blue'))
canvas.getShapes().length → 2
undo()
canvas.getShapes().length → 1
undo()
canvas.getShapes().length → 0
redo()
canvas.getShapes().length → 1
```

---

### 演習 2（応用）: トランザクション付き API コマンドキュー

以下の仕様を満たすトランザクション実行エンジンを実装してください。

**仕様:**
- 複数の非同期 Command を順序実行
- 途中で失敗した場合、実行済み Command を逆順で全て undo（ロールバック）
- リトライ機能（最大3回）
- 進捗イベントの通知（onProgress コールバック）

```typescript
// ヒント: インターフェース
interface TransactionEngine {
  execute(commands: AsyncCommand[]): Promise<TransactionResult>;
}

interface TransactionResult {
  success: boolean;
  executedCount: number;
  failedCommand?: string;
  rolledBackCount?: number;
}
```

**期待される出力:**
```
正常系:
  execute([CmdA, CmdB, CmdC])
  → { success: true, executedCount: 3 }

異常系（CmdC で失敗、リトライも失敗）:
  execute([CmdA, CmdB, CmdC])
  → Retrying CmdC (attempt 1/3)
  → Retrying CmdC (attempt 2/3)
  → Retrying CmdC (attempt 3/3)
  → Rolling back CmdB
  → Rolling back CmdA
  → { success: false, executedCount: 2, failedCommand: 'CmdC', rolledBackCount: 2 }
```

---

### 演習 3（発展）: シリアライズ可能な Command でリプレイシステムを構築

以下の仕様を満たすリプレイ（操作の記録・再生）システムを実装してください。

**仕様:**
- Command を JSON にシリアライズ/デシリアライズ
- タイムスタンプ付きで操作を記録
- 記録した操作を別の Receiver に対して再生
- 再生速度の制御（1x, 2x, 0.5x）

```typescript
// ヒント: シリアライズ形式
interface SerializedCommand {
  type: string;
  params: Record<string, unknown>;
  timestamp: number;
}

interface ReplayEngine {
  record(command: Command): void;
  serialize(): string;
  deserialize(json: string): void;
  replay(receiver: TextEditor, speed?: number): Promise<void>;
}
```

**期待される出力:**
```
記録:
  record(InsertCommand(0, "Hello"))     // t=0ms
  record(InsertCommand(5, " World"))    // t=500ms
  record(DeleteCommand(5, 11))          // t=1200ms
  serialize() → '[{"type":"insert","params":{"pos":0,"text":"Hello"},"timestamp":0},...]'

再生 (speed=2x):
  t=0ms:   Insert "Hello"     → "Hello"
  t=250ms: Insert " World"    → "Hello World"
  t=600ms: Delete [5:11]      → "Hello"
```

---

## 14. FAQ

### Q1: Command パターンはどのような場面で使うべきですか？

主に次の5つの場面で有効です。

1. **Undo/Redo**: テキストエディタ、グラフィックツール、スプレッドシート。ユーザーが操作を取り消せる必要がある場面。
2. **操作のキューイング**: ジョブキュー、バッチ処理、ネットワーク不安定時のオフラインキュー。
3. **操作のログ記録**: 監査証跡（誰がいつ何をしたか）、デバッグログ、コンプライアンス要件。
4. **マクロ記録**: ユーザー操作の記録と再生、テスト自動化。
5. **トランザクション**: 複数操作のアトミック実行（全て成功 or 全てロールバック）。

単純な関数呼び出し1回で済む場面ではオーバーエンジニアリングになります。「この操作を後から取り消す必要があるか？」「操作の履歴を残す必要があるか？」を判断基準にしてください。

### Q2: Command パターンと関数型プログラミングのクロージャは何が違いますか？

クロージャも「操作をカプセル化」しますが、以下の点で Command パターンとは異なります。

| 比較項目 | Command パターン | クロージャ |
|---------|----------------|-----------|
| undo | 明示的な undo() メソッド | 逆操作を別途用意する必要あり |
| シリアライズ | 可能（クラス名 + パラメータ） | 不可能（クロージャは直列化できない） |
| 検査可能性 | describe() で内容を確認可能 | 内部を外から見れない |
| テスト | 独立してテスト可能 | テストしにくい |

TypeScript では、クラスベースと関数型（クロージャ）のハイブリッドが実用的です。シリアライズが不要な軽量な場面ではクロージャで、永続化やネットワーク送信が必要な場面ではクラスベースを使います。

### Q3: 大量のコマンド履歴によるメモリ消費をどう管理しますか？

4つの戦略があります。

1. **履歴の上限設定**: 最大100〜1000件に制限し、古いものから破棄。ほとんどのエディタはこの方式。
2. **コマンドの圧縮（Coalescing）**: 連続する同種のコマンドをマージ。例: 連続入力の文字を1つの InsertCommand にまとめる。
3. **チェックポイント**: 定期的に全状態をスナップショットし、それ以前の Command 履歴を破棄。undo はチェックポイントまでしか遡れないが、メモリは一定。
4. **遅延読込（Lazy Loading）**: 古い履歴をディスクに退避し、必要時のみ読み込む。IndexedDB や SQLite に保存。

### Q4: Command パターンと Event Sourcing の関係は？

Command パターンと Event Sourcing は密接に関連しますが、目的が異なります。

- **Command**: 「操作の意図」を表す。実行前のもので、拒否される可能性がある。
- **Event**: 「起きた事実」を表す。実行後のもので、不変（immutable）。

Event Sourcing では、Command を受け取り、バリデーション後に Event を生成します。Event を再生すれば任意の時点の状態を復元できます。Command パターンの `execute()` が Event を発行し、`undo()` は補償 Event を発行するという設計が一般的です。

### Q5: React / フロントエンドフレームワークで Command パターンをどう使いますか？

React では以下のパターンが一般的です。

```typescript
// useUndoRedo カスタムフック
function useUndoRedo<T>(initialState: T) {
  const [state, setState] = useState(initialState);
  const undoStack = useRef<T[]>([]);
  const redoStack = useRef<T[]>([]);

  const execute = useCallback((newState: T) => {
    undoStack.current.push(state);
    redoStack.current = [];
    setState(newState);
  }, [state]);

  const undo = useCallback(() => {
    const prev = undoStack.current.pop();
    if (prev !== undefined) {
      redoStack.current.push(state);
      setState(prev);
    }
  }, [state]);

  const redo = useCallback(() => {
    const next = redoStack.current.pop();
    if (next !== undefined) {
      undoStack.current.push(state);
      setState(next);
    }
  }, [state]);

  return { state, execute, undo, redo,
    canUndo: undoStack.current.length > 0,
    canRedo: redoStack.current.length > 0,
  };
}
```

Redux の `dispatch(action)` も Command パターンそのものです。Action が Command、Reducer が execute、Redux DevTools が UndoRedoManager に対応します。

---

## まとめ

| 項目 | 要点 |
|------|------|
| Command パターンの本質 | 操作をオブジェクトとしてカプセル化し、第一級市民にする |
| 4つの構成要素 | Client, Invoker, Command (interface), Receiver |
| Undo/Redo | Undo スタックと Redo スタックで双方向の操作履歴を管理 |
| MacroCommand | Composite パターンとの組合せ。複数コマンドを1つの Undo 単位に |
| 非同期キュー | コマンドの順序実行、リトライ、ロールバック |
| 粒度設計 | ユーザーの「意図」に対応する単位でコマンドを区切る |
| 関数型アプローチ | クロージャによる軽量 Command。シリアライズ不要な場面で有効 |
| Command バス | CQRS における Command の型安全なディスパッチ |
| 導入判断 | Undo/ログ/キュー/マクロのいずれかが必要なら検討 |

---

## 次に読むべきガイド

- [03-state.md](./03-state.md) -- State パターンと状態遷移（Command + State で状態付き操作管理）
- [04-iterator.md](./04-iterator.md) -- Iterator パターンとジェネレータ（Command 履歴の走査に応用）
- [00-observer.md](./00-observer.md) -- Observer パターン（Command 実行をイベントとして通知）
- [../03-functional/00-monad.md](../03-functional/00-monad.md) -- モナドパターン（Either で Command の成功/失敗を型安全に）
- [Event Sourcing / CQRS](../../03-software-design/system-design-guide/) -- Command パターンの大規模アーキテクチャへの発展

---

## 参考文献

1. **Design Patterns: Elements of Reusable Object-Oriented Software** -- Gamma, Helm, Johnson, Vlissides (GoF, 1994) -- Command パターンの原典。Chapter 5, pp.233-242
2. **Head First Design Patterns** -- Eric Freeman, Elisabeth Robson (O'Reilly, 2nd Edition, 2020) -- Command パターンの平易な解説と実践例
3. **Refactoring.Guru - Command** -- https://refactoring.guru/design-patterns/command -- 図解と多言語実装例
4. **Martin Fowler - Command Query Responsibility Segregation (CQRS)** -- https://martinfowler.com/bliki/CQRS.html -- Command パターンのアーキテクチャレベルへの発展
5. **Redux Documentation** -- https://redux.js.org/ -- Command パターンの実践的な大規模適用例（Action = Command）
