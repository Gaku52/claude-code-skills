# Command パターン

> 操作をオブジェクトとしてカプセル化し、Undo/Redo、キューイング、マクロ記録を実現する行動パターン

## この章で学ぶこと

1. **Command パターンの基本構造** — コマンドオブジェクトによる操作のカプセル化と実行の遅延・記録
2. **Undo/Redo の実装** — コマンド履歴を管理し、操作の取り消しとやり直しを実現する設計
3. **マクロとキューイング** — 複数コマンドの合成、非同期実行キュー、トランザクション的な実行

---

## 1. Command パターンの構造

```
Command パターンの構成要素:

  ┌──────────┐     ┌──────────────┐     ┌──────────────┐
  │ Invoker  │────►│   Command    │────►│  Receiver    │
  │ (実行者)  │     │ (コマンド)    │     │ (実際の処理)  │
  └──────────┘     └──────────────┘     └──────────────┘
       │                 ▲ ▲
       │           ┌─────┘ └──────┐
       │     ┌─────┴──────┐ ┌────┴───────┐
       │     │ Concrete   │ │ Concrete   │
       │     │ Command A  │ │ Command B  │
       │     └────────────┘ └────────────┘
       │
       │     ┌────────────┐
       └────►│  History   │  コマンド履歴
              │ (Undo/Redo)│  の管理
              └────────────┘

  Client → Command を生成 → Invoker に渡す → Invoker が execute()
```

---

## 2. 基本実装

```typescript
// command.ts — Command パターンの基本構造
interface Command {
  execute(): void;
  undo(): void;
  describe(): string;
}

// Receiver: テキストエディタ
class TextEditor {
  private content: string = '';
  private cursorPosition: number = 0;

  getContent(): string {
    return this.content;
  }

  getCursorPosition(): number {
    return this.cursorPosition;
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
}

// Concrete Command: テキスト挿入
class InsertTextCommand implements Command {
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

// Concrete Command: テキスト削除
class DeleteTextCommand implements Command {
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
    return `Delete from ${this.start} to ${this.end}`;
  }
}
```

---

## 3. Undo/Redo マネージャー

```typescript
// undo-redo-manager.ts — Undo/Redo の管理
class UndoRedoManager {
  private undoStack: Command[] = [];
  private redoStack: Command[] = [];
  private readonly maxHistory: number;

  constructor(maxHistory: number = 100) {
    this.maxHistory = maxHistory;
  }

  execute(command: Command): void {
    command.execute();
    this.undoStack.push(command);

    // 新しいコマンド実行時は Redo スタックをクリア
    this.redoStack = [];

    // 履歴の上限を超えたら古いものを削除
    if (this.undoStack.length > this.maxHistory) {
      this.undoStack.shift();
    }
  }

  undo(): boolean {
    const command = this.undoStack.pop();
    if (!command) return false;

    command.undo();
    this.redoStack.push(command);
    return true;
  }

  redo(): boolean {
    const command = this.redoStack.pop();
    if (!command) return false;

    command.execute();
    this.undoStack.push(command);
    return true;
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
}

// 使用例
const editor = new TextEditor();
const manager = new UndoRedoManager();

manager.execute(new InsertTextCommand(editor, 0, 'Hello'));
manager.execute(new InsertTextCommand(editor, 5, ' World'));
console.log(editor.getContent()); // "Hello World"

manager.undo();
console.log(editor.getContent()); // "Hello"

manager.redo();
console.log(editor.getContent()); // "Hello World"
```

```
Undo/Redo のスタック操作:

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
```

---

## 4. マクロコマンド

```typescript
// macro-command.ts — 複数コマンドの合成
class MacroCommand implements Command {
  private commands: Command[] = [];

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
    // 逆順で Undo
    for (let i = this.commands.length - 1; i >= 0; i--) {
      this.commands[i].undo();
    }
  }

  describe(): string {
    return `Macro [${this.commands.map(c => c.describe()).join(', ')}]`;
  }
}

// マクロの記録と再生
class MacroRecorder {
  private recording: boolean = false;
  private commands: Command[] = [];

  startRecording(): void {
    this.recording = true;
    this.commands = [];
  }

  stopRecording(): MacroCommand {
    this.recording = false;
    const macro = new MacroCommand();
    for (const cmd of this.commands) {
      macro.add(cmd);
    }
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
}

// 使用例: テキストの整形マクロ
const recorder = new MacroRecorder();
recorder.startRecording();

// 操作を記録
const cmd1 = new InsertTextCommand(editor, 0, '# ');
recorder.recordCommand(cmd1);
manager.execute(cmd1);

const cmd2 = new InsertTextCommand(editor, editor.getContent().length, '\n');
recorder.recordCommand(cmd2);
manager.execute(cmd2);

const formatMacro = recorder.stopRecording();

// マクロを再生 (別のテキストに適用)
manager.execute(formatMacro);
```

---

## 5. 非同期コマンドキュー

```typescript
// command-queue.ts — 非同期コマンドの順序実行
interface AsyncCommand {
  execute(): Promise<void>;
  undo(): Promise<void>;
  describe(): string;
  canRetry(): boolean;
}

class CommandQueue {
  private queue: AsyncCommand[] = [];
  private processing: boolean = false;
  private executed: AsyncCommand[] = [];

  enqueue(command: AsyncCommand): void {
    this.queue.push(command);
    this.processNext();
  }

  private async processNext(): Promise<void> {
    if (this.processing || this.queue.length === 0) return;

    this.processing = true;
    const command = this.queue.shift()!;

    try {
      await command.execute();
      this.executed.push(command);
      console.log(`Executed: ${command.describe()}`);
    } catch (error) {
      console.error(`Failed: ${command.describe()}`, error);

      if (command.canRetry()) {
        // リトライ: キューの先頭に戻す
        this.queue.unshift(command);
      } else {
        // 失敗時のロールバック
        await this.rollback();
      }
    } finally {
      this.processing = false;
      this.processNext();
    }
  }

  private async rollback(): Promise<void> {
    console.log('Rolling back executed commands...');
    while (this.executed.length > 0) {
      const cmd = this.executed.pop()!;
      await cmd.undo();
      console.log(`Rolled back: ${cmd.describe()}`);
    }
  }
}
```

---

## 6. 比較表

| 特性 | Command | Strategy | Observer |
|------|---------|----------|----------|
| 目的 | 操作のカプセル化 | アルゴリズムの切替 | 状態変化の通知 |
| Undo/Redo | 対応 | 非対応 | 非対応 |
| 履歴管理 | 可能 | 不要 | 不要 |
| 遅延実行 | 可能 | 即座に実行 | イベント駆動 |
| キューイング | 可能 | 不要 | 不要 |
| 複雑さ | 中〜高 | 低い | 中 |

| Undo 実装方式 | Command 履歴 | Memento (スナップショット) | Event Sourcing |
|--------------|-------------|--------------------------|----------------|
| メモリ使用 | 低い (差分) | 高い (全状態) | 中 (イベント列) |
| 実装複雑さ | 中 | 低い | 高い |
| 部分的 Undo | 困難 | 不可 | 可能 |
| 永続化 | 容易 | 容易 | 容易 |
| 適用場面 | エディタ、操作記録 | ゲームのセーブ | ドメインイベント |

---

## 7. アンチパターン

### アンチパターン 1: コマンドの粒度が不適切

```typescript
// 悪い例: 1文字ごとにコマンドを生成
// → Undo が1文字ずつ戻る、メモリ浪費
manager.execute(new InsertTextCommand(editor, 0, 'H'));
manager.execute(new InsertTextCommand(editor, 1, 'e'));
manager.execute(new InsertTextCommand(editor, 2, 'l'));
// Undo 3回で "Hel" → "He" → "H" → ""

// 良い例: 意味のある単位でコマンドを生成
// 入力をバッファリングし、一定時間の無入力でコマンドを確定
class BufferedInsertCommand implements Command {
  private buffer: string = '';
  private timer: ReturnType<typeof setTimeout> | null = null;

  appendChar(char: string): void {
    this.buffer += char;
    // 500ms の無入力でコマンド確定
    if (this.timer) clearTimeout(this.timer);
    this.timer = setTimeout(() => this.flush(), 500);
  }

  private flush(): void {
    if (this.buffer) {
      manager.execute(
        new InsertTextCommand(editor, editor.getCursorPosition(), this.buffer)
      );
      this.buffer = '';
    }
  }
  // ...
}
```

### アンチパターン 2: Undo 不可能なコマンド

```typescript
// 悪い例: undo() が実装されていない
class SendEmailCommand implements Command {
  execute(): void {
    emailService.send(this.email);  // 送信済みメールは取り消せない
  }
  undo(): void {
    // 何もしない...??
  }
}

// 良い例: 補償アクション (Compensating Action) を定義
class SendEmailCommand implements Command {
  private sentId: string | null = null;

  execute(): void {
    this.sentId = emailService.send(this.email);
  }
  undo(): void {
    if (this.sentId) {
      // 取り消しメールを送信 (補償アクション)
      emailService.sendCancellation(this.sentId);
    }
  }
}

// または Undo 不可であることを明示
interface Command {
  execute(): void;
  undo(): void;
  readonly isUndoable: boolean;
}
```

---

## 8. FAQ

### Q1: Command パターンはどのような場面で使うべきですか？

主に次の場面で有効です。(1) **Undo/Redo**: テキストエディタ、グラフィックツール、スプレッドシート。(2) **操作のキューイング**: ジョブキュー、バッチ処理。(3) **操作のログ記録**: 監査証跡、デバッグログ。(4) **マクロ記録**: ユーザー操作の記録と再生。単純な関数呼び出しで十分な場合はオーバーエンジニアリングになるため注意してください。

### Q2: Command パターンと関数型プログラミングのクロージャは何が違いますか？

クロージャも「操作をカプセル化」しますが、Command パターンは `undo()` メソッドを持つ点が本質的に異なります。また、Command オブジェクトはシリアライズ（永続化）可能であり、ネットワーク越しの送信やログ記録に適しています。TypeScript ではクラスベースの Command が一般的ですが、`{ execute, undo, describe }` のようなオブジェクトリテラルでも表現できます。

### Q3: 大量のコマンド履歴によるメモリ消費をどう管理しますか？

(1) **履歴の上限設定**: 最大100〜1000件に制限し、古いものから破棄。(2) **コマンドの圧縮**: 連続する同種のコマンドをマージ（例: 連続入力を1コマンドに）。(3) **チェックポイント**: 定期的に全状態をスナップショットし、それ以前の履歴を破棄。(4) **遅延読込**: 古い履歴をディスクに退避し、必要時に読み込む。

---

## まとめ

| 項目 | 要点 |
|------|------|
| Command パターン | 操作をオブジェクトとしてカプセル化。execute/undo のインターフェース |
| Invoker | コマンドの実行を管理。履歴スタックを保持 |
| Undo/Redo | Undo スタックと Redo スタックで双方向の操作履歴を管理 |
| MacroCommand | 複数コマンドを合成し、一括 execute/undo |
| 非同期キュー | コマンドの順序実行とロールバック |
| 粒度設計 | ユーザーにとって意味のある単位でコマンドを区切る |

---

## 次に読むべきガイド

- [03-state.md](./03-state.md) — State パターンと状態遷移
- [04-iterator.md](./04-iterator.md) — Iterator パターンとジェネレータ
- [../03-functional/00-monad.md](../03-functional/00-monad.md) — モナドパターン

---

## 参考文献

1. **Design Patterns** — Gamma, Helm, Johnson, Vlissides (GoF, 1994) — Command パターンの原典
2. **Head First Design Patterns** — Eric Freeman, Elisabeth Robson (O'Reilly, 2020) — 平易な解説
3. **Refactoring.Guru - Command** — https://refactoring.guru/design-patterns/command — 図解と多言語実装例
