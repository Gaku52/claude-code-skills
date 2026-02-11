# Iterator パターン

> コレクションの内部構造を隠蔽しつつ要素に順次アクセスする手法と、ジェネレータによる遅延評価を習得する

## この章で学ぶこと

1. **Iterator プロトコルの仕組み** — Symbol.iterator と for...of の内部動作、カスタムイテレータの実装
2. **ジェネレータ関数 (function*)** — yield による遅延評価、無限シーケンス、非同期イテレータ
3. **実用的なイテレータパターン** — ページネーション、ツリー走査、パイプラインの構築

---

## 1. Iterator プロトコルの構造

```
JavaScript/TypeScript の Iterator プロトコル:

  Iterable                    Iterator
  ┌─────────────────┐       ┌─────────────────────┐
  │ [Symbol.iterator]│──────►│  next(): {          │
  │  () => Iterator  │       │    value: T,        │
  │                  │       │    done: boolean     │
  │ for...of で使用  │       │  }                  │
  │ スプレッド演算子  │       │                     │
  │ 分割代入         │       │  return?(): Result  │
  └─────────────────┘       │  throw?(): Result   │
                             └─────────────────────┘

  for (const item of iterable) {
    // iterator.next() を繰り返し呼び出し
    // done === true で終了
  }

  呼び出しシーケンス:
  ┌──────┐    ┌──────┐    ┌──────┐    ┌──────────┐
  │next()│───►│next()│───►│next()│───►│next()    │
  │{v:1} │    │{v:2} │    │{v:3} │    │{done:true}│
  └──────┘    └──────┘    └──────┘    └──────────┘
```

---

## 2. カスタムイテレータの実装

```typescript
// range-iterator.ts — 範囲イテレータ
class Range implements Iterable<number> {
  constructor(
    private readonly start: number,
    private readonly end: number,
    private readonly step: number = 1
  ) {
    if (step === 0) throw new Error('step must not be 0');
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
}

// 使用例
for (const n of new Range(1, 5)) {
  console.log(n); // 1, 2, 3, 4
}

const numbers = [...new Range(0, 10, 2)]; // [0, 2, 4, 6, 8]
const [first, second] = new Range(10, 0, -3); // first=10, second=7
```

```typescript
// linked-list-iterator.ts — 連結リストのイテレータ
class ListNode<T> {
  constructor(
    public value: T,
    public next: ListNode<T> | null = null
  ) {}
}

class LinkedList<T> implements Iterable<T> {
  private head: ListNode<T> | null = null;

  push(value: T): void {
    const node = new ListNode(value);
    node.next = this.head;
    this.head = node;
  }

  [Symbol.iterator](): Iterator<T> {
    let current = this.head;

    return {
      next(): IteratorResult<T> {
        if (current) {
          const value = current.value;
          current = current.next;
          return { value, done: false };
        }
        return { value: undefined, done: true };
      },
    };
  }
}

// 使用例: 連結リストを for...of で走査
const list = new LinkedList<number>();
list.push(3);
list.push(2);
list.push(1);

for (const value of list) {
  console.log(value); // 1, 2, 3
}
```

---

## 3. ジェネレータ関数

```typescript
// generators.ts — ジェネレータの基本と応用

// 基本的なジェネレータ
function* fibonacci(): Generator<number> {
  let a = 0;
  let b = 1;
  while (true) {
    yield a;
    [a, b] = [b, a + b];
  }
}

// 先頭N個を取得するヘルパー
function take<T>(n: number, iterable: Iterable<T>): T[] {
  const result: T[] = [];
  for (const item of iterable) {
    result.push(item);
    if (result.length >= n) break;
  }
  return result;
}

console.log(take(8, fibonacci()));
// [0, 1, 1, 2, 3, 5, 8, 13]

// ジェネレータによるツリーの深さ優先走査
interface TreeNode<T> {
  value: T;
  children: TreeNode<T>[];
}

function* depthFirst<T>(root: TreeNode<T>): Generator<T> {
  yield root.value;
  for (const child of root.children) {
    yield* depthFirst(child); // yield* で再帰的にデリゲート
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

// 使用例
const tree: TreeNode<string> = {
  value: 'A',
  children: [
    { value: 'B', children: [
      { value: 'D', children: [] },
      { value: 'E', children: [] },
    ]},
    { value: 'C', children: [
      { value: 'F', children: [] },
    ]},
  ],
};

console.log([...depthFirst(tree)]);  // ['A', 'B', 'D', 'E', 'C', 'F']
console.log([...breadthFirst(tree)]); // ['A', 'B', 'C', 'D', 'E', 'F']
```

```
ジェネレータの実行フロー:

  function* gen() {        呼び出し側
    yield 1;               const g = gen();
    yield 2;               g.next() → { value: 1, done: false }
    yield 3;               g.next() → { value: 2, done: false }
    return 4;              g.next() → { value: 3, done: false }
  }                        g.next() → { value: 4, done: true }

  ┌─────────────────┐     ┌─────────────────┐
  │ Generator       │     │ Caller          │
  │                 │     │                 │
  │  ──── yield 1 ──┼────►│ value: 1        │
  │  (一時停止)      │     │                 │
  │                 │◄────┼── next() ───    │
  │  ──── yield 2 ──┼────►│ value: 2        │
  │  (一時停止)      │     │                 │
  │                 │◄────┼── next() ───    │
  │  ──── yield 3 ──┼────►│ value: 3        │
  │  (一時停止)      │     │                 │
  │                 │◄────┼── next() ───    │
  │  ──── return 4 ─┼────►│ done: true      │
  └─────────────────┘     └─────────────────┘
```

---

## 4. 非同期イテレータ

```typescript
// async-iterators.ts — 非同期イテレータの活用

// API ページネーションの自動走査
async function* fetchAllPages<T>(
  baseUrl: string,
  pageSize: number = 20
): AsyncGenerator<T> {
  let page = 1;
  let hasMore = true;

  while (hasMore) {
    const response = await fetch(
      `${baseUrl}?page=${page}&per_page=${pageSize}`
    );
    const data: { items: T[]; total: number } = await response.json();

    for (const item of data.items) {
      yield item;
    }

    hasMore = page * pageSize < data.total;
    page++;
  }
}

// 使用例: 全ユーザーを1件ずつ処理 (メモリ効率が良い)
for await (const user of fetchAllPages<User>('/api/users')) {
  await processUser(user);
}

// ReadableStream の非同期イテレータ
async function* readLines(stream: ReadableStream<Uint8Array>): AsyncGenerator<string> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';

      for (const line of lines) {
        yield line;
      }
    }

    if (buffer) {
      yield buffer;
    }
  } finally {
    reader.releaseLock();
  }
}
```

---

## 5. イテレータパイプライン

```typescript
// iterator-pipeline.ts — 関数型イテレータ操作
class Iter<T> implements Iterable<T> {
  constructor(private source: Iterable<T>) {}

  static from<T>(source: Iterable<T>): Iter<T> {
    return new Iter(source);
  }

  *[Symbol.iterator](): Iterator<T> {
    yield* this.source;
  }

  map<U>(fn: (item: T) => U): Iter<U> {
    const source = this.source;
    return new Iter((function* () {
      for (const item of source) {
        yield fn(item);
      }
    })());
  }

  filter(predicate: (item: T) => boolean): Iter<T> {
    const source = this.source;
    return new Iter((function* () {
      for (const item of source) {
        if (predicate(item)) yield item;
      }
    })());
  }

  take(n: number): Iter<T> {
    const source = this.source;
    return new Iter((function* () {
      let count = 0;
      for (const item of source) {
        if (count >= n) break;
        yield item;
        count++;
      }
    })());
  }

  flatMap<U>(fn: (item: T) => Iterable<U>): Iter<U> {
    const source = this.source;
    return new Iter((function* () {
      for (const item of source) {
        yield* fn(item);
      }
    })());
  }

  reduce<U>(fn: (acc: U, item: T) => U, initial: U): U {
    let result = initial;
    for (const item of this.source) {
      result = fn(result, item);
    }
    return result;
  }

  toArray(): T[] {
    return [...this.source];
  }
}

// 使用例: パイプラインで遅延評価
const result = Iter.from(fibonacci())
  .filter(n => n % 2 === 0)       // 偶数のみ
  .map(n => n * n)                 // 二乗
  .take(5)                         // 先頭5個
  .toArray();

console.log(result); // [0, 4, 64, 1024, ...]
// → fibonacci は必要な分だけ生成される (遅延評価)
```

---

## 6. 比較表

| 特性 | for ループ | Array.map/filter | Generator | Iter パイプライン |
|------|-----------|-----------------|-----------|-----------------|
| 遅延評価 | 不可 | 不可 (即座に配列生成) | 可能 | 可能 |
| 無限シーケンス | 不可 | 不可 | 可能 | 可能 |
| メモリ効率 | 良い | 中間配列が生成 | 最良 | 最良 |
| チェーン可読性 | 低い | 高い | 中 | 高い |
| 非同期対応 | 要工夫 | なし | AsyncGenerator | 拡張可能 |

| イテレータ種類 | 同期 Iterator | AsyncIterator | Generator | AsyncGenerator |
|--------------|-------------|---------------|-----------|---------------|
| プロトコル | Symbol.iterator | Symbol.asyncIterator | function* | async function* |
| 構文 | for...of | for await...of | yield | yield (async内) |
| ユースケース | コレクション走査 | API/Stream処理 | 遅延シーケンス | ページネーション |
| 実行制御 | next() | next() (Promise) | yield で停止 | yield + await |

---

## 7. アンチパターン

### アンチパターン 1: 全データをメモリに載せてから処理

```typescript
// 悪い例: 100万件を配列に全て読み込み
const allUsers = await fetchAllUsers(); // メモリに100万件!
const activeUsers = allUsers.filter(u => u.active);
const names = activeUsers.map(u => u.name);

// 良い例: ジェネレータで1件ずつ処理
async function* fetchUsersStream(): AsyncGenerator<User> {
  for await (const user of fetchAllPages<User>('/api/users')) {
    yield user;
  }
}

// メモリ使用量はページサイズ分のみ
for await (const user of fetchUsersStream()) {
  if (user.active) {
    await processUser(user.name);
  }
}
```

### アンチパターン 2: イテレータの再利用

```typescript
// 悪い例: 同じイテレータを2回消費しようとする
function* nums() { yield 1; yield 2; yield 3; }
const iter = nums();

console.log([...iter]); // [1, 2, 3]
console.log([...iter]); // [] ← 空! イテレータは使い切り

// 良い例: Iterable を保持し、必要なときに新しい Iterator を生成
const range = new Range(1, 4);
console.log([...range]); // [1, 2, 3]
console.log([...range]); // [1, 2, 3] ← Iterable なので何度でも
```

---

## 8. FAQ

### Q1: ジェネレータと普通の関数、どちらを使うべきですか？

データ量が有限で小さい場合は通常の関数（配列を返す）で十分です。以下の場合にジェネレータを検討してください。(1) 無限シーケンス（フィボナッチ数列、乱数列）、(2) 大量データの逐次処理（ファイル読み込み、API ページネーション）、(3) 途中で処理を中断する可能性がある場合。ジェネレータは「必要になるまで計算しない」遅延評価が最大のメリットです。

### Q2: for...of と forEach の違いは何ですか？

`for...of` は Iterator プロトコルに基づき、`break`/`continue`/`return` で制御フローを操作できます。`forEach` は Array 専用のメソッドで、途中で抜けることができません。また、`for...of` はカスタムイテレータや Generator をサポートしますが、`forEach` は配列のみです。`async/await` と組み合わせる場合も `for...of`（`for await...of`）が適しています。

### Q3: AsyncGenerator のエラーハンドリングはどうすべきですか？

`for await...of` のブロック内で `try/catch` を使います。また、Generator の `throw()` メソッドで外部からエラーを注入することもできます。ページネーションのような処理では、個別ページのエラーはリトライし、致命的なエラーのみ全体を中断するような設計が実用的です。`finally` ブロックでリソースのクリーンアップを忘れないでください。

---

## まとめ

| 項目 | 要点 |
|------|------|
| Iterator プロトコル | Symbol.iterator + next() で統一的な走査インターフェース |
| for...of | Iterator プロトコルに基づくループ構文。break/continue 対応 |
| Generator | function* と yield による遅延評価。無限シーケンスも表現可能 |
| yield* | 別のイテレータにデリゲート。ツリー走査の再帰に有効 |
| AsyncGenerator | async function* で非同期データの逐次処理 |
| パイプライン | map/filter/take を遅延評価で連鎖。メモリ効率が良い |

---

## 次に読むべきガイド

- [02-command.md](./02-command.md) — Command パターンと操作のカプセル化
- [03-state.md](./03-state.md) — State パターンと状態遷移
- [../03-functional/02-fp-patterns.md](../03-functional/02-fp-patterns.md) — 関数型パターン（パイプライン、合成）

---

## 参考文献

1. **MDN - Iterators and generators** — https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Iterators_and_Generators — 公式リファレンス
2. **Exploring ES6 - Iterables and iterators** — Axel Rauschmayer — 詳細な仕様解説
3. **Design Patterns** — Gamma, Helm, Johnson, Vlissides (GoF, 1994) — Iterator パターンの原典
