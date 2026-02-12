# Iterator パターン

> コレクションの内部構造を隠蔽しつつ要素に順次アクセスする手法と、ジェネレータによる遅延評価を習得する

---

## この章で学ぶこと

1. **Iterator プロトコルの仕組み** -- Symbol.iterator と for...of の内部動作、カスタムイテレータの設計と実装
2. **ジェネレータ関数 (function*)** -- yield による遅延評価、無限シーケンス、コルーチンとしてのジェネレータ
3. **非同期イテレータ (AsyncGenerator)** -- API ページネーション、ストリーム処理、for await...of の活用
4. **実用的なイテレータパターン** -- 遅延評価パイプライン、ツリー走査、外部イテレータと内部イテレータ
5. **GoF の Iterator と JavaScript の Iterator プロトコル** -- 古典的パターンと言語組込みプロトコルの対応関係

---

## 前提知識

| トピック | 必要な理解 | 参照リンク |
|---------|-----------|-----------|
| TypeScript のジェネリクス | `Iterable<T>`, `Iterator<T>`, `Generator<T>` の型パラメータ | [02-programming](../../02-programming/) |
| Promise / async-await | 非同期処理の基本、`for await...of` の構文 | [02-programming](../../02-programming/) |
| データ構造の基本 | 配列、連結リスト、ツリーの概念 | [01-cs-fundamentals](../../01-cs-fundamentals/) |
| Composite パターン | ツリー構造のパターン（走査に Iterator を利用） | [../01-structural/04-composite.md](../01-structural/04-composite.md) |

---

## なぜ Iterator パターンが必要なのか

### コレクションの内部構造への依存問題

```
配列、連結リスト、ツリー、ハッシュマップ...
コレクションの種類ごとに走査方法が異なる:

  配列:
    for (let i = 0; i < arr.length; i++) { arr[i] }

  連結リスト:
    let node = head;
    while (node) { node = node.next; }

  ツリー (深さ優先):
    function traverse(node) {
      visit(node);
      for (const child of node.children) traverse(child);
    }

  ハッシュマップ:
    for (const key of Object.keys(map)) { map[key] }

  問題:
  ┌────────────────────────────────────────────────┐
  │ 利用者がコレクションの内部構造を知る必要がある    │
  │ コレクションの種類を変えると、走査コードも変更     │
  │ 走査ロジック (DFS/BFS 等) が利用者側に散在       │
  └────────────────────────────────────────────────┘
```

### Iterator パターンによる解決

```
Iterator パターンの解決:

  ┌──────────────────┐     ┌──────────────────┐
  │   Collection     │────►│   Iterator       │
  │ (内部構造を隠蔽)  │     │ (統一アクセス)    │
  │                  │     │                  │
  │ [Symbol.iterator]│     │ + next()         │
  │   → Iterator     │     │   { value, done }│
  └──────────────────┘     └──────────────────┘

  利用者側は常に同じコード:
    for (const item of collection) {
      // 配列でも、リストでも、ツリーでも同じ
    }

  利点:
  ✓ コレクションの内部構造を知らなくてよい
  ✓ 走査アルゴリズムを複数持てる（DFS, BFS, フィルタ付き等）
  ✓ 遅延評価で無限シーケンスも表現可能
  ✓ スプレッド演算子、分割代入、Array.from が自動的に使える
```

GoF の定義:

> "Provide a way to access the elements of an aggregate object sequentially without exposing its underlying representation."
>
> -- Design Patterns: Elements of Reusable Object-Oriented Software (1994)

JavaScript/TypeScript では、Iterator パターンが **言語仕様に組み込まれている** 点が特徴的です。`Symbol.iterator` プロトコルを実装するだけで、`for...of`、スプレッド演算子、分割代入、`Array.from` などの言語機能が自動的に利用可能になります。

---

## 1. Iterator プロトコルの構造

```
JavaScript/TypeScript の Iterator プロトコル:

  ┌─────────────────────────────┐
  │        Iterable             │
  │                             │
  │  [Symbol.iterator]()        │
  │    → Iterator を返す        │
  │                             │
  │  使える構文:                 │
  │    for...of                 │
  │    [...iterable]            │
  │    const [a, b] = iterable  │
  │    Array.from(iterable)     │
  │    yield* iterable          │
  │    new Map(iterable)        │
  │    Promise.all(iterable)    │
  └──────────────┬──────────────┘
                 │ returns
                 ▼
  ┌─────────────────────────────┐
  │        Iterator             │
  │                             │
  │  next(): IteratorResult     │
  │    { value: T, done: false }│  ← 値がある
  │    { value: undefined,      │
  │      done: true }           │  ← 終了
  │                             │
  │  return?(): IteratorResult  │  ← 早期終了
  │  throw?(e): IteratorResult  │  ← エラー注入
  └─────────────────────────────┘

  呼び出しシーケンス:
  ┌──────┐    ┌──────┐    ┌──────┐    ┌──────────┐
  │next()│───►│next()│───►│next()│───►│next()    │
  │{v:1, │    │{v:2, │    │{v:3, │    │{done:    │
  │ d:F} │    │ d:F} │    │ d:F} │    │ true}    │
  └──────┘    └──────┘    └──────┘    └──────────┘

  ★ 重要: Iterable ≠ Iterator
    Iterable: [Symbol.iterator]() メソッドを持つ → 何度でも Iterator を生成
    Iterator: next() メソッドを持つ → 1回使い切り
```

---

## 2. カスタムイテレータの実装

### コード例 1: 範囲イテレータ（Range）

```typescript
// range-iterator.ts -- 範囲イテレータ

// ============================
// Iterable を実装するクラス
// ============================
class Range implements Iterable<number> {
  constructor(
    private readonly start: number,
    private readonly end: number,
    private readonly step: number = 1
  ) {
    if (step === 0) throw new Error('step must not be 0');
    if (step > 0 && start > end) throw new Error('start must be <= end for positive step');
    if (step < 0 && start < end) throw new Error('start must be >= end for negative step');
  }

  /** Iterable プロトコル: Iterator を生成して返す */
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

  /** 要素数を事前計算（遅延評価を維持しつつサイズ取得） */
  get length(): number {
    return Math.max(0, Math.ceil((this.end - this.start) / this.step));
  }

  /** 要素が含まれるか判定（O(1)） */
  includes(value: number): boolean {
    if (this.step > 0) {
      if (value < this.start || value >= this.end) return false;
    } else {
      if (value > this.start || value <= this.end) return false;
    }
    return (value - this.start) % this.step === 0;
  }
}

// ============================
// 使用例
// ============================

// for...of で走査
for (const n of new Range(1, 5)) {
  console.log(n); // 1, 2, 3, 4
}

// スプレッド演算子
const numbers = [...new Range(0, 10, 2)]; // [0, 2, 4, 6, 8]

// 分割代入
const [first, second] = new Range(10, 0, -3); // first=10, second=7

// Array.from
const arr = Array.from(new Range(1, 6)); // [1, 2, 3, 4, 5]

// ★ Iterable なので何度でも走査可能
const range = new Range(1, 4);
console.log([...range]); // [1, 2, 3]
console.log([...range]); // [1, 2, 3] ← 2回目も同じ結果
```

### コード例 2: 連結リストのイテレータ

```typescript
// linked-list-iterator.ts -- 連結リストのイテレータ

// ============================
// Node と LinkedList の定義
// ============================
class ListNode<T> {
  constructor(
    public value: T,
    public next: ListNode<T> | null = null
  ) {}
}

class LinkedList<T> implements Iterable<T> {
  private head: ListNode<T> | null = null;
  private _size: number = 0;

  push(value: T): void {
    const node = new ListNode(value);
    node.next = this.head;
    this.head = node;
    this._size++;
  }

  get size(): number {
    return this._size;
  }

  /** 順方向のイテレータ */
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

  /** 逆順イテレータ（複数の走査方法を提供） */
  reversed(): Iterable<T> {
    const items = [...this]; // 一度配列に
    let index = items.length - 1;

    return {
      [Symbol.iterator](): Iterator<T> {
        return {
          next(): IteratorResult<T> {
            if (index >= 0) {
              return { value: items[index--], done: false };
            }
            return { value: undefined, done: true };
          },
        };
      },
    };
  }

  /** 条件付きフィルタイテレータ */
  filter(predicate: (value: T) => boolean): Iterable<T> {
    const source = this;
    return {
      *[Symbol.iterator](): Iterator<T> {
        for (const item of source) {
          if (predicate(item)) yield item;
        }
      },
    };
  }
}

// ============================
// 使用例
// ============================
const list = new LinkedList<number>();
list.push(3);
list.push(2);
list.push(1);

// 順方向
for (const value of list) {
  console.log(value); // 1, 2, 3
}

// 逆順
for (const value of list.reversed()) {
  console.log(value); // 3, 2, 1
}

// フィルタ付き
for (const value of list.filter(v => v % 2 !== 0)) {
  console.log(value); // 1, 3
}
```

---

## 3. ジェネレータ関数

### コード例 3: ジェネレータの基本と応用

```typescript
// generators.ts -- ジェネレータの基本と応用

// ============================
// 基本的なジェネレータ
// ============================
function* fibonacci(): Generator<number> {
  let a = 0;
  let b = 1;
  while (true) {
    yield a;
    [a, b] = [b, a + b];
  }
}

// 先頭 N 個を取得するヘルパー
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

// ============================
// ジェネレータによるツリーの走査
// ============================
interface TreeNode<T> {
  value: T;
  children: TreeNode<T>[];
}

/** 深さ優先走査（前順: pre-order） */
function* depthFirst<T>(root: TreeNode<T>): Generator<T> {
  yield root.value;
  for (const child of root.children) {
    yield* depthFirst(child); // yield* で再帰的にデリゲート
  }
}

/** 深さ優先走査（後順: post-order） */
function* depthFirstPostOrder<T>(root: TreeNode<T>): Generator<T> {
  for (const child of root.children) {
    yield* depthFirstPostOrder(child);
  }
  yield root.value;
}

/** 幅優先走査 (BFS) */
function* breadthFirst<T>(root: TreeNode<T>): Generator<T> {
  const queue: TreeNode<T>[] = [root];
  while (queue.length > 0) {
    const node = queue.shift()!;
    yield node.value;
    queue.push(...node.children);
  }
}

/** レベルごとのグループ化走査 */
function* levelOrder<T>(root: TreeNode<T>): Generator<T[]> {
  let currentLevel: TreeNode<T>[] = [root];
  while (currentLevel.length > 0) {
    const values = currentLevel.map(n => n.value);
    yield values;
    const nextLevel: TreeNode<T>[] = [];
    for (const node of currentLevel) {
      nextLevel.push(...node.children);
    }
    currentLevel = nextLevel;
  }
}

// ============================
// 使用例
// ============================
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

console.log([...depthFirst(tree)]);
// ['A', 'B', 'D', 'E', 'C', 'F']

console.log([...depthFirstPostOrder(tree)]);
// ['D', 'E', 'B', 'F', 'C', 'A']

console.log([...breadthFirst(tree)]);
// ['A', 'B', 'C', 'D', 'E', 'F']

console.log([...levelOrder(tree)]);
// [['A'], ['B', 'C'], ['D', 'E', 'F']]
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

  ★ yield は「一時停止 + 値の送出」
  ★ next() は「再開 + 値の受信」
  ★ return の値は for...of では取得されない
    （done: true の value は無視される）

  yield* によるデリゲート:
  function* outer() {
    yield 'a';
    yield* inner();   // inner のすべての yield を外に転送
    yield 'c';
  }
  function* inner() { yield 'b1'; yield 'b2'; }

  [...outer()] → ['a', 'b1', 'b2', 'c']
```

---

## 4. 非同期イテレータ

### コード例 4: API ページネーションの自動走査

```typescript
// async-iterators.ts -- 非同期イテレータの活用

// ============================
// API ページネーション
// ============================
interface PageResponse<T> {
  items: T[];
  total: number;
  page: number;
  perPage: number;
}

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

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    const data: PageResponse<T> = await response.json();

    for (const item of data.items) {
      yield item; // 1件ずつ yield（メモリ効率が良い）
    }

    hasMore = page * pageSize < data.total;
    page++;
  }
}

// 使用例: 全ユーザーを1件ずつ処理（メモリ効率が良い）
interface User { id: string; name: string; active: boolean; }

async function processAllUsers(): Promise<void> {
  let count = 0;
  for await (const user of fetchAllPages<User>('/api/users')) {
    if (user.active) {
      await processUser(user);
      count++;
    }
  }
  console.log(`Processed ${count} active users`);
}

// ============================
// ReadableStream のラインリーダー
// ============================
async function* readLines(
  stream: ReadableStream<Uint8Array>
): AsyncGenerator<string> {
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

    // 最後のバッファ（改行なし）
    if (buffer) {
      yield buffer;
    }
  } finally {
    reader.releaseLock();
  }
}

// ============================
// Server-Sent Events (SSE) のストリーム
// ============================
async function* streamSSE(url: string): AsyncGenerator<{
  event: string;
  data: string;
}> {
  const response = await fetch(url);
  if (!response.body) throw new Error('No response body');

  for await (const line of readLines(response.body)) {
    if (line.startsWith('event: ')) {
      const event = line.slice(7);
      // 次の行が data:
      // (簡略化: 実際の SSE パーサーはもっと複雑)
      continue;
    }
    if (line.startsWith('data: ')) {
      yield { event: 'message', data: line.slice(6) };
    }
  }
}

// 使用例: SSE ストリームの購読
async function watchUpdates(): Promise<void> {
  for await (const { event, data } of streamSSE('/api/events')) {
    console.log(`[${event}] ${data}`);
    // 必要に応じて break で中断可能
  }
}
```

```
同期 vs 非同期イテレータの対応関係:

  同期                         非同期
  ─────────────────────────    ─────────────────────────
  Iterable                     AsyncIterable
  [Symbol.iterator]()          [Symbol.asyncIterator]()
  Iterator                     AsyncIterator
  next(): IteratorResult       next(): Promise<IteratorResult>
  for...of                     for await...of
  function*                    async function*
  yield                        yield (async コンテキスト内)
  yield*                       yield*

  ★ for await...of は以下と等価:
  const iter = asyncIterable[Symbol.asyncIterator]();
  while (true) {
    const { value, done } = await iter.next();
    if (done) break;
    // value を処理
  }
```

---

## 5. イテレータパイプライン（遅延評価）

### コード例 5: 関数型イテレータ操作ライブラリ

```typescript
// iterator-pipeline.ts -- 関数型イテレータ操作

// ============================
// 遅延評価パイプラインクラス
// ============================
class Iter<T> implements Iterable<T> {
  constructor(private source: Iterable<T>) {}

  static from<T>(source: Iterable<T>): Iter<T> {
    return new Iter(source);
  }

  /** 無限の繰り返しイテレータ */
  static repeat<T>(value: T): Iter<T> {
    return new Iter((function* () {
      while (true) yield value;
    })());
  }

  /** 自然数列 (0, 1, 2, ...) */
  static naturals(): Iter<number> {
    return new Iter((function* () {
      let n = 0;
      while (true) yield n++;
    })());
  }

  *[Symbol.iterator](): Iterator<T> {
    yield* this.source;
  }

  /** 値の変換 */
  map<U>(fn: (item: T, index: number) => U): Iter<U> {
    const source = this.source;
    return new Iter((function* () {
      let i = 0;
      for (const item of source) {
        yield fn(item, i++);
      }
    })());
  }

  /** フィルタリング */
  filter(predicate: (item: T) => boolean): Iter<T> {
    const source = this.source;
    return new Iter((function* () {
      for (const item of source) {
        if (predicate(item)) yield item;
      }
    })());
  }

  /** 先頭 N 個を取得 */
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

  /** 先頭 N 個をスキップ */
  skip(n: number): Iter<T> {
    const source = this.source;
    return new Iter((function* () {
      let count = 0;
      for (const item of source) {
        if (count >= n) yield item;
        count++;
      }
    })());
  }

  /** 条件を満たす間だけ取得 */
  takeWhile(predicate: (item: T) => boolean): Iter<T> {
    const source = this.source;
    return new Iter((function* () {
      for (const item of source) {
        if (!predicate(item)) break;
        yield item;
      }
    })());
  }

  /** フラットマップ */
  flatMap<U>(fn: (item: T) => Iterable<U>): Iter<U> {
    const source = this.source;
    return new Iter((function* () {
      for (const item of source) {
        yield* fn(item);
      }
    })());
  }

  /** 隣接要素のペア */
  pairwise(): Iter<[T, T]> {
    const source = this.source;
    return new Iter((function* () {
      let prev: T | undefined;
      let first = true;
      for (const item of source) {
        if (!first) {
          yield [prev!, item] as [T, T];
        }
        prev = item;
        first = false;
      }
    })());
  }

  /** チャンク分割 */
  chunk(size: number): Iter<T[]> {
    const source = this.source;
    return new Iter((function* () {
      let chunk: T[] = [];
      for (const item of source) {
        chunk.push(item);
        if (chunk.length >= size) {
          yield chunk;
          chunk = [];
        }
      }
      if (chunk.length > 0) yield chunk;
    })());
  }

  /** 重複除去 */
  distinct(): Iter<T> {
    const source = this.source;
    return new Iter((function* () {
      const seen = new Set<T>();
      for (const item of source) {
        if (!seen.has(item)) {
          seen.add(item);
          yield item;
        }
      }
    })());
  }

  /** 畳み込み（即座に評価） */
  reduce<U>(fn: (acc: U, item: T) => U, initial: U): U {
    let result = initial;
    for (const item of this.source) {
      result = fn(result, item);
    }
    return result;
  }

  /** 配列に変換（即座に評価） */
  toArray(): T[] {
    return [...this.source];
  }

  /** 最初の要素を取得 */
  first(): T | undefined {
    for (const item of this.source) {
      return item;
    }
    return undefined;
  }

  /** 要素数をカウント */
  count(): number {
    let n = 0;
    for (const _ of this.source) n++;
    return n;
  }

  /** 全要素が条件を満たすか */
  every(predicate: (item: T) => boolean): boolean {
    for (const item of this.source) {
      if (!predicate(item)) return false;
    }
    return true;
  }

  /** いずれかの要素が条件を満たすか */
  some(predicate: (item: T) => boolean): boolean {
    for (const item of this.source) {
      if (predicate(item)) return true;
    }
    return false;
  }
}

// ============================
// 使用例: パイプラインで遅延評価
// ============================

// フィボナッチ数列から偶数の二乗を5個取得
const result = Iter.from(fibonacci())
  .filter(n => n % 2 === 0)       // 偶数のみ
  .map(n => n * n)                 // 二乗
  .take(5)                         // 先頭5個
  .toArray();

console.log(result); // [0, 4, 64, 17956, ...]
// → fibonacci は必要な分だけ生成される（遅延評価）

// 自然数から素数を生成
function isPrime(n: number): boolean {
  if (n < 2) return false;
  for (let i = 2; i <= Math.sqrt(n); i++) {
    if (n % i === 0) return false;
  }
  return true;
}

const primes = Iter.naturals()
  .skip(2)
  .filter(isPrime)
  .take(10)
  .toArray();

console.log(primes); // [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

// チャンク分割 + マップ
const chunks = Iter.from(new Range(1, 11))
  .chunk(3)
  .map(chunk => chunk.reduce((a, b) => a + b, 0))
  .toArray();

console.log(chunks); // [6, 15, 24, 10]
// [1+2+3, 4+5+6, 7+8+9, 10]
```

```
遅延評価の動作イメージ:

  fibonacci() → filter(even) → map(square) → take(3) → toArray()

  評価は「右から左」に要求が伝搬し、
  「左から右」に値が1つずつ流れる:

  fib     filter    map      take    result
  ─────   ─────     ─────    ─────   ─────
  0    →  0      →  0     →  [0]
  1    →  (skip)
  1    →  (skip)
  2    →  2      →  4     →  [0,4]
  3    →  (skip)
  5    →  (skip)
  8    →  8      →  64    →  [0,4,64]  ← take(3) 完了!

  ★ fibonacci は 8 までしか計算されない（遅延評価のメリット）
  ★ 中間配列は一切生成されない
```

---

## 6. Python のイテレータとジェネレータ

### コード例 6: Python の Iterator プロトコル

```python
# python_iterators.py -- Python のイテレータパターン

from __future__ import annotations
from typing import Iterator, Iterable, TypeVar, Callable, Generator
from dataclasses import dataclass

T = TypeVar("T")
U = TypeVar("U")


# ============================
# カスタム Range (Python 組込みの range と同等)
# ============================
class MyRange:
    """Python の range() を再実装"""

    def __init__(self, start: int, stop: int, step: int = 1):
        self.start = start
        self.stop = stop
        self.step = step

    def __iter__(self) -> Iterator[int]:
        """__iter__ = [Symbol.iterator] に相当"""
        current = self.start
        while (self.step > 0 and current < self.stop) or \
              (self.step < 0 and current > self.stop):
            yield current  # Python ではジェネレータで簡潔に書ける
            current += self.step

    def __len__(self) -> int:
        return max(0, (self.stop - self.start + self.step - 1) // self.step)

    def __contains__(self, value: int) -> bool:
        if self.step > 0:
            return self.start <= value < self.stop and (value - self.start) % self.step == 0
        else:
            return self.stop < value <= self.start and (self.start - value) % (-self.step) == 0


# ============================
# ジェネレータの応用: パイプライン
# ============================
def take(n: int, iterable: Iterable[T]) -> Generator[T, None, None]:
    """先頭 N 個を取得"""
    count = 0
    for item in iterable:
        if count >= n:
            return
        yield item
        count += 1


def chunk(iterable: Iterable[T], size: int) -> Generator[list[T], None, None]:
    """チャンク分割"""
    buf: list[T] = []
    for item in iterable:
        buf.append(item)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def flatten(iterable: Iterable[Iterable[T]]) -> Generator[T, None, None]:
    """ネストしたイテラブルをフラット化"""
    for inner in iterable:
        yield from inner  # yield from = yield* に相当


# ============================
# Python のイテレータ式 (Generator Expression)
# ============================
def demonstrate_generators():
    # リスト内包表記（即座に全要素を生成）
    squares_list = [x**2 for x in range(10)]  # list

    # ジェネレータ式（遅延評価）
    squares_gen = (x**2 for x in range(10))   # generator

    # メモリ効率の違い:
    # squares_list: 10個のint を全てメモリに保持
    # squares_gen:  1個ずつ生成、メモリはO(1)

    # パイプライン（関数の合成）
    import itertools
    result = list(
        itertools.islice(                    # take(5)
            filter(lambda x: x % 2 == 0,    # filter(even)
                (x**2 for x in range(100))   # map(square)
            ),
            5
        )
    )
    print(result)  # [0, 4, 16, 36, 64]


# ============================
# 使用例
# ============================
if __name__ == "__main__":
    # カスタム Range
    for n in MyRange(1, 5):
        print(n, end=" ")  # 1 2 3 4
    print()

    # チャンク分割
    for c in chunk(range(1, 11), 3):
        print(c)
    # [1, 2, 3]
    # [4, 5, 6]
    # [7, 8, 9]
    # [10]

    # itertools の活用
    demonstrate_generators()
```

---

## 7. 外部イテレータ vs 内部イテレータ

### コード例 7: 両方のスタイルの比較と使い分け

```typescript
// external-vs-internal.ts -- 外部イテレータと内部イテレータ

// ============================
// 外部イテレータ (External Iterator)
// ============================
// 利用者が明示的に next() を呼んで制御する
class ExternalIterator<T> {
  private iterator: Iterator<T>;

  constructor(iterable: Iterable<T>) {
    this.iterator = iterable[Symbol.iterator]();
  }

  /** 次の要素があるか */
  hasNext(): boolean {
    const result = this.iterator.next();
    if (result.done) return false;
    // peek のために値を戻す必要があるが、Iterator は巻き戻せない
    // → 実用的には1要素先読みバッファが必要
    return true;
  }

  /** 次の要素を取得 */
  next(): T {
    const result = this.iterator.next();
    if (result.done) throw new Error('No more elements');
    return result.value;
  }
}

// 先読み付き外部イテレータ
class PeekableIterator<T> implements Iterable<T> {
  private iterator: Iterator<T>;
  private buffer: T[] = [];

  constructor(source: Iterable<T>) {
    this.iterator = source[Symbol.iterator]();
  }

  /** 次の要素を消費せずに確認 */
  peek(): T | undefined {
    if (this.buffer.length === 0) {
      const result = this.iterator.next();
      if (result.done) return undefined;
      this.buffer.push(result.value);
    }
    return this.buffer[0];
  }

  /** 次の要素を取得して消費 */
  next(): T | undefined {
    if (this.buffer.length > 0) {
      return this.buffer.shift();
    }
    const result = this.iterator.next();
    return result.done ? undefined : result.value;
  }

  hasNext(): boolean {
    return this.peek() !== undefined;
  }

  [Symbol.iterator](): Iterator<T> {
    return {
      next: () => {
        const value = this.next();
        if (value === undefined) {
          return { value: undefined, done: true };
        }
        return { value, done: false };
      },
    };
  }
}

// ============================
// 内部イテレータ (Internal Iterator)
// ============================
// コレクション側が走査を制御し、コールバックを呼ぶ
class InternalIterator<T> {
  constructor(private items: T[]) {}

  forEach(callback: (item: T, index: number) => void): void {
    for (let i = 0; i < this.items.length; i++) {
      callback(this.items[i], i);
    }
  }

  map<U>(fn: (item: T) => U): U[] {
    const result: U[] = [];
    this.forEach(item => result.push(fn(item)));
    return result;
  }

  find(predicate: (item: T) => boolean): T | undefined {
    for (const item of this.items) {
      if (predicate(item)) return item;
    }
    return undefined;
  }
}

// ============================
// 外部イテレータの活用例: トークナイザー
// ============================
function* tokenize(input: string): Generator<{ type: string; value: string }> {
  const patterns = [
    { type: 'number', regex: /^\d+/ },
    { type: 'operator', regex: /^[+\-*/]/ },
    { type: 'paren', regex: /^[()]/ },
    { type: 'whitespace', regex: /^\s+/ },
  ];

  let pos = 0;
  while (pos < input.length) {
    let matched = false;
    for (const { type, regex } of patterns) {
      const match = input.slice(pos).match(regex);
      if (match) {
        if (type !== 'whitespace') {
          yield { type, value: match[0] };
        }
        pos += match[0].length;
        matched = true;
        break;
      }
    }
    if (!matched) {
      throw new Error(`Unexpected character at position ${pos}: '${input[pos]}'`);
    }
  }
}

// PeekableIterator でパーサーを実装
function parseExpression(tokens: PeekableIterator<{ type: string; value: string }>): number {
  let result = parseTerm(tokens);

  while (tokens.peek()?.type === 'operator' &&
         ['+', '-'].includes(tokens.peek()!.value)) {
    const op = tokens.next()!.value;
    const right = parseTerm(tokens);
    result = op === '+' ? result + right : result - right;
  }

  return result;
}

function parseTerm(tokens: PeekableIterator<{ type: string; value: string }>): number {
  const token = tokens.next();
  if (!token) throw new Error('Unexpected end of input');
  if (token.type === 'number') return parseInt(token.value);
  if (token.type === 'paren' && token.value === '(') {
    const result = parseExpression(tokens);
    tokens.next(); // consume ')'
    return result;
  }
  throw new Error(`Unexpected token: ${token.value}`);
}

// 使用例
const tokens = new PeekableIterator(tokenize('3 + 5 - 2'));
console.log(parseExpression(tokens)); // 6
```

```
外部 vs 内部イテレータの比較:

  外部イテレータ (for...of, next())
  ┌──────────┐                    ┌──────────┐
  │ 利用者   │── next() ─────────►│ Iterator │
  │          │◄── { value, done } │          │
  │ 制御権は │                    │ 値を提供 │
  │ 利用者側 │                    │          │
  └──────────┘                    └──────────┘

  内部イテレータ (forEach, map)
  ┌──────────┐                    ┌───────────┐
  │ 利用者   │── callback ───────►│ Collection│
  │          │                    │           │
  │ コール   │◄── callback(item) ─│ 走査を    │
  │ バックを │                    │ 制御する  │
  │ 渡すだけ │                    │           │
  └──────────┘                    └───────────┘

  使い分け:
  外部: 複数イテレータの同時走査、peek、早期終了
  内部: シンプルな全要素処理、関数型スタイル
```

---

## 8. 深掘り: Iterator の設計判断

### イテレータの使い捨て問題

```
Iterator は1回しか走査できない（使い捨て）:
  const gen = fibonacci();
  [...gen]  // [0, 1, 1, 2, ...]
  [...gen]  // [] ← 空! 使い切り

Iterable は何度でも走査可能:
  const range = new Range(1, 5);
  [...range]  // [1, 2, 3, 4]
  [...range]  // [1, 2, 3, 4] ← 何度でもOK

理由: Iterable は [Symbol.iterator]() で毎回新しい Iterator を生成する
     Generator は1つの Iterator インスタンスで、状態を内部に保持

  ★ 設計指針:
    再利用するコレクション → Iterable インターフェースを実装（class + [Symbol.iterator]）
    1回限りのストリーム → Generator 関数で十分
```

### 遅延評価 vs 即時評価

```
即時評価 (Array.map/filter):
  [1,2,3,4,5].filter(x => x > 2).map(x => x * 2)
  ステップ1: filter で中間配列 [3,4,5] を生成
  ステップ2: map で結果配列 [6,8,10] を生成
  → 2つの配列がメモリに存在

遅延評価 (Iterator パイプライン):
  Iter.from([1,2,3,4,5]).filter(x => x > 2).map(x => x * 2).toArray()
  → 中間配列なし、1要素ずつ処理
  → 1 → filter(skip) → 2 → filter(skip) → 3 → filter(pass) → map(6) → ...

判断基準:
  データが小さい(< 1000件) → 即時評価でOK（可読性優先）
  データが大きい(> 10000件) → 遅延評価が有効
  無限シーケンス → 遅延評価が必須
  複数回走査 → 即時評価（配列に変換）
```

---

## 9. 比較表

### 走査方法の比較

| 特性 | for ループ | Array.map/filter | Generator | Iter パイプライン |
|------|-----------|-----------------|-----------|-----------------|
| 遅延評価 | 不可 | 不可（即座に配列生成） | 可能 | 可能 |
| 無限シーケンス | 不可 | 不可 | 可能 | 可能 |
| メモリ効率 | 良い | 中間配列が生成 | 最良 | 最良 |
| チェーン可読性 | 低い | 高い | 中 | 高い |
| 非同期対応 | 要工夫 | なし | AsyncGenerator | 拡張可能 |
| 早期終了 (break) | 可能 | 不可（forEach） | 可能 | 可能 |
| TypeScript 型推論 | 完全 | 完全 | 完全 | 完全 |

### イテレータの種類

| イテレータ種類 | 同期 Iterator | AsyncIterator | Generator | AsyncGenerator |
|--------------|-------------|---------------|-----------|---------------|
| プロトコル | Symbol.iterator | Symbol.asyncIterator | function* | async function* |
| 構文 | for...of | for await...of | yield | yield (async内) |
| ユースケース | コレクション走査 | API/Stream処理 | 遅延シーケンス | ページネーション |
| 実行制御 | next() | next() (Promise) | yield で停止 | yield + await |
| リソース解放 | return() | return() | try/finally | try/finally |

### 言語間のイテレータ対応

| 概念 | JavaScript/TS | Python | Rust | Java |
|------|--------------|--------|------|------|
| Iterable | Symbol.iterator | `__iter__` | `IntoIterator` | `Iterable<T>` |
| Iterator | `{ next() }` | `__next__` | `Iterator` trait | `Iterator<T>` |
| Generator | `function*` | `yield` | -- (Iterator で代替) | -- |
| 遅延パイプライン | 自前実装 | `itertools` | `.iter().map().filter()` | `Stream API` |
| 非同期イテレータ | `async function*` | `async for` | `Stream` | `Flow (Kotlin)` |

---

## 10. アンチパターン

### アンチパターン 1: 全データをメモリに載せてから処理

```typescript
// ============================
// [NG] 100万件を配列に全て読み込み
// ============================
const allUsers = await fetchAllUsers(); // メモリに100万件!
const activeUsers = allUsers.filter(u => u.active);
const names = activeUsers.map(u => u.name);
// → 中間配列が2つ生成される
// → メモリ使用量: 100万件 × 3 配列

// ============================
// [OK] ジェネレータで1件ずつ処理
// ============================
async function* fetchActiveUserNames(): AsyncGenerator<string> {
  for await (const user of fetchAllPages<User>('/api/users')) {
    if (user.active) {
      yield user.name;
    }
  }
}

// メモリ使用量はページサイズ分のみ
const names2: string[] = [];
for await (const name of fetchActiveUserNames()) {
  names2.push(name);
  if (names2.length >= 100) break; // 早期終了も可能
}
```

### アンチパターン 2: イテレータの再利用

```typescript
// ============================
// [NG] 同じイテレータを2回消費しようとする
// ============================
function* nums() { yield 1; yield 2; yield 3; }
const iter = nums();

console.log([...iter]); // [1, 2, 3]
console.log([...iter]); // [] ← 空! イテレータは使い切り

// ============================
// [OK 方法1] Iterable を保持し、必要なときに新しい Iterator を生成
// ============================
const range = new Range(1, 4);
console.log([...range]); // [1, 2, 3]
console.log([...range]); // [1, 2, 3] ← Iterable なので何度でも

// ============================
// [OK 方法2] ジェネレータをファクトリ関数でラップ
// ============================
function numsIterable(): Iterable<number> {
  return {
    *[Symbol.iterator]() { yield 1; yield 2; yield 3; }
  };
}

const reusable = numsIterable();
console.log([...reusable]); // [1, 2, 3]
console.log([...reusable]); // [1, 2, 3] ← OK
```

### アンチパターン 3: 非同期処理での forEach / map の誤用

```typescript
// ============================
// [NG] forEach 内で async/await が期待通り動かない
// ============================
const urls = ['url1', 'url2', 'url3'];

// forEach は同期的に全コールバックを起動する
urls.forEach(async (url) => {
  const data = await fetch(url);
  console.log(data); // 順序不定、エラーも捕捉できない
});
console.log('完了'); // ← forEach のコールバックより先に実行される!

// ============================
// [OK] for...of + await で逐次処理
// ============================
for (const url of urls) {
  const data = await fetch(url);
  console.log(data); // 順序保証、try/catch で捕捉可能
}
console.log('完了'); // ← 全 fetch 完了後に実行

// [OK] 並列実行が必要な場合は Promise.all
const results = await Promise.all(urls.map(url => fetch(url)));
```

---

## 11. 演習問題

### 演習 1（基礎）: 二分木のイテレータ

以下の仕様を満たす二分木のイテレータを実装してください。

**仕様:**
- 二分探索木 (BST) クラスを定義
- 3つの走査方法を Generator で実装: `inOrder()`, `preOrder()`, `postOrder()`
- デフォルトの `[Symbol.iterator]` は中順 (in-order) 走査

```typescript
// ヒント
interface BSTNode<T> {
  value: T;
  left: BSTNode<T> | null;
  right: BSTNode<T> | null;
}
```

**期待される出力:**
```
BST に 5, 3, 7, 1, 4, 6, 8 を挿入

inOrder:   [1, 3, 4, 5, 6, 7, 8]  // 昇順ソート
preOrder:  [5, 3, 1, 4, 7, 6, 8]  // 根 → 左 → 右
postOrder: [1, 4, 3, 6, 8, 7, 5]  // 左 → 右 → 根

for (const value of bst) { ... }   // inOrder で走査
[...bst]                            // [1, 3, 4, 5, 6, 7, 8]
```

---

### 演習 2（応用）: 非同期パイプライン

以下の仕様を満たす非同期イテレータパイプラインを実装してください。

**仕様:**
- `AsyncIter<T>` クラスを実装
- `map`, `filter`, `take`, `buffer` (N件ずつバッチ化) を遅延評価で提供
- `for await...of` で消費可能

**期待される出力:**
```
const pipeline = AsyncIter.from(fetchAllPages<User>('/api/users'))
  .filter(user => user.active)
  .map(user => user.name)
  .buffer(10)  // 10件ずつバッチ化
  .take(5);    // 5バッチ（最大50件）

for await (const batch of pipeline) {
  console.log(`Processing batch of ${batch.length} names`);
  await saveBatch(batch);
}
```

---

### 演習 3（発展）: コルーチンによるジョブスケジューラ

以下の仕様を満たすコルーチンベースのジョブスケジューラを実装してください。

**仕様:**
- Generator の `next(value)` による双方向通信を活用
- ジョブは Generator 関数で定義し、`yield` で実行権を返す
- スケジューラが複数ジョブをラウンドロビンで実行

```typescript
// ヒント: ジョブの定義
function* downloadJob(url: string): Generator<string, void, void> {
  yield `Starting download: ${url}`;
  yield `Downloading... 50%`;
  yield `Downloading... 100%`;
  yield `Download complete: ${url}`;
}
```

**期待される出力:**
```
scheduler.add(downloadJob('file1.zip'));
scheduler.add(downloadJob('file2.zip'));
scheduler.run();

// ラウンドロビン実行:
// [Job1] Starting download: file1.zip
// [Job2] Starting download: file2.zip
// [Job1] Downloading... 50%
// [Job2] Downloading... 50%
// [Job1] Downloading... 100%
// [Job2] Downloading... 100%
// [Job1] Download complete: file1.zip
// [Job2] Download complete: file2.zip
```

---

## 12. FAQ

### Q1: ジェネレータと普通の関数、どちらを使うべきですか？

データ量が有限で小さい場合は通常の関数（配列を返す）で十分です。以下の場合にジェネレータを検討してください。

1. **無限シーケンス**: フィボナッチ数列、乱数列、連番など終わりのないデータ
2. **大量データの逐次処理**: ファイル読み込み、API ページネーション、ログ解析
3. **途中で処理を中断する可能性**: 条件に合う最初の要素を見つけたら停止
4. **メモリ制約が厳しい場面**: 中間配列を生成したくない

ジェネレータの最大のメリットは「**必要になるまで計算しない**」遅延評価です。

### Q2: for...of と forEach の違いは何ですか？

| 比較項目 | `for...of` | `Array.forEach` |
|---------|-----------|----------------|
| 対象 | 全 Iterable | Array のみ |
| break | 可能 | 不可能 |
| continue | 可能 | return で代替 |
| async/await | 正常動作 | 期待通り動かない |
| Generator 対応 | 可能 | 不可能 |
| return | ループを抜ける | コールバックを抜ける（ループは継続） |

基本方針: `for...of` を優先し、`forEach` は副作用のない単純な全要素処理のみに使用。

### Q3: AsyncGenerator のエラーハンドリングはどうすべきですか？

`for await...of` のブロック内で `try/catch` を使います。また、Generator の `throw()` メソッドで外部からエラーを注入することもできます。

```typescript
// 実践的なエラーハンドリング
async function* resilientFetch<T>(
  urls: string[]
): AsyncGenerator<T> {
  for (const url of urls) {
    try {
      const res = await fetch(url);
      if (!res.ok) {
        console.warn(`Skipping ${url}: ${res.status}`);
        continue; // 個別ページのエラーはスキップ
      }
      const data = await res.json();
      yield data;
    } catch (err) {
      console.warn(`Network error for ${url}:`, err);
      // 致命的でなければ continue、致命的なら throw
    }
  }
}
```

ページネーションのような処理では、個別ページのエラーはリトライまたはスキップし、致命的なエラーのみ全体を中断するような設計が実用的です。`finally` ブロックでリソースのクリーンアップ（ストリームの解放、コネクションのクローズ）を忘れないでください。

### Q4: Iterator パターンは関数型プログラミングとどう関係しますか？

Iterator パターンは関数型プログラミングの **遅延リスト (Lazy List)** と本質的に同じ概念です。Haskell のリストは遅延評価で、必要な要素のみ計算されます。JavaScript の Generator はこの概念を命令型言語に持ち込んだものです。

Rust の `Iterator` trait は、`map`, `filter`, `fold` などの関数型操作を遅延評価で提供しており、Iterator パターンと関数型プログラミングの融合の好例です。

### Q5: Generator の yield* と yield の違いは何ですか？

`yield` は1つの値を送出します。`yield*` は別の Iterable のすべての値を順に送出します（デリゲート）。

```typescript
function* example() {
  yield 1;               // 1つの値
  yield* [2, 3, 4];      // 配列の全要素を順に
  yield* anotherGen();   // 別の Generator の全要素を順に
  yield 5;
}
// 結果: 1, 2, 3, 4, (anotherGen の全出力), 5
```

`yield*` はツリーの再帰走査で特に有用です。子ノードのイテレータに走査を委譲する場合に、明示的なループなしで自然に書けます。

---

## まとめ

| 項目 | 要点 |
|------|------|
| Iterator プロトコル | `Symbol.iterator` + `next()` で統一的な走査インターフェース |
| Iterable vs Iterator | Iterable は何度でも走査可能、Iterator は1回使い切り |
| for...of | Iterator プロトコルに基づくループ構文。break/continue/await 対応 |
| Generator | `function*` と `yield` による遅延評価。無限シーケンスも表現可能 |
| yield* | 別のイテレータにデリゲート。ツリー走査の再帰に有効 |
| AsyncGenerator | `async function*` で非同期データの逐次処理 |
| パイプライン | map/filter/take を遅延評価で連鎖。中間配列なし |
| 外部 vs 内部 | 外部: 利用者が制御（peek, 複数同時走査）、内部: コレクションが制御（forEach） |

---

## 次に読むべきガイド

- [02-command.md](./02-command.md) -- Command パターンと操作のカプセル化（Command 履歴の走査に Iterator を活用）
- [03-state.md](./03-state.md) -- State パターンと状態遷移
- [../01-structural/04-composite.md](../01-structural/04-composite.md) -- Composite パターン（ツリー走査に Iterator を活用）
- [../03-functional/02-fp-patterns.md](../03-functional/02-fp-patterns.md) -- 関数型パターン（パイプライン、合成）
- [../03-functional/00-monad.md](../03-functional/00-monad.md) -- モナドパターン（Iterator は List モナドと関連）

---

## 参考文献

1. **Design Patterns: Elements of Reusable Object-Oriented Software** -- Gamma, Helm, Johnson, Vlissides (GoF, 1994) -- Iterator パターンの原典。Chapter 5, pp.257-271
2. **MDN - Iterators and generators** -- https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Iterators_and_Generators -- JavaScript のイテレータとジェネレータの公式リファレンス
3. **Exploring ES6 - Iterables and iterators** -- Axel Rauschmayer -- ES6 仕様に基づく詳細な解説
4. **Python itertools documentation** -- https://docs.python.org/3/library/itertools.html -- Python のイテレータユーティリティ
5. **Rust Iterator trait documentation** -- https://doc.rust-lang.org/std/iter/trait.Iterator.html -- 遅延評価イテレータの優れた実装例
