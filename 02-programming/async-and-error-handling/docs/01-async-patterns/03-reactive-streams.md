# Reactive Streams

> Reactive Streams は「非同期データストリーム」を宣言的に処理するパターン。RxJS の Observable、バックプレッシャー、イベント駆動アーキテクチャの基盤を理解する。

## この章で学ぶこと

- [ ] Observable パターンの仕組みを理解する
- [ ] RxJS のオペレータとパイプラインを把握する
- [ ] バックプレッシャーの概念を学ぶ
- [ ] Subject の種類と使い分けを理解する
- [ ] Angular・React でのリアクティブパターンを習得する
- [ ] テスト・デバッグの手法を身につける

---

## 1. Promise vs Observable

```
Promise:
  → 単一の値（1回だけ解決）
  → 作成時に実行開始（eager）
  → キャンセル不可

Observable:
  → 複数の値（時間をかけて流れるストリーム）
  → 購読時に実行開始（lazy）
  → キャンセル可能（unsubscribe）

  Promise:    ──────●              （1つの値）
  Observable: ──●──●──●──●──│     （複数の値、完了あり）
              ──●──●──✗              （エラーで終了）

用途:
  Promise: API呼び出し、DBクエリ（1回の結果）
  Observable: WebSocket、ユーザー入力、タイマー（連続的なイベント）
```

### 1.1 Observable のライフサイクル

```
Observable のライフサイクル:

  作成 (Creation)
    │
    ▼
  購読 (Subscription)  ← subscribe() の呼び出し
    │
    ▼
  値の発行 (Emission)
    │  next(value)  ──→ Observer の next コールバック
    │  next(value)  ──→ Observer の next コールバック
    │  ...
    │
    ├── complete()  ──→ Observer の complete コールバック ──→ 終了
    │
    └── error(err)  ──→ Observer の error コールバック ──→ 終了

  購読解除 (Unsubscription)  ← unsubscribe() の呼び出し
    │
    ▼
  リソース解放 (Teardown)  ← teardown ロジックの実行
```

### 1.2 Hot Observable vs Cold Observable

```
Cold Observable:
  → 購読するたびに新しいデータストリームが作成される
  → 各購読者が独立したストリームを受け取る
  → 例: HTTPリクエスト、ファイル読み取り

  Subscriber A: ──1──2──3──4──│
  Subscriber B:    ──1──2──3──4──│  (独立したストリーム)

Hot Observable:
  → データソースが購読に関係なく値を発行し続ける
  → 途中から購読すると、過去の値は受け取れない
  → 例: WebSocket、マウスイベント、株価フィード

  Source:       ──1──2──3──4──5──6──│
  Subscriber A: ──1──2──3──4──5──6──│  (最初から購読)
  Subscriber B:       ──3──4──5──6──│  (途中から購読)
```

```typescript
// Cold Observable の例
const cold$ = new Observable(subscriber => {
  // 購読するたびに新しいランダム値
  subscriber.next(Math.random());
  subscriber.complete();
});

cold$.subscribe(v => console.log('A:', v)); // A: 0.123...
cold$.subscribe(v => console.log('B:', v)); // B: 0.456... (異なる値)

// Hot Observable の例（Subject を使用）
const hot$ = new Subject<number>();

hot$.subscribe(v => console.log('A:', v));
hot$.next(1); // A: 1
hot$.next(2); // A: 2

hot$.subscribe(v => console.log('B:', v));
hot$.next(3); // A: 3, B: 3 (Bは3からしか受け取れない)
```

---

## 2. RxJS の基本

```typescript
import { Observable, of, from, interval, fromEvent } from 'rxjs';
import { map, filter, take, debounceTime, switchMap } from 'rxjs/operators';

// Observable の作成
const numbers$ = of(1, 2, 3, 4, 5);
const array$ = from([10, 20, 30]);
const timer$ = interval(1000); // 1秒ごとに 0, 1, 2, ...

// パイプライン（オペレータチェーン）
numbers$.pipe(
  filter(n => n % 2 === 0),  // 偶数のみ
  map(n => n * 10),          // 10倍
).subscribe(value => console.log(value)); // 20, 40

// 検索ボックスの実践例
const searchInput = document.getElementById('search');
fromEvent(searchInput, 'input').pipe(
  debounceTime(300),                           // 300ms 入力停止を待つ
  map(event => (event.target as HTMLInputElement).value),
  filter(query => query.length >= 2),          // 2文字以上
  switchMap(query => fetch(`/api/search?q=${query}`).then(r => r.json())),
  // switchMap: 新しい値が来たら前のリクエストをキャンセル
).subscribe(results => {
  renderSearchResults(results);
});
```

### 2.1 Observable の作成方法

```typescript
import {
  Observable, of, from, interval, timer, fromEvent,
  defer, range, EMPTY, NEVER, throwError,
  generate, iif
} from 'rxjs';
import { ajax } from 'rxjs/ajax';

// 1. カスタム Observable
const custom$ = new Observable<number>(subscriber => {
  subscriber.next(1);
  subscriber.next(2);
  subscriber.next(3);
  setTimeout(() => {
    subscriber.next(4);
    subscriber.complete();
  }, 1000);

  // teardown ロジック（購読解除時に実行）
  return () => {
    console.log('クリーンアップ処理');
  };
});

// 2. 静的生成関数
const values$ = of('a', 'b', 'c');                    // 同期的に3つの値を発行
const arr$ = from([1, 2, 3]);                          // 配列からObservable
const promise$ = from(fetch('/api/data'));              // PromiseからObservable
const iter$ = from(new Map([['a', 1], ['b', 2]]));    // IterableからObservable

// 3. タイマー系
const interval$ = interval(1000);                       // 0, 1, 2, ... (1秒間隔)
const timerOnce$ = timer(3000);                         // 3秒後に0を発行
const timerRepeat$ = timer(0, 1000);                    // 即座に開始、1秒間隔

// 4. イベント系
const clicks$ = fromEvent(document, 'click');
const resize$ = fromEvent(window, 'resize');
const keydown$ = fromEvent<KeyboardEvent>(document, 'keydown');

// 5. AJAX
const data$ = ajax.getJSON('/api/users');
const post$ = ajax({
  url: '/api/users',
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: { name: 'Tanaka', email: 'tanaka@example.com' },
});

// 6. 条件分岐
const source$ = iif(
  () => Math.random() > 0.5,
  of('heads'),
  of('tails'),
);

// 7. 遅延生成（購読時に初めてObservableを作成）
const deferred$ = defer(() => {
  const timestamp = Date.now();
  return of(timestamp);
});

// 8. ジェネレータ風
const fib$ = generate(
  [0, 1],                           // 初期値
  ([a, b]) => a < 100,              // 条件
  ([a, b]) => [b, a + b] as [number, number],  // 更新
  ([a, b]) => a,                     // 結果の選択
);
// → 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89
```

### 2.2 Observer パターンの詳細

```typescript
// Observer インターフェース
interface Observer<T> {
  next: (value: T) => void;
  error: (err: any) => void;
  complete: () => void;
}

// 完全な Observer を渡す
const subscription = numbers$.subscribe({
  next: value => console.log('値:', value),
  error: err => console.error('エラー:', err),
  complete: () => console.log('完了'),
});

// 部分的な Observer（省略可能）
numbers$.subscribe(
  value => console.log(value),          // next のみ
);

numbers$.subscribe({
  next: value => console.log(value),
  error: err => console.error(err),     // complete 省略
});

// Subscription の管理
const sub = interval(1000).subscribe(v => console.log(v));

// 5秒後に購読解除
setTimeout(() => {
  sub.unsubscribe(); // リソース解放
  console.log('購読解除しました');
}, 5000);

// 複数の Subscription をまとめて管理
import { Subscription } from 'rxjs';

const parentSub = new Subscription();

parentSub.add(interval(1000).subscribe(v => console.log('A:', v)));
parentSub.add(interval(2000).subscribe(v => console.log('B:', v)));
parentSub.add(interval(3000).subscribe(v => console.log('C:', v)));

// まとめて購読解除
setTimeout(() => parentSub.unsubscribe(), 10000);
```

---

## 3. 主要オペレータ

```
変換:
  map       — 値を変換
  switchMap — 新しい Observable に切り替え（前をキャンセル）
  mergeMap  — 並行して Observable を実行
  concatMap — 直列に Observable を実行
  exhaustMap — 現在の Observable が完了するまで新しいものを無視
  scan      — 累積値を計算（reduce のストリーム版）
  pluck     — オブジェクトの特定プロパティを抽出
  pairwise  — 直前の値と現在の値をペアにする

フィルタリング:
  filter       — 条件に合う値のみ通す
  take         — 最初のN個だけ取得
  takeUntil    — 別のObservableが発行するまで取得
  takeWhile    — 条件がtrueの間取得
  skip         — 最初のN個をスキップ
  debounceTime — 一定時間入力がなかったら通す
  throttleTime — 一定時間に1つだけ通す
  distinctUntilChanged — 値が変わった時だけ通す
  first        — 最初の値（条件付き可）
  last         — 最後の値（条件付き可）
  elementAt    — N番目の値

結合:
  merge       — 複数の Observable を合流
  combineLatest — 各 Observable の最新値を組み合わせ
  zip         — 各 Observable の値を1対1で組み合わせ
  forkJoin    — 全ての Observable の最後の値を取得（Promise.allに近い）
  concat      — 直列に結合（前が完了してから次）
  race        — 最も早く値を発行した Observable を採用
  withLatestFrom — メインストリームの発行時に他の最新値を付加

エラー:
  catchError  — エラーをハンドリングして回復
  retry       — エラー時にリトライ
  retryWhen   — 条件付きリトライ（RxJS 7 で非推奨、retry に統合）

ユーティリティ:
  tap         — 副作用（デバッグ、ログ）
  delay       — 値の発行を遅延
  timeout     — 一定時間値が来なければエラー
  finalize    — 完了/エラー/購読解除時のクリーンアップ
  share       — Cold を Hot に変換（マルチキャスト）
  shareReplay — 最新N個の値をリプレイ
```

### 3.1 変換オペレータの詳細

```typescript
import {
  of, from, interval, fromEvent, timer
} from 'rxjs';
import {
  map, switchMap, mergeMap, concatMap, exhaustMap,
  scan, pairwise, bufferTime, groupBy, toArray
} from 'rxjs/operators';

// === switchMap: 最新のリクエストのみ（検索、オートコンプリート） ===
const search$ = fromEvent<Event>(searchInput, 'input').pipe(
  debounceTime(300),
  map(e => (e.target as HTMLInputElement).value),
  switchMap(query =>
    // 新しい入力が来ると前のリクエストをキャンセル
    fetch(`/api/search?q=${query}`).then(r => r.json())
  ),
);

// === mergeMap: 並行実行（メール一括送信、ファイル並行ダウンロード） ===
const sendEmails$ = from(emailList).pipe(
  mergeMap(
    email => sendEmail(email),
    5,  // 最大並行数を5に制限
  ),
);

// === concatMap: 直列実行（順序保証が必要な場合） ===
const uploadFiles$ = from(files).pipe(
  concatMap(file =>
    // 1つずつ順番にアップロード（前が完了してから次）
    uploadFile(file)
  ),
);

// === exhaustMap: 二重送信防止（フォーム送信ボタン） ===
const submitForm$ = fromEvent(submitBtn, 'click').pipe(
  exhaustMap(() =>
    // 前のリクエストが進行中なら新しいクリックを無視
    fetch('/api/submit', { method: 'POST', body: formData })
      .then(r => r.json())
  ),
);

// === scan: 累積計算（状態管理） ===
const actions$ = new Subject<{ type: string; payload: any }>();
const state$ = actions$.pipe(
  scan((state, action) => {
    switch (action.type) {
      case 'INCREMENT':
        return { ...state, count: state.count + 1 };
      case 'DECREMENT':
        return { ...state, count: state.count - 1 };
      case 'SET_NAME':
        return { ...state, name: action.payload };
      default:
        return state;
    }
  }, { count: 0, name: '' }),
);

// === pairwise: 直前の値と比較 ===
const scrollPosition$ = fromEvent(window, 'scroll').pipe(
  map(() => window.scrollY),
  pairwise(),
  map(([prev, curr]) => ({
    direction: curr > prev ? 'down' : 'up',
    delta: Math.abs(curr - prev),
  })),
);

// === bufferTime: 一定時間ごとにバッチ処理 ===
const events$ = fromEvent(document, 'mousemove').pipe(
  bufferTime(1000),  // 1秒ごとにイベントの配列として発行
  filter(events => events.length > 0),
  map(events => ({
    count: events.length,
    avgX: events.reduce((sum, e: any) => sum + e.clientX, 0) / events.length,
  })),
);

// === groupBy: ストリームをグループ分け ===
interface LogEntry {
  level: 'info' | 'warn' | 'error';
  message: string;
}

const logs$ = from<LogEntry[]>([
  { level: 'info', message: 'Started' },
  { level: 'error', message: 'Failed' },
  { level: 'info', message: 'Processing' },
  { level: 'warn', message: 'Slow query' },
  { level: 'error', message: 'Timeout' },
]);

logs$.pipe(
  groupBy(log => log.level),
  mergeMap(group$ =>
    group$.pipe(
      toArray(),
      map(entries => ({ level: group$.key, entries })),
    )
  ),
).subscribe(group => {
  console.log(`${group.level}: ${group.entries.length} entries`);
});
```

### 3.2 flattening オペレータの比較

```
switchMap vs mergeMap vs concatMap vs exhaustMap:

入力:    ──A─────B─────C──│

switchMap（最新のみ）:
  A: ──a1──a2──(キャンセル)
  B:       ──b1──b2──(キャンセル)
  C:             ──c1──c2──c3──│
  出力: ──a1──a2──b1──b2──c1──c2──c3──│

mergeMap（並行）:
  A: ──a1──a2──a3──│
  B:       ──b1──b2──b3──│
  C:             ──c1──c2──c3──│
  出力: ──a1──a2──b1──a3──b2──c1──b3──c2──c3──│

concatMap（直列）:
  A: ──a1──a2──a3──│
  B:               ──b1──b2──b3──│
  C:                             ──c1──c2──c3──│
  出力: ──a1──a2──a3──b1──b2──b3──c1──c2──c3──│

exhaustMap（進行中は無視）:
  A: ──a1──a2──a3──│
  B:  (無視)
  C:             ──c1──c2──c3──│
  出力: ──a1──a2──a3──c1──c2──c3──│

使い分け:
  switchMap  → 検索、オートコンプリート（最新のみ必要）
  mergeMap   → 並行ダウンロード（全結果必要、順序不問）
  concatMap  → ファイル順序処理（順序保証必要）
  exhaustMap → フォーム送信（二重送信防止）
```

### 3.3 結合オペレータの詳細

```typescript
import {
  merge, combineLatest, zip, forkJoin, concat, race, withLatestFrom
} from 'rxjs';

// === merge: 複数ストリームの合流 ===
const keyboard$ = fromEvent(document, 'keydown');
const mouse$ = fromEvent(document, 'click');
const touch$ = fromEvent(document, 'touchstart');

const userActivity$ = merge(keyboard$, mouse$, touch$).pipe(
  throttleTime(1000),
  tap(() => resetIdleTimer()),
);

// === combineLatest: 各ストリームの最新値を組み合わせ ===
const selectedCategory$ = new BehaviorSubject<string>('all');
const searchQuery$ = new BehaviorSubject<string>('');
const sortOrder$ = new BehaviorSubject<string>('newest');

const filteredProducts$ = combineLatest([
  selectedCategory$,
  searchQuery$,
  sortOrder$,
]).pipe(
  debounceTime(100),
  switchMap(([category, query, sort]) =>
    fetch(`/api/products?category=${category}&q=${query}&sort=${sort}`)
      .then(r => r.json())
  ),
);

// === zip: 1対1で組み合わせ ===
const names$ = of('Alice', 'Bob', 'Charlie');
const ages$ = of(25, 30, 35);
const cities$ = of('Tokyo', 'Osaka', 'Kyoto');

zip(names$, ages$, cities$).pipe(
  map(([name, age, city]) => ({ name, age, city })),
).subscribe(person => console.log(person));
// { name: 'Alice', age: 25, city: 'Tokyo' }
// { name: 'Bob', age: 30, city: 'Osaka' }
// { name: 'Charlie', age: 35, city: 'Kyoto' }

// === forkJoin: 全完了後に最後の値（Promise.all 相当） ===
const dashboardData$ = forkJoin({
  users: ajax.getJSON('/api/users'),
  orders: ajax.getJSON('/api/orders'),
  stats: ajax.getJSON('/api/stats'),
  notifications: ajax.getJSON('/api/notifications'),
}).pipe(
  catchError(err => {
    console.error('ダッシュボードデータ取得失敗:', err);
    return of({ users: [], orders: [], stats: null, notifications: [] });
  }),
);

// === race: 最速のストリームを採用 ===
const primary$ = ajax.getJSON('https://primary-api.com/data');
const fallback$ = ajax.getJSON('https://fallback-api.com/data');

const data$ = race(primary$, fallback$); // 先に応答したほうを使用

// === withLatestFrom: メインストリーム発行時に他の最新値を付加 ===
const saveButton$ = fromEvent(saveBtn, 'click');
const formValue$ = new BehaviorSubject(getFormValues());

saveButton$.pipe(
  withLatestFrom(formValue$),
  switchMap(([_, formData]) =>
    fetch('/api/save', { method: 'POST', body: JSON.stringify(formData) })
  ),
).subscribe();
```

---

## 4. Subject の種類

```
Subject の種類と特徴:

Subject (基本):
  購読前の値は受け取れない
  A subscribes → next(1) → next(2) → B subscribes → next(3)
  A: 1, 2, 3
  B:       3

BehaviorSubject (最新値を保持):
  購読時に最新値を即座に受け取る。初期値が必要
  A subscribes(初期値0) → next(1) → next(2) → B subscribes → next(3)
  A: 0, 1, 2, 3
  B:          2, 3

ReplaySubject (N個の過去値をリプレイ):
  指定した数だけ過去の値をバッファして新しい購読者に送る
  A subscribes → next(1) → next(2) → next(3) → B subscribes(replay=2)
  A: 1, 2, 3
  B:          2, 3  (最新2個がリプレイされる)

AsyncSubject (最後の値のみ):
  complete() 時に最後の値だけを発行
  next(1) → next(2) → next(3) → complete()
  A: 3
  B: 3 (complete 後に購読しても最後の値を受け取る)
```

```typescript
import { Subject, BehaviorSubject, ReplaySubject, AsyncSubject } from 'rxjs';

// === BehaviorSubject: 現在の状態管理に最適 ===
interface AppState {
  user: User | null;
  theme: 'light' | 'dark';
  language: string;
}

class StateService {
  private state$ = new BehaviorSubject<AppState>({
    user: null,
    theme: 'light',
    language: 'ja',
  });

  // 現在の状態を取得（同期的）
  get currentState(): AppState {
    return this.state$.getValue();
  }

  // 状態のストリームを取得
  select<K extends keyof AppState>(key: K): Observable<AppState[K]> {
    return this.state$.pipe(
      map(state => state[key]),
      distinctUntilChanged(),
    );
  }

  // 状態を更新
  update(partial: Partial<AppState>): void {
    this.state$.next({
      ...this.currentState,
      ...partial,
    });
  }
}

const stateService = new StateService();

// テーマの変更を監視
stateService.select('theme').subscribe(theme => {
  document.body.className = `theme-${theme}`;
});

// ユーザー情報の変更を監視
stateService.select('user').subscribe(user => {
  if (user) {
    console.log(`Welcome, ${user.name}`);
  }
});

// === ReplaySubject: イベントの履歴を保持 ===
class EventBus {
  private events$ = new ReplaySubject<AppEvent>(10); // 最新10件を保持

  emit(event: AppEvent): void {
    this.events$.next(event);
  }

  on(type: string): Observable<AppEvent> {
    return this.events$.pipe(
      filter(event => event.type === type),
    );
  }

  // 過去のイベントも含めて取得
  history(): Observable<AppEvent> {
    return this.events$.asObservable();
  }
}

// === AsyncSubject: 完了時の最終結果 ===
class ConfigLoader {
  private config$ = new AsyncSubject<Config>();

  async load(): Promise<void> {
    try {
      const config = await fetch('/api/config').then(r => r.json());
      this.config$.next(config);
      this.config$.complete(); // complete() で最後の値が発行される
    } catch (err) {
      this.config$.error(err);
    }
  }

  getConfig(): Observable<Config> {
    return this.config$.asObservable();
  }
}
```

---

## 5. 実践例: リアルタイムダッシュボード

```typescript
import { combineLatest, timer, Subject, BehaviorSubject } from 'rxjs';
import {
  switchMap, catchError, retry, map, share,
  distinctUntilChanged, tap, takeUntil, startWith, scan
} from 'rxjs/operators';

// === ダッシュボードサービス ===
class DashboardService {
  private destroy$ = new Subject<void>();
  private refreshTrigger$ = new BehaviorSubject<void>(undefined);

  // 統計情報（5秒ごと + 手動リフレッシュ）
  readonly stats$ = this.refreshTrigger$.pipe(
    switchMap(() => timer(0, 5000)),
    switchMap(() =>
      fetch('/api/stats')
        .then(r => r.json())
        .catch(() => ({ error: true }))
    ),
    retry({ count: 3, delay: 2000 }),
    catchError(() => of({ error: true, data: null })),
    share(), // マルチキャスト（複数の購読者で共有）
    takeUntil(this.destroy$),
  );

  // アラート（10秒ごと）
  readonly alerts$ = timer(0, 10000).pipe(
    switchMap(() =>
      fetch('/api/alerts')
        .then(r => r.json())
        .catch(() => [])
    ),
    catchError(() => of([])),
    share(),
    takeUntil(this.destroy$),
  );

  // WebSocket でリアルタイム更新
  readonly liveEvents$ = new Observable<ServerEvent>(subscriber => {
    const ws = new WebSocket('wss://api.example.com/events');

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        subscriber.next(data);
      } catch (err) {
        console.error('WebSocket parse error:', err);
      }
    };

    ws.onerror = (err) => subscriber.error(err);
    ws.onclose = () => subscriber.complete();

    return () => ws.close();
  }).pipe(
    retry({ count: 5, delay: (error, retryCount) => timer(Math.min(1000 * Math.pow(2, retryCount), 30000)) }),
    share(),
    takeUntil(this.destroy$),
  );

  // ダッシュボードの統合状態
  readonly dashboardState$ = combineLatest([
    this.stats$.pipe(startWith(null)),
    this.alerts$.pipe(startWith([])),
    this.liveEvents$.pipe(
      scan((events: ServerEvent[], event) => [...events.slice(-50), event], []),
      startWith([]),
    ),
  ]).pipe(
    map(([stats, alerts, events]) => ({
      stats,
      alerts,
      events,
      lastUpdated: new Date(),
    })),
    distinctUntilChanged((prev, curr) =>
      JSON.stringify(prev) === JSON.stringify(curr)
    ),
  );

  refresh(): void {
    this.refreshTrigger$.next();
  }

  destroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }
}
```

---

## 6. バックプレッシャー

```
バックプレッシャー（Backpressure）:
  → 生産者（Producer）が消費者（Consumer）より速い場合の制御

  生産者: ──●●●●●●●●●●──→ 速い
  消費者: ──●───●───●───→ 遅い
                         → メモリ溢れ / 遅延蓄積

対策:
  1. バッファリング: 一時的にキューに溜める（メモリ限界あり）
  2. ドロップ: 古い値を捨てる（最新のみ保持）
  3. サンプリング: 一定間隔で最新値を取得
  4. スロットリング: 一定時間に1つだけ通す
  5. ウィンドウイング: 時間やカウントでグループ化

RxJS でのバックプレッシャー:
  → bufferTime: 時間ごとにバッチ処理
  → bufferCount: 個数ごとにバッチ処理
  → throttleTime: 一定間隔で値を通す
  → sampleTime: 一定間隔で最新値を取得
  → auditTime: 値が来てから一定時間後に最新値を通す
  → debounceTime: 値が来てから一定時間待ってから最新値を通す
  → window/windowTime: 時間窓でグループ化
```

### 6.1 バックプレッシャーの実践例

```typescript
import {
  fromEvent, interval, Subject
} from 'rxjs';
import {
  bufferTime, bufferCount, throttleTime, sampleTime,
  auditTime, debounceTime, windowTime, mergeAll,
  tap, filter, map, scan
} from 'rxjs/operators';

// === マウス移動の間引き ===
const mouseMove$ = fromEvent<MouseEvent>(document, 'mousemove');

// throttleTime: 最初の値を通し、指定時間は無視
mouseMove$.pipe(
  throttleTime(16), // ~60fps
  map(e => ({ x: e.clientX, y: e.clientY })),
).subscribe(pos => updateCursor(pos));

// sampleTime: 一定間隔で最新値を取得
mouseMove$.pipe(
  sampleTime(100), // 100msごとに最新の位置
  map(e => ({ x: e.clientX, y: e.clientY })),
).subscribe(pos => sendAnalytics(pos));

// auditTime: 値が来てから一定時間後に最新値を通す
mouseMove$.pipe(
  auditTime(200),
).subscribe(e => updateTooltip(e));

// === ログの一括送信 ===
const logStream$ = new Subject<LogEntry>();

// 100件ごと、または5秒ごとにバッチ送信
logStream$.pipe(
  bufferTime(5000, undefined, 100), // 5秒 or 100件
  filter(batch => batch.length > 0),
).subscribe(async batch => {
  await fetch('/api/logs', {
    method: 'POST',
    body: JSON.stringify(batch),
  });
});

// === ウィンドウでのメトリクス集計 ===
const requestStream$ = new Subject<{ endpoint: string; duration: number }>();

requestStream$.pipe(
  windowTime(60000), // 1分間のウィンドウ
  mergeAll(),
  scan((acc, req) => ({
    count: acc.count + 1,
    totalDuration: acc.totalDuration + req.duration,
    maxDuration: Math.max(acc.maxDuration, req.duration),
  }), { count: 0, totalDuration: 0, maxDuration: 0 }),
).subscribe(metrics => {
  console.log(`Requests/min: ${metrics.count}`);
  console.log(`Avg duration: ${metrics.totalDuration / metrics.count}ms`);
  console.log(`Max duration: ${metrics.maxDuration}ms`);
});

// === debounceTime vs throttleTime の違い ===
//
// debounceTime(300):
//   入力: ──a─b─c───────d─e──────│
//   出力: ──────────c─────────e──│
//   → 入力が落ち着いてから発行（検索に最適）
//
// throttleTime(300):
//   入力: ──a─b─c───────d─e──────│
//   出力: ──a───────────d────────│
//   → 最初の値をすぐ通し、指定時間待つ（スクロールに最適）
//
// auditTime(300):
//   入力: ──a─b─c───────d─e──────│
//   出力: ──────c───────────e────│
//   → 値が来てから指定時間後に最新値を通す
//
// sampleTime(300):
//   入力: ──a─b─c───────d─e──────│
//   出力: ──b─────c──────e───────│
//   → 一定間隔で最新値をサンプリング
```

---

## 7. エラーハンドリング

```typescript
import { of, throwError, timer, EMPTY, Observable } from 'rxjs';
import {
  catchError, retry, retryWhen, delay, take,
  tap, finalize, timeout, switchMap
} from 'rxjs/operators';

// === 基本的なエラーハンドリング ===
const data$ = ajax.getJSON('/api/data').pipe(
  catchError(err => {
    console.error('API error:', err);
    return of({ fallback: true, data: [] }); // フォールバック値
  }),
);

// === リトライ戦略 ===
// 単純なリトライ
const withRetry$ = ajax.getJSON('/api/unstable').pipe(
  retry(3), // 3回リトライ（合計4回試行）
  catchError(err => {
    console.error('全リトライ失敗:', err);
    return EMPTY;
  }),
);

// 指数バックオフ付きリトライ（RxJS 7+）
const withBackoff$ = ajax.getJSON('/api/unstable').pipe(
  retry({
    count: 5,
    delay: (error, retryCount) => {
      const delayMs = Math.min(1000 * Math.pow(2, retryCount - 1), 30000);
      console.log(`リトライ ${retryCount}: ${delayMs}ms後`);
      return timer(delayMs);
    },
    resetOnSuccess: true,
  }),
  catchError(err => {
    notifyUser('サービスに接続できません');
    return EMPTY;
  }),
);

// === 条件付きリトライ ===
const smartRetry$ = ajax('/api/data').pipe(
  retry({
    count: 3,
    delay: (error, retryCount) => {
      // 4xx エラーはリトライしない
      if (error.status >= 400 && error.status < 500) {
        return throwError(() => error);
      }
      // 5xx エラーのみリトライ
      return timer(1000 * retryCount);
    },
  }),
);

// === タイムアウト ===
const withTimeout$ = ajax.getJSON('/api/slow-endpoint').pipe(
  timeout({
    each: 5000, // 各値の発行間隔が5秒を超えたらエラー
    with: () => throwError(() => new Error('Request timeout')),
  }),
  catchError(err => {
    if (err.message === 'Request timeout') {
      return of({ timeout: true });
    }
    return throwError(() => err);
  }),
);

// === finalize: クリーンアップ ===
function loadData(): Observable<Data> {
  showLoadingSpinner();

  return ajax.getJSON<Data>('/api/data').pipe(
    retry(2),
    catchError(err => {
      showErrorNotification(err.message);
      return EMPTY;
    }),
    finalize(() => {
      hideLoadingSpinner(); // 成功/失敗/購読解除に関わらず実行
    }),
  );
}

// === エラーの分類と処理 ===
class ApiService {
  request<T>(url: string): Observable<T> {
    return ajax.getJSON<T>(url).pipe(
      catchError(err => {
        switch (err.status) {
          case 401:
            this.authService.logout();
            return throwError(() => new UnauthorizedError());
          case 403:
            return throwError(() => new ForbiddenError());
          case 404:
            return throwError(() => new NotFoundError(url));
          case 429:
            // レート制限: Retry-After ヘッダーを尊重
            const retryAfter = parseInt(err.response?.headers?.get('Retry-After') || '5');
            return timer(retryAfter * 1000).pipe(
              switchMap(() => this.request<T>(url)),
            );
          default:
            return throwError(() => new ApiError(err.message, err.status));
        }
      }),
    );
  }
}
```

---

## 8. Angular でのリアクティブパターン

```typescript
// === Angular コンポーネントでの使用 ===
@Component({
  selector: 'app-user-list',
  template: `
    <input [formControl]="searchControl" placeholder="検索...">

    <div *ngIf="loading$ | async" class="spinner">読み込み中...</div>

    <ul>
      <li *ngFor="let user of users$ | async; trackBy: trackById">
        {{ user.name }} - {{ user.email }}
      </li>
    </ul>

    <div *ngIf="error$ | async as error" class="error">
      {{ error.message }}
    </div>
  `,
})
export class UserListComponent implements OnInit, OnDestroy {
  private destroy$ = new Subject<void>();

  searchControl = new FormControl('');
  loading$ = new BehaviorSubject<boolean>(false);
  error$ = new BehaviorSubject<Error | null>(null);

  users$: Observable<User[]>;

  constructor(private userService: UserService) {}

  ngOnInit(): void {
    this.users$ = this.searchControl.valueChanges.pipe(
      startWith(''),
      debounceTime(300),
      distinctUntilChanged(),
      tap(() => {
        this.loading$.next(true);
        this.error$.next(null);
      }),
      switchMap(query =>
        this.userService.searchUsers(query).pipe(
          catchError(err => {
            this.error$.next(err);
            return of([]);
          }),
          finalize(() => this.loading$.next(false)),
        )
      ),
      takeUntil(this.destroy$),
    );
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  trackById(index: number, user: User): number {
    return user.id;
  }
}

// === Angular サービスでのキャッシュ ===
@Injectable({ providedIn: 'root' })
export class UserService {
  private cache$ = new Map<string, Observable<User>>();

  constructor(private http: HttpClient) {}

  getUser(id: string): Observable<User> {
    if (!this.cache$.has(id)) {
      this.cache$.set(id,
        this.http.get<User>(`/api/users/${id}`).pipe(
          shareReplay({ bufferSize: 1, refCount: true }),
          // refCount: true → 購読者が0になったらキャッシュを破棄
        )
      );
    }
    return this.cache$.get(id)!;
  }

  searchUsers(query: string): Observable<User[]> {
    return this.http.get<User[]>(`/api/users`, {
      params: { q: query },
    });
  }

  // リアクティブなCRUD
  private refresh$ = new Subject<void>();

  users$ = this.refresh$.pipe(
    startWith(undefined),
    switchMap(() => this.http.get<User[]>('/api/users')),
    shareReplay(1),
  );

  createUser(user: CreateUserDto): Observable<User> {
    return this.http.post<User>('/api/users', user).pipe(
      tap(() => this.refresh$.next()), // 作成後にリフレッシュ
    );
  }
}
```

---

## 9. React でのリアクティブパターン

```typescript
import { useEffect, useState, useRef, useMemo } from 'react';
import { Subject, BehaviorSubject, Observable, Subscription } from 'rxjs';
import { debounceTime, distinctUntilChanged, switchMap, catchError } from 'rxjs/operators';

// === カスタムフック: useObservable ===
function useObservable<T>(observable$: Observable<T>, initialValue: T): T {
  const [value, setValue] = useState<T>(initialValue);

  useEffect(() => {
    const subscription = observable$.subscribe({
      next: setValue,
      error: err => console.error('Observable error:', err),
    });
    return () => subscription.unsubscribe();
  }, [observable$]);

  return value;
}

// === カスタムフック: useSubject ===
function useSubject<T>(): [Subject<T>, (value: T) => void] {
  const subjectRef = useRef<Subject<T>>();
  if (!subjectRef.current) {
    subjectRef.current = new Subject<T>();
  }

  const emit = useMemo(
    () => (value: T) => subjectRef.current!.next(value),
    [],
  );

  useEffect(() => {
    return () => subjectRef.current!.complete();
  }, []);

  return [subjectRef.current, emit];
}

// === 検索コンポーネント ===
function SearchComponent() {
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const searchSubject = useRef(new Subject<string>());

  useEffect(() => {
    const subscription = searchSubject.current.pipe(
      debounceTime(300),
      distinctUntilChanged(),
      tap(() => {
        setLoading(true);
        setError(null);
      }),
      switchMap(query =>
        from(fetch(`/api/search?q=${query}`).then(r => r.json())).pipe(
          catchError(err => {
            setError(err.message);
            return of([]);
          }),
        )
      ),
      tap(() => setLoading(false)),
    ).subscribe(setResults);

    return () => subscription.unsubscribe();
  }, []);

  return (
    <div>
      <input
        type="text"
        onChange={e => searchSubject.current.next(e.target.value)}
        placeholder="検索..."
      />
      {loading && <div className="spinner">検索中...</div>}
      {error && <div className="error">{error}</div>}
      <ul>
        {results.map(r => (
          <li key={r.id}>{r.title}</li>
        ))}
      </ul>
    </div>
  );
}

// === WebSocket フック ===
function useWebSocket<T>(url: string): {
  messages$: Observable<T>;
  send: (data: any) => void;
  status: 'connecting' | 'connected' | 'disconnected';
} {
  const [status, setStatus] = useState<'connecting' | 'connected' | 'disconnected'>('connecting');
  const wsRef = useRef<WebSocket | null>(null);
  const messages$ = useMemo(() => new Subject<T>(), []);

  useEffect(() => {
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => setStatus('connected');
    ws.onclose = () => setStatus('disconnected');
    ws.onmessage = (event) => {
      try {
        messages$.next(JSON.parse(event.data));
      } catch (err) {
        console.error('Parse error:', err);
      }
    };

    return () => {
      ws.close();
      messages$.complete();
    };
  }, [url]);

  const send = (data: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  };

  return { messages$: messages$.asObservable(), send, status };
}
```

---

## 10. テストとデバッグ

```typescript
import { TestScheduler } from 'rxjs/testing';
import { map, filter, delay, debounceTime, switchMap } from 'rxjs/operators';

// === Marble Testing ===
describe('RxJS オペレータのテスト', () => {
  let scheduler: TestScheduler;

  beforeEach(() => {
    scheduler = new TestScheduler((actual, expected) => {
      expect(actual).toEqual(expected);
    });
  });

  // map オペレータのテスト
  it('values を 10 倍にする', () => {
    scheduler.run(({ cold, expectObservable }) => {
      const source$ = cold(' -a-b-c-|', { a: 1, b: 2, c: 3 });
      const expected = '     -a-b-c-|';
      const result$ = source$.pipe(map(x => x * 10));

      expectObservable(result$).toBe(expected, { a: 10, b: 20, c: 30 });
    });
  });

  // filter オペレータのテスト
  it('偶数のみ通す', () => {
    scheduler.run(({ cold, expectObservable }) => {
      const source$ = cold(' -a-b-c-d-|', { a: 1, b: 2, c: 3, d: 4 });
      const expected = '     ---b---d-|';
      const result$ = source$.pipe(filter(x => x % 2 === 0));

      expectObservable(result$).toBe(expected, { b: 2, d: 4 });
    });
  });

  // debounceTime のテスト
  it('300ms のデバウンスが正しく機能する', () => {
    scheduler.run(({ cold, expectObservable }) => {
      const source$ = cold(' -a--b-----c--|');
      const expected = '     ---- 300ms b 196ms c--|';
      // bはaの後300ms経過で発行、cはbの後300ms以上経過で発行

      const result$ = source$.pipe(debounceTime(300));
      expectObservable(result$).toBe(expected);
    });
  });

  // エラーのテスト
  it('エラーをキャッチしてフォールバック値を返す', () => {
    scheduler.run(({ cold, expectObservable }) => {
      const source$ = cold(' -a-b-#', { a: 1, b: 2 }, new Error('fail'));
      const expected = '     -a-b-(c|)';

      const result$ = source$.pipe(
        catchError(() => of(0)),
      );

      expectObservable(result$).toBe(expected, { a: 1, b: 2, c: 0 });
    });
  });
});

// === Marble Syntax 一覧 ===
// -  : 1フレーム（10ms in virtual time）
// a  : 値の発行（値はオブジェクトリテラルで定義）
// |  : complete
// #  : error
// ^  : subscribe ポイント（hot observableで使用）
// !  : unsubscribe ポイント
// () : 同期グループ（同じフレームで複数の値を発行）

// === tap でデバッグ ===
const debuggedStream$ = source$.pipe(
  tap({
    next: val => console.log('[DEBUG] next:', val),
    error: err => console.error('[DEBUG] error:', err),
    complete: () => console.log('[DEBUG] complete'),
    subscribe: () => console.log('[DEBUG] subscribe'),
    unsubscribe: () => console.log('[DEBUG] unsubscribe'),
    finalize: () => console.log('[DEBUG] finalize'),
  }),
  map(transform),
  tap(val => console.log('[DEBUG] after map:', val)),
);
```

---

## 11. Reactive Streams Specification（Java/Kotlin）

```
Reactive Streams 仕様（JVM）:

  Publisher<T>
    └── subscribe(Subscriber<T>)

  Subscriber<T>
    ├── onSubscribe(Subscription)
    ├── onNext(T)
    ├── onError(Throwable)
    └── onComplete()

  Subscription
    ├── request(long n)    ← バックプレッシャーの核心
    └── cancel()

  Processor<T, R>
    └── Publisher<R> + Subscriber<T>

実装ライブラリ:
  → Project Reactor (Spring WebFlux)
  → RxJava 3
  → Akka Streams
  → Kotlin Coroutines Flow
```

```kotlin
// Kotlin Flow の例（Reactive Streams の軽量版）
import kotlinx.coroutines.flow.*

// Flow の作成
fun fibonacci(): Flow<Long> = flow {
    var a = 0L
    var b = 1L
    while (true) {
        emit(a)
        val temp = a + b
        a = b
        b = temp
    }
}

// 使用
suspend fun main() {
    fibonacci()
        .take(10)
        .filter { it % 2 == 0L }
        .map { it * it }
        .collect { println(it) }
}

// StateFlow（BehaviorSubject 相当）
class UserViewModel : ViewModel() {
    private val _state = MutableStateFlow(UserState())
    val state: StateFlow<UserState> = _state.asStateFlow()

    fun loadUser(id: String) {
        viewModelScope.launch {
            _state.update { it.copy(loading = true) }
            try {
                val user = userRepository.getUser(id)
                _state.update { it.copy(user = user, loading = false) }
            } catch (e: Exception) {
                _state.update { it.copy(error = e.message, loading = false) }
            }
        }
    }
}

// SharedFlow（Subject 相当）
class EventBus {
    private val _events = MutableSharedFlow<AppEvent>(
        replay = 0,
        extraBufferCapacity = 64,
        onBufferOverflow = BufferOverflow.DROP_OLDEST,
    )
    val events: SharedFlow<AppEvent> = _events.asSharedFlow()

    suspend fun emit(event: AppEvent) {
        _events.emit(event)
    }
}
```

---

## 12. パフォーマンス最適化

```typescript
// === share / shareReplay でマルチキャストを活用 ===

// NG: 各 subscribe が独立した HTTP リクエストを発行
const user$ = ajax.getJSON('/api/user/1');
user$.subscribe(u => updateHeader(u));   // リクエスト1
user$.subscribe(u => updateSidebar(u));  // リクエスト2（無駄）

// OK: shareReplay で結果を共有
const user$ = ajax.getJSON('/api/user/1').pipe(
  shareReplay({ bufferSize: 1, refCount: true }),
);
user$.subscribe(u => updateHeader(u));   // リクエスト1
user$.subscribe(u => updateSidebar(u));  // キャッシュから取得（リクエストなし）

// === メモリリーク防止 ===

// NG: takeUntil なしで購読
class MyComponent {
  ngOnInit() {
    interval(1000).subscribe(v => this.update(v));
    // コンポーネント破棄後もメモリリーク
  }
}

// OK: takeUntil で自動解除
class MyComponent implements OnDestroy {
  private destroy$ = new Subject<void>();

  ngOnInit() {
    interval(1000).pipe(
      takeUntil(this.destroy$),
    ).subscribe(v => this.update(v));
  }

  ngOnDestroy() {
    this.destroy$.next();
    this.destroy$.complete();
  }
}

// === 不要な再計算を避ける ===
const expensive$ = source$.pipe(
  distinctUntilChanged(), // 値が変わった時だけ
  map(computeExpensiveResult),
  shareReplay(1),         // 結果を共有
);

// === observeOn / subscribeOn でスケジューリング ===
import { asyncScheduler, asapScheduler, animationFrameScheduler } from 'rxjs';
import { observeOn, subscribeOn } from 'rxjs/operators';

// アニメーションフレームで描画
source$.pipe(
  observeOn(animationFrameScheduler), // requestAnimationFrame に同期
).subscribe(value => {
  updateUI(value); // 60fps でスムーズな描画
});
```

---

## 13. いつ使うか

```
Observable が適切:
  ✓ WebSocket のメッセージストリーム
  ✓ ユーザー入力（検索、スクロール、リサイズ）
  ✓ リアルタイムデータ（株価、チャット）
  ✓ 複数のイベントソースの結合
  ✓ Angular の HttpClient / Forms
  ✓ 複雑な非同期オーケストレーション
  ✓ バックプレッシャー制御が必要な場面

Promise/async-await が適切:
  ✓ 単発のAPI呼び出し
  ✓ DBクエリ
  ✓ ファイル操作
  ✓ シンプルな非同期処理
  ✓ Node.js のサーバーサイド処理

Signals（Angular/Solid/Preact）が適切:
  ✓ UI の状態管理
  ✓ 派生値の計算
  ✓ 同期的なリアクティビティ

原則:
  → 単発 → Promise
  → ストリーム → Observable
  → UI状態 → Signals (利用可能なら)
  → 迷ったら Promise（シンプルさ優先）
```

---

## まとめ

| 概念 | Promise | Observable |
|------|---------|-----------|
| 値の数 | 1つ | 0〜無限 |
| 実行 | eager | lazy |
| キャンセル | 不可 | 可能 |
| オペレータ | 限定的 | 豊富 |
| 用途 | 単発のI/O | ストリーム |
| バックプレッシャー | なし | あり |
| マルチキャスト | N/A | share / Subject |

| Subject | 特徴 | 初期値 | リプレイ |
|---------|------|--------|---------|
| Subject | 基本 | なし | なし |
| BehaviorSubject | 最新値保持 | 必要 | 1個 |
| ReplaySubject | N個リプレイ | なし | N個 |
| AsyncSubject | 最後の値 | なし | complete時 |

| オペレータ | 用途 | キャンセル | 順序保証 |
|-----------|------|-----------|---------|
| switchMap | 検索 | する | 最新のみ |
| mergeMap | 並行処理 | しない | 不定 |
| concatMap | 順序処理 | しない | 保証 |
| exhaustMap | 二重送信防止 | 無視 | 最初のみ |

---

## 次に読むべきガイド
→ [[../02-error-handling/00-exceptions.md]] — 例外処理

---

## 参考文献
1. RxJS Documentation. rxjs.dev.
2. Reactive Streams Specification. reactive-streams.org.
3. Ben Lesh. "RxJS: Observable, Observer, and Subscription." rxjs.dev.
4. Angular Documentation. "Observables in Angular." angular.dev.
5. Kotlin Documentation. "Asynchronous Flow." kotlinlang.org.
6. Erik Meijer. "Your Mouse is a Database." ACM Queue, 2012.
