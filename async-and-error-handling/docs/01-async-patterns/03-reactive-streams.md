# Reactive Streams

> Reactive Streams は「非同期データストリーム」を宣言的に処理するパターン。RxJS の Observable、バックプレッシャー、イベント駆動アーキテクチャの基盤を理解する。

## この章で学ぶこと

- [ ] Observable パターンの仕組みを理解する
- [ ] RxJS のオペレータとパイプラインを把握する
- [ ] バックプレッシャーの概念を学ぶ

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

---

## 3. 主要オペレータ

```
変換:
  map       — 値を変換
  switchMap — 新しい Observable に切り替え（前をキャンセル）
  mergeMap  — 並行して Observable を実行
  concatMap — 直列に Observable を実行

フィルタリング:
  filter       — 条件に合う値のみ通す
  take         — 最初のN個だけ取得
  debounceTime — 一定時間入力がなかったら通す
  distinctUntilChanged — 値が変わった時だけ通す

結合:
  merge       — 複数の Observable を合流
  combineLatest — 各 Observable の最新値を組み合わせ
  zip         — 各 Observable の値を1対1で組み合わせ
  forkJoin    — 全ての Observable の最後の値を取得（Promise.allに近い）

エラー:
  catchError  — エラーをハンドリングして回復
  retry       — エラー時にリトライ
  retryWhen   — 条件付きリトライ
```

```typescript
// 実践例: リアルタイムダッシュボード
import { combineLatest, timer } from 'rxjs';
import { switchMap, catchError, retry } from 'rxjs/operators';

const dashboard$ = combineLatest([
  timer(0, 5000).pipe(  // 5秒ごとに更新
    switchMap(() => fetch('/api/stats').then(r => r.json())),
    retry(3),
    catchError(() => of({ error: true })),
  ),
  timer(0, 10000).pipe( // 10秒ごとに更新
    switchMap(() => fetch('/api/alerts').then(r => r.json())),
    catchError(() => of([])),
  ),
]).subscribe(([stats, alerts]) => {
  updateDashboard(stats, alerts);
});

// 購読解除（コンポーネント破棄時）
dashboard$.unsubscribe();
```

---

## 4. バックプレッシャー

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

RxJS でのバックプレッシャー:
  → bufferTime: 時間ごとにバッチ処理
  → throttleTime: 一定間隔で値を通す
  → sampleTime: 一定間隔で最新値を取得
  → auditTime: 値が来てから一定時間後に最新値を通す
```

---

## 5. いつ使うか

```
Observable が適切:
  ✓ WebSocket のメッセージストリーム
  ✓ ユーザー入力（検索、スクロール、リサイズ）
  ✓ リアルタイムデータ（株価、チャット）
  ✓ 複数のイベントソースの結合

Promise/async-await が適切:
  ✓ 単発のAPI呼び出し
  ✓ DBクエリ
  ✓ ファイル操作
  ✓ シンプルな非同期処理

原則:
  → 単発 → Promise
  → ストリーム → Observable
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

---

## 次に読むべきガイド
→ [[../02-error-handling/00-exceptions.md]] — 例外処理

---

## 参考文献
1. RxJS Documentation. rxjs.dev.
2. Reactive Streams Specification. reactive-streams.org.
