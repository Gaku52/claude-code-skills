# ブラウザのイベントループ

> ブラウザのイベントループはJSの実行モデルの核心。タスクキュー、マイクロタスクキュー、requestAnimationFrame、requestIdleCallbackの実行順序を正確に理解する。

## この章で学ぶこと

- [ ] ブラウザのイベントループの実行順序を理解する
- [ ] マクロタスクとマイクロタスクの違いを把握する
- [ ] rAFとrICの使い分けを学ぶ

---

## 1. イベントループの構造

```
ブラウザのイベントループ:

  ┌─────────────────────────────────────────┐
  │ 1つのイベントループサイクル               │
  │                                         │
  │ ① タスクキューから1つのタスクを取得・実行│
  │    └→ setTimeout, setInterval,          │
  │       I/Oコールバック, UI events        │
  │                                         │
  │ ② マイクロタスクキューを全て実行         │
  │    └→ Promise.then, queueMicrotask,     │
  │       MutationObserver                  │
  │                                         │
  │ ③ レンダリング更新（必要な場合）         │
  │    ├→ requestAnimationFrame             │
  │    ├→ Style 再計算                      │
  │    ├→ Layout                            │
  │    ├→ Paint                             │
  │    └→ Composite                         │
  │                                         │
  │ ④ requestIdleCallback（余裕がある場合）  │
  │                                         │
  │ → ①に戻る                               │
  └─────────────────────────────────────────┘

重要なポイント:
  → タスクは1つずつ実行（次のタスクの前にレンダリングの機会）
  → マイクロタスクは全て実行（途中でレンダリングなし）
  → レンダリングは毎回ではない（ブラウザが判断、通常60fps）
```

---

## 2. タスク vs マイクロタスク

```
タスク（マクロタスク）:
  → setTimeout, setInterval
  → setImmediate（Node.js）
  → I/O コールバック
  → UI イベントハンドラー（click, scroll等）
  → MessageChannel

マイクロタスク:
  → Promise.then / catch / finally
  → queueMicrotask()
  → MutationObserver
  → async/await のawait後の処理

実行順序クイズ:
  console.log('1');

  setTimeout(() => console.log('2'), 0);

  Promise.resolve().then(() => console.log('3'));

  queueMicrotask(() => console.log('4'));

  console.log('5');

  出力: 1, 5, 3, 4, 2

  解説:
  同期: 1, 5
  マイクロタスク: 3, 4（Promiseとqueueはどちらもマイクロ）
  タスク: 2（setTimeout は次のタスクサイクル）

さらに複雑な例:
  setTimeout(() => {
    console.log('timeout 1');
    Promise.resolve().then(() => console.log('promise in timeout'));
  }, 0);

  setTimeout(() => {
    console.log('timeout 2');
  }, 0);

  Promise.resolve().then(() => {
    console.log('promise 1');
  });

  出力: promise 1, timeout 1, promise in timeout, timeout 2

  解説:
  ① 同期コード完了
  ② マイクロタスク: promise 1
  ③ タスク1: timeout 1 → マイクロタスク: promise in timeout
  ④ タスク2: timeout 2
```

---

## 3. requestAnimationFrame

```
rAF = 次のレンダリングフレームの前に実行

  タイミング:
  ┌──────┐ ┌───────────┐ ┌───────────┐ ┌──────────┐
  │ Task │→│ Microtasks│→│    rAF    │→│ Rendering│
  └──────┘ └───────────┘ └───────────┘ └──────────┘

  用途:
  → DOM変更のバッチ処理
  → アニメーションの更新
  → レイアウト読み取りの集約

  // DOM変更をrAFにまとめる
  function updateDOM(changes) {
    requestAnimationFrame(() => {
      changes.forEach(change => applyChange(change));
    });
  }

  // スムーズスクロール
  function smoothScrollTo(target) {
    const start = window.scrollY;
    const distance = target - start;
    const duration = 500;
    let startTime;

    function step(timestamp) {
      if (!startTime) startTime = timestamp;
      const progress = Math.min((timestamp - startTime) / duration, 1);
      window.scrollTo(0, start + distance * easeOutCubic(progress));
      if (progress < 1) requestAnimationFrame(step);
    }

    requestAnimationFrame(step);
  }

rAF vs setTimeout:
  setTimeout(() => {}, 16):
    → フレームとずれる可能性
    → 非アクティブタブでも実行され続ける

  requestAnimationFrame():
    → フレームに正確に同期
    → 非アクティブタブで停止（省電力）
    → レンダリング直前に実行
```

---

## 4. requestIdleCallback

```
rIC = ブラウザがアイドル状態の時に実行

  タイミング:
  フレーム1                              フレーム2
  ┌───────────────────┬────────┐        ┌────────
  │ Task + Rendering  │ Idle   │        │
  └───────────────────┴────┬───┘        └────────
                           │
                      rIC実行（余った時間で）

  requestIdleCallback((deadline) => {
    // deadline.timeRemaining() でこのアイドル期間の残り時間を確認
    while (deadline.timeRemaining() > 0 && tasks.length > 0) {
      performTask(tasks.shift());
    }

    // タスクが残っていれば次のアイドル時間に続行
    if (tasks.length > 0) {
      requestIdleCallback(processRemainingTasks);
    }
  }, { timeout: 2000 }); // 最大2秒待ち（保証）

用途:
  → アナリティクスの送信
  → 低優先度のデータフェッチ
  → 遅延初期化（非クリティカルな機能）
  → 大量データの段階的処理

注意:
  → DOM操作は避ける（レイアウトが強制される可能性）
  → DOM変更はrAFで行う
  → Safari未サポート（polyfillが必要）
```

---

## 5. 実行順序のまとめ

```
完全な実行順序:

  1. 同期コード（コールスタック上で即実行）
  2. マイクロタスクキュー（全て実行）
  3. レンダリング（ブラウザ判断、通常60fps）
     a. requestAnimationFrame
     b. Style/Layout/Paint/Composite
  4. requestIdleCallback（余裕がある場合）
  5. 次のタスクキューのタスク → 2に戻る

  優先度:
  同期 > マイクロタスク > rAF > タスク > rIC

API選択ガイド:
  ┌─────────────────┬──────────────────────────┐
  │ 用途            │ 使うべきAPI               │
  ├─────────────────┼──────────────────────────┤
  │ 即座に実行      │ queueMicrotask           │
  │ DOM更新前       │ requestAnimationFrame    │
  │ 遅延実行        │ setTimeout               │
  │ 低優先度処理    │ requestIdleCallback      │
  │ アニメーション  │ rAF or CSS Animation     │
  └─────────────────┴──────────────────────────┘
```

---

## まとめ

| 概念 | 実行タイミング | 用途 |
|------|-------------|------|
| マイクロタスク | タスク完了直後、全て実行 | Promise, async/await |
| タスク | 1つずつ、レンダリング挟む | setTimeout, I/O |
| rAF | レンダリング直前 | アニメーション、DOM操作 |
| rIC | アイドル時間 | 低優先度処理 |

---

## 次に読むべきガイド
→ [[02-web-workers.md]] — Web Workers

---

## 参考文献
1. Jake Archibald. "Tasks, microtasks, queues and schedules." 2015.
2. HTML Living Standard. "Event loops." WHATWG, 2024.
