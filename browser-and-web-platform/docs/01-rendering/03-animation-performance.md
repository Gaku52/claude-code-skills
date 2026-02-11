# アニメーションパフォーマンス

> 60fpsのスムーズなアニメーションを実現するための手法を体系的に学ぶ。CSS Transitions/Animations、requestAnimationFrame、Web Animations API、FLIP技法を理解する。

## この章で学ぶこと

- [ ] 60fpsアニメーションの原理を理解する
- [ ] CSS vs JavaScriptアニメーションの使い分けを把握する
- [ ] FLIP技法とView Transitionsを学ぶ

---

## 1. 60fps の原則

```
60fps = 16.67ms/フレーム

  ブラウザの1フレーム:
  ┌──────┬───────┬────────┬───────┬───────────┐
  │ JS   │ Style │ Layout │ Paint │ Composite │
  └──────┴───────┴────────┴───────┴───────────┘
  ← ────────── 16.67ms 以内 ────────────── →

  最速のアニメーション（Composite のみ）:
  ┌──────┬───────────┐
  │ JS   │ Composite │  ← Layout/Paint スキップ
  └──────┴───────────┘
  対象: transform, opacity のみ

  使うべきプロパティ:
  ✓ transform: translateX(), translateY()  — 移動
  ✓ transform: scale()                    — 拡大/縮小
  ✓ transform: rotate()                   — 回転
  ✓ opacity                               — 透明度

  避けるべきプロパティ:
  ✗ top, left, right, bottom  — Layout発生
  ✗ width, height             — Layout発生
  ✗ margin, padding           — Layout発生
  ✗ border-width              — Layout発生
```

---

## 2. CSS Transitions / Animations

```css
/* Transition — 状態変化のアニメーション */
.button {
  background: #3b82f6;
  transform: scale(1);
  transition: transform 200ms ease-out, opacity 200ms ease-out;
}

.button:hover {
  transform: scale(1.05);
}

/* Animation — キーフレームアニメーション */
@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.card {
  animation: fadeIn 300ms ease-out forwards;
}

/* スクロール連動アニメーション */
@keyframes reveal {
  from { opacity: 0; transform: translateY(50px); }
  to { opacity: 1; transform: translateY(0); }
}

.reveal {
  animation: reveal linear both;
  animation-timeline: view();
  animation-range: entry 0% entry 100%;
}
```

```
イージング関数:
  ease:        ゆっくり始まり → 加速 → ゆっくり終わる
  ease-in:     ゆっくり始まる
  ease-out:    ゆっくり終わる（推奨: UIアニメーション）
  ease-in-out: 両端がゆっくり
  linear:      一定速度
  cubic-bezier(x1, y1, x2, y2): カスタム

  推奨:
  UI操作の応答:  ease-out, 150-300ms
  入場アニメーション: ease-out, 200-400ms
  退場アニメーション: ease-in, 150-250ms
```

---

## 3. requestAnimationFrame

```javascript
// requestAnimationFrame（rAF）— JSアニメーションの基盤
function animate() {
  // 次のフレーム描画前に呼ばれる
  element.style.transform = `translateX(${x}px)`;
  x += 2;

  if (x < 300) {
    requestAnimationFrame(animate);
  }
}

requestAnimationFrame(animate);

// タイムスタンプベース（推奨）
function animate(timestamp) {
  if (!startTime) startTime = timestamp;
  const elapsed = timestamp - startTime;
  const progress = Math.min(elapsed / duration, 1);

  // イージング適用
  const eased = easeOutCubic(progress);
  element.style.transform = `translateX(${eased * 300}px)`;

  if (progress < 1) {
    requestAnimationFrame(animate);
  }
}

function easeOutCubic(t) {
  return 1 - Math.pow(1 - t, 3);
}

// setInterval vs rAF:
// ✗ setInterval(fn, 16) — フレームとずれる、非アクティブタブで無駄
// ✓ requestAnimationFrame — フレームに同期、非アクティブで停止
```

---

## 4. FLIP 技法

```
FLIP = First, Last, Invert, Play
→ レイアウト変更を transform アニメーションに変換

  問題: width/height の変更は Layout + Paint が発生
  解決: transform で見た目だけ変更（Composite のみ）

手順:
  F (First):  変更前の位置・サイズを記録
  L (Last):   変更を適用し、最終位置・サイズを記録
  I (Invert): 差分を計算し、transform で元に戻す
  P (Play):   transform を解除してアニメーション

実装例:
  function flipAnimate(element, changeFn) {
    // First: 現在の位置を記録
    const first = element.getBoundingClientRect();

    // Last: DOM変更を適用
    changeFn();
    const last = element.getBoundingClientRect();

    // Invert: 差分を計算して元の位置に戻す
    const deltaX = first.left - last.left;
    const deltaY = first.top - last.top;
    const deltaW = first.width / last.width;
    const deltaH = first.height / last.height;

    element.style.transform = `translate(${deltaX}px, ${deltaY}px) scale(${deltaW}, ${deltaH})`;
    element.style.transformOrigin = 'top left';

    // Play: 次のフレームでtransformを解除
    requestAnimationFrame(() => {
      element.style.transition = 'transform 300ms ease-out';
      element.style.transform = '';
      element.addEventListener('transitionend', () => {
        element.style.transition = '';
        element.style.transformOrigin = '';
      }, { once: true });
    });
  }

View Transitions API（ネイティブFLIP）:
  document.startViewTransition(() => {
    // DOM変更
    updateDOM();
  });

  /* CSS でアニメーションをカスタマイズ */
  ::view-transition-old(root) {
    animation: fade-out 300ms ease-out;
  }
  ::view-transition-new(root) {
    animation: fade-in 300ms ease-out;
  }
```

---

## 5. prefers-reduced-motion

```css
/* アクセシビリティ: アニメーション軽減の設定を尊重 */
@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
    scroll-behavior: auto !important;
  }
}

/* より洗練されたアプローチ: 動きは残すが控えめに */
@media (prefers-reduced-motion: reduce) {
  .card {
    transition-duration: 0ms;  /* 即座に変化 */
  }
}

@media (prefers-reduced-motion: no-preference) {
  .card {
    transition: transform 300ms ease-out;
  }
}
```

---

## まとめ

| 手法 | 用途 | パフォーマンス |
|------|------|-------------|
| CSS Transition | 状態変化 | 高（Compositeのみ可） |
| CSS Animation | キーフレーム | 高 |
| rAF | 複雑なJSアニメーション | 中（実装次第） |
| FLIP | レイアウト変更のアニメーション | 高 |
| View Transitions | ページ遷移 | 高（ネイティブ） |

---

## 次に読むべきガイド
→ [[../02-javascript-runtime/00-v8-engine.md]] — V8エンジン

---

## 参考文献
1. Paul Lewis. "FLIP Your Animations." aerotwist.com, 2015.
2. web.dev. "View Transitions API." Google, 2024.
