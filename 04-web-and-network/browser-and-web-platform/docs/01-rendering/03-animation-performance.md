# アニメーションパフォーマンス

> 60fpsのスムーズなアニメーションを実現するための手法を体系的に学ぶ。CSS Transitions/Animations、requestAnimationFrame、Web Animations API、FLIP技法、View Transitions APIを深く理解し、パフォーマンス計測と最適化の全体像を把握する。

## 前提知識

- ペイントとコンポジティングの仕組み → 参照: [ペイントとコンポジティング](./02-paint-and-compositing.md)
- CSSアニメーション/トランジションの基本構文
- requestAnimationFrameの概念

## この章で学ぶこと

- [ ] 60fpsアニメーションの原理とフレームバジェットを理解する
- [ ] CSS Transitions/Animations の内部動作と最適化手法を習得する
- [ ] requestAnimationFrame の正しい使い方とタイミングモデルを把握する
- [ ] FLIP 技法の原理と応用パターンを学ぶ
- [ ] Web Animations API の統一的なアニメーション制御を理解する
- [ ] View Transitions API によるページ遷移アニメーションを学ぶ
- [ ] パフォーマンス計測ツールとデバッグ手法を身につける
- [ ] アクセシビリティを考慮したアニメーション設計を習得する

---

## 1. 60fps の原則とフレームバジェット

### 1.1 なぜ60fpsなのか

人間の視覚系は、およそ10fps以上で連続した画像を「動き」として知覚する。しかし、滑らかな動きとして認識されるには少なくとも24fps（映画の標準フレームレート）が必要であり、インタラクティブなUIにおいてはさらに高いフレームレートが求められる。

多くのディスプレイのリフレッシュレートは60Hzであり、これは1秒間に60回画面を更新することを意味する。ブラウザがこのリフレッシュレートに同期してフレームを描画できなければ、ユーザーは「カクつき」（ジャンク）を感じる。

```
フレームレートとユーザー体験の関係:

  fps    フレーム間隔     ユーザー体験
  ─────────────────────────────────────────────────
  10fps    100.0ms      スライドショーのような印象
  24fps     41.7ms      映画的。動きは認識できるがUI向きではない
  30fps     33.3ms      やや滑らか。モバイルゲームの下限
  60fps     16.7ms      十分に滑らか。Webアニメーションの標準目標
  90fps     11.1ms      非常に滑らか。VR/ARの最低要件
  120fps     8.3ms      極めて滑らか。ProMotionディスプレイ対応

  ※ 人間がジャンクを感知するのはおよそ3フレーム以上のドロップ時
```

### 1.2 1フレーム 16.67ms の内訳

ブラウザが1フレームを描画するまでに実行するパイプラインは以下の通りである。

```
┌─────────────────────────────────────────────────────────────┐
│                   1フレーム = 16.67ms                        │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Input        rAF         Style        Layout     Paint     │
│  Events    Callbacks    Recalc       (Reflow)   (Repaint)   │
│  ┌────┐    ┌────┐      ┌────┐       ┌────┐     ┌────┐     │
│  │ 1  │ →  │ 2  │  →   │ 3  │  →    │ 4  │  →  │ 5  │     │
│  │ ms │    │ ms │      │ ms │       │ ms │     │ ms │     │
│  └────┘    └────┘      └────┘       └────┘     └────┘     │
│                                                              │
│  → Composite (GPU) → Display                                │
│    ┌────┐                                                    │
│    │ 6  │  合成レイヤーの結合                                 │
│    └────┘                                                    │
│                                                              │
│  理想: 全ステップが16.67ms以内に完了すること                   │
│  実用: JSの処理は10ms以内に収めるのが安全                      │
└──────────────────────────────────────────────────────────────┘

各ステップの詳細:
  1. Input Events     : タッチ、クリック、スクロール等のイベント処理
  2. rAF Callbacks    : requestAnimationFrame で登録されたコールバック
  3. Style Recalc     : CSSルールの再計算（セレクタマッチング等）
  4. Layout (Reflow)  : 要素の位置・サイズの再計算
  5. Paint (Repaint)  : ピクセルの描画命令生成
  6. Composite        : GPUレイヤーの合成・表示
```

### 1.3 レンダリングパスとプロパティの関係

CSSプロパティの変更は、レンダリングパイプラインのどの段階をトリガーするかによって、パフォーマンスへの影響が大きく異なる。

```
アニメーション対象プロパティとレンダリングコスト:

  プロパティ        Layout  Paint  Composite  コスト
  ──────────────────────────────────────────────────
  transform          -       -       ✓        最低 ★★★
  opacity            -       -       ✓        最低 ★★★
  filter             -       ✓       ✓        低   ★★☆
  background-color   -       ✓       ✓        低   ★★☆
  box-shadow         -       ✓       ✓        低   ★★☆
  color              -       ✓       ✓        低   ★★☆
  border-radius      -       ✓       ✓        低   ★★☆
  width / height     ✓       ✓       ✓        高   ★☆☆
  top / left         ✓       ✓       ✓        高   ★☆☆
  margin / padding   ✓       ✓       ✓        高   ★☆☆
  font-size          ✓       ✓       ✓        高   ★☆☆
  display            ✓       ✓       ✓        高   ★☆☆
  border-width       ✓       ✓       ✓        高   ★☆☆

  ★★★ = Compositeのみ → GPUで完結、メインスレッド不要
  ★★☆ = Paint + Composite → メインスレッドで描画命令生成
  ★☆☆ = Layout + Paint + Composite → 全要素の再計算が発生
```

### 1.4 will-change によるレイヤー昇格

`will-change` プロパティを使うと、ブラウザに対してどのプロパティがアニメーションされるかをヒントとして伝えられる。これにより、事前にGPUレイヤーとして昇格（promote）させ、合成処理の準備を行える。

```css
/* 適切な使用例: アニメーション直前に適用 */
.card:hover {
  will-change: transform;
}

.card.animating {
  will-change: transform, opacity;
  transition: transform 300ms ease-out, opacity 300ms ease-out;
}

/* アニメーション完了後に解除 */
.card.animating {
  /* transitionend イベントで will-change を解除する */
}
```

```javascript
// JavaScript で will-change を適切に管理する例
const card = document.querySelector('.card');

card.addEventListener('mouseenter', () => {
  // ホバー時にレイヤー昇格を予約
  card.style.willChange = 'transform, opacity';
});

card.addEventListener('transitionend', () => {
  // アニメーション完了後にリソースを解放
  card.style.willChange = 'auto';
});
```

```
will-change 使用時のメモリ影響:

  要素数    will-change なし   will-change あり   メモリ増加
  ──────────────────────────────────────────────────────────
  10個        基準値             +2-5MB           軽微
  100個       基準値             +20-50MB         注意
  1000個      基準値             +200-500MB       危険 ⚠

  重要: will-change は必要な時だけ、必要な要素にだけ適用する
  常時適用はGPUメモリを大量消費し、逆効果になる
```

---

## 2. CSS Transitions の深い理解

### 2.1 Transition の内部モデル

CSS Transitionは、プロパティ値の変化を検出すると自動的に補間アニメーションを生成する仕組みである。ブラウザ内部では以下のステップで処理される。

```
CSS Transition の処理フロー:

  1. プロパティ値の変化を検出
     ┌──────────────────────────────┐
     │ .box { left: 0; }           │
     │ .box:hover { left: 100px; } │  ← 値が変化した！
     └──────────────────────────────┘
                    │
                    ▼
  2. transition 宣言をチェック
     ┌──────────────────────────────────────────┐
     │ transition: left 300ms ease-out 0ms;      │
     │             ^^^^  ^^^^  ^^^^^^^^  ^^^     │
     │           property duration timing delay  │
     └──────────────────────────────────────────┘
                    │
                    ▼
  3. 補間値を計算し、フレームごとに描画
     ┌─────────────────────────────────────┐
     │  t=0ms:    left: 0px               │
     │  t=50ms:   left: 28px  (ease-out)  │
     │  t=100ms:  left: 52px              │
     │  t=150ms:  left: 72px              │
     │  t=200ms:  left: 88px              │
     │  t=250ms:  left: 96px              │
     │  t=300ms:  left: 100px (完了)      │
     └─────────────────────────────────────┘
```

### 2.2 transition プロパティの詳細構文

```css
/* 個別プロパティ指定 */
.element {
  transition-property: transform, opacity, background-color;
  transition-duration: 300ms, 200ms, 150ms;
  transition-timing-function: ease-out, ease, linear;
  transition-delay: 0ms, 50ms, 100ms;
}

/* ショートハンド */
.element {
  transition:
    transform 300ms ease-out 0ms,
    opacity 200ms ease 50ms,
    background-color 150ms linear 100ms;
}

/* 全プロパティ一括指定（注意が必要） */
.element {
  transition: all 300ms ease-out;
  /* 意図しないプロパティもアニメーションされる可能性がある */
  /* 例: font-size の変更もアニメーションされてしまう */
}
```

### 2.3 transition イベントの活用

```javascript
const element = document.querySelector('.animated');

// Transition開始時（各プロパティごとに発火）
element.addEventListener('transitionstart', (e) => {
  console.log(`開始: ${e.propertyName}, 時間: ${e.elapsedTime}s`);
});

// Transition実行中（各イテレーション完了時。通常はtransitionでは1回）
element.addEventListener('transitionrun', (e) => {
  console.log(`実行中: ${e.propertyName}`);
});

// Transition完了時
element.addEventListener('transitionend', (e) => {
  console.log(`完了: ${e.propertyName}, 時間: ${e.elapsedTime}s`);
  // 完了後のクリーンアップ処理
  element.classList.remove('animating');
  element.style.willChange = 'auto';
});

// Transitionキャンセル時（途中で値が変わった場合）
element.addEventListener('transitioncancel', (e) => {
  console.log(`キャンセル: ${e.propertyName}`);
});
```

### 2.4 複数プロパティのステージングアニメーション

```css
/* カードの入場アニメーション: 段階的に要素を表示 */
.card {
  opacity: 0;
  transform: translateY(30px);
}

.card.visible {
  opacity: 1;
  transform: translateY(0);
  /* opacity が先に完了し、その後 transform が完了する演出 */
  transition:
    opacity 200ms ease-out 0ms,
    transform 350ms cubic-bezier(0.34, 1.56, 0.64, 1) 50ms;
}

/* リストアイテムの連鎖アニメーション */
.list-item {
  opacity: 0;
  transform: translateX(-20px);
  transition: opacity 300ms ease-out, transform 300ms ease-out;
}

.list-item.visible {
  opacity: 1;
  transform: translateX(0);
}

/* CSS変数を使ったスタッガー遅延 */
.list-item:nth-child(1) { transition-delay: calc(0 * 50ms); }
.list-item:nth-child(2) { transition-delay: calc(1 * 50ms); }
.list-item:nth-child(3) { transition-delay: calc(2 * 50ms); }
.list-item:nth-child(4) { transition-delay: calc(3 * 50ms); }
.list-item:nth-child(5) { transition-delay: calc(4 * 50ms); }

/* あるいは CSS カスタムプロパティでインラインに設定 */
.list-item {
  transition-delay: calc(var(--index) * 50ms);
}
```

```javascript
// JavaScript でスタッガー遅延を設定
const items = document.querySelectorAll('.list-item');
items.forEach((item, index) => {
  item.style.setProperty('--index', index);
});

// IntersectionObserver でビューポート進入時にアニメーション
const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        observer.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.1 }
);

items.forEach((item) => observer.observe(item));
```

---

## 3. CSS Animations の高度な活用

### 3.1 @keyframes の補間モデル

CSS Animationsは `@keyframes` ルールで定義されたキーフレーム間を補間してアニメーションを生成する。Transitionsが2つの状態間の遷移であるのに対し、Animationsは任意の数の中間状態を定義できる。

```css
/* 基本的なキーフレーム定義 */
@keyframes slideInFromLeft {
  0% {
    transform: translateX(-100%);
    opacity: 0;
  }
  60% {
    transform: translateX(10%);
    opacity: 1;
  }
  80% {
    transform: translateX(-5%);
  }
  100% {
    transform: translateX(0);
  }
}

.panel {
  animation: slideInFromLeft 500ms cubic-bezier(0.22, 1, 0.36, 1) forwards;
}

/* 複数アニメーションの組み合わせ */
@keyframes scaleUp {
  from { transform: scale(0.8); }
  to { transform: scale(1); }
}

@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

.modal {
  animation:
    scaleUp 300ms cubic-bezier(0.34, 1.56, 0.64, 1) forwards,
    fadeIn 200ms ease-out forwards;
}
```

### 3.2 animation-fill-mode の挙動

```
animation-fill-mode の動作比較:

  ──────── 遅延期間 ──── アニメーション期間 ──── 完了後 ────

  none:
  [ 初期値          ][ 0% → → → → → 100% ][ 初期値          ]
                      ^^^^^^^^^^^^^^^^^^
                      アニメーション中のみ変化

  forwards:
  [ 初期値          ][ 0% → → → → → 100% ][ 100%の値を保持  ]
                                            ^^^^^^^^^^^^^^^^^
                                            最終フレームで固定

  backwards:
  [ 0%の値を適用    ][ 0% → → → → → 100% ][ 初期値          ]
   ^^^^^^^^^^^^^^^^^
   遅延中に開始フレームを適用

  both:
  [ 0%の値を適用    ][ 0% → → → → → 100% ][ 100%の値を保持  ]
   ^^^^^^^^^^^^^^^^^                        ^^^^^^^^^^^^^^^^^
   遅延中も完了後も適用
```

```css
/* 実用例: モーダルの表示/非表示 */
@keyframes modalOpen {
  from {
    opacity: 0;
    transform: scale(0.9) translateY(20px);
  }
  to {
    opacity: 1;
    transform: scale(1) translateY(0);
  }
}

@keyframes modalClose {
  from {
    opacity: 1;
    transform: scale(1) translateY(0);
  }
  to {
    opacity: 0;
    transform: scale(0.9) translateY(20px);
  }
}

.modal.opening {
  animation: modalOpen 300ms ease-out forwards;
}

.modal.closing {
  animation: modalClose 200ms ease-in forwards;
}
```

### 3.3 animation-composition: 累積と合成

```css
/* animation-composition でアニメーション効果を累積できる */
@keyframes bounce {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-20px); }
}

@keyframes wobble {
  0%, 100% { transform: rotate(0deg); }
  25% { transform: rotate(-5deg); }
  75% { transform: rotate(5deg); }
}

.icon {
  /* replace: 後のアニメーションが前のものを置き換える（デフォルト） */
  animation-composition: replace;

  /* add: 変換が加算される */
  animation-composition: add;

  /* accumulate: 数値が累積される */
  animation-composition: accumulate;
}
```

### 3.4 スクロール連動アニメーション (Scroll-driven Animations)

CSS Scroll-driven Animationsは、スクロール位置をアニメーションのタイムラインとして使用する新しいCSS仕様である。

```css
/* スクロール進捗インジケータ */
@keyframes progressBar {
  from { transform: scaleX(0); }
  to { transform: scaleX(1); }
}

.progress-bar {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 4px;
  background: #3b82f6;
  transform-origin: left;
  animation: progressBar linear;
  animation-timeline: scroll(root block);
}

/* 要素がビューポートに入る時のアニメーション */
@keyframes reveal {
  from {
    opacity: 0;
    transform: translateY(50px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.reveal-on-scroll {
  animation: reveal linear both;
  animation-timeline: view();
  animation-range: entry 0% entry 100%;
}

/* パララックス効果 */
@keyframes parallax {
  from { transform: translateY(100px); }
  to { transform: translateY(-100px); }
}

.parallax-layer {
  animation: parallax linear;
  animation-timeline: scroll(root);
}
```

```
Scroll Timeline と View Timeline の違い:

  scroll()                          view()
  ──────────────────────────────    ──────────────────────────────
  スクロールコンテナ全体の            要素がビューポートを
  スクロール位置に連動                横切る進捗に連動

  ┌──────────────────┐              ┌──────────────────┐
  │   scroll: 0%     │              │                  │
  │   ┌────────────┐ │              │  ← entry 0%     │
  │   │ コンテンツ  │ │              │  ┌────────────┐ │
  │   │            │ │              │  │ 要素        │ │ entry 100%
  │   │            │ │              │  │            │ │
  │   └────────────┘ │              │  │            │ │ exit 0%
  │   scroll: 100%   │              │  └────────────┘ │
  └──────────────────┘              │  ← exit 100%   │
                                    └──────────────────┘

  animation-range の指定:
    entry 0%   : 要素の先端がビューポート下端に到達
    entry 100% : 要素の末端がビューポート下端を通過
    exit 0%    : 要素の先端がビューポート上端に到達
    exit 100%  : 要素の末端がビューポート上端を通過
```

---

## 4. requestAnimationFrame の徹底理解

### 4.1 rAF のタイミングモデル

`requestAnimationFrame` (rAF) は、ブラウザの描画サイクルに同期してコールバックを実行する仕組みである。`setInterval` や `setTimeout` と異なり、ディスプレイのリフレッシュレートに合わせた最適なタイミングで呼び出される。

```
rAF のタイミング（60Hz ディスプレイの場合）:

  時間軸 (ms)
  0        16.67     33.33     50.00     66.67
  │         │         │         │         │
  ▼         ▼         ▼         ▼         ▼
  ┌─────────┬─────────┬─────────┬─────────┐
  │ Frame 1 │ Frame 2 │ Frame 3 │ Frame 4 │
  │ rAF(cb) │ rAF(cb) │ rAF(cb) │ rAF(cb) │
  └─────────┴─────────┴─────────┴─────────┘
       │          │          │          │
       ▼          ▼          ▼          ▼
    cb(0)     cb(16.67)  cb(33.33)  cb(50.00)
              timestamp  timestamp  timestamp

  setInterval(fn, 16) との比較:
  ┌─────────┬─────────┬─────────┬─────────┐
  │         │   ↑ ずれ │       ↑ ずれ     │
  │ Frame 1 │ Frame 2 │ Frame 3 │ Frame 4 │
  │ fn()    │   fn()  │ fn() fn │ fn()    │
  └─────────┴─────────┴─────────┴─────────┘
  ※ setInterval はフレームとずれ、1フレームに2回呼ばれたり
    スキップされたりする
```

### 4.2 基本的な rAF アニメーションパターン

```javascript
// パターン1: 基本的なアニメーションループ
function basicAnimation() {
  let x = 0;
  const element = document.querySelector('.box');

  function animate() {
    x += 2;
    element.style.transform = `translateX(${x}px)`;

    if (x < 300) {
      requestAnimationFrame(animate);
    }
  }

  requestAnimationFrame(animate);
}

// パターン2: タイムスタンプベース（推奨）
function timestampAnimation() {
  const element = document.querySelector('.box');
  const duration = 1000; // 1秒
  const distance = 300;  // 300px移動
  let startTime = null;

  function animate(timestamp) {
    if (startTime === null) startTime = timestamp;
    const elapsed = timestamp - startTime;
    const progress = Math.min(elapsed / duration, 1);

    // イージング関数を適用
    const eased = easeOutCubic(progress);
    element.style.transform = `translateX(${eased * distance}px)`;

    if (progress < 1) {
      requestAnimationFrame(animate);
    }
  }

  requestAnimationFrame(animate);
}

// パターン3: キャンセル可能なアニメーション
function cancellableAnimation() {
  const element = document.querySelector('.box');
  let animationId = null;
  let startTime = null;
  const duration = 2000;

  function animate(timestamp) {
    if (startTime === null) startTime = timestamp;
    const progress = Math.min((timestamp - startTime) / duration, 1);

    element.style.transform = `translateX(${progress * 300}px)`;

    if (progress < 1) {
      animationId = requestAnimationFrame(animate);
    }
  }

  // 開始
  animationId = requestAnimationFrame(animate);

  // 停止（任意のタイミングで呼べる）
  function stop() {
    if (animationId !== null) {
      cancelAnimationFrame(animationId);
      animationId = null;
    }
  }

  return { stop };
}
```

### 4.3 イージング関数ライブラリ

```javascript
// 基本イージング関数集
const Easing = {
  // 線形
  linear: (t) => t,

  // Quad（2次曲線）
  easeInQuad: (t) => t * t,
  easeOutQuad: (t) => t * (2 - t),
  easeInOutQuad: (t) => t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t,

  // Cubic（3次曲線）
  easeInCubic: (t) => t * t * t,
  easeOutCubic: (t) => 1 - Math.pow(1 - t, 3),
  easeInOutCubic: (t) =>
    t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2,

  // Elastic（弾性）
  easeOutElastic: (t) => {
    const c4 = (2 * Math.PI) / 3;
    return t === 0 ? 0 : t === 1 ? 1 :
      Math.pow(2, -10 * t) * Math.sin((t * 10 - 0.75) * c4) + 1;
  },

  // Bounce（バウンス）
  easeOutBounce: (t) => {
    const n1 = 7.5625;
    const d1 = 2.75;
    if (t < 1 / d1) {
      return n1 * t * t;
    } else if (t < 2 / d1) {
      return n1 * (t -= 1.5 / d1) * t + 0.75;
    } else if (t < 2.5 / d1) {
      return n1 * (t -= 2.25 / d1) * t + 0.9375;
    } else {
      return n1 * (t -= 2.625 / d1) * t + 0.984375;
    }
  },

  // Back（引き戻し）
  easeOutBack: (t) => {
    const c1 = 1.70158;
    const c3 = c1 + 1;
    return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
  },

  // Spring（バネ）
  spring: (t) => {
    return 1 - (Math.cos(t * Math.PI * 4.5) * Math.exp(-t * 6));
  }
};

// 使用例
function springAnimation(element, fromX, toX, duration) {
  let startTime = null;

  function animate(timestamp) {
    if (startTime === null) startTime = timestamp;
    const progress = Math.min((timestamp - startTime) / duration, 1);
    const eased = Easing.spring(progress);
    const x = fromX + (toX - fromX) * eased;

    element.style.transform = `translateX(${x}px)`;

    if (progress < 1) {
      requestAnimationFrame(animate);
    }
  }

  requestAnimationFrame(animate);
}
```

### 4.4 rAF のバッチ処理とスケジューリング

```javascript
// 複数のアニメーションを効率的にバッチ処理するスケジューラ
class AnimationScheduler {
  constructor() {
    this.animations = new Map();
    this.isRunning = false;
    this.frameId = null;
  }

  add(id, updateFn) {
    this.animations.set(id, updateFn);
    if (!this.isRunning) {
      this.start();
    }
  }

  remove(id) {
    this.animations.delete(id);
    if (this.animations.size === 0) {
      this.stop();
    }
  }

  start() {
    this.isRunning = true;
    const tick = (timestamp) => {
      // 全アニメーションを1フレーム内でバッチ処理
      for (const [id, updateFn] of this.animations) {
        const shouldContinue = updateFn(timestamp);
        if (!shouldContinue) {
          this.animations.delete(id);
        }
      }

      if (this.animations.size > 0) {
        this.frameId = requestAnimationFrame(tick);
      } else {
        this.isRunning = false;
      }
    };

    this.frameId = requestAnimationFrame(tick);
  }

  stop() {
    if (this.frameId !== null) {
      cancelAnimationFrame(this.frameId);
      this.frameId = null;
    }
    this.isRunning = false;
    this.animations.clear();
  }
}

// 使用例
const scheduler = new AnimationScheduler();

// 複数要素のアニメーションを登録
document.querySelectorAll('.particle').forEach((el, i) => {
  let startTime = null;
  scheduler.add(`particle-${i}`, (timestamp) => {
    if (startTime === null) startTime = timestamp;
    const progress = (timestamp - startTime) / 2000;
    if (progress >= 1) return false; // アニメーション完了

    const x = Math.sin(progress * Math.PI * 2 + i) * 100;
    const y = Math.cos(progress * Math.PI * 2 + i) * 100;
    el.style.transform = `translate(${x}px, ${y}px)`;
    return true; // 継続
  });
});
```

---

## 5. FLIP 技法の原理と応用

### 5.1 FLIP の概念

FLIPは "First, Last, Invert, Play" の頭文字をとった技法であり、Paul Lewisが提唱した。レイアウト変更を伴うアニメーション（要素の移動、リサイズ等）を、パフォーマンスに優れた `transform` アニメーションに変換するテクニックである。

```
FLIP 技法の4ステップ:

  Step 1: First（記録）
  ┌────────────────────────┐
  │  ┌──┐                  │  getBoundingClientRect() で
  │  │A │  x=10, y=20      │  現在の位置・サイズを記録
  │  └──┘  w=100, h=50     │
  └────────────────────────┘

  Step 2: Last（変更を適用）
  ┌────────────────────────┐
  │            ┌──────┐    │  DOM変更を適用し、
  │            │  A   │    │  最終位置を取得
  │            └──────┘    │  x=200, y=100
  └────────────────────────┘  w=150, h=75

  Step 3: Invert（逆変換）
  ┌────────────────────────┐
  │  ┌──┐                  │  transform で元の位置に戻す
  │  │A │  ← transform で  │  deltaX = 10 - 200 = -190
  │  └──┘    元の位置に     │  deltaY = 20 - 100 = -80
  └────────────────────────┘  scaleX = 100/150, scaleY = 50/75

  Step 4: Play（再生）
  ┌────────────────────────┐
  │  ┌──┐  →  →  ┌──────┐ │  transform を解除して
  │  │A │  →  →  │  A   │ │  CSS Transition で
  │  └──┘  →  →  └──────┘ │  自然にアニメーション
  └────────────────────────┘
```

### 5.2 汎用 FLIP ヘルパー関数

```javascript
/**
 * 汎用FLIP アニメーション関数
 * @param {HTMLElement} element - アニメーション対象要素
 * @param {Function} changeFn - DOM変更を行う関数
 * @param {Object} options - アニメーションオプション
 */
function flipAnimate(element, changeFn, options = {}) {
  const {
    duration = 300,
    easing = 'cubic-bezier(0.2, 0, 0.2, 1)',
    onComplete = null,
    scaleCorrection = true
  } = options;

  // First: 現在の位置・サイズを記録
  const first = element.getBoundingClientRect();

  // Last: DOM変更を適用し、最終位置を取得
  changeFn();
  const last = element.getBoundingClientRect();

  // 差分を計算
  const deltaX = first.left - last.left;
  const deltaY = first.top - last.top;
  const scaleX = first.width / last.width;
  const scaleY = first.height / last.height;

  // 変化がない場合はスキップ
  if (deltaX === 0 && deltaY === 0 && scaleX === 1 && scaleY === 1) {
    if (onComplete) onComplete();
    return;
  }

  // Invert: transform で元の位置に戻す
  const transform = scaleCorrection
    ? `translate(${deltaX}px, ${deltaY}px) scale(${scaleX}, ${scaleY})`
    : `translate(${deltaX}px, ${deltaY}px)`;

  element.style.transform = transform;
  element.style.transformOrigin = 'top left';

  // ブラウザに Invert 状態を認識させる
  // getComputedStyle().transform を読むことで強制的に再計算
  void element.offsetHeight;

  // Play: transform を解除してアニメーション
  element.style.transition = `transform ${duration}ms ${easing}`;
  element.style.transform = '';

  // 完了後のクリーンアップ
  function handleTransitionEnd(e) {
    if (e.propertyName !== 'transform') return;
    element.removeEventListener('transitionend', handleTransitionEnd);
    element.style.transition = '';
    element.style.transformOrigin = '';
    if (onComplete) onComplete();
  }

  element.addEventListener('transitionend', handleTransitionEnd);
}

// 使用例: リストのソートアニメーション
function sortListWithFlip(container, compareFn) {
  const items = Array.from(container.children);

  // First: 全要素の位置を記録
  const firstRects = new Map();
  items.forEach(item => {
    firstRects.set(item, item.getBoundingClientRect());
  });

  // Last: ソートを適用
  items.sort(compareFn);
  items.forEach(item => container.appendChild(item));

  // 各要素にFLIPを適用
  items.forEach(item => {
    const first = firstRects.get(item);
    const last = item.getBoundingClientRect();

    const deltaX = first.left - last.left;
    const deltaY = first.top - last.top;

    if (deltaX === 0 && deltaY === 0) return;

    item.style.transform = `translate(${deltaX}px, ${deltaY}px)`;

    requestAnimationFrame(() => {
      item.style.transition = 'transform 300ms ease-out';
      item.style.transform = '';

      item.addEventListener('transitionend', () => {
        item.style.transition = '';
      }, { once: true });
    });
  });
}
```

### 5.3 FLIP によるリスト項目の追加・削除

```javascript
// リスト項目の追加アニメーション
function addItemWithFlip(container, newElement, referenceNode = null) {
  // First: 既存要素の位置を記録
  const existingItems = Array.from(container.children);
  const firstRects = new Map();
  existingItems.forEach(item => {
    firstRects.set(item, item.getBoundingClientRect());
  });

  // Last: 新要素を挿入
  if (referenceNode) {
    container.insertBefore(newElement, referenceNode);
  } else {
    container.appendChild(newElement);
  }

  // 新要素のアニメーション
  newElement.style.opacity = '0';
  newElement.style.transform = 'scale(0.8)';

  requestAnimationFrame(() => {
    newElement.style.transition = 'opacity 200ms ease-out, transform 200ms ease-out';
    newElement.style.opacity = '1';
    newElement.style.transform = 'scale(1)';
  });

  // 既存要素のFLIPアニメーション
  existingItems.forEach(item => {
    const first = firstRects.get(item);
    const last = item.getBoundingClientRect();
    const deltaY = first.top - last.top;

    if (deltaY === 0) return;

    item.style.transform = `translateY(${deltaY}px)`;

    requestAnimationFrame(() => {
      item.style.transition = 'transform 300ms ease-out';
      item.style.transform = '';
      item.addEventListener('transitionend', () => {
        item.style.transition = '';
      }, { once: true });
    });
  });
}

// リスト項目の削除アニメーション
function removeItemWithFlip(container, targetElement) {
  // First: 全要素の位置を記録
  const items = Array.from(container.children);
  const firstRects = new Map();
  items.forEach(item => {
    firstRects.set(item, item.getBoundingClientRect());
  });

  // 削除対象のフェードアウト
  targetElement.style.transition = 'opacity 150ms ease-in, transform 150ms ease-in';
  targetElement.style.opacity = '0';
  targetElement.style.transform = 'scale(0.8)';

  targetElement.addEventListener('transitionend', () => {
    // 要素を削除
    container.removeChild(targetElement);

    // 残りの要素にFLIPを適用
    items.filter(item => item !== targetElement).forEach(item => {
      const first = firstRects.get(item);
      const last = item.getBoundingClientRect();
      const deltaY = first.top - last.top;

      if (deltaY === 0) return;

      item.style.transform = `translateY(${deltaY}px)`;

      requestAnimationFrame(() => {
        item.style.transition = 'transform 300ms ease-out';
        item.style.transform = '';
        item.addEventListener('transitionend', () => {
          item.style.transition = '';
        }, { once: true });
      });
    });
  }, { once: true });
}
```

---

## 6. Web Animations API (WAAPI)

### 6.1 WAAPI の概要と利点

Web Animations APIは、CSS AnimationsとCSS Transitionsの基盤となるアニメーションモデルを直接JavaScriptから操作できるAPIである。CSSベースのアニメーションのパフォーマンス特性を維持しつつ、JavaScriptによる動的な制御を可能にする。

```
CSSアニメーション vs rAF vs WAAPI の比較:

  特性              CSS Animation    rAF          WAAPI
  ──────────────────────────────────────────────────────────
  宣言的             ✓               -            -
  プログラム制御      △               ✓            ✓
  パフォーマンス      高(GPU)          中(CPU)      高(GPU)
  一時停止/再開      △               手動実装      ✓ (組み込み)
  逆再生             -               手動実装      ✓ (組み込み)
  完了Promise        -               手動実装      ✓ (組み込み)
  再生速度変更       -               手動実装      ✓ (組み込み)
  タイムライン同期   ✓ (CSS)          -            ✓ (JS)
  複数要素の連携     △               手動実装      ✓ (GroupEffect)
  キーフレーム動的変更 -              ✓            ✓
  ブラウザ互換性     広い             広い          広い (2024+)
```

### 6.2 基本的なWAAPI の使い方

```javascript
// Element.animate() による基本アニメーション
const element = document.querySelector('.box');

const animation = element.animate(
  // キーフレーム配列
  [
    { transform: 'translateX(0)', opacity: 1 },
    { transform: 'translateX(300px)', opacity: 0.5 },
    { transform: 'translateX(300px) rotate(180deg)', opacity: 1 }
  ],
  // タイミングオプション
  {
    duration: 1000,
    easing: 'cubic-bezier(0.4, 0, 0.2, 1)',
    iterations: 1,
    fill: 'forwards',
    delay: 0
  }
);

// アニメーションの制御
animation.pause();           // 一時停止
animation.play();            // 再生
animation.reverse();         // 逆再生
animation.cancel();          // キャンセル
animation.finish();          // 即完了

// 再生速度の変更
animation.playbackRate = 2;  // 2倍速
animation.playbackRate = 0.5; // 半速
animation.playbackRate = -1;  // 逆再生

// 現在の進捗を取得・設定
console.log(animation.currentTime);  // 現在の経過時間(ms)
animation.currentTime = 500;         // 500msの位置にシーク

// 完了を待つ Promise
animation.finished.then(() => {
  console.log('アニメーション完了');
  element.classList.add('final-state');
});

// async/await パターン
async function animateSequence() {
  const el = document.querySelector('.box');

  await el.animate(
    [{ transform: 'scale(1)' }, { transform: 'scale(1.2)' }],
    { duration: 200, fill: 'forwards' }
  ).finished;

  await el.animate(
    [{ transform: 'scale(1.2)' }, { transform: 'scale(1)' }],
    { duration: 150, fill: 'forwards' }
  ).finished;

  console.log('全アニメーション完了');
}
```

### 6.3 WAAPI によるステージングアニメーション

```javascript
// 複数要素の連鎖アニメーション
function staggeredReveal(elements, options = {}) {
  const {
    duration = 400,
    stagger = 50,
    easing = 'cubic-bezier(0.22, 1, 0.36, 1)',
    distance = 30
  } = options;

  const animations = Array.from(elements).map((el, index) => {
    return el.animate(
      [
        {
          opacity: 0,
          transform: `translateY(${distance}px)`
        },
        {
          opacity: 1,
          transform: 'translateY(0)'
        }
      ],
      {
        duration,
        easing,
        delay: index * stagger,
        fill: 'both'
      }
    );
  });

  // 全アニメーション完了を待つ
  return Promise.all(animations.map(a => a.finished));
}

// 使用例
const cards = document.querySelectorAll('.card');
staggeredReveal(cards, { stagger: 80, distance: 40 }).then(() => {
  console.log('全カードの表示アニメーション完了');
});
```

### 6.4 WAAPI のキーフレーム記法

```javascript
// 記法1: 配列形式（各フレームを個別オブジェクトで定義）
element.animate([
  { transform: 'rotate(0deg)', offset: 0 },
  { transform: 'rotate(360deg)', offset: 0.7 },
  { transform: 'rotate(360deg) scale(1.2)', offset: 0.85 },
  { transform: 'rotate(360deg) scale(1)', offset: 1 }
], { duration: 800 });

// 記法2: オブジェクト形式（プロパティごとに値の配列）
element.animate({
  transform: [
    'translateX(0)',
    'translateX(200px)',
    'translateX(200px) rotate(90deg)',
    'translateX(0) rotate(0deg)'
  ],
  opacity: [1, 0.8, 0.6, 1],
  offset: [0, 0.3, 0.7, 1],
  easing: ['ease-in', 'ease-out', 'ease-in-out']
}, { duration: 1200, iterations: Infinity });

// composite オプションによるアニメーション合成
const baseAnimation = element.animate(
  [{ transform: 'translateX(0)' }, { transform: 'translateX(200px)' }],
  { duration: 2000, iterations: Infinity, composite: 'replace' }
);

const additiveAnimation = element.animate(
  [{ transform: 'translateY(0)' }, { transform: 'translateY(50px)' }],
  { duration: 1000, iterations: Infinity, composite: 'add' }
);
// 結果: X方向とY方向の動きが合成される
```

---

## 7. View Transitions API

### 7.1 同一ドキュメント内のView Transitions

View Transitions APIは、DOM変更前後の状態を自動的にスナップショットとして取得し、クロスフェードやスライド等の遷移アニメーションを実現する。FLIPと同様の概念をブラウザネイティブで提供するものである。

```javascript
// 基本的なView Transition
async function updateContent(newContent) {
  // startViewTransition は DOM 変更を引数として受け取る
  const transition = document.startViewTransition(() => {
    document.querySelector('.content').innerHTML = newContent;
  });

  // 遷移の完了を待つことが可能
  await transition.finished;
  console.log('View Transition 完了');
}

// View Transition のライフサイクル
const transition = document.startViewTransition(async () => {
  // この関数内でDOM変更を行う
  await fetchAndUpdateDOM();
});

// 各フェーズのPromise
transition.ready.then(() => {
  // 擬似要素ツリーが構築され、アニメーション開始直前
  console.log('アニメーション準備完了');
});

transition.updateCallbackDone.then(() => {
  // DOM更新コールバックが完了した
  console.log('DOM更新完了');
});

transition.finished.then(() => {
  // 全アニメーションが完了した
  console.log('View Transition 完了');
});
```

### 7.2 View Transition のCSS制御

```css
/* デフォルトのクロスフェードをカスタマイズ */
::view-transition-old(root) {
  animation-duration: 250ms;
  animation-timing-function: ease-in;
}

::view-transition-new(root) {
  animation-duration: 250ms;
  animation-timing-function: ease-out;
}

/* 特定の要素に名前を付けて個別制御 */
.hero-image {
  view-transition-name: hero;
}

.page-title {
  view-transition-name: title;
}

/* 名前付き要素のアニメーションをカスタマイズ */
::view-transition-old(hero) {
  animation: slideOutLeft 300ms ease-in forwards;
}

::view-transition-new(hero) {
  animation: slideInRight 300ms ease-out forwards;
}

::view-transition-group(title) {
  animation-duration: 400ms;
  animation-timing-function: cubic-bezier(0.22, 1, 0.36, 1);
}

@keyframes slideOutLeft {
  from { transform: translateX(0); opacity: 1; }
  to { transform: translateX(-100%); opacity: 0; }
}

@keyframes slideInRight {
  from { transform: translateX(100%); opacity: 0; }
  to { transform: translateX(0); opacity: 1; }
}
```

```
View Transition の擬似要素ツリー:

  ::view-transition
  ├── ::view-transition-group(root)
  │   ├── ::view-transition-image-pair(root)
  │   │   ├── ::view-transition-old(root)    ← 変更前のスナップショット
  │   │   └── ::view-transition-new(root)    ← 変更後のスナップショット
  │   │
  ├── ::view-transition-group(hero)
  │   ├── ::view-transition-image-pair(hero)
  │   │   ├── ::view-transition-old(hero)
  │   │   └── ::view-transition-new(hero)
  │   │
  └── ::view-transition-group(title)
      ├── ::view-transition-image-pair(title)
      │   ├── ::view-transition-old(title)
      │   └── ::view-transition-new(title)

  各グループは独立してアニメーション可能
  デフォルトでは全要素がクロスフェード
  view-transition-name で個別制御が可能
```

### 7.3 MPA (Multi-Page Application) でのView Transitions

```html
<!-- ページ間遷移でのView Transitions -->
<!-- ページ A の head に追加 -->
<meta name="view-transition" content="same-origin" />

<!-- ページ B の head にも追加 -->
<meta name="view-transition" content="same-origin" />
```

```css
/* ページ間で共有する要素に同じ名前を付ける */
/* ページ A */
.product-image-123 {
  view-transition-name: product-123;
}

/* ページ B */
.product-detail-image {
  view-transition-name: product-123;
}

/* ナビゲーション方向に応じたアニメーション */
@view-transition {
  navigation: auto;
}

/* 進む方向のアニメーション */
::view-transition-old(root) {
  animation: slide-out-to-left 300ms ease-in;
}
::view-transition-new(root) {
  animation: slide-in-from-right 300ms ease-out;
}

/* Navigation API と組み合わせて戻る方向を検出 */
@keyframes slide-out-to-left {
  to { transform: translateX(-30%); opacity: 0; }
}

@keyframes slide-in-from-right {
  from { transform: translateX(30%); opacity: 0; }
}
```

---

## 8. パフォーマンス計測とデバッグ

### 8.1 Chrome DevTools によるアニメーション分析

```
DevTools Performance パネルの読み方:

  ┌────────────────────────────────────────────────────────┐
  │ FPS グラフ                                              │
  │ ████████████████████████ ██ ███████████████████████████ │
  │ 60fps                   ↑ドロップ                      │
  │                         ここでジャンク発生               │
  ├────────────────────────────────────────────────────────┤
  │ Main スレッド                                           │
  │ ┌──────┐ ┌──┐ ┌────────────────────┐ ┌──────┐        │
  │ │ Task │ │rAF│ │   Long Task        │ │ Task │        │
  │ │ 3ms  │ │2ms│ │   52ms (> 50ms)   │ │ 4ms  │        │
  │ └──────┘ └──┘ └────────────────────┘ └──────┘        │
  │                 ↑ これがジャンクの原因                   │
  ├────────────────────────────────────────────────────────┤
  │ GPU スレッド                                            │
  │ ┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐          │
  │ │Comp │     │Comp │     │Comp │     │Comp │          │
  │ │ 2ms │     │ 2ms │     │ 8ms │     │ 2ms │          │
  │ └─────┘     └─────┘     └─────┘     └─────┘          │
  └────────────────────────────────────────────────────────┘

  確認すべきポイント:
  1. 赤い三角マーク → Long Task（50ms以上）の発生箇所
  2. FPSグラフの谷 → フレームドロップの発生箇所
  3. Layout / Paint の発生 → 不必要なリフロー・リペイント
  4. Forced Reflow 警告 → レイアウトスラッシングの疑い
```

### 8.2 Performance API によるプログラム的計測

```javascript
// フレームレートの計測
class FPSMonitor {
  constructor() {
    this.frames = [];
    this.isRunning = false;
  }

  start() {
    this.isRunning = true;
    this.frames = [];
    this.lastTime = performance.now();
    this.tick();
  }

  tick() {
    if (!this.isRunning) return;

    const now = performance.now();
    const delta = now - this.lastTime;
    this.lastTime = now;

    this.frames.push({
      timestamp: now,
      frameDuration: delta,
      fps: 1000 / delta
    });

    // 直近60フレームのみ保持
    if (this.frames.length > 60) {
      this.frames.shift();
    }

    requestAnimationFrame(() => this.tick());
  }

  stop() {
    this.isRunning = false;
  }

  getAverageFPS() {
    if (this.frames.length === 0) return 0;
    const totalFPS = this.frames.reduce((sum, f) => sum + f.fps, 0);
    return totalFPS / this.frames.length;
  }

  getDroppedFrames() {
    // 16.67ms * 1.5 = 25ms 以上のフレームをドロップとみなす
    return this.frames.filter(f => f.frameDuration > 25).length;
  }

  getReport() {
    const durations = this.frames.map(f => f.frameDuration);
    return {
      averageFPS: this.getAverageFPS().toFixed(1),
      droppedFrames: this.getDroppedFrames(),
      totalFrames: this.frames.length,
      maxFrameDuration: Math.max(...durations).toFixed(1),
      minFrameDuration: Math.min(...durations).toFixed(1),
      p95FrameDuration: this.percentile(durations, 95).toFixed(1)
    };
  }

  percentile(arr, p) {
    const sorted = [...arr].sort((a, b) => a - b);
    const index = Math.ceil(sorted.length * p / 100) - 1;
    return sorted[index];
  }
}

// 使用例
const monitor = new FPSMonitor();
monitor.start();

// アニメーション実行...

setTimeout(() => {
  monitor.stop();
  console.table(monitor.getReport());
}, 3000);
```

### 8.3 PerformanceObserver でのロングタスク検出

```javascript
// Long Task の検出と記録
const longTaskObserver = new PerformanceObserver((list) => {
  for (const entry of list.getEntries()) {
    console.warn(
      `Long Task 検出: ${entry.duration.toFixed(1)}ms`,
      `名前: ${entry.name}`,
      `開始: ${entry.startTime.toFixed(1)}ms`
    );

    // 50ms以上のタスクはアニメーションに影響を与える可能性がある
    if (entry.duration > 100) {
      console.error(
        `致命的なLong Task: ${entry.duration.toFixed(1)}ms - ` +
        `約${Math.floor(entry.duration / 16.67)}フレーム分のジャンクが発生`
      );
    }
  }
});

longTaskObserver.observe({ type: 'longtask', buffered: true });

// Layout Shift の検出
const clsObserver = new PerformanceObserver((list) => {
  for (const entry of list.getEntries()) {
    if (!entry.hadRecentInput) {
      console.warn(`Layout Shift: ${entry.value.toFixed(4)}`, entry.sources);
    }
  }
});

clsObserver.observe({ type: 'layout-shift', buffered: true });
```

### 8.4 フレームタイミングの可視化

```javascript
// アニメーションフレームのタイミングをオーバーレイ表示
class FrameTimingOverlay {
  constructor() {
    this.canvas = document.createElement('canvas');
    this.canvas.width = 200;
    this.canvas.height = 80;
    this.canvas.style.cssText = `
      position: fixed; top: 10px; right: 10px; z-index: 99999;
      background: rgba(0,0,0,0.7); border-radius: 4px;
      pointer-events: none;
    `;
    document.body.appendChild(this.canvas);
    this.ctx = this.canvas.getContext('2d');
    this.frameTimes = [];
    this.lastTime = 0;
    this.running = false;
  }

  start() {
    this.running = true;
    this.lastTime = performance.now();
    this.loop();
  }

  loop() {
    if (!this.running) return;
    const now = performance.now();
    const dt = now - this.lastTime;
    this.lastTime = now;
    this.frameTimes.push(dt);
    if (this.frameTimes.length > 100) this.frameTimes.shift();
    this.draw();
    requestAnimationFrame(() => this.loop());
  }

  draw() {
    const { ctx, canvas } = this;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // FPS テキスト
    const fps = this.frameTimes.length > 0
      ? (1000 / (this.frameTimes.reduce((a, b) => a + b) / this.frameTimes.length))
      : 0;
    ctx.fillStyle = fps >= 55 ? '#4ade80' : fps >= 30 ? '#fbbf24' : '#ef4444';
    ctx.font = '14px monospace';
    ctx.fillText(`${fps.toFixed(0)} FPS`, 10, 18);

    // バーグラフ
    const barWidth = (canvas.width - 20) / this.frameTimes.length;
    this.frameTimes.forEach((dt, i) => {
      const height = Math.min(dt / 33.33 * 40, 50);
      const x = 10 + i * barWidth;
      const y = canvas.height - 10 - height;

      ctx.fillStyle = dt <= 16.67 ? '#4ade80' :
                      dt <= 25 ? '#fbbf24' : '#ef4444';
      ctx.fillRect(x, y, Math.max(barWidth - 1, 1), height);
    });

    // 16.67ms ライン
    const lineY = canvas.height - 10 - (16.67 / 33.33 * 40);
    ctx.strokeStyle = 'rgba(255,255,255,0.3)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(10, lineY);
    ctx.lineTo(canvas.width - 10, lineY);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  stop() {
    this.running = false;
    this.canvas.remove();
  }
}
```

---

## 9. prefers-reduced-motion と アクセシビリティ

### 9.1 モーション酔いとアクセシビリティ

前庭障害（vestibular disorders）を持つユーザーは、画面上の動きによって頭痛、めまい、吐き気を経験する場合がある。`prefers-reduced-motion` メディアクエリを使うことで、こうしたユーザーに対してアニメーションを軽減できる。

```css
/* アプローチ1: 動きを完全に除去 */
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

/* アプローチ2: 動きの種類を選択的に制御（推奨） */
@media (prefers-reduced-motion: reduce) {
  /* 移動・回転・スケールは除去 */
  .animated-element {
    transform: none !important;
    transition: opacity 200ms ease-out;
  }

  /* フェード効果は残す（動きではないので問題が少ない） */
  .fade-in {
    transition: opacity 200ms ease-out;
  }

  /* スクロール連動アニメーションは除去 */
  .scroll-animation {
    animation: none !important;
  }
}

/* アプローチ3: モーション前提ではなく、オプトインにする */
/* ベース: アニメーションなし */
.card {
  /* 静的なスタイルのみ */
}

/* 動きを許可したユーザーにのみアニメーションを適用 */
@media (prefers-reduced-motion: no-preference) {
  .card {
    transition: transform 300ms ease-out, opacity 300ms ease-out;
  }

  .card:hover {
    transform: translateY(-4px);
  }
}
```

### 9.2 JavaScript での prefers-reduced-motion 検出

```javascript
// メディアクエリの状態を検出
const prefersReducedMotion = window.matchMedia(
  '(prefers-reduced-motion: reduce)'
);

// 初期値を取得
if (prefersReducedMotion.matches) {
  console.log('ユーザーはモーション軽減を要求している');
}

// 設定変更を監視
prefersReducedMotion.addEventListener('change', (event) => {
  if (event.matches) {
    // モーション軽減に切り替わった → アニメーションを停止
    stopAllAnimations();
  } else {
    // モーション軽減が解除された → アニメーションを有効化
    enableAnimations();
  }
});

// アニメーション関数にモーション軽減を組み込む
function animateElement(element, keyframes, options) {
  if (prefersReducedMotion.matches) {
    // モーション軽減時: 即座に最終状態を適用
    const lastKeyframe = keyframes[keyframes.length - 1];
    Object.assign(element.style, lastKeyframe);
    return { finished: Promise.resolve() };
  }

  return element.animate(keyframes, options);
}

// フレームワーク向けカスタムフック例（React風の擬似コード）
function useReducedMotion() {
  const query = window.matchMedia('(prefers-reduced-motion: reduce)');
  let matches = query.matches;

  query.addEventListener('change', (e) => {
    matches = e.matches;
    // 状態更新をトリガー
  });

  return matches;
}
```

---

## 10. アニメーション設計のパターン集

### 10.1 マイクロインタラクション

```css
/* ボタンの押下フィードバック */
.button {
  transform: scale(1);
  transition: transform 100ms ease-out;
}

.button:active {
  transform: scale(0.96);
  transition-duration: 50ms;
}

/* トグルスイッチ */
.toggle-track {
  width: 48px;
  height: 24px;
  border-radius: 12px;
  background: #d1d5db;
  transition: background-color 200ms ease;
  position: relative;
}

.toggle-track.active {
  background: #3b82f6;
}

.toggle-thumb {
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: white;
  position: absolute;
  top: 2px;
  left: 2px;
  transition: transform 200ms cubic-bezier(0.34, 1.56, 0.64, 1);
  box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}

.toggle-track.active .toggle-thumb {
  transform: translateX(24px);
}

/* スケルトンスクリーンのシマー効果 */
@keyframes shimmer {
  0% {
    background-position: -200% 0;
  }
  100% {
    background-position: 200% 0;
  }
}

.skeleton {
  background: linear-gradient(
    90deg,
    #f0f0f0 25%,
    #e0e0e0 50%,
    #f0f0f0 75%
  );
  background-size: 200% 100%;
  animation: shimmer 1.5s ease-in-out infinite;
  border-radius: 4px;
}

/* ローディングスピナー */
@keyframes spin {
  to { transform: rotate(360deg); }
}

.spinner {
  width: 24px;
  height: 24px;
  border: 3px solid #e5e7eb;
  border-top-color: #3b82f6;
  border-radius: 50%;
  animation: spin 0.6s linear infinite;
}
```

### 10.2 ページ遷移パターン

```javascript
// スライド遷移（SPA向け）
async function slideTransition(direction, updateFn) {
  if (!document.startViewTransition) {
    // View Transitions API 非対応時のフォールバック
    updateFn();
    return;
  }

  // 方向に応じたCSSクラスを設定
  document.documentElement.dataset.transition = direction;

  const transition = document.startViewTransition(async () => {
    await updateFn();
  });

  await transition.finished;
  delete document.documentElement.dataset.transition;
}
```

```css
/* スライド方向の制御 */
[data-transition="forward"] ::view-transition-old(root) {
  animation: slide-out-left 300ms ease-in forwards;
}

[data-transition="forward"] ::view-transition-new(root) {
  animation: slide-in-right 300ms ease-out forwards;
}

[data-transition="back"] ::view-transition-old(root) {
  animation: slide-out-right 300ms ease-in forwards;
}

[data-transition="back"] ::view-transition-new(root) {
  animation: slide-in-left 300ms ease-out forwards;
}

@keyframes slide-out-left {
  to { transform: translateX(-30%); opacity: 0; }
}
@keyframes slide-in-right {
  from { transform: translateX(30%); opacity: 0; }
}
@keyframes slide-out-right {
  to { transform: translateX(30%); opacity: 0; }
}
@keyframes slide-in-left {
  from { transform: translateX(-30%); opacity: 0; }
}
```

---

## 11. アンチパターンと回避策

### 11.1 アンチパターン1: レイアウトスラッシング

レイアウトスラッシング（Layout Thrashing）は、DOM読み取りとDOM書き込みを交互に繰り返すことで、ブラウザが各書き込みの度に強制的にレイアウト再計算を行ってしまう問題である。これはアニメーション中にフレームバジェットを大幅に超過させる最も一般的な原因の一つである。

```javascript
// ---- 悪い例: レイアウトスラッシング ----
function badResizeItems(items) {
  items.forEach((item) => {
    // 読み取り → 強制レイアウト発生！
    const height = item.offsetHeight;
    // 書き込み → レイアウトを無効化
    item.style.height = (height * 1.2) + 'px';
    // 次のループの読み取りで再び強制レイアウト...
  });
  // N個の要素があれば N回の強制レイアウトが発生する
}

// ---- 良い例: 読み取りと書き込みを分離 ----
function goodResizeItems(items) {
  // Phase 1: 全ての読み取りをまとめて行う
  const heights = items.map((item) => item.offsetHeight);

  // Phase 2: 全ての書き込みをまとめて行う（レイアウトは1回のみ）
  items.forEach((item, i) => {
    item.style.height = (heights[i] * 1.2) + 'px';
  });
}
```

```
レイアウトスラッシングの影響:

  悪い例（交互に読み書き）:
  ┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐
  │Read││Write│Read││Write│Read││Write│Read││Write│
  │    ││+   ││    ││+   ││    ││+   ││    ││+   │
  │    ││Lay ││    ││Lay ││    ││Lay ││    ││Lay │
  └────┘└────┘└────┘└────┘└────┘└────┘└────┘└────┘
  合計: 4回のレイアウト計算（各Write時に強制発生）

  良い例（読み取りをバッチ処理）:
  ┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────────┐
  │Read││Read││Read││Read││Write│Write│Write│Write│+Layout│
  └────┘└────┘└────┘└────┘└────┘└────┘└────┘└────┘└──────┘
  合計: 1回のレイアウト計算（フレーム描画時に1回のみ）
```

```javascript
// fastdom ライブラリのようなパターンで読み書きを分離
class DOMBatcher {
  constructor() {
    this.reads = [];
    this.writes = [];
    this.scheduled = false;
  }

  read(fn) {
    this.reads.push(fn);
    this.schedule();
  }

  write(fn) {
    this.writes.push(fn);
    this.schedule();
  }

  schedule() {
    if (this.scheduled) return;
    this.scheduled = true;

    requestAnimationFrame(() => {
      // まず全ての読み取りを実行
      const readResults = this.reads.map((fn) => fn());
      this.reads = [];

      // 次に全ての書き込みを実行
      this.writes.forEach((fn) => fn());
      this.writes = [];

      this.scheduled = false;
    });
  }
}

const batcher = new DOMBatcher();

// 使用例: 読み取りと書き込みを安全に分離
function animateCards(cards) {
  cards.forEach((card) => {
    batcher.read(() => {
      const rect = card.getBoundingClientRect();
      batcher.write(() => {
        card.style.transform = `translateY(${rect.top * 0.1}px)`;
      });
    });
  });
}
```

### 11.2 アンチパターン2: will-change の乱用

`will-change` を全要素に永続的に適用すると、GPUメモリを大量消費し、かえってパフォーマンスが低下する。

```css
/* ---- 悪い例: will-change を全要素に常時適用 ---- */
* {
  will-change: transform, opacity;
  /* 全要素がGPUレイヤーに昇格 → メモリ枯渇 */
}

.every-list-item {
  will-change: transform;
  /* 1000個のリストアイテムそれぞれがレイヤーになる */
}

/* ---- 良い例: 必要な時に必要な要素にだけ適用 ---- */
.card {
  /* 通常時は will-change なし */
}

.card:hover {
  will-change: transform;
  /* ホバー時にのみ昇格 */
}

/* さらに良い例: JavaScript で動的に制御 */
```

```javascript
// will-change の適切なライフサイクル管理
class WillChangeManager {
  constructor(element, properties) {
    this.element = element;
    this.properties = properties;
    this.isActive = false;
  }

  // アニメーション開始の少し前に準備
  prepare() {
    if (this.isActive) return;
    this.element.style.willChange = this.properties;
    this.isActive = true;
  }

  // アニメーション完了後に解除
  cleanup() {
    if (!this.isActive) return;
    this.element.style.willChange = 'auto';
    this.isActive = false;
  }

  // transitionend と連動する自動管理
  autoManage() {
    this.element.addEventListener('mouseenter', () => this.prepare());
    this.element.addEventListener('transitionend', () => this.cleanup());
    this.element.addEventListener('mouseleave', () => {
      // マウスが離れた後、遷移が終わればクリーンアップ
      requestAnimationFrame(() => {
        if (!this.element.matches(':hover')) {
          this.cleanup();
        }
      });
    });
  }
}
```

### 11.3 アンチパターン3: setInterval によるアニメーション

```javascript
// ---- 悪い例: setInterval でアニメーション ----
let x = 0;
const intervalId = setInterval(() => {
  x += 2;
  element.style.left = x + 'px'; // Layout を毎回トリガー
  if (x >= 300) clearInterval(intervalId);
}, 16); // 16msはフレームとずれる

// 問題点:
// 1. setInterval のタイミングはフレームと同期しない
// 2. 非アクティブタブでも実行され続ける
// 3. 処理が遅延した場合、コールバックが溜まる
// 4. left プロパティは Layout を毎フレーム発生させる

// ---- 良い例: rAF + transform ----
let startTime = null;
const duration = 2500;

function animate(timestamp) {
  if (startTime === null) startTime = timestamp;
  const progress = Math.min((timestamp - startTime) / duration, 1);
  const eased = Easing.easeOutCubic(progress);

  element.style.transform = `translateX(${eased * 300}px)`;

  if (progress < 1) {
    requestAnimationFrame(animate);
  }
}

requestAnimationFrame(animate);
```

---

## 12. エッジケース分析

### 12.1 エッジケース1: 高リフレッシュレートディスプレイ

120Hzや144Hzのディスプレイでは、1フレームあたりの時間がそれぞれ8.3msや6.9msに短縮される。固定値ベースのアニメーション（フレームごとに一定量移動する方式）は、これらのディスプレイで予期せぬ速度変化を起こす。

```javascript
// ---- 悪い例: フレーム単位の固定値移動 ----
function badAnimate() {
  x += 5; // 60Hzでは300px/s、120Hzでは600px/s になってしまう
  element.style.transform = `translateX(${x}px)`;
  if (x < 300) requestAnimationFrame(badAnimate);
}

// ---- 良い例: 経過時間ベースの移動（デルタタイム） ----
function goodAnimate(timestamp) {
  if (!lastTimestamp) lastTimestamp = timestamp;
  const deltaTime = timestamp - lastTimestamp;
  lastTimestamp = timestamp;

  // 速度: 300px/秒（ディスプレイリフレッシュレートに依存しない）
  const speed = 300; // px per second
  x += speed * (deltaTime / 1000);

  element.style.transform = `translateX(${Math.min(x, 300)}px)`;
  if (x < 300) requestAnimationFrame(goodAnimate);
}

// ---- 最良の例: duration ベースの正規化 ----
function bestAnimate(timestamp) {
  if (!startTime) startTime = timestamp;
  const elapsed = timestamp - startTime;
  const progress = Math.min(elapsed / 1000, 1); // 1秒間で完了
  const eased = Easing.easeOutCubic(progress);

  element.style.transform = `translateX(${eased * 300}px)`;
  if (progress < 1) requestAnimationFrame(bestAnimate);
}
// この方式なら60Hz, 120Hz, 144Hz いずれでも同じ1秒で300px移動する
```

```
リフレッシュレート別のフレームバジェット比較:

  リフレッシュ  フレーム間隔   JS予算    フレーム/秒
  レート                     (目安)
  ───────────────────────────────────────────────
  60Hz         16.67ms       10ms       60
  90Hz         11.11ms       7ms        90
  120Hz         8.33ms       5ms       120
  144Hz         6.94ms       4ms       144
  240Hz         4.17ms       2ms       240

  重要: 高リフレッシュレートでは JS の処理時間の許容値が
  大幅に狭まる。複雑な計算はWorkerに移すことを検討すべき。
```

### 12.2 エッジケース2: タブ非アクティブ時のアニメーション

ブラウザはパフォーマンスとバッテリー消費を最適化するため、非アクティブタブでの `requestAnimationFrame` コールバックの頻度を大幅に制限する（通常1fps以下）。これにより、タブを切り替えて戻った際にアニメーションが急激にジャンプする可能性がある。

```javascript
// タブ切り替え時のアニメーション管理
class VisibilityAwareAnimation {
  constructor(animateFn) {
    this.animateFn = animateFn;
    this.isRunning = false;
    this.lastTimestamp = null;
    this.pausedAt = null;
    this.totalPausedDuration = 0;

    // Page Visibility API で監視
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) {
        this.onHidden();
      } else {
        this.onVisible();
      }
    });
  }

  start() {
    this.isRunning = true;
    this.lastTimestamp = null;
    this.totalPausedDuration = 0;
    this.tick();
  }

  tick() {
    if (!this.isRunning) return;

    requestAnimationFrame((timestamp) => {
      if (this.lastTimestamp === null) {
        this.lastTimestamp = timestamp;
      }

      // 非アクティブ期間を除いた正味の経過時間を計算
      const adjustedTime = timestamp - this.totalPausedDuration;
      const shouldContinue = this.animateFn(adjustedTime);

      if (shouldContinue && this.isRunning) {
        this.lastTimestamp = timestamp;
        this.tick();
      }
    });
  }

  onHidden() {
    // タブが非アクティブになった時刻を記録
    this.pausedAt = performance.now();
  }

  onVisible() {
    // タブがアクティブに戻った時、非アクティブ期間を加算
    if (this.pausedAt !== null) {
      this.totalPausedDuration += performance.now() - this.pausedAt;
      this.pausedAt = null;
    }
  }

  stop() {
    this.isRunning = false;
  }
}

// 使用例
const anim = new VisibilityAwareAnimation((adjustedTime) => {
  const progress = Math.min(adjustedTime / 3000, 1);
  element.style.transform = `translateX(${progress * 300}px)`;
  return progress < 1;
});
anim.start();
```

### 12.3 エッジケース3: transform と子要素への影響

`transform` プロパティは新しいスタッキングコンテキストとコンテインメントブロックを生成するため、子要素の `position: fixed` の基準が変わるなどの副作用がある。

```css
/* 問題: transform を持つ親の中で fixed が期待通り動かない */
.parent {
  transform: translateX(0); /* これだけで fixed の基準が変わる */
}

.parent .fixed-child {
  position: fixed; /* ビューポート基準ではなく、.parent 基準になる */
  top: 0;
  left: 0;
}

/* 対策1: fixed要素をtransform要素の外に移動 */
/* 対策2: ポータルパターンでDOMの別の場所にマウント */
/* 対策3: fixedの代わりにstickyを使用（用途による） */
```

```
transform が生成するコンテキスト:

  transform なし:
  ┌─ viewport ──────────────────────────┐
  │  ┌─ parent ──────────┐              │
  │  │  ┌─ child ──────┐ │              │
  │  │  │ fixed: top 0  │ │   ← viewport基準 │
  │  │  └──────────────┘ │              │
  │  └───────────────────┘              │
  └─────────────────────────────────────┘
  child は viewport の top:0 に表示される

  transform あり:
  ┌─ viewport ──────────────────────────┐
  │  ┌─ parent (transform) ──────────┐  │
  │  │  ┌─ child ──────┐            │  │
  │  │  │ fixed: top 0  │ ← parent基準 │  │
  │  │  └──────────────┘            │  │
  │  └──────────────────────────────┘  │
  └─────────────────────────────────────┘
  child は parent の top:0 に表示される（意図と異なる可能性）
```

---

## 13. 比較表

### 13.1 アニメーション手法の総合比較

| 特性 | CSS Transition | CSS Animation | rAF | WAAPI | FLIP | View Transitions |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| 宣言的記述 | ✓ | ✓ | - | - | - | ✓ (CSS側) |
| GPUアクセラレーション | ✓ | ✓ | △ | ✓ | ✓ | ✓ |
| 動的キーフレーム | - | - | ✓ | ✓ | - | - |
| 一時停止/再開 | - | ✓ (play-state) | 手動 | ✓ | - | - |
| 完了検出 | イベント | イベント | 手動 | Promise | イベント | Promise |
| 逆再生 | △ | ✓ (direction) | 手動 | ✓ | - | - |
| 複雑なシーケンス | △ (delay) | ✓ | ✓ | ✓ | - | △ |
| DOM変更連動 | - | - | ✓ | ✓ | ✓ | ✓ |
| スクロール連動 | - | ✓ (scroll-timeline) | ✓ | ✓ | - | - |
| メインスレッド負荷 | 低 | 低 | 高 | 低 | 中 | 低 |
| 学習コスト | 低 | 低 | 中 | 中 | 高 | 中 |
| ブラウザ対応 | 全ブラウザ | 全ブラウザ | 全ブラウザ | 広い | 全ブラウザ | Chrome/Edge主体 |

### 13.2 イージング関数の推奨用途

| 用途 | 推奨イージング | 推奨時間 | cubic-bezier 近似値 |
|------|:---:|:---:|:---:|
| ボタンホバー | ease-out | 100-150ms | (0, 0, 0.2, 1) |
| ボタン押下 | ease-out | 50-100ms | (0, 0, 0.2, 1) |
| 要素の入場 | ease-out (decelerate) | 200-350ms | (0.22, 1, 0.36, 1) |
| 要素の退場 | ease-in (accelerate) | 150-250ms | (0.4, 0, 1, 1) |
| 移動（入退場） | ease-in-out | 250-400ms | (0.4, 0, 0.2, 1) |
| オーバーシュート | back(ease-out) | 300-500ms | (0.34, 1.56, 0.64, 1) |
| 弾み | bounce | 500-800ms | JS実装推奨 |
| スプリング | spring | 400-700ms | JS実装推奨 |
| スクロール連動 | linear | - | (0, 0, 1, 1) |
| ローディング回転 | linear | 600-1000ms | (0, 0, 1, 1) |
| モーダル表示 | decelerate + overshoot | 250-350ms | (0.34, 1.56, 0.64, 1) |
| モーダル非表示 | accelerate | 150-200ms | (0.4, 0, 1, 1) |

---

## 14. 演習

### 14.1 演習1: 基礎 - CSS Transitionによるカードホバーエフェクト

以下の要件を満たすカードコンポーネントのアニメーションを実装せよ。

**要件:**
- ホバー時にカードが4px上に移動し、影が深くなる
- transform と box-shadow の両方をアニメーションする
- ease-out イージングで200msかけて遷移する
- `prefers-reduced-motion: reduce` の場合はアニメーションを無効にする
- ホバー解除時は少し遅め（250ms）で元に戻る

```css
/* 解答例 */
.card {
  background: white;
  border-radius: 8px;
  padding: 24px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.12);
  transition:
    transform 250ms ease-out,
    box-shadow 250ms ease-out;
}

.card:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
  transition-duration: 200ms;
}

@media (prefers-reduced-motion: reduce) {
  .card {
    transition: none;
  }
  .card:hover {
    transform: none;
    /* 影の変化だけは残す（非モーション的な変更） */
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  }
}
```

**確認ポイント:**
- Chrome DevTools の Rendering タブで "Paint flashing" を有効にし、ホバー時にカード全体ではなく影の部分のみがリペイントされることを確認する
- Performance タブで録画し、Layout イベントが発生していないことを確認する

### 14.2 演習2: 中級 - FLIP によるリスト並べ替えアニメーション

以下の要件を満たすソート可能なリストを実装せよ。

**要件:**
- ボタンクリックでリスト項目を名前順/数値順に切り替える
- FLIP技法を用いて、各項目が現在位置から新しい位置へ滑らかに移動する
- アニメーション時間は300ms、ease-out イージングとする
- 同時に複数のソートが要求された場合、前のアニメーションをキャンセルして新しいソートを開始する

```javascript
// 解答例
class AnimatedSortableList {
  constructor(container) {
    this.container = container;
    this.isAnimating = false;
    this.pendingSort = null;
  }

  async sort(compareFn) {
    if (this.isAnimating) {
      // 前のアニメーション中なら、完了後に新しいソートを適用
      this.pendingSort = compareFn;
      return;
    }

    this.isAnimating = true;
    const items = Array.from(this.container.children);

    // First: 全要素の現在位置を記録
    const firstPositions = new Map();
    items.forEach(item => {
      firstPositions.set(item, item.getBoundingClientRect());
    });

    // Last: ソートを適用
    const sorted = [...items].sort(compareFn);
    sorted.forEach(item => this.container.appendChild(item));

    // Invert + Play
    const animations = sorted.map(item => {
      const first = firstPositions.get(item);
      const last = item.getBoundingClientRect();
      const deltaX = first.left - last.left;
      const deltaY = first.top - last.top;

      if (deltaX === 0 && deltaY === 0) return null;

      return item.animate(
        [
          { transform: `translate(${deltaX}px, ${deltaY}px)` },
          { transform: 'translate(0, 0)' }
        ],
        {
          duration: 300,
          easing: 'cubic-bezier(0.2, 0, 0.2, 1)'
        }
      );
    }).filter(Boolean);

    // 全アニメーション完了を待つ
    await Promise.all(animations.map(a => a.finished));
    this.isAnimating = false;

    // 待機中のソートがあれば実行
    if (this.pendingSort) {
      const nextSort = this.pendingSort;
      this.pendingSort = null;
      this.sort(nextSort);
    }
  }
}

// 使用例
const list = new AnimatedSortableList(document.querySelector('.list'));

document.querySelector('#sort-name').addEventListener('click', () => {
  list.sort((a, b) => a.textContent.localeCompare(b.textContent));
});

document.querySelector('#sort-value').addEventListener('click', () => {
  list.sort((a, b) => {
    return parseInt(a.dataset.value) - parseInt(b.dataset.value);
  });
});
```

### 14.3 演習3: 上級 - Web Animations API + View Transitions による画像ギャラリー

以下の要件を満たす画像ギャラリーのトランジションシステムを実装せよ。

**要件:**
- サムネイルクリックでフルサイズ画像に遷移する
- View Transitions API を使い、サムネイルからフルサイズへの滑らかな遷移を実現する
- View Transitions 非対応ブラウザにはWAAPIによるフォールバックを提供する
- 画像が読み込まれる前はスケルトンスクリーンを表示する
- `prefers-reduced-motion` を尊重する

```javascript
// 解答例
class GalleryTransition {
  constructor() {
    this.prefersReducedMotion = window.matchMedia(
      '(prefers-reduced-motion: reduce)'
    );
  }

  async openFullSize(thumbnail, fullSizeUrl) {
    // スケルトンを表示
    const skeleton = this.createSkeleton(thumbnail);
    document.body.appendChild(skeleton);

    // 画像のプリロード
    const img = new Image();
    const imageLoaded = new Promise((resolve) => {
      img.onload = resolve;
      img.src = fullSizeUrl;
    });

    if (document.startViewTransition && !this.prefersReducedMotion.matches) {
      // View Transitions API が利用可能な場合
      return this.openWithViewTransition(thumbnail, img, imageLoaded, skeleton);
    } else {
      // フォールバック: WAAPI を使用
      return this.openWithWAAPI(thumbnail, img, imageLoaded, skeleton);
    }
  }

  async openWithViewTransition(thumbnail, img, imageLoaded, skeleton) {
    // サムネイルに view-transition-name を設定
    thumbnail.style.viewTransitionName = 'gallery-image';

    await imageLoaded;

    const transition = document.startViewTransition(() => {
      skeleton.remove();
      thumbnail.style.viewTransitionName = '';

      const fullView = this.createFullView(img);
      fullView.style.viewTransitionName = 'gallery-image';
      document.body.appendChild(fullView);
    });

    await transition.finished;
  }

  async openWithWAAPI(thumbnail, img, imageLoaded, skeleton) {
    const thumbnailRect = thumbnail.getBoundingClientRect();

    await imageLoaded;
    skeleton.remove();

    const fullView = this.createFullView(img);
    document.body.appendChild(fullView);
    const fullRect = fullView.getBoundingClientRect();

    // スケール差を計算
    const scaleX = thumbnailRect.width / fullRect.width;
    const scaleY = thumbnailRect.height / fullRect.height;
    const translateX = thumbnailRect.left - fullRect.left +
      (thumbnailRect.width - fullRect.width) / 2;
    const translateY = thumbnailRect.top - fullRect.top +
      (thumbnailRect.height - fullRect.height) / 2;

    if (this.prefersReducedMotion.matches) {
      fullView.style.opacity = '1';
      return;
    }

    await fullView.animate(
      [
        {
          transform: `translate(${translateX}px, ${translateY}px) scale(${scaleX}, ${scaleY})`,
          opacity: 0.8
        },
        {
          transform: 'translate(0, 0) scale(1, 1)',
          opacity: 1
        }
      ],
      {
        duration: 350,
        easing: 'cubic-bezier(0.22, 1, 0.36, 1)',
        fill: 'forwards'
      }
    ).finished;
  }

  createSkeleton(thumbnail) {
    const rect = thumbnail.getBoundingClientRect();
    const skeleton = document.createElement('div');
    skeleton.className = 'skeleton gallery-skeleton';
    skeleton.style.cssText = `
      position: fixed;
      top: ${rect.top}px; left: ${rect.left}px;
      width: ${rect.width}px; height: ${rect.height}px;
    `;
    return skeleton;
  }

  createFullView(img) {
    const container = document.createElement('div');
    container.className = 'gallery-full-view';
    container.appendChild(img);
    return container;
  }
}
```

---

## 15. FAQ

### Q1: CSS Transition と CSS Animation はどう使い分けるべきか？

**回答:** 基本原則は「2状態間の遷移ならTransition、それ以上ならAnimation」である。

- **Transition の適用場面:** ホバーエフェクト、ボタンの状態変化、メニューの開閉、ツールチップの表示/非表示など、明確な開始状態と終了状態がある場合。トリガー（`:hover`, クラスの付与など）に対する自動的な反応として最適。
- **Animation の適用場面:** ローディングスピナー、パルスエフェクト、複数の中間状態を経る入場アニメーション、無限ループのアニメーションなど。`@keyframes` で複数の状態を定義する必要がある場合。

性能面では両者に大きな差はない。いずれも対象プロパティが `transform` / `opacity` であればCompositorスレッドで処理される。コードの可読性と保守性を基準に選択するのが良い。

### Q2: requestAnimationFrame と Web Animations API のどちらを使うべきか？

**回答:** 可能な限りWAAPIを使用することを推奨する。

WAAPIはブラウザのアニメーションエンジンに直接処理を委譲するため、メインスレッドをブロックしない。一方、rAFのコールバック内で行うDOM操作はメインスレッドで実行されるため、他のJavaScript処理と競合する可能性がある。

ただし、以下の場合はrAFが適している:
- 物理シミュレーション（衝突判定やバネ物理など、各フレームで動的に計算が必要な場合）
- Canvas / WebGL 描画
- 外部データ（マウス座標、センサーデータ等）に応じたリアルタイム更新
- 複雑な条件分岐を含むアニメーションロジック

### Q3: アニメーション中にスクロールが発生するとパフォーマンスが低下するのはなぜか？

**回答:** スクロール処理とアニメーション処理は同じメインスレッドで実行されるため、フレームバジェットを奪い合う形になる。特に以下の状況で問題が顕著になる:

1. **スクロールイベントリスナー内でのDOM操作:** `scroll` イベントは高頻度で発火し、そのハンドラ内でレイアウトを変更すると強制リフローが多発する。
2. **パッシブでないスクロールリスナー:** `passive: false` のスクロールリスナーは、ブラウザがスクロールをCompositorに委譲できなくなるため、メインスレッドでの処理を待つことになる。
3. **固定位置要素のリペイント:** `position: fixed` の要素があると、スクロールのたびにリペイントが発生する場合がある。

対策として、スクロール連動アニメーションにはCSS Scroll-driven Animationsの使用を検討すべきである。これはCompositorスレッドで動作するため、メインスレッドへの負荷がない。

### Q4: FLIP技法でscaleを使うと子要素のテキストが歪むのを防ぐにはどうすればよいか？

**回答:** FLIP技法で `scale()` を使用すると、その要素の子要素も同様にスケーリングされ、テキストやアイコンが歪んで見える場合がある。これを防ぐには、子要素に逆スケールを適用する「Counter-Scale」パターンを使用する。

```javascript
// Counter-Scaleパターン
function flipWithCounterScale(parent, changeFn) {
  const first = parent.getBoundingClientRect();
  changeFn();
  const last = parent.getBoundingClientRect();

  const scaleX = first.width / last.width;
  const scaleY = first.height / last.height;

  parent.style.transform = `scale(${scaleX}, ${scaleY})`;

  // 子要素に逆スケールを適用
  const children = parent.querySelectorAll('.preserve-scale');
  children.forEach(child => {
    child.style.transform = `scale(${1/scaleX}, ${1/scaleY})`;
  });

  requestAnimationFrame(() => {
    parent.style.transition = 'transform 300ms ease-out';
    parent.style.transform = '';

    children.forEach(child => {
      child.style.transition = 'transform 300ms ease-out';
      child.style.transform = '';
    });
  });
}
```

### Q5: モバイルデバイスでアニメーションが滑らかにならない場合の対処法は？

**回答:** モバイルデバイスのGPUやCPUはデスクトップに比べて性能が限定的である。以下の対策を段階的に適用することを推奨する。

1. **transform / opacity のみ使用:** Layout や Paint をトリガーするプロパティを避ける。
2. **要素数の削減:** 同時にアニメーションする要素数を最小限にする（理想は10要素以下）。
3. **解像度の考慮:** 高解像度デバイスではピクセル数が多い分、描画負荷が高い。大きなアニメーション領域を避ける。
4. **box-shadow / filter の回避:** これらはPaintが重い処理であり、アニメーション中のリペイントを増やす。
5. **will-change の節度ある利用:** モバイルではGPUメモリが限られるため、必要な要素にだけ適用する。

---

## 16. まとめ

### 60fpsアニメーション達成のチェックリスト

| チェック項目 | 判定基準 |
|------|------|
| transform / opacity のみ使用 | Layout/Paint を発生させていないか |
| rAFまたはWAAPIを使用 | setInterval/setTimeout を使っていないか |
| タイムスタンプベースの計算 | フレーム単位の固定値移動をしていないか |
| will-change の適切な管理 | 必要な時だけ、必要な要素にのみ適用しているか |
| レイアウトスラッシング回避 | 読み取りと書き込みを分離しているか |
| prefers-reduced-motion 対応 | モーション軽減設定を尊重しているか |
| タブ非アクティブ時の対処 | Page Visibility API で制御しているか |
| ロングタスクの排除 | メインスレッドを50ms以上占有していないか |
| DevTools での検証 | Performance パネルでジャンクがないことを確認したか |

### 技術選択のフローチャート

```
アニメーション手法の選択:

  開始
   │
   ├── 2状態間の単純な遷移？
   │    └── Yes → CSS Transition
   │
   ├── キーフレームが必要？
   │    └── Yes → CSS Animation
   │         └── スクロール連動？ → animation-timeline: scroll()/view()
   │
   ├── DOM変更に連動した位置移動？
   │    ├── View Transitions API 対応ブラウザ？
   │    │    └── Yes → View Transitions API
   │    └── No → FLIP 技法
   │
   ├── JS制御が必要？（一時停止、逆再生、動的キーフレーム）
   │    └── Yes → Web Animations API
   │
   ├── 物理シミュレーション / Canvas？
   │    └── Yes → requestAnimationFrame
   │
   └── 上記いずれにも該当しない
        └── CSS Transition（シンプルさ優先）
```

---

## FAQ

### Q1: CSS AnimationsとWeb Animations APIはどう使い分けるべきか？

**A:** 使い分けの基準は「動的制御の必要性」と「複雑さ」である。

```
選定フローチャート:

  アニメーションの要件
   │
   ├─ 静的なホバーエフェクトや入退場アニメーション
   │   └→ CSS Animations/Transitions
   │      理由: 宣言的でシンプル、will-changeによる自動最適化
   │
   ├─ 途中で速度変更・一時停止・逆再生が必要
   │   └→ Web Animations API
   │      理由: .playbackRate、.pause()、.reverse() が使える
   │
   ├─ タイムライン全体の進捗を外部から制御したい
   │   └→ Web Animations API
   │      理由: .currentTime で直接シーク可能
   │
   └─ 物理演算・衝突判定など複雑なロジック
       └→ requestAnimationFrame + 自前の計算
          理由: フレームごとの完全な制御が可能
```

**具体例:**

| ユースケース | 推奨手法 | 理由 |
|---|---|---|
| ボタンホバー時の色変化 | CSS Transition | 最もシンプル、パフォーマンス最適 |
| ローディングスピナー | CSS Animation | ループアニメーション、宣言的 |
| モーダルの開閉 | WAAPI | 開閉状態の動的制御が必要 |
| スクロール連動視差効果 | Scroll-driven Animations | 専用API、最適化済み |
| ゲームキャラクターの動き | rAF + Canvas | 物理演算が必要 |

**併用パターン:**
```javascript
// CSS Animationsを定義しておき、WA APIで制御する
const elem = document.querySelector('.box');
const animation = elem.getAnimations()[0]; // CSSアニメーションを取得

// JavaScriptから動的に制御
animation.playbackRate = 2.0; // 2倍速
animation.currentTime = 500;  // 500ms地点にシーク
```

---

### Q2: 60fpsを達成するための具体的なチェックリストは？

**A:** 以下の7つのステップを順に確認する。

```
┌─────────────────────────────────────────────────────────────┐
│          60fps達成のための7ステップチェックリスト             │
├─────────────────────────────────────────────────────────────┤
│ ✅ Step 1: アニメーション対象プロパティの確認                 │
│    → transform / opacity のみを使用しているか？              │
│    → width/height/top/left をアニメーションしていないか？    │
│                                                              │
│ ✅ Step 2: will-change の適切な設定                          │
│    → アニメーション前に will-change を設定済みか？           │
│    → アニメーション終了後に will-change を削除しているか？   │
│                                                              │
│ ✅ Step 3: レイヤー化の確認                                  │
│    → DevTools > Layers パネルで独立レイヤーになっているか？  │
│    → 不要なレイヤーが大量に作られていないか？                │
│                                                              │
│ ✅ Step 4: JavaScriptの処理時間                              │
│    → rAFコールバック内の処理が10ms以内か？                   │
│    → 強制同期レイアウト(FSL)を引き起こしていないか？          │
│                                                              │
│ ✅ Step 5: ペイント範囲の最小化                              │
│    → DevTools > Rendering > Paint flashing で確認            │
│    → 必要以上に広範囲を再描画していないか？                  │
│                                                              │
│ ✅ Step 6: ガベージコレクションの回避                        │
│    → アニメーション中にオブジェクト生成していないか？        │
│    → 配列のプッシュ/スプライスを繰り返していないか？         │
│                                                              │
│ ✅ Step 7: パフォーマンス計測                                │
│    → DevTools > Performance でフレームドロップを確認          │
│    → FPS Meter で実測値をモニタリング                        │
└─────────────────────────────────────────────────────────────┘
```

**デバッグワークフロー:**

```javascript
// 1. Performance APIで実測
const observer = new PerformanceObserver((list) => {
  for (const entry of list.getEntries()) {
    if (entry.duration > 16.67) {
      console.warn(`⚠️ Long frame: ${entry.duration.toFixed(2)}ms`);
      console.log('Start time:', entry.startTime);
      console.log('Entry type:', entry.entryType);
    }
  }
});
observer.observe({ entryTypes: ['measure', 'longtask'] });

// 2. rAF内で処理時間を計測
function animate() {
  const start = performance.now();

  // アニメーション処理
  updatePositions();

  const duration = performance.now() - start;
  if (duration > 10) {
    console.warn(`⚠️ JS処理が長い: ${duration.toFixed(2)}ms`);
  }

  requestAnimationFrame(animate);
}

// 3. Chrome DevTools のタイムライン分析
// Performance > Record > アニメーション実行 > 停止
// フレームバーが赤色 = フレームドロップ発生
// Summary でどの処理が重いかを特定
```

**よくある落とし穴:**

| 問題 | 症状 | 解決策 |
|---|---|---|
| 強制同期レイアウト | offsetWidth読み取り直後にスタイル変更 | バッチ処理（読み取り→書き込み） |
| メモリリーク | 長時間実行でカクつき悪化 | removeEventListener、WeakMap使用 |
| 過剰なレイヤー化 | メモリ使用量増加 | will-changeを必要最小限に |
| Paint範囲が広い | 全画面再描画 | contain: layout paint を使用 |

---

### Q3: JavaScriptアニメーションライブラリはどう選ぶべきか？

**A:** ユースケースとバンドルサイズのトレードオフで判断する。

```
ライブラリ選定マトリクス:

                     複雑さ
                       ↑
                       │
  GSAP (TweenMax)      │  Mo.js
  ~30KB (gzip)         │  ~20KB
  ┌──────────┐        │  ┌──────────┐
  │フル機能   │        │  │モーション │
  │タイムライン│        │  │グラフィクス│
  └──────────┘        │  └──────────┘
                       │
  ─────────────────────┼─────────────────────→
                       │              バンドルサイズ
  Anime.js             │  Popmotion
  ~9KB                 │  ~5KB (tree-shakable)
  ┌──────────┐        │  ┌──────────┐
  │軽量バランス│        │  │最軽量     │
  │型          │        │  │関数型     │
  └──────────┘        │  └──────────┘
                       │
                       ↓
                    シンプル
```

**選定フローチャート:**

```
  要件の確認
   │
   ├─ SVGモーフィングやパス描画が必要
   │   └→ GSAP (DrawSVG, MorphSVG) or Mo.js
   │
   ├─ 複雑なタイムライン制御・シーケンス
   │   └→ GSAP (Timeline API が最強)
   │
   ├─ 物理演算ベースの自然な動き（慣性・バネ）
   │   └→ Popmotion (spring, inertia)
   │
   ├─ 軽量でモダンなAPI、TypeScript対応
   │   └→ Motion One (~5KB, WAAPI wrapper)
   │
   └─ バンドルサイズを最小化したい
       └→ CSS Animations + WAAPI (ライブラリ不要)
```

**ベンチマーク比較 (2024年基準):**

| ライブラリ | バンドルサイズ | パフォーマンス | 学習コスト | 推奨ケース |
|---|---|---|---|---|
| **GSAP** | ~30KB (gzip) | ★★★★★ | 中 | エンタープライズ、複雑なアニメーション |
| **Anime.js** | ~9KB | ★★★★☆ | 低 | 汎用的な用途、バランス重視 |
| **Popmotion** | ~5KB | ★★★★★ | 中 | 物理演算、インタラクティブUI |
| **Motion One** | ~5KB | ★★★★★ | 低 | 最新プロジェクト、WAAPI活用 |
| **Velocity.js** | ~15KB | ★★★☆☆ | 低 | jQueryからの移行 (非推奨) |
| **Framer Motion** | ~60KB | ★★★★☆ | 中 | React専用、宣言的API |

**実装例の比較:**

```javascript
// 1. GSAP (最も直感的、機能豊富)
gsap.to('.box', {
  x: 100,
  rotation: 360,
  duration: 1,
  ease: 'elastic.out(1, 0.3)',
  onComplete: () => console.log('done')
});

// 2. Anime.js (シンプル、軽量)
anime({
  targets: '.box',
  translateX: 100,
  rotate: 360,
  duration: 1000,
  easing: 'easeOutElastic(1, .3)',
  complete: () => console.log('done')
});

// 3. Popmotion (物理演算特化)
import { animate, spring } from 'popmotion';
animate({
  from: 0,
  to: 100,
  type: spring({ stiffness: 100, damping: 10 }),
  onUpdate: (x) => {
    box.style.transform = `translateX(${x}px)`;
  }
});

// 4. Motion One (WAAPI wrapper、最軽量)
import { animate } from 'motion';
animate('.box',
  { x: 100, rotate: 360 },
  { duration: 1, easing: 'ease-out' }
);

// 5. Web Animations API (ライブラリ不要)
document.querySelector('.box').animate(
  [
    { transform: 'translateX(0) rotate(0deg)' },
    { transform: 'translateX(100px) rotate(360deg)' }
  ],
  { duration: 1000, easing: 'ease-out' }
);
```

**2026年の推奨:**
- **新規プロジェクト**: Motion One または WAAPI直接利用（バンドルサイズ最小）
- **複雑なアニメーション**: GSAP（実績とエコシステム）
- **React**: Framer Motion（宣言的API）
- **Vue**: @vueuse/motion（Composition API対応）

---

## まとめ

### アニメーションパフォーマンス最適化の全体像

| カテゴリ | 重要ポイント | 推奨手法 |
|---|---|---|
| **基本原則** | 60fps = 16.67ms/frame、JS処理は10ms以内に収める | transform/opacityのみアニメーション |
| **CSS手法** | Transitions/Animations/Scroll-driven | 静的アニメーションはCSSで宣言的に |
| **JavaScript手法** | rAF/WAAPI/FLIP技法 | 動的制御が必要な場合のみJSを使用 |
| **レイヤー最適化** | will-change、contain、合成レイヤー | 事前レイヤー化でペイント回避 |
| **計測とデバッグ** | DevTools Performance/Rendering、FPS Meter | フレームドロップの早期発見 |
| **アクセシビリティ** | prefers-reduced-motion、代替UI | 視覚過敏ユーザーへの配慮 |
| **最新API** | View Transitions、Scroll-driven | ネイティブ機能で高パフォーマンス |

### キーポイント

1. **transformとopacityを最優先する**
   - GPU合成のみで完結し、Layout/Paintをスキップ
   - will-change で事前にレイヤー化することで初回フレームも最適化
   - width/height/top/left は避け、scaleX/scaleY/translateで代替

2. **FLIP技法でレイアウト変更を吸収する**
   - First/Last/Invert/Play の4ステップで、高コストなレイアウト変更を低コストなtransformに変換
   - 要素の追加・削除・並び替えなど、DOMの構造変更を伴うアニメーションに有効
   - View Transitions API登場後はそちらを優先（よりシンプルな実装）

3. **計測なしに最適化せず、ボトルネックを特定してから改善する**
   - DevTools Performanceで「どの処理が重いか」を可視化
   - FPS Meterで実測値をモニタリング
   - Long Tasks APIで16.67msを超える処理を検出
   - パフォーマンスは環境依存が大きいため、ターゲットデバイスで必ず計測すること

---

## 次に読むべきガイド

- [V8エンジンの内部動作](../02-javascript-runtime/00-v8-engine.md)
- 合成レイヤーとGPU加速
- レイアウトとリフローの最適化

---

## 参考文献

1. Paul Lewis. "FLIP Your Animations." aerotwist.com, 2015. FLIPテクニックの提唱者による原典。アニメーションのパフォーマンスをtransformベースに変換する手法を解説。
2. Google Developers. "Rendering Performance." web.dev, 2023. レンダリングパイプラインの各段階と、60fpsを達成するための最適化手法を体系的にまとめた公式ガイド。
3. Jake Archibald. "View Transitions API." Chrome for Developers, 2023. View Transitions APIの仕様策定に関わったエンジニアによる詳細な解説。SPA・MPA両方のユースケースをカバー。
4. MDN Web Docs. "Web Animations API." Mozilla, 2024. WAAPIの仕様、メソッド、プロパティの完全なリファレンス。各ブラウザの互換性情報も含む。
5. CSS Working Group. "CSS Scroll-driven Animations." W3C, 2024. scroll()およびview()タイムラインの仕様書。animation-rangeの詳細な定義を含む。
6. Steve Souders. "High Performance Web Sites." O'Reilly Media, 2007. Webパフォーマンス最適化の古典的名著。フロントエンドパフォーマンスの基本原則を確立した。
