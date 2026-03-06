# アニメーションパフォーマンス

> 60fpsのスムーズなアニメーションを実現するための手法を体系的に学ぶ。CSS Transitions/Animations、requestAnimationFrame、Web Animations API、FLIP技法、View Transitions APIを深く理解し、パフォーマンス計測と最適化の全体像を把握する。

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
