# Paint と Compositing

> Paintはレイアウトをピクセルへ変換し、CompositingはGPUでレイヤーを合成する。レイヤー昇格、will-change、GPU加速の仕組みを理解し、スムーズなUIを実現する。

## この章で学ぶこと

- [ ] Paint と Composite の役割の違いを理解する
- [ ] レイヤー昇格の条件とコストを把握する
- [ ] GPU加速を活用したパフォーマンス最適化を学ぶ

---

## 1. Paint（ペイント）

```
Paint = レイアウト情報を描画命令に変換

  描画命令の例:
  1. 背景色を描画（#ffffff）
  2. テキストを描画（"Hello", x:10, y:20, font:16px）
  3. ボーダーを描画（1px solid #ccc）
  4. 画像を描画（src, x:0, y:100, w:300, h:200）

  Paintの段階:
  ① Paint Records の生成（描画命令リスト）
  ② Rasterization（ラスタライズ = ピクセル化）

  タイルベースラスタライズ:
  → 画面を小さなタイル（256×256px）に分割
  → ビューポート内のタイルを優先的にラスタライズ
  → スクロールで必要なタイルを順次ラスタライズ

  ┌──────┬──────┬──────┐
  │ tile │ tile │ tile │ ← ビューポート内 = 優先
  ├──────┼──────┼──────┤
  │ tile │ tile │ tile │ ← ビューポート内 = 優先
  ├──────┼──────┼──────┤
  │ tile │ tile │ tile │ ← ビューポート外 = 後回し
  └──────┴──────┴──────┘

  ラスタースレッド:
  → メインスレッドとは別のスレッドで実行
  → 複数のラスタースレッドが並行処理
  → GPUでのラスタライズも可能（GPU rasterization）
```

---

## 2. Compositing（合成）

```
Compositing = 複数のレイヤーを重ね合わせて最終画像を生成

  なぜレイヤーが必要か:
  → レイヤーごとに独立して再描画可能
  → transform/opacity の変更はレイヤー単位 → 再Paintなし
  → GPU で高速に合成

  Compositing の流れ:
  ┌─────────────────────────────────────┐
  │ メインスレッド                       │
  │ → Paint Records 生成                │
  │ → Compositor Thread にコミット       │
  └──────────────┬──────────────────────┘
                 ↓
  ┌──────────────▼──────────────────────┐
  │ Compositor Thread（合成スレッド）     │
  │ → レイヤーをタイルに分割             │
  │ → ラスタースレッドでピクセル化       │
  │ → Draw Quads（描画コマンド）生成     │
  └──────────────┬──────────────────────┘
                 ↓
  ┌──────────────▼──────────────────────┐
  │ GPU プロセス                        │
  │ → テクスチャの合成                  │
  │ → 画面に表示                        │
  └─────────────────────────────────────┘

  Compositor Thread の利点:
  → メインスレッドが JS で忙しくても合成は継続
  → transform/opacity のアニメーションはメインスレッド不要
  → スクロールもCompositor Thread で処理可能
```

---

## 3. レイヤー昇格

```
レイヤー昇格 = 要素を独立した合成レイヤーに昇格

  自動昇格の条件:
  → transform: translate3d() / translateZ(0)
  → will-change: transform / opacity
  → position: fixed（多くのブラウザ）
  → <video>, <canvas> 要素
  → CSS animation / transition（transform/opacityの場合）
  → 3D transform を持つ要素
  → 昇格したレイヤーの上に重なる要素（暗黙的昇格）

  暗黙的昇格の問題:
  → 要素Aが昇格 → Aの上に重なる要素Bも昇格が必要
  → 「レイヤー爆発」を引き起こす可能性

  .base {
    transform: translateZ(0);  /* 昇格 */
    position: relative;
    z-index: 1;
  }

  .overlap {
    /* baseの上に重なる → 暗黙的に昇格 */
    /* z-index で解決可能 */
  }

will-change のベストプラクティス:
  // JS で動的に設定・解除
  element.addEventListener('mouseenter', () => {
    element.style.willChange = 'transform';
  });

  element.addEventListener('transitionend', () => {
    element.style.willChange = 'auto';  // 解除
  });

  // CSS で常時設定（頻繁にアニメーションする要素のみ）
  .frequently-animated {
    will-change: transform;
  }

  ✗ * { will-change: transform; }  ← 絶対NG
  ✗ 大量の要素に will-change       ← GPUメモリ枯渇
```

---

## 4. DevToolsでの確認

```
Chrome DevTools:

  ① Layers パネル:
     → DevTools > More tools > Layers
     → 全レイヤーの3Dビュー
     → 各レイヤーのメモリ消費
     → 昇格理由の確認

  ② Rendering タブ:
     → DevTools > More tools > Rendering
     → Paint flashing: 再Paintされた領域を緑で表示
     → Layer borders: レイヤー境界をオレンジで表示
     → FPS meter: フレームレートの表示

  ③ Performance パネル:
     → 録画 → Frames セクション
     → 各フレームの処理時間の内訳
     → Layout / Paint / Composite の時間確認

レイヤーのメモリ計算:
  レイヤーサイズ: width × height × 4 bytes（RGBA）
  例: 1920 × 1080 のレイヤー = 約8MB
  → 10個のフルスクリーンレイヤー = 80MB のGPUメモリ
```

---

## 5. 最適化戦略

```
① Composite-only アニメーション:
  transform: translateX(), scale(), rotate()
  opacity: 0 〜 1

② contain プロパティ:
  .widget {
    contain: layout paint;
    /* この要素の変更は外に影響しない */
    /* → ブラウザが最適化可能 */
  }

  contain の値:
  layout:  レイアウト計算を要素内に限定
  paint:   ペイントを要素内に限定
  size:    要素サイズを子に依存させない
  content: layout + paint の短縮
  strict:  layout + paint + size の短縮

③ content-visibility:
  .below-fold {
    content-visibility: auto;
    contain-intrinsic-size: 0 500px;
    /* ビューポート外のレンダリングをスキップ */
  }

  効果:
  → 初期レンダリング時間の大幅削減
  → 長いページで特に効果的
  → ビューポートに近づくと自動でレンダリング
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| Paint | 描画命令の生成 + ラスタライズ（タイルベース） |
| Composite | GPUでレイヤー合成（メインスレッド外） |
| レイヤー昇格 | transform/will-change で昇格、メモリに注意 |
| 最適化 | transform/opacity のみ、contain、content-visibility |

---

## 次に読むべきガイド
→ [[03-animation-performance.md]] — アニメーションパフォーマンス

---

## 参考文献
1. Surma. "The Anatomy of a Frame." aerotwist.com, 2019.
2. web.dev. "Stick to Compositor-Only Properties." Google, 2024.
