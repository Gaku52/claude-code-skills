# レンダリングパイプライン

> ブラウザのレンダリングパイプラインはDOMからピクセルへの変換プロセス。Style → Layout → Paint → Composite の各段階を理解し、パフォーマンスのボトルネックを特定・改善する力を身につける。

## この章で学ぶこと

- [ ] レンダリングパイプラインの各段階を理解する
- [ ] Reflow（Layout）と Repaint の違いを把握する
- [ ] Composite のみで完結するアニメーションを学ぶ

---

## 1. パイプラインの全体像

```
レンダリングパイプライン:

  DOM + CSSOM
       ↓
  ① Style（スタイル計算）
       ↓ Computed Styles
  ② Layout（レイアウト / Reflow）
       ↓ Layout Tree + 座標情報
  ③ Paint（ペイント / Repaint）
       ↓ Paint Records（描画命令リスト）
  ④ Composite（合成）
       ↓ GPU でレイヤーを合成
  画面表示

  各段階のコスト:
  ① Style:     低〜中（セレクタマッチング）
  ② Layout:    高（座標計算、全体に影響しうる）
  ③ Paint:     中（ピクセルの描画）
  ④ Composite: 低（GPUでレイヤー合成）

変更の種類と影響範囲:
  ┌─────────────────┬───────┬────────┬───────┬──────────┐
  │ CSSプロパティ    │ Style │ Layout │ Paint │ Composite│
  ├─────────────────┼───────┼────────┼───────┼──────────┤
  │ width, height   │ ✓     │ ✓      │ ✓     │ ✓        │
  │ margin, padding │ ✓     │ ✓      │ ✓     │ ✓        │
  │ top, left       │ ✓     │ ✓      │ ✓     │ ✓        │
  │ color           │ ✓     │        │ ✓     │ ✓        │
  │ background      │ ✓     │        │ ✓     │ ✓        │
  │ box-shadow      │ ✓     │        │ ✓     │ ✓        │
  │ transform       │ ✓     │        │       │ ✓ ←最速  │
  │ opacity         │ ✓     │        │       │ ✓ ←最速  │
  └─────────────────┴───────┴────────┴───────┴──────────┘

  → transform と opacity は Layout/Paint をスキップ
  → GPU のみで処理 → 最も高速なアニメーション
```

---

## 2. Layout（Reflow）

```
Layout = 要素の位置とサイズを計算

  計算する情報:
  → 各要素のx, y座標
  → 幅と高さ
  → マージン、パディング、ボーダー

  Layout が発生するケース:
  → DOM要素の追加/削除
  → 要素のサイズ変更（width, height, padding, margin）
  → テキスト内容の変更
  → フォントサイズの変更
  → ウィンドウリサイズ
  → スクロール（場合による）

  Layout を強制する JS プロパティ（強制同期レイアウト）:
  element.offsetWidth    → 読み取りのためにレイアウトを強制実行
  element.offsetHeight
  element.clientWidth
  element.getBoundingClientRect()
  window.getComputedStyle()

  Layout Thrashing（レイアウトスラッシング）:
  // 悪い例: 読み書きの交互 → 毎回レイアウト再計算
  for (const el of elements) {
    el.style.width = el.offsetWidth + 10 + 'px';  // 読み→書き→読み→書き...
  }

  // 良い例: 読みをまとめてから書く
  const widths = elements.map(el => el.offsetWidth);
  elements.forEach((el, i) => {
    el.style.width = widths[i] + 10 + 'px';
  });
```

---

## 3. Paint

```
Paint = レイアウト情報をピクセルに変換

  描画する内容:
  → テキストの描画
  → 背景色/画像
  → ボーダー
  → ボックスシャドウ
  → border-radius

  Paint Order（描画順序）:
  1. background-color
  2. background-image
  3. border
  4. children
  5. outline

  Repaint が発生するケース:
  → color の変更
  → background の変更
  → visibility の変更
  → box-shadow の変更
  → border-radius の変更
  → Layoutが変わらない見た目の変更

  Paint は Layout より軽いが、面積が大きいと重くなる
```

---

## 4. Composite（合成）

```
Composite = 複数のレイヤーをGPUで合成

  レイヤー昇格の条件:
  → transform: translate3d() / translateZ()
  → will-change: transform / opacity
  → position: fixed
  → <video>, <canvas>, <iframe>
  → CSS animation / transition（transform/opacity）

  レイヤーのメリット:
  ✓ GPU で独立に合成 → メインスレッドをブロックしない
  ✓ transform/opacity の変更はレイヤー単位
  ✓ 60fps アニメーションの実現

  レイヤーのデメリット:
  ✗ メモリ消費（各レイヤーがGPUメモリを使用）
  ✗ 大量のレイヤー → GPU メモリ不足
  ✗ 不要なレイヤー昇格 → パフォーマンス低下

  will-change の使い方:
  .animated {
    will-change: transform;  /* 事前にレイヤー昇格 */
  }

  注意:
  → 必要な要素にのみ使用
  → * { will-change: transform; } は絶対NG
  → アニメーション完了後に解除するのが理想

DevTools で確認:
  → Chrome DevTools > Rendering > Layer borders
  → 緑の枠 = Composite レイヤー
  → Chrome DevTools > Layers パネル
```

---

## 5. 60fps を実現するためのルール

```
60fps = 1フレーム 16.67ms 以内に処理完了

  1フレームの予算:
  ┌──────────────────────────────────────┐
  │ JS (< 10ms) │ Style │ Layout │ Paint │ Composite │
  └──────────────────────────────────────┘
  ← ────────── 16.67ms ────────────── →

最適化のルール:
  ① アニメーションは transform / opacity のみ
     // 悪い: left を使う（Layout + Paint + Composite）
     .box { left: 100px; }

     // 良い: transform を使う（Composite のみ）
     .box { transform: translateX(100px); }

  ② JS の実行を短くする（< 10ms/フレーム）
     → 重い処理は requestIdleCallback / Web Worker へ

  ③ 強制同期レイアウトを避ける
     → offsetWidth 等の読み取りを最小限に

  ④ Paint 範囲を小さくする
     → contain: layout paint; で影響範囲を制限

  ⑤ レイヤー数を適切に管理
     → 多すぎるレイヤーはGPUメモリを圧迫
```

---

## まとめ

| 段階 | 内容 | コスト |
|------|------|--------|
| Style | CSS計算 | 低〜中 |
| Layout | 位置・サイズ計算 | 高 |
| Paint | ピクセル描画 | 中 |
| Composite | GPU合成 | 低 |

---

## 次に読むべきガイド
→ [[01-css-layout-engine.md]] — CSSレイアウトエンジン

---

## 参考文献
1. Paul Lewis. "Rendering Performance." web.dev, 2015.
2. CSS Triggers. "csstriggers.com." 2024.
