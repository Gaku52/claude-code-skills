# Paint と Compositing

> Paint はレイアウト情報をピクセルへ変換し、Compositing は GPU でレイヤーを合成して最終画像を生成する。レイヤー昇格、will-change、GPU アクセラレーション、合成戦略の仕組みを理解することは、スムーズな UI を実現するうえで不可欠である。本ガイドでは、ブラウザのレンダリングパイプライン後半を構成する Paint フェーズと Compositing フェーズについて、その内部動作からパフォーマンス最適化手法まで体系的に解説する。

## この章で学ぶこと

- [ ] Paint と Composite の役割の違いを説明できる
- [ ] ブラウザのレンダリングパイプライン全体における Paint/Composite の位置づけを理解する
- [ ] レイヤー昇格の条件、メリット、コストを把握する
- [ ] GPU アクセラレーションの仕組みと適用範囲を理解する
- [ ] `will-change` プロパティの正しい使い方を習得する
- [ ] `contain` / `content-visibility` による最適化を実践できる
- [ ] DevTools を使ったペイント・合成のプロファイリングができる
- [ ] アンチパターンを認識して回避できる

---

## 1. レンダリングパイプラインにおける位置づけ

Paint と Compositing は、ブラウザのレンダリングパイプラインの後半に位置する。前半のフェーズ（DOM 構築、CSSOM 構築、スタイル計算、レイアウト）が「何をどこに配置するか」を決定するのに対し、後半の Paint と Compositing は「どのようにピクセル化して画面に表示するか」を担う。

```
レンダリングパイプライン全体像
===========================================================================

 ┌──────────┐   ┌──────────┐   ┌────────────┐   ┌──────────┐
 │   DOM    │──▶│  CSSOM   │──▶│   Style    │──▶│  Layout  │
 │ 構築     │   │ 構築     │   │   計算     │   │ (Reflow) │
 └──────────┘   └──────────┘   └────────────┘   └────┬─────┘
                                                      │
                                                      ▼
 ┌───────────────────────────────────────────────────────────┐
 │                 Paint Phase                                │
 │  ┌─────────────────┐   ┌──────────────────────┐          │
 │  │ Paint Records   │──▶│ Rasterization        │          │
 │  │ (描画命令リスト)│   │ (ピクセル化)         │          │
 │  └─────────────────┘   └──────────┬───────────┘          │
 └───────────────────────────────────┼───────────────────────┘
                                     │
                                     ▼
 ┌───────────────────────────────────────────────────────────┐
 │              Compositing Phase                             │
 │  ┌─────────────────┐   ┌──────────────────────┐          │
 │  │ Layer 合成      │──▶│ GPU テクスチャ       │          │
 │  │ (Draw Quads)    │   │ 画面表示             │          │
 │  └─────────────────┘   └──────────────────────┘          │
 └───────────────────────────────────────────────────────────┘

===========================================================================
```

### 1.1 各フェーズで発生するコスト

CSS プロパティの変更が引き起こすレンダリング処理の範囲は、変更されるプロパティによって大きく異なる。以下に、プロパティ変更とトリガーされるフェーズの関係を示す。

| 変更されるプロパティ | Layout | Paint | Composite | 具体例 |
|:---|:---:|:---:|:---:|:---|
| `width`, `height`, `margin`, `padding` | Yes | Yes | Yes | ボックスサイズの変更 |
| `top`, `left` (positioned) | Yes | Yes | Yes | 位置の変更 |
| `color`, `background-color` | No | Yes | Yes | 色の変更 |
| `box-shadow`, `border-radius` | No | Yes | Yes | 装飾の変更 |
| `transform` | No | No | Yes | 移動・回転・拡大 |
| `opacity` | No | No | Yes | 透明度の変更 |
| `filter` (GPU 対応) | No | No | Yes | ぼかし・色調変更 |

この表から明らかなとおり、`transform` と `opacity` の変更は Layout も Paint もスキップし、Composite のみで完結する。これが「Compositor-only プロパティ」と呼ばれ、高パフォーマンスなアニメーションの基盤となる理由である。

---

## 2. Paint（ペイント）の詳細

### 2.1 Paint Records の生成

Paint フェーズの最初のステップは、レイアウトツリーを走査して Paint Records（描画命令リスト）を生成することである。Paint Records は、各要素の描画に必要な情報を順序付きリストとして保持する。

```
Paint Records の構造（概念図）
===========================================================================

  PaintRecord {
    type: "drawRect"
    rect: { x: 0, y: 0, width: 300, height: 200 }
    color: "#ffffff"
    zOrder: 0
  }

  PaintRecord {
    type: "drawText"
    text: "Hello, World!"
    position: { x: 16, y: 32 }
    font: { family: "Arial", size: "16px", weight: "normal" }
    color: "#333333"
    zOrder: 1
  }

  PaintRecord {
    type: "drawBorder"
    rect: { x: 0, y: 0, width: 300, height: 200 }
    border: { width: "1px", style: "solid", color: "#cccccc" }
    zOrder: 2
  }

  PaintRecord {
    type: "drawImage"
    src: "photo.jpg"
    rect: { x: 0, y: 200, width: 300, height: 200 }
    zOrder: 3
  }

===========================================================================

  描画順序（Stacking Context に従う）:
  1. ルート要素の背景とボーダー
  2. z-index が負の子要素
  3. フロー内ブロックレベル要素
  4. フロート要素
  5. フロー内インライン要素
  6. z-index: 0 の positioned 要素
  7. z-index が正の子要素
```

### 2.2 ラスタライズ（Rasterization）

Paint Records が生成されると、次にラスタライズが行われる。ラスタライズは、ベクターベースの描画命令を実際のピクセルデータ（ビットマップ）に変換するプロセスである。

#### タイルベースラスタライズ

現代のブラウザは、ページ全体を一度にラスタライズするのではなく、画面をタイル（通常 256x256 ピクセル）に分割してラスタライズを行う。

```
タイルベースラスタライズの優先順位
===========================================================================

  ビューポート（画面に見えている領域）
  ┌─────────────────────────────────────────┐
  │                                         │
  │  ┌──────┬──────┬──────┬──────┐         │
  │  │ P:1  │ P:1  │ P:1  │ P:1  │  ← 最優先でラスタライズ
  │  ├──────┼──────┼──────┼──────┤         │
  │  │ P:1  │ P:1  │ P:1  │ P:1  │  ← 最優先でラスタライズ
  │  ├──────┼──────┼──────┼──────┤         │
  │  │ P:1  │ P:1  │ P:1  │ P:1  │  ← 最優先でラスタライズ
  │  └──────┴──────┴──────┴──────┘         │
  │                                         │
  └─────────────────────────────────────────┘
  ┌──────┬──────┬──────┬──────┐
  │ P:2  │ P:2  │ P:2  │ P:2  │  ← ビューポート近傍（次に優先）
  ├──────┼──────┼──────┼──────┤
  │ P:3  │ P:3  │ P:3  │ P:3  │  ← 少し離れた領域
  ├──────┼──────┼──────┼──────┤
  │ P:4  │ P:4  │ P:4  │ P:4  │  ← さらに離れた領域
  ├──────┼──────┼──────┼──────┤
  │ P:5  │ P:5  │ P:5  │ P:5  │  ← 最後にラスタライズ
  └──────┴──────┴──────┴──────┘

  P:N = Priority（優先度）。N が小さいほど高優先
  → ユーザーのスクロール方向を予測して事前ラスタライズも行われる

===========================================================================
```

#### ラスタースレッドの並行処理

ラスタライズはメインスレッドとは独立したラスタースレッドで実行される。Chromium では複数のラスタースレッドが並行してタイルを処理する。

```javascript
// ラスタースレッドの動作を概念的に示すコード（実際のブラウザ内部の擬似コード）
// ※ これは教育目的の疑似実装であり、実際のブラウザ実装とは異なる

class RasterThread {
  constructor(id, gpuContext) {
    this.id = id;
    this.gpuContext = gpuContext;
    this.taskQueue = [];
  }

  processTile(tile) {
    // タイル内の Paint Records を取得
    const records = tile.getPaintRecords();

    // GPU ラスタライズの場合: GPU コンテキストを使用
    if (this.gpuContext) {
      const texture = this.gpuContext.createTexture(
        tile.width,
        tile.height
      );
      for (const record of records) {
        this.gpuContext.drawToTexture(texture, record);
      }
      return texture;
    }

    // ソフトウェアラスタライズの場合: CPU でビットマップ生成
    const bitmap = new Bitmap(tile.width, tile.height);
    for (const record of records) {
      bitmap.draw(record);
    }
    return bitmap;
  }
}

// Chromium の場合、通常 4 つのラスタースレッドが並行動作
// モバイルデバイスでは 2 つに制限されることが多い
```

### 2.3 GPU ラスタライズと Software ラスタライズ

ラスタライズには 2 つの方式がある。

| 項目 | Software Rasterization | GPU Rasterization |
|:---|:---|:---|
| 実行場所 | CPU（ラスタースレッド） | GPU |
| ビットマップ生成 | CPU がピクセルデータを生成 | GPU シェーダーがテクスチャを生成 |
| VRAM 転送 | CPU → GPU へのコピーが必要 | GPU 上で直接生成（転送不要） |
| 適したケース | シンプルなページ、GPU 非対応デバイス | 複雑な描画、高 DPI ディスプレイ |
| Chromium デフォルト | 以前のデフォルト | 現在のデフォルト（Android / Desktop） |
| テキスト描画品質 | 高品質（CPU フォントレンダラー使用） | やや劣る場合あり（改善中） |

Chromium では `chrome://gpu` ページで現在のラスタライズ方式を確認できる。`Rasterization: Hardware accelerated` と表示されていれば GPU ラスタライズが有効である。

### 2.4 Repaint（再ペイント）の条件

以下の操作は Repaint をトリガーする。Repaint は Layout の再計算を伴わないが、ピクセルの再生成が必要になるため、パフォーマンスコストがかかる。

```css
/* Repaint をトリガーするプロパティの例 */
.element {
  /* 色関連 */
  color: red;              /* テキスト色の変更 */
  background-color: blue;  /* 背景色の変更 */
  border-color: green;     /* ボーダー色の変更 */

  /* 視覚効果 */
  box-shadow: 0 2px 8px rgba(0,0,0,0.2); /* 影の変更 */
  text-decoration: underline;             /* テキスト装飾の変更 */
  outline: 2px solid red;                 /* アウトラインの変更 */
  background-image: url("new.jpg");       /* 背景画像の変更 */

  /* visibility の変更（display:none とは異なりレイアウトに影響しない） */
  visibility: hidden;
}
```

---

## 3. Compositing（合成）の詳細

### 3.1 Compositing の基本概念

Compositing は、複数のレイヤー（合成レイヤー）を GPU 上で重ね合わせ、最終的な画面表示を生成するプロセスである。各レイヤーは独立したテクスチャとして GPU メモリ（VRAM）に保持され、合成時に z-order に従って重ね合わされる。

```
合成レイヤーの重ね合わせ（概念図）
===========================================================================

  GPU メモリ上のレイヤー:

  Layer 3 (z-index: 100) ─── ポップアップメニュー
  ┌──────────┐
  │ Menu     │
  │ Item 1   │
  │ Item 2   │
  └──────────┘
                    ↓ 合成
  Layer 2 (z-index: 10) ─── ヘッダー（position: fixed）
  ┌──────────────────────────────────────┐
  │ Header    [Logo]    [Nav]    [User]  │
  └──────────────────────────────────────┘
                    ↓ 合成
  Layer 1 (z-index: 1) ─── コンテンツ（transform アニメーション中）
  ┌──────────────────────────────────────┐
  │ Main Content Area                    │
  │                                      │
  │  ┌─────────┐  ┌─────────┐          │
  │  │ Card 1  │  │ Card 2  │          │
  │  └─────────┘  └─────────┘          │
  └──────────────────────────────────────┘
                    ↓ 合成
  Layer 0 (root) ─── ルートレイヤー
  ┌──────────────────────────────────────┐
  │ body background (#f5f5f5)            │
  │                                      │
  └──────────────────────────────────────┘

  最終画面 = Layer 0 + Layer 1 + Layer 2 + Layer 3 を GPU が合成
  → 各レイヤーのテクスチャを alpha blending で重ね合わせ

===========================================================================
```

### 3.2 Compositor Thread の役割

Compositing は、メインスレッドとは別の Compositor Thread（合成スレッド）で実行される。これは非常に重要な設計上の決定であり、以下の利点をもたらす。

```
スレッド間の役割分担
===========================================================================

  メインスレッド                    Compositor Thread
  ┌──────────────────────┐         ┌──────────────────────┐
  │ ・JavaScript 実行    │         │ ・レイヤーの合成      │
  │ ・DOM 操作           │         │ ・タイル管理          │
  │ ・スタイル計算       │         │ ・スクロール処理      │
  │ ・レイアウト計算     │         │ ・transform アニメ    │
  │ ・Paint Records 生成 │         │ ・opacity アニメ      │
  │ ・イベントハンドラ   │         │ ・Draw Quads 生成     │
  └──────────┬───────────┘         └──────────┬───────────┘
             │                                 │
             │    コミット（同期ポイント）       │
             │ ────────────────────────────▶   │
             │                                 │
             │                                 ▼
             │                     ┌──────────────────────┐
             │                     │ Raster Threads       │
             │                     │ （タイルのピクセル化）│
             │                     └──────────┬───────────┘
             │                                 │
             │                                 ▼
             │                     ┌──────────────────────┐
             │                     │ GPU Process          │
             │                     │ （テクスチャ合成）   │
             │                     │ （画面表示）         │
             │                     └──────────────────────┘

===========================================================================

  メインスレッドがビジー状態でも:
  → スクロールは Compositor Thread で処理される（Non-fast scrollable region 外）
  → transform/opacity アニメーションは Compositor Thread で継続
  → ユーザーは「カクつき」を感じにくい
```

### 3.3 Draw Quads と Display Compositor

Compositor Thread がレイヤーの合成を行う際に生成するのが Draw Quads である。Draw Quads は、GPU に対する最終的な描画命令であり、各タイルのテクスチャをどの位置にどのサイズで描画するかを指定する。

```javascript
// Draw Quad の概念的な構造（教育目的の疑似コード）
const drawQuad = {
  type: "TileDrawQuad",
  // タイルのテクスチャ（GPU メモリ上のビットマップ）
  texture: gpuTextureHandle,
  // 描画先の矩形（画面座標系）
  destRect: { x: 0, y: 0, width: 256, height: 256 },
  // テクスチャ内の参照範囲（UV 座標）
  texCoordRect: { u0: 0.0, v0: 0.0, u1: 1.0, v1: 1.0 },
  // 変換行列（transform の適用）
  transformMatrix: [
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    100, 50, 0, 1  // translateX(100px) translateY(50px)
  ],
  // 透明度
  opacity: 0.9,
  // ブレンドモード
  blendMode: "normal"
};
```

---

## 4. レイヤー昇格の詳細

### 4.1 明示的レイヤー昇格

特定の CSS プロパティを適用すると、要素は独立した合成レイヤーに昇格する。これを明示的レイヤー昇格と呼ぶ。

```css
/* 方法 1: will-change プロパティ（推奨） */
.promoted-element {
  will-change: transform;
}

/* 方法 2: 3D transform（レガシーハック） */
.promoted-element-legacy {
  transform: translateZ(0);
  /* または */
  transform: translate3d(0, 0, 0);
}

/* 方法 3: CSS アニメーション中の要素（自動） */
.animated-element {
  animation: slideIn 0.3s ease-out;
}

@keyframes slideIn {
  from { transform: translateX(-100%); }
  to   { transform: translateX(0); }
}

/* 方法 4: position: fixed（多くのブラウザで自動昇格） */
.fixed-header {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
}

/* 方法 5: 特定の HTML 要素（自動昇格） */
/* <video>, <canvas>, <iframe> は自動的にレイヤー昇格される */
```

### 4.2 暗黙的レイヤー昇格（Implicit Compositing）

暗黙的レイヤー昇格は、意図せずに発生するレイヤー昇格であり、パフォーマンス問題の原因となることが多い。

```
暗黙的レイヤー昇格のメカニズム
===========================================================================

  ケース: 要素 A が昇格済み、要素 B が A の上に重なる場合

  通常の描画順序:
  ┌──────────────────────────┐
  │ Layer 0 (root)           │
  │  ┌────────────────────┐  │
  │  │ Element A (z:1)    │  │ ← 昇格済み（独自レイヤー）
  │  │ transform: ...     │  │
  │  └────────────────────┘  │
  │       ┌──────────────┐   │
  │       │ Element B    │   │ ← A と重なっている
  │       │ (z:2)        │   │
  │       └──────────────┘   │
  └──────────────────────────┘

  問題:
  → A は独自レイヤーで GPU テクスチャとして保持
  → B は A の上に描画される必要がある
  → B がルートレイヤーに残ると、A のテクスチャの上に正しく描画できない
  → よって B も独自レイヤーに昇格が必要（暗黙的昇格）

  結果:
  Layer 0 (root)  ← ルートレイヤー
  Layer 1 (A)     ← 明示的に昇格
  Layer 2 (B)     ← 暗黙的に昇格（メモリ消費が増加）

===========================================================================
```

### 4.3 レイヤー爆発（Layer Explosion）

暗黙的レイヤー昇格が連鎖すると、レイヤー爆発が発生する。これは大量のレイヤーが生成され、GPU メモリが枯渇する深刻な問題である。

```html
<!-- アンチパターン: レイヤー爆発を引き起こすコード -->
<style>
  .base {
    position: relative;
    z-index: 1;
    /* この要素が昇格すると、上に重なる全要素が暗黙的に昇格 */
    will-change: transform;
  }

  .item {
    position: relative;
    z-index: 2; /* base より上 → 暗黙的昇格の対象 */
    width: 200px;
    height: 100px;
    margin: 4px;
  }
</style>

<!-- 1000 個の .item が全て暗黙的に昇格 -->
<!-- 各 200x100x4 = 80KB → 合計約 80MB の GPU メモリ消費 -->
<div class="base">Base Element</div>
<div class="item">Item 1</div>
<div class="item">Item 2</div>
<!-- ... 998 個続く ... -->
<div class="item">Item 1000</div>
```

```css
/* 修正: z-index を適切に設定してレイヤー爆発を防ぐ */
.base {
  position: relative;
  z-index: 2;         /* item より上に設定 */
  will-change: transform;
}

.item {
  position: relative;
  z-index: 1;         /* base より下 → 暗黙的昇格は発生しない */
  width: 200px;
  height: 100px;
  margin: 4px;
}
```

### 4.4 レイヤー昇格のコスト

レイヤー昇格にはメモリコストが伴う。各レイヤーは GPU メモリ上にテクスチャとして保持されるため、レイヤーの数とサイズが増えるほど消費メモリが増加する。

```
レイヤーのメモリ消費計算
===========================================================================

  基本計算式:
  メモリ = width(px) x height(px) x 4(bytes/pixel, RGBA) x devicePixelRatio^2

  例 1: 標準的なカード要素（通常ディスプレイ）
  幅: 300px, 高さ: 200px, DPR: 1
  メモリ = 300 x 200 x 4 x 1 = 240,000 bytes ≈ 234 KB

  例 2: 同じカード要素（Retina ディスプレイ, DPR: 2）
  メモリ = 300 x 200 x 4 x 4 = 960,000 bytes ≈ 937 KB
  → DPR 2 のデバイスでは 4 倍のメモリ

  例 3: フルスクリーンレイヤー（1920x1080, DPR: 1）
  メモリ = 1920 x 1080 x 4 = 8,294,400 bytes ≈ 7.9 MB

  例 4: フルスクリーンレイヤー（1920x1080, DPR: 2）
  メモリ = 3840 x 2160 x 4 = 33,177,600 bytes ≈ 31.6 MB

  モバイルデバイスの GPU メモリ上限（参考値）:
  ローエンド: 128〜256 MB
  ミドルレンジ: 512 MB〜1 GB
  ハイエンド: 2〜4 GB

===========================================================================
```

---

## 5. GPU アクセラレーションの仕組み

### 5.1 GPU が得意な処理

GPU（Graphics Processing Unit）は、大量の並列演算に特化したプロセッサである。以下の処理は GPU で高速に実行できる。

- **テクスチャの合成**: 複数のビットマップを重ね合わせる処理
- **行列変換**: transform（移動、回転、拡大縮小、傾斜）の適用
- **透明度処理**: opacity の変更と alpha blending
- **フィルター処理**: blur、brightness、contrast などの CSS フィルター
- **3D 変換**: perspective、rotateX/Y/Z などの 3D 変換

### 5.2 Compositor-Only プロパティ

以下のプロパティは、メインスレッドに戻ることなく Compositor Thread と GPU のみで処理できる。これを「Compositor-Only プロパティ」と呼ぶ。

```css
/* Compositor-Only プロパティ（高パフォーマンスアニメーション向け） */

/* 1. transform - あらゆる変換 */
.move    { transform: translateX(100px); }
.rotate  { transform: rotate(45deg); }
.scale   { transform: scale(1.5); }
.skew    { transform: skewX(10deg); }
.matrix  { transform: matrix(1, 0, 0, 1, 100, 50); }
.combine { transform: translate(100px, 50px) rotate(45deg) scale(1.2); }

/* 2. opacity - 透明度 */
.fade    { opacity: 0.5; }

/* 3. filter - 一部のフィルター（GPU 対応ブラウザ） */
.blur    { filter: blur(4px); }

/* 4. backdrop-filter - 背景フィルター（GPU 対応ブラウザ） */
.glass   { backdrop-filter: blur(10px); }
```

```javascript
// パフォーマンスの違いを示す比較コード

// --- 悪い例: left/top アニメーション（Layout + Paint + Composite） ---
function animateBad(element) {
  let position = 0;
  function frame() {
    position += 2;
    element.style.left = position + "px"; // Layout をトリガー
    if (position < 300) {
      requestAnimationFrame(frame);
    }
  }
  requestAnimationFrame(frame);
}

// --- 良い例: transform アニメーション（Composite のみ） ---
function animateGood(element) {
  let position = 0;
  function frame() {
    position += 2;
    element.style.transform = `translateX(${position}px)`; // Composite のみ
    if (position < 300) {
      requestAnimationFrame(frame);
    }
  }
  requestAnimationFrame(frame);
}

// --- 最良の例: CSS Animation / Web Animations API ---
function animateBest(element) {
  element.animate(
    [
      { transform: "translateX(0)" },
      { transform: "translateX(300px)" }
    ],
    {
      duration: 500,
      easing: "ease-out",
      fill: "forwards"
    }
  );
  // Web Animations API はブラウザが最適化しやすい
  // Compositor Thread で完全にオフメインスレッド実行される
}
```

### 5.3 GPU アクセラレーションの有効化確認

```javascript
// Chrome DevTools Console で確認する方法

// 1. GPU 情報の確認
// chrome://gpu にアクセスして以下を確認:
// - Canvas: Hardware accelerated
// - Compositing: Hardware accelerated
// - Rasterization: Hardware accelerated
// - Video Decode: Hardware accelerated

// 2. 要素のレイヤー情報を確認（DevTools Layers パネル）
// DevTools > More tools > Layers
// → 各レイヤーの昇格理由、メモリサイズ、描画回数を確認

// 3. Performance パネルでフレーム分析
// DevTools > Performance > Record
// → 各フレームの Composite 時間を確認
// → 16.67ms（60fps）以内に収まっているか確認
```

---

## 6. will-change プロパティの深掘り

### 6.1 will-change の目的と仕組み

`will-change` は、ブラウザに対して「この要素のこのプロパティが近い将来変更される」ことを事前に通知するためのプロパティである。ブラウザはこのヒントを受け取ると、事前に最適化準備（レイヤー昇格、GPU テクスチャの確保など）を行う。

```css
/* will-change の基本構文 */
.element {
  will-change: auto;          /* デフォルト値。ヒントなし */
  will-change: transform;     /* transform の変更を予告 */
  will-change: opacity;       /* opacity の変更を予告 */
  will-change: transform, opacity; /* 複数プロパティの予告 */
  will-change: scroll-position;    /* スクロール位置の変更を予告 */
  will-change: contents;      /* 要素コンテンツの変更を予告 */
}
```

### 6.2 will-change が引き起こす内部動作

`will-change: transform` を設定すると、ブラウザ内部で以下の処理が即座に実行される。

```
will-change 設定時のブラウザ内部動作
===========================================================================

  will-change: transform を設定した瞬間:

  1. 新しい Stacking Context（スタッキングコンテキスト）の生成
     → z-index: auto でも新しいスタッキングコンテキストが作られる
     → 子要素の z-index の基準点が変わる

  2. 新しい Containing Block の生成（fixed 配置の子要素に対して）
     → position: fixed の子要素が、viewport ではなく
       will-change 要素を基準にする場合がある

  3. 合成レイヤーの生成
     → GPU テクスチャが確保される
     → VRAM が消費される

  4. 新しい Offset Parent の生成
     → offsetParent が変わる可能性がある

  注意すべき副作用:
  ┌────────────────────────────────────────────────────┐
  │ will-change: transform を親に設定すると...          │
  │                                                    │
  │ .parent { will-change: transform; }                │
  │                                                    │
  │   .child {                                         │
  │     position: fixed;                               │
  │     top: 0;                                        │
  │     /* viewport 基準ではなく .parent 基準になる！ */│
  │   }                                                │
  └────────────────────────────────────────────────────┘

===========================================================================
```

### 6.3 will-change のベストプラクティス

```javascript
// パターン 1: イベント駆動での動的設定・解除（推奨）
// ホバー時のアニメーション準備
const card = document.querySelector(".card");

card.addEventListener("mouseenter", () => {
  // マウスが乗った瞬間に昇格を準備
  card.style.willChange = "transform, box-shadow";
});

card.addEventListener("mouseleave", () => {
  // マウスが離れた時点ではまだ解除しない
  // （トランジション完了後に解除）
});

card.addEventListener("transitionend", () => {
  // トランジション完了後に解除
  card.style.willChange = "auto";
});
```

```javascript
// パターン 2: スクロール連動アニメーションの場合
const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        // ビューポートに入る直前に昇格を準備
        entry.target.style.willChange = "transform, opacity";
      } else {
        // ビューポートから出た後に解除
        entry.target.style.willChange = "auto";
      }
    });
  },
  {
    // ビューポートの上下 200px 手前から検知
    rootMargin: "200px 0px"
  }
);

document.querySelectorAll(".animate-on-scroll").forEach((el) => {
  observer.observe(el);
});
```

```css
/* パターン 3: 常時アニメーションする要素（CSS のみ） */
/* ローディングスピナーなど、常にアニメーションしている要素に限定 */
.spinner {
  will-change: transform;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

/* パターン 4: ページ内ナビゲーション（親要素でのヒント） */
/* 子要素がホバー時にアニメーションする場合、親のホバーで準備 */
.card-grid:hover .card {
  will-change: transform;
}
```

### 6.4 will-change アンチパターン集

```css
/* ===== アンチパターン 1: 全要素への適用 ===== */
/* GPU メモリを大量消費し、逆にパフォーマンス低下を引き起こす */
* {
  will-change: transform;  /* 絶対にやってはいけない */
}

/* 何が起こるか:
   → 全要素が合成レイヤーに昇格
   → GPU メモリが枯渇（特にモバイルで致命的）
   → レイヤー合成のオーバーヘッドが増大
   → 結果的にパフォーマンスが悪化
*/

/* ===== アンチパターン 2: 不要な常時設定 ===== */
.button {
  will-change: transform, opacity, color, background-color, box-shadow;
  /* 問題: 使わないかもしれないプロパティまで列挙
     → 無駄なリソース消費 */
}

/* 修正 */
.button {
  transition: transform 0.2s, opacity 0.2s;
  /* will-change は JavaScript で動的に設定・解除 */
}

/* ===== アンチパターン 3: CSS 内での安易な使用 ===== */
.modal {
  will-change: transform;
  /* モーダルは開閉時のみアニメーション
     → 常時設定は無駄なリソース消費 */
}

/* 修正: モーダルが開く直前に動的設定 */
.modal.is-opening {
  will-change: transform, opacity;
}
```

---

## 7. CSS contain プロパティによる最適化

### 7.1 contain の概要

CSS `contain` プロパティは、ブラウザに対して要素のレンダリングの独立性をヒントとして伝える。これにより、要素内部の変更が外部に影響しないことをブラウザが保証でき、最適化の機会が増える。

```css
/* contain の値と効果 */

/* layout: レイアウト計算を要素内に封じ込め */
.widget {
  contain: layout;
  /* 効果:
     → この要素内部のレイアウト変更が親や兄弟要素に伝播しない
     → フロートやクリアの影響が外部に漏れない
     → ブラウザは要素外のレイアウト再計算をスキップできる */
}

/* paint: 描画を要素の境界内に封じ込め */
.sidebar {
  contain: paint;
  /* 効果:
     → この要素の子孫は要素境界の外側に描画されない
     → overflow: hidden と似た効果だが、ブラウザ最適化のヒントとして機能
     → 要素がビューポート外の場合、子孫の Paint をスキップ可能 */
}

/* size: 要素のサイズを子コンテンツに依存させない */
.fixed-size-container {
  contain: size;
  width: 300px;
  height: 200px;
  /* 効果:
     → 子要素の変更がこの要素のサイズに影響しない
     → 注意: 明示的なサイズ指定が必須 */
}

/* style: カウンターやquotesのスコープを制限 */
.isolated {
  contain: style;
  /* 効果:
     → CSS カウンターの値が外部に漏れない
     → quotes の状態が外部に影響しない */
}

/* content: layout + paint の短縮形 */
.card {
  contain: content;
}

/* strict: size + layout + paint の短縮形（最も強力） */
.tile {
  contain: strict;
  width: 200px;
  height: 150px;
}
```

### 7.2 contain の使いどころ

```html
<!-- 実用例: カード型レイアウトでの contain 活用 -->
<style>
  .card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 16px;
    padding: 16px;
  }

  .card {
    contain: content;
    /* カード内部の変更が他のカードに影響しないことを保証
       → カード内のDOM変更時に、他のカードの再レイアウトをスキップ */
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    overflow: hidden;
  }

  .card__image {
    width: 100%;
    aspect-ratio: 16 / 9;
    object-fit: cover;
  }

  .card__body {
    padding: 16px;
  }

  .card__title {
    font-size: 1.125rem;
    font-weight: 600;
    margin-bottom: 8px;
  }

  .card__description {
    font-size: 0.875rem;
    color: #666;
    line-height: 1.5;
  }
</style>

<div class="card-grid">
  <article class="card">
    <img class="card__image" src="photo1.jpg" alt="Card 1" />
    <div class="card__body">
      <h3 class="card__title">Card Title 1</h3>
      <p class="card__description">Description text...</p>
    </div>
  </article>
  <!-- 数十〜数百枚のカード -->
</div>
```

### 7.3 contain と will-change の組み合わせ

```css
/* contain と will-change を組み合わせた最適化 */
.animated-widget {
  contain: layout paint;     /* レイアウト・描画の封じ込め */
  /* will-change は JavaScript で動的設定 */
}

/* Intersection Observer と組み合わせた最適化パターン */
.lazy-section {
  contain: content;
  content-visibility: auto;
  contain-intrinsic-size: 0 500px; /* 推定サイズを指定 */
}
```

---

## 8. content-visibility による遅延レンダリング

### 8.1 content-visibility の概要

`content-visibility` は、ビューポート外の要素のレンダリングを遅延させることで、初期ページロードのパフォーマンスを大幅に向上させるプロパティである。

```css
/* content-visibility の3つの値 */

/* visible: デフォルト値。通常どおりレンダリング */
.normal {
  content-visibility: visible;
}

/* hidden: 要素のコンテンツを完全に非表示（display: none に近い）*/
/* ただし、要素自体のボックスは維持される */
.hidden-content {
  content-visibility: hidden;
  /* 用途: タブの非アクティブパネルなど
     → display: none と異なり、再表示時のレンダリングコストが低い
     → レンダリング状態がキャッシュされるため */
}

/* auto: ビューポート外の場合にレンダリングをスキップ */
.lazy-render {
  content-visibility: auto;
  contain-intrinsic-size: auto 300px;
  /* ビューポート外の要素:
     → Style, Layout, Paint をすべてスキップ
     → スクロールで近づくと自動でレンダリング開始
     → contain-intrinsic-size で推定サイズを指定し、スクロールバーの安定性を確保 */
}
```

### 8.2 content-visibility: auto の効果測定

```
content-visibility: auto 適用前後の比較（長いページの場合）
===========================================================================

  ページ構成: 50 セクション、各セクション内に複数要素

  ┌─ ビューポート ──────────────────┐
  │  Section 1  ← 通常レンダリング  │
  │  Section 2  ← 通常レンダリング  │
  │  Section 3  ← 通常レンダリング  │
  └────────────────────────────────┘
     Section 4  ← content-visibility: auto でスキップ
     Section 5  ← content-visibility: auto でスキップ
     ...
     Section 50 ← content-visibility: auto でスキップ

  パフォーマンス効果（参考値）:
  ┌───────────────────┬───────────┬──────────────┐
  │ 指標              │ 適用前    │ 適用後       │
  ├───────────────────┼───────────┼──────────────┤
  │ 初期レンダリング  │ 800ms     │ 120ms        │
  │ DOM ノード処理数  │ 5000      │ 300          │
  │ Layout 計算       │ 200ms     │ 30ms         │
  │ Paint 処理        │ 150ms     │ 25ms         │
  │ メモリ消費        │ 50MB      │ 15MB         │
  └───────────────────┴───────────┴──────────────┘

  ※ 数値はページ構成やデバイスにより大きく異なる

===========================================================================
```

### 8.3 content-visibility の実装例

```html
<!-- 実装例: 長いドキュメントページ -->
<style>
  .doc-section {
    content-visibility: auto;
    contain-intrinsic-size: auto 500px;
    /* auto キーワードにより、一度レンダリングされた後は
       実際のサイズをブラウザが記憶する */
    padding: 2rem;
    border-bottom: 1px solid #e5e7eb;
  }

  /* content-visibility: hidden を活用したタブ切り替え */
  .tab-panel {
    content-visibility: hidden;
    /* レンダリング状態がキャッシュされるため
       タブ切り替え時の再レンダリングが高速 */
  }

  .tab-panel.is-active {
    content-visibility: visible;
  }
</style>

<article>
  <section class="doc-section">
    <h2>Section 1: Introduction</h2>
    <p>This section is visible in the viewport...</p>
  </section>

  <section class="doc-section">
    <h2>Section 2: Getting Started</h2>
    <p>This section may be below the fold...</p>
  </section>

  <!-- 多数のセクションが続く -->
</article>
```

```javascript
// content-visibility と IntersectionObserver の連携
// より細かい制御が必要な場合

class LazyRenderController {
  constructor(selector, options = {}) {
    this.elements = document.querySelectorAll(selector);
    this.rendered = new WeakSet();

    this.observer = new IntersectionObserver(
      (entries) => this.handleIntersection(entries),
      {
        rootMargin: options.rootMargin || "200px 0px",
        threshold: options.threshold || 0
      }
    );

    this.init();
  }

  init() {
    this.elements.forEach((el) => {
      // 初期状態では content-visibility: auto を使用
      el.style.contentVisibility = "auto";
      el.style.containIntrinsicSize = "auto 300px";
      this.observer.observe(el);
    });
  }

  handleIntersection(entries) {
    entries.forEach((entry) => {
      if (entry.isIntersecting && !this.rendered.has(entry.target)) {
        // 一度表示された要素は通常レンダリングに切り替え
        this.rendered.add(entry.target);
        // 動的コンテンツのロードなどをトリガー
        this.loadContent(entry.target);
      }
    });
  }

  loadContent(element) {
    const lazyContent = element.dataset.lazySrc;
    if (lazyContent) {
      // 動的にコンテンツをロード
      fetch(lazyContent)
        .then((res) => res.text())
        .then((html) => {
          element.innerHTML = html;
        });
    }
  }

  destroy() {
    this.observer.disconnect();
  }
}

// 使用例
const controller = new LazyRenderController(".doc-section", {
  rootMargin: "300px 0px"
});
```

---

## 9. DevTools によるペイントとコンポジティングの分析

### 9.1 Chrome DevTools: Layers パネル

Layers パネルは、ページ上の全合成レイヤーを 3D ビューで可視化するツールである。

```
Chrome DevTools Layers パネルの使い方
===========================================================================

  アクセス方法:
  1. DevTools を開く（F12 または Cmd+Option+I）
  2. 右上の「⋮」→「More tools」→「Layers」

  表示される情報:
  ┌─────────────────────────────────────────────────────┐
  │ Layers パネル                                       │
  │                                                     │
  │ ┌─────────────────────┐ ┌───────────────────────┐  │
  │ │                     │ │ Details               │  │
  │ │   3D View           │ │                       │  │
  │ │                     │ │ Size: 1920 x 1080     │  │
  │ │   ┌─────┐          │ │ Memory: 7.9 MB        │  │
  │ │   │ L3  │          │ │                       │  │
  │ │   └─────┘          │ │ Compositing Reasons:  │  │
  │ │  ┌──────────────┐  │ │ - Has a will-change:  │  │
  │ │  │ Layer 2      │  │ │   transform property  │  │
  │ │  └──────────────┘  │ │                       │  │
  │ │ ┌────────────────┐ │ │ Paint Count: 3        │  │
  │ │ │ Layer 1        │ │ │                       │  │
  │ │ └────────────────┘ │ │ Slow scroll regions:  │  │
  │ │ ┌────────────────┐ │ │ none                  │  │
  │ │ │ Root Layer     │ │ │                       │  │
  │ │ └────────────────┘ │ └───────────────────────┘  │
  │ └─────────────────────┘                            │
  └─────────────────────────────────────────────────────┘

  確認すべきポイント:
  ① レイヤー数が妥当か（数十個以下が理想）
  ② 各レイヤーのメモリ消費が適正か
  ③ 不要な暗黙的昇格がないか（Compositing Reasons を確認）
  ④ Paint Count が異常に多くないか

===========================================================================
```

### 9.2 Chrome DevTools: Rendering タブ

```
Rendering タブの各機能
===========================================================================

  アクセス方法:
  DevTools > 「⋮」 > More tools > Rendering

  ┌─────────────────────────────────────────────────────┐
  │ Rendering                                           │
  │                                                     │
  │ [x] Paint flashing                                  │
  │     → 再 Paint された領域を緑色でハイライト         │
  │     → 不要な Repaint を発見するのに有用             │
  │                                                     │
  │ [x] Layout shift regions                            │
  │     → レイアウトシフトが発生した領域を青色で表示    │
  │     → CLS（Cumulative Layout Shift）の原因特定に有用│
  │                                                     │
  │ [x] Layer borders                                   │
  │     → レイヤー境界をオレンジ色の線で表示            │
  │     → タイル境界を水色の線で表示                    │
  │     → レイヤーの分割状況を直感的に確認              │
  │                                                     │
  │ [x] Frame Rendering Stats                           │
  │     → FPS メーター、GPU メモリ使用量を表示          │
  │     → リアルタイムのフレームレート監視              │
  │                                                     │
  │ [ ] Scrolling performance issues                    │
  │     → スクロールパフォーマンスに影響する領域を表示  │
  │     → touch / wheel イベントリスナーの影響を可視化  │
  │                                                     │
  │ [ ] Core Web Vitals                                 │
  │     → LCP, FID, CLS をリアルタイムで表示            │
  └─────────────────────────────────────────────────────┘

===========================================================================
```

### 9.3 Performance パネルでのフレーム分析

```javascript
// Performance パネルの読み方（概念的な説明）

/*
  Performance Recording の構造:

  ┌──────────────────────────────────────────────────────┐
  │ Timeline（時間軸）                                   │
  │ ├── Frames（各フレームのタイミング）                 │
  │ ├── Main（メインスレッドの活動）                     │
  │ │   ├── JavaScript 実行                              │
  │ │   ├── Recalculate Style                           │
  │ │   ├── Layout                                       │
  │ │   ├── Update Layer Tree                           │
  │ │   ├── Paint                                        │
  │ │   └── Composite Layers                            │
  │ ├── Compositor（合成スレッド）                       │
  │ ├── Raster（ラスタースレッド）                       │
  │ └── GPU（GPU プロセス）                              │
  └──────────────────────────────────────────────────────┘

  フレーム分析の手順:
  1. 「Record」ボタンを押して操作を記録
  2. Frames セクションで各フレームの長さを確認
     → 16.67ms を超えるフレームを探す
  3. 長いフレームを選択して Main セクションを確認
     → 何が時間を消費しているか特定
  4. Paint や Composite Layers の時間を確認

  理想的なフレームの構成:
  16.67ms（60fps ターゲット）の内訳:
  ┌────────────────────────────────────────────┐
  │ JS     │ Style  │ Layout │ Paint │ Comp.  │
  │ 4ms    │ 1ms    │ 2ms    │ 1ms   │ 0.5ms  │
  │ ────── │ ────── │ ────── │ ───── │ ────── │
  │ 残り 8.17ms は余裕（idle time）            │
  └────────────────────────────────────────────┘
*/
```

---

## 10. 合成戦略の設計パターン

### 10.1 スクロール連動アニメーションの最適化

```javascript
// スクロール連動パララックスの最適化実装
class OptimizedParallax {
  constructor(container) {
    this.container = container;
    this.layers = container.querySelectorAll("[data-parallax-speed]");
    this.ticking = false;

    this.init();
  }

  init() {
    // 各レイヤーに will-change を事前設定（常にスクロール連動するため）
    this.layers.forEach((layer) => {
      layer.style.willChange = "transform";
    });

    // passive: true でスクロールイベントを登録
    // → Compositor Thread のスクロール処理をブロックしない
    window.addEventListener("scroll", () => this.onScroll(), {
      passive: true
    });
  }

  onScroll() {
    if (!this.ticking) {
      // requestAnimationFrame で次のフレームにバッチ処理
      requestAnimationFrame(() => {
        this.updatePositions();
        this.ticking = false;
      });
      this.ticking = true;
    }
  }

  updatePositions() {
    const scrollY = window.scrollY;

    this.layers.forEach((layer) => {
      const speed = parseFloat(layer.dataset.parallaxSpeed);
      // transform のみを使用（Compositor-Only）
      const offset = scrollY * speed;
      layer.style.transform = `translate3d(0, ${offset}px, 0)`;
    });
  }

  destroy() {
    // クリーンアップ: will-change を解除
    this.layers.forEach((layer) => {
      layer.style.willChange = "auto";
    });
  }
}

// 使用例
// <div class="parallax-container">
//   <div data-parallax-speed="0.5">Slow layer</div>
//   <div data-parallax-speed="0.8">Medium layer</div>
//   <div data-parallax-speed="1.2">Fast layer</div>
// </div>
const parallax = new OptimizedParallax(
  document.querySelector(".parallax-container")
);
```

### 10.2 リスト仮想化とレイヤー戦略

大量のリストアイテムを表示する場合、仮想化（ウィンドウイング）と合成レイヤー戦略を組み合わせることが重要である。

```javascript
// 仮想スクロールリストの合成レイヤー最適化
class VirtualizedList {
  constructor(container, options) {
    this.container = container;
    this.itemHeight = options.itemHeight;
    this.totalItems = options.totalItems;
    this.overscan = options.overscan || 5; // バッファ行数
    this.renderItem = options.renderItem;

    // スクロールコンテナの設定
    this.container.style.overflow = "auto";
    this.container.style.position = "relative";
    // contain でレイアウトの影響範囲を限定
    this.container.style.contain = "strict";

    // 全体の高さを持つスペーサー
    this.spacer = document.createElement("div");
    this.spacer.style.height = `${this.totalItems * this.itemHeight}px`;
    this.spacer.style.position = "relative";
    this.container.appendChild(this.spacer);

    // 表示中のアイテムを保持するコンテナ
    this.viewport = document.createElement("div");
    // transform で位置をオフセット（Compositor-Only）
    this.viewport.style.willChange = "transform";
    this.viewport.style.position = "absolute";
    this.viewport.style.top = "0";
    this.viewport.style.left = "0";
    this.viewport.style.right = "0";
    this.spacer.appendChild(this.viewport);

    this.container.addEventListener("scroll", () => this.onScroll(), {
      passive: true
    });

    this.render();
  }

  onScroll() {
    requestAnimationFrame(() => this.render());
  }

  render() {
    const scrollTop = this.container.scrollTop;
    const viewportHeight = this.container.clientHeight;

    const startIndex = Math.max(
      0,
      Math.floor(scrollTop / this.itemHeight) - this.overscan
    );
    const endIndex = Math.min(
      this.totalItems - 1,
      Math.ceil((scrollTop + viewportHeight) / this.itemHeight)
        + this.overscan
    );

    // transform でオフセット（Layout を回避）
    const offsetY = startIndex * this.itemHeight;
    this.viewport.style.transform = `translateY(${offsetY}px)`;

    // 必要なアイテムのみ DOM に存在させる
    this.viewport.innerHTML = "";
    for (let i = startIndex; i <= endIndex; i++) {
      const item = this.renderItem(i);
      item.style.height = `${this.itemHeight}px`;
      // 各アイテムには contain: content を適用
      item.style.contain = "content";
      this.viewport.appendChild(item);
    }
  }
}

// 使用例
const list = new VirtualizedList(
  document.querySelector("#list-container"),
  {
    itemHeight: 60,
    totalItems: 10000,
    renderItem: (index) => {
      const div = document.createElement("div");
      div.className = "list-item";
      div.textContent = `Item ${index + 1}`;
      return div;
    }
  }
);
```
