# レンダリングパイプライン

> ブラウザがHTML/CSSを受け取ってから画面上のピクセルとして描画するまでの全工程を、DOM構築からComposite（合成）まで段階的に解説する。各段階の役割・コスト・最適化戦略を理解し、60fpsを安定して維持できるフロントエンドエンジニアを目指す。

---

## この章で学ぶこと

- [ ] レンダリングパイプライン全6段階（DOM → CSSOM → Render Tree → Layout → Paint → Composite）の役割と相互関係を説明できる
- [ ] 各段階で発生するボトルネックを DevTools を使って特定できる
- [ ] Layout Thrashing を検出し、修正できる
- [ ] Composite のみで完結するアニメーションを設計できる
- [ ] will-change / contain / content-visibility を適切に使い分けられる
- [ ] 主要ブラウザエンジン（Blink, Gecko, WebKit）の差異を把握している

---

## 前提知識

| 項目 | 推奨レベル |
|------|-----------|
| HTML/CSS 基礎 | セレクタ優先度、ボックスモデルを理解している |
| JavaScript 基礎 | DOM 操作、イベントループの概念を理解している |
| ブラウザ DevTools | Elements パネル、Performance パネルの基本操作ができる |

---

## 1. パイプラインの全体像

### 1.1 6段階の概要

ブラウザがネットワークからHTMLを受信してから画面にピクセルを描画するまでの工程は、大きく6つの段階に分けられる。

```
レンダリングパイプライン全体像:

  ネットワークから HTML/CSS/JS を受信
       │
       ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  ① DOM 構築                                                 │
  │  HTMLバイト列 → 文字列 → トークン → ノード → DOM ツリー       │
  └─────────────────────┬───────────────────────────────────────┘
                        │
  ┌─────────────────────▼───────────────────────────────────────┐
  │  ② CSSOM 構築                                               │
  │  CSSバイト列 → 文字列 → トークン → ノード → CSSOM ツリー      │
  └─────────────────────┬───────────────────────────────────────┘
                        │
                        │  DOM + CSSOM
                        ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  ③ Render Tree 構築                                         │
  │  可視要素のみを対象に、DOMノードとスタイル情報を結合          │
  │  display:none → 除外 / visibility:hidden → 含む              │
  └─────────────────────┬───────────────────────────────────────┘
                        │
                        ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  ④ Layout（Reflow）                                         │
  │  各要素の正確な位置(x,y)とサイズ(width,height)を計算         │
  │  ビューポートからの相対位置、ボックスモデル解決              │
  └─────────────────────┬───────────────────────────────────────┘
                        │
                        ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  ⑤ Paint（Repaint）                                         │
  │  レイアウト情報を元にピクセルレベルの描画命令を生成          │
  │  テキスト描画、色、影、ボーダー、画像の塗りつぶし            │
  └─────────────────────┬───────────────────────────────────────┘
                        │
                        ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  ⑥ Composite（合成）                                        │
  │  複数のペイントレイヤーを GPU 上で重ね合わせて最終画像を生成 │
  │  transform / opacity はこの段階のみで処理可能                │
  └─────────────────────────────────────────────────────────────┘
       │
       ▼
    画面表示（ディスプレイリフレッシュに同期）
```

### 1.2 各段階のコスト比較表

| 段階 | 処理内容 | 典型的コスト | 実行スレッド | 影響範囲 |
|------|---------|-------------|-------------|---------|
| DOM 構築 | HTML パース → ツリー構築 | 中 | メインスレッド | - |
| CSSOM 構築 | CSS パース → ツリー構築 | 低〜中 | メインスレッド | - |
| Render Tree | DOM + CSSOM マージ | 低 | メインスレッド | - |
| Layout | 座標・サイズ計算 | **高** | メインスレッド | 子孫要素に連鎖 |
| Paint | ピクセル描画命令生成 | 中 | メインスレッド | レイヤー単位 |
| Composite | GPU でレイヤー合成 | **低** | コンポジタースレッド/GPU | レイヤー単位 |

### 1.3 CSS プロパティ変更と影響段階の対応表

どの CSS プロパティを変更するかによって、パイプラインのどの段階から再実行が必要かが決まる。

```
CSS プロパティ変更時のパイプライン再実行マップ:

  ┌─────────────────────┬────────┬────────┬───────┬───────────┐
  │ CSS プロパティ       │ Style  │ Layout │ Paint │ Composite │
  ├─────────────────────┼────────┼────────┼───────┼───────────┤
  │ width / height      │   ✓    │   ✓    │   ✓   │     ✓     │
  │ margin / padding    │   ✓    │   ✓    │   ✓   │     ✓     │
  │ top / left / right  │   ✓    │   ✓    │   ✓   │     ✓     │
  │ font-size           │   ✓    │   ✓    │   ✓   │     ✓     │
  │ display             │   ✓    │   ✓    │   ✓   │     ✓     │
  │ float / position    │   ✓    │   ✓    │   ✓   │     ✓     │
  ├─────────────────────┼────────┼────────┼───────┼───────────┤
  │ color               │   ✓    │        │   ✓   │     ✓     │
  │ background-color    │   ✓    │        │   ✓   │     ✓     │
  │ background-image    │   ✓    │        │   ✓   │     ✓     │
  │ box-shadow          │   ✓    │        │   ✓   │     ✓     │
  │ border-radius       │   ✓    │        │   ✓   │     ✓     │
  │ outline             │   ✓    │        │   ✓   │     ✓     │
  │ visibility          │   ✓    │        │   ✓   │     ✓     │
  ├─────────────────────┼────────┼────────┼───────┼───────────┤
  │ transform           │   ✓    │        │       │  ✓ ← 最速 │
  │ opacity             │   ✓    │        │       │  ✓ ← 最速 │
  │ filter (GPU対応)    │   ✓    │        │       │  ✓ ← 最速 │
  └─────────────────────┴────────┴────────┴───────┴───────────┘

  凡例: ✓ = その段階が再実行される
  → transform / opacity / filter は Layout・Paint をスキップ
  → GPU のみで処理されるため最も高速なアニメーション向きプロパティ
```

---

## 2. DOM 構築

### 2.1 HTML パースの流れ

ブラウザのHTMLパーサはネットワークから受信したバイトストリームを段階的に処理し、DOM ツリーを構築する。

```
HTML パース処理の流れ:

  バイト列          文字列            トークン           ノード           DOM ツリー
  (Bytes)          (Characters)     (Tokens)          (Nodes)          (DOM Tree)

  3C 68 74   →    "<html>"    →    StartTag:html  →   HTMLElement  →      html
  6D 6C 3E                                                               /    \
  3C 68 65   →    "<head>"    →    StartTag:head  →   HTMLElement  →  head    body
  61 64 3E                                                              |       |
  ...        →    "<title>"   →    StartTag:title →   HTMLElement  → title    div
             →    "Hello"     →    Character      →   TextNode    →  "Hello"  ...

  重要: パースは逐次的（インクリメンタル）に行われる
  → ネットワークからデータを受信するたびに部分的に DOM を構築
  → 全HTML の受信完了を待たない
```

### 2.2 パーサブロッキング

`<script>` タグに遭遇すると、HTMLパーサは一時停止する。これはスクリプトがDOMを操作する可能性があるためである。

```javascript
// コード例1: script タグの配置によるパース影響

// 悪い例: <head> 内に同期スクリプト
// → DOM構築が完全にブロックされる
`<head>
  <script src="heavy-library.js"></script>  <!-- パーサがここで停止 -->
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <!-- heavy-library.js の読み込み・実行が完了するまで DOM 構築されない -->
  <div id="app">...</div>
</body>`

// 良い例: async/defer を活用
`<head>
  <script src="analytics.js" async></script>   <!-- DOMパースと並行 -->
  <script src="app.js" defer></script>          <!-- DOM構築完了後に実行 -->
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <div id="app">...</div>  <!-- すぐに DOM 構築される -->
</body>`
```

**async と defer の違い:**

| 属性 | ダウンロード | 実行タイミング | 実行順序 | 用途 |
|------|------------|--------------|---------|------|
| (なし) | パース停止 → ダウンロード | ダウンロード直後 | 記述順 | レガシー対応 |
| `async` | パースと並行 | ダウンロード完了直後 | 不定 | 独立スクリプト（Analytics等） |
| `defer` | パースと並行 | DOMContentLoaded 直前 | 記述順 | DOM依存スクリプト |

### 2.3 Speculative Parsing（投機的パース）

モダンブラウザは、メインパーサがスクリプトの実行を待っている間に、先読みスキャナ（Preload Scanner）を使って後続のリソース参照を検出し、事前にダウンロードを開始する。

```javascript
// コード例2: Preload Scanner が検出するリソース

`<head>
  <script src="app.js"></script>        <!-- メインパーサはここで停止 -->
  <!-- ↓ Preload Scanner はここから先を走査し、以下を事前ダウンロード -->
  <link rel="stylesheet" href="main.css">
  <script src="utils.js" defer></script>
  <link rel="preload" href="hero.webp" as="image">
</head>
<body>
  <img src="logo.png" alt="Logo">      <!-- これも事前ダウンロード対象 -->
</body>`

// Preload Scanner を無効化してしまうアンチパターン:
// → JS で動的に <script> や <link> を挿入すると、
//   Preload Scanner は検出できない

// 悪い例: 動的挿入
const script = document.createElement('script');
script.src = 'critical-module.js';      // Preload Scanner に見えない
document.head.appendChild(script);

// 改善: <link rel="preload"> を HTML に記述
`<link rel="preload" href="critical-module.js" as="script">`
```

---

## 3. CSSOM 構築

### 3.1 CSS パースとツリー構築

CSS ファイルもHTMLと同様にバイト列からパースされ、ツリー構造（CSSOM: CSS Object Model）に変換される。

```
CSSOM ツリー構築の概念図:

  CSS ソース:
  ┌────────────────────────────────┐
  │ body { font-size: 16px; }      │
  │ .container { width: 80%; }     │
  │ .container p { color: #333; }  │
  │ .container p .highlight {      │
  │   background: yellow;          │
  │ }                              │
  └────────────────────────────────┘

            ↓ パース & カスケード処理

  CSSOM ツリー:
                    [StyleSheet]
                         │
                     [body]
                  font-size: 16px
                         │
                   [.container]
                    width: 80%
                   (font-size: 16px を継承)
                         │
                      [p]
                   color: #333
                  (font-size: 16px を継承)
                         │
                  [.highlight]
                background: yellow
                (color: #333 を継承)
                (font-size: 16px を継承)

  特徴:
  → CSS はレンダーブロッキングリソース
  → CSSOM が完成しないと Render Tree を構築できない
  → カスケード（優先度解決）、継承、デフォルト値の適用が含まれる
```

### 3.2 CSS がレンダーブロッキングである理由

CSS はレンダーブロッキングリソースとして扱われる。これは、CSSOMが不完全な状態でレンダリングを進めると、スタイルが適用されていない状態（FOUC: Flash of Unstyled Content）が発生するためである。

```javascript
// コード例3: Critical CSS のインライン化によるレンダーブロッキング緩和

// 手順1: ファーストビューに必要なCSS（Critical CSS）をインライン化
`<head>
  <!-- Critical CSS: ファーストビューの描画に必要な最小限のスタイル -->
  <style>
    body { margin: 0; font-family: sans-serif; }
    .hero { height: 100vh; display: flex; align-items: center; }
    .hero h1 { font-size: 3rem; color: #1a1a1a; }
    .nav { position: fixed; top: 0; width: 100%; background: #fff; }
  </style>

  <!-- 残りの CSS は非同期で読み込み -->
  <link rel="preload" href="full-styles.css" as="style"
        onload="this.onload=null;this.rel='stylesheet'">
  <noscript>
    <link rel="stylesheet" href="full-styles.css">
  </noscript>
</head>`

// 手順2: Critical CSS の抽出は自動化ツールで行う
// - critical (npm package)
// - critters (webpack plugin)
// - PurgeCSS + 手動選定
```

### 3.3 セレクタマッチングの方向

ブラウザのセレクタマッチングは **右から左** に評価される。これはパフォーマンス上の理由による。

```
セレクタマッチング方向の理解:

  CSS: .sidebar .menu li a { color: blue; }

  マッチング順序（右から左）:
  1. まず全ての <a> タグを収集
  2. その中から親に <li> を持つものをフィルタ
  3. さらに先祖に .menu を持つものをフィルタ
  4. さらに先祖に .sidebar を持つものをフィルタ

  なぜ右から左なのか:
  → 左から右だと .sidebar から全子孫を走査する必要があり非効率
  → 右から左なら、候補を早期に絞り込める

  セレクタ効率の比較:
  ┌────────────────────────────────┬──────────────┐
  │ セレクタ                       │ 効率          │
  ├────────────────────────────────┼──────────────┤
  │ #main-title                    │ 最速（ID）    │
  │ .btn-primary                   │ 速い（Class） │
  │ button                         │ 普通（Tag）   │
  │ div.wrapper > ul > li > a      │ 遅い（深い）  │
  │ div * a                        │ 非常に遅い    │
  │ [data-active="true"]           │ 遅い（属性）  │
  └────────────────────────────────┴──────────────┘

  ただし:
  → モダンブラウザはセレクタマッチングを高度に最適化している
  → 数千要素レベルでないと体感差は出にくい
  → BEM 記法の .block__element--modifier は効率的
```

---

## 4. Render Tree 構築

### 4.1 DOM + CSSOM の結合

Render Tree は DOM と CSSOM を結合して生成される。画面上に表示される要素のみが含まれる。

```
Render Tree 構築プロセス:

  DOM ツリー:                    CSSOM ツリー:
  html                           body { font: 16px; }
  ├── head                       .visible { color: blue; }
  │   ├── meta                   .hidden { display: none; }
  │   └── title                  .invisible { visibility: hidden; }
  └── body
      ├── div.visible
      │   └── "Hello"
      ├── div.hidden
      │   └── "Secret"
      ├── div.invisible
      │   └── "Ghost"
      └── script

                 ↓ 結合（Attachment）

  Render Tree:
  [RenderView] ─── ビューポート
  └── [RenderBody] ─── font: 16px
      ├── [RenderBlock: div.visible] ─── color: blue
      │   └── [RenderText: "Hello"]
      └── [RenderBlock: div.invisible] ─── visibility: hidden
          └── [RenderText: "Ghost"]

  除外されたもの:
  ✗ <head> 配下（meta, title, script）→ 非表示要素
  ✗ div.hidden → display: none は Render Tree に含まれない
  ✗ <script> → 表示要素ではない

  重要な違い:
  → display: none → Render Tree から完全に除外（レイアウトスペースなし）
  → visibility: hidden → Render Tree に含まれる（レイアウトスペースあり）
  → opacity: 0 → Render Tree に含まれる（レイアウトスペースあり、イベント受付）
```

### 4.2 Render Tree と DOM の不一致

Render Tree は DOM と1対1に対応しない場合がある。

```
DOM と Render Tree の不一致パターン:

  1. display: none
     DOM: <div style="display:none">text</div>
     Render Tree: （存在しない）

  2. ::before / ::after 疑似要素
     DOM: <p class="note">本文</p>
     CSS: .note::before { content: "注: "; }
     Render Tree:
       [RenderBlock: p.note]
       ├── [RenderInline: ::before] → "注: "
       └── [RenderText: "本文"]
     → DOM には存在しないが Render Tree には存在する

  3. Anonymous Box（匿名ボックス）
     DOM: <div>テキスト <span>要素</span> テキスト</div>
     Render Tree:
       [RenderBlock: div]
       ├── [RenderText: "テキスト "]        ← 匿名インラインボックス
       ├── [RenderInline: span]
       │   └── [RenderText: "要素"]
       └── [RenderText: " テキスト"]        ← 匿名インラインボックス

  4. float / position: absolute
     → 通常のフローから外れるが Render Tree には存在する
     → ただし、レイアウト計算では別系統で処理される
```

---

## 5. Layout（Reflow）

### 5.1 レイアウト計算の詳細

Layout 段階では、Render Tree の各ノードに対して正確な幾何学情報（位置とサイズ）を計算する。この処理は「Reflow」とも呼ばれる。

```
Layout 計算で決定される情報:

  各 Render Object に対して:
  ┌──────────────────────────────────────────┐
  │  x 座標     : ビューポート左端からの距離   │
  │  y 座標     : ビューポート上端からの距離   │
  │  width      : コンテンツ幅 + padding + border │
  │  height     : コンテンツ高 + padding + border │
  │  margin     : 外側の余白                   │
  │  scrollWidth: スクロール可能な幅           │
  │  scrollHeight: スクロール可能な高さ         │
  └──────────────────────────────────────────┘

  ボックスモデル:
  ┌────────────────────────────────────────┐
  │              margin-top                │
  │  ┌──────────────────────────────────┐  │
  │  │          border-top              │  │
  │  │  ┌──────────────────────────┐    │  │
  │  │  │      padding-top         │    │  │
  │  │  │  ┌──────────────────┐    │    │  │
  │  │  │  │                  │    │    │  │
  │  │  │  │    content       │    │    │  │
  │  │  │  │  (width x height)│    │    │  │
  │  │  │  │                  │    │    │  │
  │  │  │  └──────────────────┘    │    │  │
  │  │  │      padding-bottom      │    │  │
  │  │  └──────────────────────────┘    │  │
  │  │          border-bottom           │  │
  │  └──────────────────────────────────┘  │
  │              margin-bottom             │
  └────────────────────────────────────────┘

  box-sizing による違い:
  → content-box（デフォルト）: width = コンテンツ幅のみ
  → border-box: width = コンテンツ + padding + border
```

### 5.2 Global Layout と Incremental Layout

Layout には2つのモードがある。

```
Layout のモード:

  1. Global Layout（グローバルレイアウト）
     → ビューポート全体の再計算
     → 発生条件:
        ・初回レンダリング
        ・ウィンドウリサイズ
        ・フォントサイズの変更（html/body レベル）
        ・メディアクエリのブレークポイント通過
     → コスト: 高い（全要素を再計算）

  2. Incremental Layout（インクリメンタルレイアウト）
     → 変更された要素とその影響範囲のみ再計算
     → 発生条件:
        ・特定要素のサイズ/位置変更
        ・DOM ノードの追加/削除
        ・テキスト内容の変更
     → コスト: 変更範囲に依存

  影響の伝播パターン:
  ┌──────────────────────────────────────────────────┐
  │                                                  │
  │   [parent]  ← width 変更                          │
  │   ├── [child-1] ← width が % 指定なら再計算       │
  │   │   └── [grandchild] ← 同様に連鎖              │
  │   ├── [child-2] ← 同様                           │
  │   └── [child-3] ← 同様                           │
  │                                                  │
  │   → 親の変更は子孫に連鎖的に伝播する              │
  │   → 子の変更が親のサイズに影響することもある       │
  │     （auto height の場合など）                     │
  └──────────────────────────────────────────────────┘
```

### 5.3 Layout Thrashing（レイアウトスラッシング）

Layout Thrashing は、JavaScriptがレイアウト情報の読み取りとDOM変更を交互に行うことで、ブラウザが毎回強制的に同期レイアウト（Forced Synchronous Layout）を実行してしまう現象である。

```javascript
// コード例4: Layout Thrashing の検出と修正

// --- アンチパターン1: 読み書き交互（Layout Thrashing） ---
function resizeAllBad(elements) {
  for (const el of elements) {
    // offsetWidth を読む → ブラウザは最新のレイアウトを計算（強制同期レイアウト）
    const currentWidth = el.offsetWidth;
    // width を書く → レイアウトを無効化
    el.style.width = (currentWidth * 1.1) + 'px';
    // 次のループで再び offsetWidth を読む → 再び強制同期レイアウト
    // → N要素あれば N回の Layout が実行される
  }
}

// --- 修正パターン: 読みをまとめてから書く（バッチ処理） ---
function resizeAllGood(elements) {
  // Phase 1: 全要素の幅を一括で読み取る（Layout は1回だけ）
  const widths = [];
  for (const el of elements) {
    widths.push(el.offsetWidth);
  }

  // Phase 2: 全要素の幅を一括で書き込む（Layout は次フレームまで遅延）
  for (let i = 0; i < elements.length; i++) {
    elements[i].style.width = (widths[i] * 1.1) + 'px';
  }
}

// --- 修正パターン（応用）: requestAnimationFrame を使う ---
function resizeAllRAF(elements) {
  const widths = elements.map(el => el.offsetWidth);

  requestAnimationFrame(() => {
    elements.forEach((el, i) => {
      el.style.width = (widths[i] * 1.1) + 'px';
    });
  });
}

// --- 修正パターン（ライブラリ）: fastdom を使う ---
// fastdom はDOM読み書きを自動的にバッチ化する
// npm install fastdom
import fastdom from 'fastdom';

function resizeAllFastdom(elements) {
  elements.forEach(el => {
    fastdom.measure(() => {
      const width = el.offsetWidth;
      fastdom.mutate(() => {
        el.style.width = (width * 1.1) + 'px';
      });
    });
  });
}
```

### 5.4 強制同期レイアウトを引き起こす API

以下のJavaScript API を呼び出すと、ブラウザは最新のレイアウト情報を返すために同期的にレイアウトを再計算する。

| カテゴリ | プロパティ / メソッド |
|---------|---------------------|
| 要素の寸法 | `offsetWidth`, `offsetHeight`, `offsetTop`, `offsetLeft` |
| クライアント領域 | `clientWidth`, `clientHeight`, `clientTop`, `clientLeft` |
| スクロール | `scrollWidth`, `scrollHeight`, `scrollTop`, `scrollLeft` |
| 矩形情報 | `getBoundingClientRect()`, `getClientRects()` |
| ウィンドウ | `window.getComputedStyle()`, `window.scrollX`, `window.scrollY` |
| フォーカス | `element.focus()` （一部ブラウザ） |
| その他 | `window.innerHeight`, `window.innerWidth` |

---

## 6. Paint（ペイント）

### 6.1 Paint の処理内容

Paint 段階では、Layout で計算された幾何学情報を元に、実際のピクセルレベルの描画命令（Paint Records）を生成する。

```
Paint 段階の描画対象:

  描画順序（Stacking Order に従う）:
  ┌─────────────────────────────────────────────┐
  │  1. 要素の background-color                  │
  │  2. 要素の background-image                  │
  │  3. 要素の border                            │
  │  4. 子要素（再帰的に同じ順序で描画）          │
  │  5. 要素の outline                           │
  └─────────────────────────────────────────────┘

  Stacking Context（スタッキングコンテキスト）:
  → z-index を持つ positioned 要素
  → opacity < 1 の要素
  → transform を持つ要素
  → filter を持つ要素
  → will-change を持つ要素
  → isolation: isolate の要素

  Paint の影響範囲:
  → 変更された要素が属するレイヤー全体が再描画
  → レイヤーが分離されていれば、他のレイヤーは再描画不要
```

### 6.2 Repaint が発生する操作

```
Repaint のみが発生するケース（Layout は不要）:

  → color の変更
  → background-color / background-image の変更
  → visibility: visible ↔ hidden の切り替え
  → box-shadow の変更
  → border-color の変更
  → border-radius の変更
  → outline の変更
  → text-decoration の変更

  ポイント:
  → 要素の幾何学的性質（位置・サイズ）が変わらない視覚変更
  → Layout より軽いが、大きな領域の Repaint は依然として高コスト
  → 特に複雑な box-shadow や gradient は Paint コストが高い
```

### 6.3 Paint の最適化: CSS contain プロパティ

`contain` プロパティを使うと、要素のPaint（およびLayout）の影響範囲をブラウザに明示的に伝えることができる。

```css
/* コード例5: contain プロパティによる Paint 最適化 */

/* layout: この要素の内部レイアウト変更は外部に影響しない */
.card {
  contain: layout;
}

/* paint: この要素の内部の Paint は要素の境界外にはみ出さない */
.widget {
  contain: paint;
}

/* size: この要素のサイズは子要素に依存しない（明示的に指定する） */
.fixed-box {
  contain: size;
  width: 300px;
  height: 200px;
}

/* strict: layout + paint + size の全てを含む（最も強力な封じ込め） */
.isolated-component {
  contain: strict;
  width: 400px;
  height: 300px;
}

/* content: layout + paint（size を含まない、より実用的） */
.article-card {
  contain: content;
}

/* content-visibility: 画面外の要素のレンダリングを完全にスキップ */
.long-list-item {
  content-visibility: auto;
  contain-intrinsic-size: 0 200px; /* レイアウト用の推定サイズ */
}
```

```
contain プロパティの効果まとめ:

  ┌──────────────┬─────────────────────────────────────────────┐
  │  値          │  効果                                        │
  ├──────────────┼─────────────────────────────────────────────┤
  │  layout      │  内部レイアウト変更が外部に波及しない         │
  │  paint       │  内部描画が要素境界の外にクリップされる       │
  │  size        │  要素サイズが子要素から独立（要 width/height）│
  │  style       │  CSS カウンタ等のスタイルが外部に漏れない     │
  │  content     │  layout + paint（推奨: 汎用的に使える）       │
  │  strict      │  layout + paint + size（最大の封じ込め）      │
  └──────────────┴─────────────────────────────────────────────┘

  content-visibility: auto の効果:
  → 画面外の要素は Layout / Paint / Composite 全てスキップ
  → スクロールして画面内に入った時点でレンダリング開始
  → 長いリストやフィード型UIで劇的な初期表示速度改善
  → contain-intrinsic-size でスクロールバーの高さを安定化
```

---

## 7. Composite（合成）

### 7.1 レイヤーの概念

Composite 段階では、Paint で生成された描画結果を「レイヤー」として管理し、GPU 上で合成して最終的な画面を生成する。

```
レイヤー合成の概念図:

  画面に表示される最終結果:
  ┌─────────────────────────────────────┐
  │                                     │
  │   ┌───────────────────────┐         │
  │   │  レイヤー3: モーダル   │←── z: 3 │
  │   │  (transform 付き)      │         │
  │   └───────────────────────┘         │
  │                                     │
  │   ┌─────────────────────────────┐   │
  │   │  レイヤー2: ヘッダー        │←── z: 2 (position: fixed)
  │   └─────────────────────────────┘   │
  │                                     │
  │   ┌─────────────────────────────┐   │
  │   │  レイヤー1: メインコンテンツ │←── z: 1 │
  │   │                             │   │
  │   │  テキスト、画像、カード...   │   │
  │   │                             │   │
  │   └─────────────────────────────┘   │
  │                                     │
  │   レイヤー0: 背景                    │←── z: 0 │
  └─────────────────────────────────────┘

  GPU 合成の流れ:
  1. 各レイヤーを個別にラスタライズ（ピクセル化）
  2. レイヤーをテクスチャとして GPU にアップロード
  3. z-order に従ってレイヤーを重ね合わせ
  4. 最終画像をフレームバッファに出力
  5. ディスプレイに表示
```

### 7.2 レイヤー昇格（Layer Promotion）の条件

特定の条件を満たす要素は自動的に独立したコンポジットレイヤーに昇格する。

```
レイヤー昇格が発生する条件:

  明示的な昇格:
  ┌────────────────────────────────────────────────────┐
  │ will-change: transform                              │
  │ will-change: opacity                                │
  │ will-change: filter                                 │
  │ transform: translate3d(...) / translateZ(...)        │
  │ backface-visibility: hidden                         │
  └────────────────────────────────────────────────────┘

  暗黙的な昇格:
  ┌────────────────────────────────────────────────────┐
  │ position: fixed の要素                              │
  │ <video> / <canvas> / <iframe> 要素                  │
  │ CSS animation / transition (transform/opacity)      │
  │ z-index で上位レイヤーと重なる場合（暗黙的昇格）     │
  │ filter プロパティを持つ要素                          │
  │ mix-blend-mode を持つ要素                           │
  │ isolation: isolate の要素                            │
  │ clip-path / mask を持つ要素                          │
  │ backdrop-filter を持つ要素                           │
  └────────────────────────────────────────────────────┘

  暗黙的昇格（Layer Squashing 関連）:
  → あるレイヤーの上に重なる要素は、自動的にレイヤー昇格される
  → これを「暗黙的コンポジット」と呼ぶ
  → ブラウザは Layer Squashing で不要なレイヤーの統合を試みる
```

### 7.3 will-change の正しい使い方

```javascript
// コード例6: will-change のベストプラクティス

// --- 正しい使い方: アニメーション直前に適用、終了後に解除 ---
const card = document.querySelector('.card');

card.addEventListener('mouseenter', () => {
  // ホバー直前にレイヤーを準備
  card.style.willChange = 'transform';
});

card.addEventListener('transitionend', () => {
  // トランジション完了後にレイヤーを解放
  card.style.willChange = 'auto';
});

// --- CSS で常時適用する場合（頻繁にアニメーションする要素のみ） ---
/*
.frequently-animated {
  will-change: transform, opacity;
}
*/

// --- アンチパターン: 全要素に will-change を適用 ---
/*
  決してやってはいけない:
  * {
    will-change: transform;
  }

  理由:
  → 全要素がレイヤー昇格 → GPU メモリの大量消費
  → モバイルデバイスでメモリ不足によるクラッシュの原因
  → ブラウザの最適化を妨害
*/

// --- will-change を CSS から適用する推奨パターン ---
/*
.card {
  transition: transform 0.3s ease;
}
.card:hover {
  will-change: transform;
}
.card:active {
  transform: scale(1.05);
}
*/
```

### 7.4 Compositor Thread の役割

Composite 処理はメインスレッドとは独立した Compositor Thread（合成スレッド）で実行される。

```
スレッドモデルの理解:

  メインスレッド:
  ┌──────────────────────────────────────────────┐
  │  JavaScript → Style → Layout → Paint          │
  │  （重い処理があるとフレーム落ちの原因になる）   │
  └──────────────┬───────────────────────────────┘
                 │ Paint Records + Layer情報
                 ▼
  コンポジタースレッド:
  ┌──────────────────────────────────────────────┐
  │  Composite（GPU 合成）                        │
  │  → メインスレッドの負荷に影響されない          │
  │  → transform / opacity の変更はここだけで処理  │
  │  → スクロール処理もここで処理可能              │
  └──────────────────────────────────────────────┘

  これが意味すること:
  → メインスレッドで重いJSが実行されていても
    transform / opacity アニメーションは滑らかに動く
  → スクロールもコンポジタースレッドで処理されるため
    JS の実行がスクロールをブロックしにくい
    （ただし scroll イベントハンドラがある場合は例外）

  注意: scroll イベントハンドラと passive オプション
  document.addEventListener('scroll', handler, { passive: true });
  → passive: true を指定すると、ハンドラ内で preventDefault() を
    呼ばないことをブラウザに保証 → スクロールがブロックされない
```

---

## 8. 60fps を実現するためのルール

### 8.1 フレームバジェット

```
1フレームの時間配分（60fps の場合）:

  1秒 / 60フレーム = 16.67ms / フレーム

  理想的な時間配分:
  ┌────────────────────────────────────────────────────────┐
  │                    16.67ms                              │
  ├──────────┬────────┬────────┬───────┬──────────┬────────┤
  │ Input    │  JS    │ Style  │Layout │  Paint   │Composite│
  │ handling │(<10ms) │        │       │          │        │
  │  (~1ms)  │        │(~1ms)  │(~2ms) │ (~2ms)   │(~1ms)  │
  ├──────────┴────────┴────────┴───────┴──────────┴────────┤
  │ 合計: ~17ms → ギリギリ                                  │
  │ JS が 10ms を超えると → フレーム落ち（ジャンク）         │
  └────────────────────────────────────────────────────────┘

  120Hz ディスプレイの場合:
  → 1フレーム = 8.33ms → さらにシビアな予算
  → JS は 5ms 以内に抑える必要がある

  フレーム落ちの視覚的影響:
  ┌──────────────────────────────────────────┐
  │ 60fps: ●●●●●●●●●●●● 滑らか              │
  │ 30fps: ●─●─●─●─●─●─ カクカク感じ始める   │
  │ 15fps: ●───●───●───● 明らかにカクカク     │
  │  5fps: ●─────────●── スライドショー状態    │
  └──────────────────────────────────────────┘
```

### 8.2 アニメーション最適化の比較

```javascript
// コード例7: アニメーション手法の比較

// --- 手法1: left/top を使うアニメーション（非推奨） ---
// パイプライン: Style → Layout → Paint → Composite（全段階実行）
function animateWithPosition(element) {
  let pos = 0;
  function frame() {
    pos += 2;
    element.style.left = pos + 'px';  // Layout + Paint + Composite
    if (pos < 300) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

// --- 手法2: transform を使うアニメーション（推奨） ---
// パイプライン: Style → Composite（Layout と Paint をスキップ）
function animateWithTransform(element) {
  let pos = 0;
  function frame() {
    pos += 2;
    element.style.transform = `translateX(${pos}px)`;  // Composite のみ
    if (pos < 300) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

// --- 手法3: CSS Animation（最も推奨） ---
// ブラウザが最適化しやすい
/*
@keyframes slideRight {
  from { transform: translateX(0); }
  to   { transform: translateX(300px); }
}

.animated-element {
  animation: slideRight 0.5s ease-out forwards;
}
*/

// --- 手法4: Web Animations API（プログラマティックに制御） ---
function animateWithWAAPI(element) {
  const animation = element.animate([
    { transform: 'translateX(0)' },
    { transform: 'translateX(300px)' }
  ], {
    duration: 500,
    easing: 'ease-out',
    fill: 'forwards'
  });

  animation.onfinish = () => {
    console.log('Animation completed');
  };

  return animation;  // pause(), cancel(), reverse() が可能
}
```

### 8.3 アニメーション手法の比較表

| 手法 | Pipeline 段階 | 滑らかさ | JS実行中の動作 | 制御性 | 推奨度 |
|------|-------------|---------|--------------|--------|-------|
| `left`/`top` 変更 | Style→Layout→Paint→Comp | 低 | カクつく | 高 | 非推奨 |
| `transform` (JS) | Style→Comp | 高 | 滑らか | 高 | 推奨 |
| CSS Animation | Style→Comp | 最高 | 滑らか | 低 | 最推奨 |
| Web Animations API | Style→Comp | 最高 | 滑らか | 高 | 最推奨 |
| `setTimeout`/`setInterval` | 全段階 | 最低 | 停止する | 高 | 非推奨 |

### 8.4 重い JS 処理の分割

```javascript
// コード例8: 長時間実行タスクの分割手法

// --- アンチパターン2: メインスレッドを長時間ブロック ---
function processLargeArrayBad(items) {
  // 10万件のデータを一度に処理 → メインスレッドが数百ms ブロック
  // → アニメーション停止、入力無反応
  for (const item of items) {
    heavyComputation(item);
  }
}

// --- 修正: チャンク分割 + requestIdleCallback ---
function processLargeArrayGood(items) {
  const CHUNK_SIZE = 100;
  let index = 0;

  function processChunk(deadline) {
    // deadline.timeRemaining() でフレームの残り時間をチェック
    while (index < items.length && deadline.timeRemaining() > 1) {
      const end = Math.min(index + CHUNK_SIZE, items.length);
      for (let i = index; i < end; i++) {
        heavyComputation(items[i]);
      }
      index = end;
    }

    if (index < items.length) {
      requestIdleCallback(processChunk);
    }
  }

  requestIdleCallback(processChunk);
}

// --- 修正: Web Worker にオフロード ---
// main.js
const worker = new Worker('compute-worker.js');

worker.postMessage({ items: largeArray });
worker.onmessage = (event) => {
  const results = event.data;
  updateUI(results);  // 結果をUIに反映
};

// compute-worker.js
self.onmessage = (event) => {
  const { items } = event.data;
  const results = items.map(item => heavyComputation(item));
  self.postMessage(results);
};

// --- 修正: scheduler.yield()（新しいAPI） ---
async function processWithYield(items) {
  for (let i = 0; i < items.length; i++) {
    heavyComputation(items[i]);

    // 定期的にメインスレッドに制御を戻す
    if (i % 100 === 0 && 'scheduler' in globalThis) {
      await scheduler.yield();
    }
  }
}
```

---

## 9. ブラウザエンジン別の差異

### 9.1 主要エンジンの比較

| 特性 | Blink (Chrome/Edge) | Gecko (Firefox) | WebKit (Safari) |
|------|-------------------|-----------------|-----------------|
| Layout エンジン | LayoutNG | Gecko Layout | WebCore Layout |
| Paint 方式 | Skia (GPU加速) | WebRender (GPU) | CoreGraphics |
| Compositor | cc (Chromium Compositor) | WebRender | CA (Core Animation) |
| スレッドモデル | マルチプロセス・マルチスレッド | マルチプロセス | マルチプロセス（制限あり） |
| Layer 管理 | 暗黙的昇格あり | 手動管理寄り | Core Animation 依存 |
| will-change 対応 | 完全対応 | 完全対応 | 部分的（過去にバグあり） |
| content-visibility | 対応 | 対応 | 部分対応 |
| contain プロパティ | 完全対応 | 完全対応 | 完全対応 |

### 9.2 Chrome DevTools でのパイプライン解析

```
Chrome DevTools を使ったレンダリングパイプライン分析手順:

  1. Performance パネル:
     → F12 → Performance タブ → Record
     → 操作を実行 → Stop
     → Main セクションで各フレームのタスクを確認
     → 黄色 = JS / 紫色 = Layout / 緑色 = Paint

  2. Rendering パネル（詳細設定）:
     → F12 → Ctrl+Shift+P → "Show Rendering"
     → Paint flashing: 再描画領域を緑でハイライト
     → Layout Shift Regions: CLS の発生箇所を可視化
     → Layer borders: コンポジットレイヤーの境界を表示
     → FPS meter: リアルタイム FPS 表示

  3. Layers パネル:
     → F12 → Ctrl+Shift+P → "Show Layers"
     → 3D ビューでレイヤーの重なりを確認
     → 各レイヤーのメモリ使用量を確認
     → レイヤー昇格の理由（Compositing Reasons）を確認

  4. Performance Monitor:
     → F12 → Ctrl+Shift+P → "Show Performance Monitor"
     → リアルタイムで以下を監視:
        ・CPU usage
        ・JS heap size
        ・DOM Nodes count
        ・Layouts / sec
        ・Style recalcs / sec
```

---

## 10. 実践的な最適化テクニック

### 10.1 CSS contain と content-visibility の活用

```css
/* コード例9: 仮想リスト風の最適化 */

/* 長いリストの各アイテムに content-visibility を適用 */
.feed-item {
  content-visibility: auto;
  contain-intrinsic-size: 0 120px;  /* 推定高さを指定 */
}

/* カードコンポーネントの封じ込め */
.card {
  contain: content;  /* layout + paint */
  /* → カード内部の変更がカード外に波及しない */
}

/* サイドバーウィジェット */
.sidebar-widget {
  contain: strict;
  width: 300px;
  height: 250px;
  /* → 完全に独立したレンダリングコンテキスト */
}

/* タブの非表示コンテンツ */
.tab-panel[hidden] {
  content-visibility: hidden;
  /* display:none と違い、状態を保持したままレンダリングをスキップ */
  /* → タブ切り替え時の再レンダリングコストが低い */
}
```

### 10.2 DOM 操作のバッチ処理

```javascript
// コード例10: DocumentFragment を使ったバッチ DOM 操作

// --- 悪い例: 1要素ずつ追加 ---
function addItemsBad(container, items) {
  items.forEach(item => {
    const li = document.createElement('li');
    li.textContent = item.name;
    li.className = 'list-item';
    container.appendChild(li);
    // → 毎回 Layout が再計算される可能性
  });
}

// --- 良い例: DocumentFragment でまとめて追加 ---
function addItemsGood(container, items) {
  const fragment = document.createDocumentFragment();

  items.forEach(item => {
    const li = document.createElement('li');
    li.textContent = item.name;
    li.className = 'list-item';
    fragment.appendChild(li);  // オフスクリーンなのでLayoutなし
  });

  container.appendChild(fragment);  // 1回のDOM操作でまとめて追加
}

// --- 良い例: innerHTML を使う（大量要素の場合に最速） ---
function addItemsFastest(container, items) {
  const html = items.map(item =>
    `<li class="list-item">${escapeHtml(item.name)}</li>`
  ).join('');

  container.insertAdjacentHTML('beforeend', html);
}

// HTML エスケープ関数（XSS 防止）
function escapeHtml(str) {
  const div = document.createElement('div');
  div.appendChild(document.createTextNode(str));
  return div.innerHTML;
}
```

---

## FAQ

### Q1. レンダリングパイプラインのボトルネックを特定するにはどうすればいいですか？

**A.** Chrome DevTools の Performance パネルを使います。

```
手順:
1. F12 → Performance タブ → Record（または Ctrl+E）
2. ページ操作を実行（スクロール、アニメーション、インタラクション）
3. Stop → Main セクションでフレームを確認

読み方:
→ 黄色（JavaScript）が長い → JS処理が重い（チャンク分割 / Web Worker 検討）
→ 紫色（Layout）が長い → Reflow が頻発（Layout Thrashing の可能性）
→ 緑色（Paint）が長い → 複雑な描画（box-shadow、gradient、大きな領域）
→ 赤い三角マーク → フレーム落ち（16.67ms 超過）

具体的な診断:
・Layout が 5ms 以上 → contain プロパティで封じ込め
・Paint が連続発生 → will-change / transform アニメーション化
・Script が 50ms 以上 → requestIdleCallback / Web Worker にオフロード
・Recalculate Style が頻発 → CSS セレクタの深さを減らす

補助ツール:
・Rendering パネル → Paint flashing で再描画領域を可視化
・Layers パネル → レイヤー構成とメモリ使用量を確認
・Performance Monitor → リアルタイムで Layouts/sec を監視
```

### Q2. 仮想DOM（React/Vue）とレンダリングパイプラインの関係を教えてください

**A.** 仮想DOMは「DOM操作の最適化レイヤー」であり、ブラウザのレンダリングパイプラインとは別階層です。

```
関係性の整理:

  [React/Vue Component] ← アプリケーション層
       ↓ state変更
  [Virtual DOM diff] ← 仮想DOM層
       ↓ 差分検出
  [最小限のDOM操作] ← DOM API呼び出し
       ↓
  [レンダリングパイプライン] ← ブラウザ層
   Style → Layout → Paint → Composite

仮想DOMが解決する問題:
→ 開発者が無駄な DOM 操作を書いてしまうのを防ぐ
→ 大量の state 変更を1つのバッチ更新にまとめる
→ React 18 の Concurrent Mode では優先度制御も可能

仮想DOMが解決しない問題:
→ Layout Thrashing（読み書き分離は開発者の責任）
→ 重い Paint（CSS プロパティの選択は開発者の責任）
→ 不要なレイヤー昇格（will-change の乱用）

パフォーマンスの鍵:
→ 仮想DOM は DOM 操作の回数を減らすが、各操作のコストは変わらない
→ Layout / Paint のコストが高い場合、仮想DOMだけでは不十分
→ 結論: 仮想DOM + contain + transform/opacity アニメーション が理想
```

### Q3. 60fps を維持するための最も重要な最適化ポイントは何ですか？

**A.** **アニメーションは transform / opacity のみで実装する** ことです。

```
理由:
→ transform / opacity は Composite 段階のみで処理される
→ Layout と Paint をスキップするため、16.67ms のフレーム予算を大幅に節約
→ Compositor Thread で処理されるため、重いJS実行中も滑らか

具体的なルール:
┌──────────────────────────────────────────────────────────┐
│ DO（推奨）                                                │
├──────────────────────────────────────────────────────────┤
│ ✓ transform: translateX/Y/Z, scale, rotate, skew         │
│ ✓ opacity                                                 │
│ ✓ filter（一部GPU対応のもの: blur, brightness など）      │
│ ✓ will-change: transform, opacity（直前に適用）           │
│ ✓ CSS Animation / Transition                             │
│ ✓ Web Animations API                                     │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ DON'T（非推奨）                                           │
├──────────────────────────────────────────────────────────┤
│ ✗ left / top / right / bottom                            │
│ ✗ width / height（アニメーションでの変更）                │
│ ✗ margin / padding（アニメーションでの変更）              │
│ ✗ setTimeout / setInterval                               │
│ ✗ jQuery.animate()（内部で left/top を使用）             │
└──────────────────────────────────────────────────────────┘

例: 要素を右に 300px 移動する
  悪い: element.style.left = '300px'; → Layout + Paint + Composite
  良い: element.style.transform = 'translateX(300px)'; → Composite のみ

追加の最適化:
→ content-visibility: auto で画面外要素をスキップ
→ contain: content で内部変更を封じ込め
→ 重い処理は Web Worker にオフロード
→ requestIdleCallback で低優先度タスクを処理
```

---

## まとめ

### レンダリングパイプライン全体のキーポイント

| 段階 | 役割 | 発生条件 | コスト | 最適化手法 |
|------|------|---------|-------|----------|
| **DOM構築** | HTMLパース | HTML受信時 | 中 | async/defer、Preload Scanner 活用 |
| **CSSOM構築** | CSSパース | CSS受信時 | 低〜中 | Critical CSS インライン化、メディアクエリ活用 |
| **Render Tree** | DOM+CSSOMマージ | 両方完成時 | 低 | display:none で不要要素を除外 |
| **Layout** | 座標・サイズ計算 | 幾何学的変更 | **高** | contain で封じ込め、読み書き分離 |
| **Paint** | ピクセル描画命令 | 視覚的変更 | 中 | レイヤー分離、box-shadow 削減 |
| **Composite** | GPU レイヤー合成 | 常時実行 | **低** | transform/opacity アニメーション |

### 3つの最重要原則

1. **アニメーションは transform / opacity のみで実装する**
   - Layout と Paint をスキップし、Composite のみで処理
   - メインスレッドの負荷に影響されず、常に60fps維持が可能
   - will-change を直前に適用してレイヤー昇格を準備

2. **Layout Thrashing を絶対に発生させない**
   - DOM の読み取り（offsetWidth 等）と書き込み（style変更）を分離
   - fastdom ライブラリで自動的にバッチ処理
   - requestAnimationFrame で書き込みタイミングを制御

3. **contain / content-visibility で影響範囲を限定する**
   - コンポーネント単位で `contain: content` を適用
   - 長いリストは `content-visibility: auto` で画面外をスキップ
   - ブラウザの最適化を助け、数千要素でも滑らかに

---

## パフォーマンス最適化の実践

### 実践1: Critical Rendering Path の最適化

Critical Rendering Path（クリティカルレンダリングパス）とは、ブラウザが最初のピクセルを画面に描画するまでに必要な最短経路のことである。この経路を最適化することで、First Contentful Paint（FCP）や Largest Contentful Paint（LCP）を大幅に改善できる。

**Critical CSS のインライン化:**

ファーストビューに必要な CSS のみを `<style>` タグで HTML 内にインライン化し、残りの CSS は非同期で読み込む。これにより、外部 CSS ファイルのダウンロード完了を待たずにレンダリングを開始できる。

```html
<head>
  <!-- ファーストビューに必要な CSS をインライン化 -->
  <style>
    /* Critical CSS: ヘッダー、ヒーロー、ナビゲーション */
    .header { display: flex; align-items: center; height: 64px; }
    .hero { min-height: 400px; background: #f0f0f0; }
    .nav { display: flex; gap: 16px; }
  </style>

  <!-- 残りの CSS を非同期読み込み -->
  <link rel="preload" href="/styles/main.css" as="style"
        onload="this.onload=null;this.rel='stylesheet'">
  <noscript><link rel="stylesheet" href="/styles/main.css"></noscript>
</head>
```

**リソースヒントの活用:**

```html
<!-- DNS 事前解決 -->
<link rel="dns-prefetch" href="https://api.example.com">

<!-- TCP 接続の事前確立 -->
<link rel="preconnect" href="https://cdn.example.com" crossorigin>

<!-- 重要リソースの事前読み込み -->
<link rel="preload" href="/fonts/main.woff2" as="font" type="font/woff2" crossorigin>
<link rel="preload" href="/images/hero.webp" as="image">

<!-- 次のページの事前取得 -->
<link rel="prefetch" href="/about.html">
```

### 実践2: レイアウトシフトの防止

Cumulative Layout Shift（CLS）はユーザー体験を大きく損なう指標であり、レンダリングパイプラインの Layout 段階と密接に関連する。レイアウトシフトを防止するには、要素のサイズを事前に確保することが最も効果的である。

```html
<!-- 画像: width/height 属性で aspect-ratio を確保 -->
<img src="photo.jpg" width="800" height="600" alt="説明"
     style="max-width: 100%; height: auto;">

<!-- 動的コンテンツ: min-height で領域を確保 -->
<div class="ad-slot" style="min-height: 250px;">
  <!-- 広告がロードされるまでスペースを確保 -->
</div>

<!-- Web フォント: size-adjust で代替フォントとのサイズ差を軽減 -->
<style>
@font-face {
  font-family: 'CustomFont';
  src: url('/fonts/custom.woff2') format('woff2');
  font-display: swap;
  size-adjust: 105%;
  ascent-override: 90%;
  descent-override: 20%;
}
</style>
```

**レイアウトシフトのデバッグ:**

```javascript
// PerformanceObserver でレイアウトシフトを検出
const observer = new PerformanceObserver((list) => {
  for (const entry of list.getEntries()) {
    if (!entry.hadRecentInput) {
      console.log('Layout shift detected:', {
        value: entry.value,
        sources: entry.sources?.map(s => ({
          node: s.node,
          previousRect: s.previousRect,
          currentRect: s.currentRect,
        })),
      });
    }
  }
});
observer.observe({ type: 'layout-shift', buffered: true });
```

### 実践3: 大量要素のレンダリング最適化

数千〜数万の要素を持つリスト（テーブル、フィード、チャットログなど）では、全要素を同時にレンダリングするとLayout・Paint のコストが爆発的に増加する。以下の3つの手法を状況に応じて使い分ける。

**手法1: content-visibility による遅延レンダリング**

```css
.list-item {
  content-visibility: auto;
  contain-intrinsic-size: auto 80px; /* 推定高さを指定 */
}
```

`content-visibility: auto` は画面外の要素のレンダリング（Style/Layout/Paint）を完全にスキップし、要素がビューポートに近づいた時点で初めてレンダリングを実行する。`contain-intrinsic-size` で推定サイズを指定することで、スクロールバーの位置計算が正確になる。10,000件のリストで、初回レンダリングが最大7倍高速化される事例が報告されている。

**手法2: 仮想スクロール（Virtual Scrolling）**

仮想スクロールは、ビューポートに表示される範囲の要素のみをDOMに存在させる手法である。スクロール位置に応じてDOM要素を動的に生成・破棄し、数十万件のデータでもDOMノード数を数十個に抑える。

```javascript
class VirtualList {
  constructor(container, items, itemHeight) {
    this.container = container;
    this.items = items;
    this.itemHeight = itemHeight;
    this.visibleCount = Math.ceil(container.clientHeight / itemHeight) + 2;

    // スクロール領域の全体高さを設定
    this.spacer = document.createElement('div');
    this.spacer.style.height = `${items.length * itemHeight}px`;
    container.appendChild(this.spacer);

    container.addEventListener('scroll', () => this.render(), { passive: true });
    this.render();
  }

  render() {
    const scrollTop = this.container.scrollTop;
    const startIndex = Math.floor(scrollTop / this.itemHeight);
    const endIndex = Math.min(startIndex + this.visibleCount, this.items.length);

    // 既存のアイテムをクリアして再描画
    const fragment = document.createDocumentFragment();
    for (let i = startIndex; i < endIndex; i++) {
      const el = document.createElement('div');
      el.className = 'virtual-item';
      el.style.position = 'absolute';
      el.style.top = `${i * this.itemHeight}px`;
      el.style.height = `${this.itemHeight}px`;
      el.textContent = this.items[i];
      fragment.appendChild(el);
    }

    // バッチ更新
    requestAnimationFrame(() => {
      this.spacer.querySelectorAll('.virtual-item').forEach(el => el.remove());
      this.spacer.appendChild(fragment);
    });
  }
}
```

**手法3: Intersection Observer による遅延初期化**

画面外の要素は軽量なプレースホルダーとして描画し、ビューポートに入った時点で実際のコンテンツを初期化する。

```javascript
const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      const el = entry.target;
      // 重いコンポーネントの初期化
      initializeComponent(el);
      observer.unobserve(el);
    }
  });
}, {
  rootMargin: '200px', // 200px 手前から初期化開始
});

document.querySelectorAll('.lazy-component').forEach(el => {
  observer.observe(el);
});
```

### 実践4: DevTools によるパフォーマンス分析ワークフロー

実際の開発でレンダリングパフォーマンスを分析する際の推奨ワークフローを示す。

1. **計測環境の準備**: シークレットウィンドウで拡張機能の影響を排除し、CPU スロットリング（4x slowdown）を有効にして低スペック端末を模倣する

2. **Performance パネルで記録**: 問題のある操作（スクロール、アニメーション、画面遷移）を実行しながらプロファイルを記録する

3. **フレームチャートの分析**: 16.67ms を超えるフレーム（赤色バー）を特定し、その中の Layout/Paint/Composite の内訳を確認する

4. **ボトルネックの特定**: Layout が支配的であれば Layout Thrashing を疑い、Paint が支配的であれば box-shadow や filter の過剰使用を確認する

5. **改善と再計測**: 修正後に同じ操作でプロファイルを取り、フレーム時間の改善を定量的に確認する

```javascript
// プログラムからのパフォーマンス計測
performance.mark('animation-start');

// アニメーション処理
requestAnimationFrame(() => {
  // DOM更新処理
  updateAnimatedElements();

  performance.mark('animation-end');
  performance.measure('animation-duration', 'animation-start', 'animation-end');

  const measure = performance.getEntriesByName('animation-duration')[0];
  if (measure.duration > 16.67) {
    console.warn(`フレーム予算超過: ${measure.duration.toFixed(2)}ms`);
  }
});
```

---

## 次に読むべきガイド

→ [CSSレイアウトエンジン](./01-css-layout-engine.md) — レイアウト計算の詳細

→ [ペイントとコンポジティング](./02-paint-and-compositing.md) — 描画プロセスの詳細

---

## 参考文献

### 公式ドキュメント・仕様

- [HTML Standard - 8.2 Parsing HTML documents](https://html.spec.whatwg.org/multipage/parsing.html)
  HTML パーサの動作仕様

- [CSS Containment Module Level 2](https://www.w3.org/TR/css-contain-2/)
  contain プロパティと content-visibility の仕様

- [Chromium Design Docs - How Blink Works](https://docs.google.com/document/d/1aitSOucL0VHZa9Z2vbRJSyAIsAz24kX8LFByQ5xQnUg/edit)
  Blink レンダリングエンジンの内部設計

### パフォーマンス最適化ガイド

- [Chrome Developers - Rendering Performance](https://developer.chrome.com/docs/lighthouse/performance/rendering/)
  レンダリングパフォーマンスの総合ガイド

- [web.dev - Optimize Cumulative Layout Shift](https://web.dev/articles/optimize-cls)
  Layout Shift の検出と修正方法

- [web.dev - content-visibility: the new CSS property](https://web.dev/articles/content-visibility)
  content-visibility の実践的な活用法

### DevTools 活用リソース

- [Chrome DevTools - Performance features reference](https://developer.chrome.com/docs/devtools/performance/reference/)
  Performance パネルの詳細な使い方

- [Firefox Developer Tools - Performance](https://firefox-source-docs.mozilla.org/devtools-user/performance/)
  Firefox DevTools のパフォーマンス解析

- [Chromium Blog - Inside look at modern web browser (part 3)](https://developer.chrome.com/blog/inside-browser-part3/)
  レンダリングパイプラインの詳細解説（図解付き）

### その他の重要リソース

- [Paul Irish - What Forces Layout / Reflow](https://gist.github.com/paulirish/5d52fb081b3570c81e3a)
  強制同期レイアウトを引き起こすプロパティの完全リスト

- [CSS Triggers](https://csstriggers.com/)
  各CSSプロパティがパイプラインのどの段階に影響するかの一覧表

- [Compositor Thread Architecture](https://blog.chromium.org/2014/05/a-faster-smoother-web.html)
  Chromium のコンポジタースレッドアーキテクチャ
