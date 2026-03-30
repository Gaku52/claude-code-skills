# CSSレイアウトエンジン

> CSSレイアウトエンジンは、HTML要素の位置・寸法・配置順序を決定するブラウザ内部の中核コンポーネントである。Box Model、Normal Flow、Flexbox、Grid、Positioning、およびレイアウト計算アルゴリズムの内部動作を深く理解することで、意図通りのレイアウトを効率的かつパフォーマンスに優れた形で構築できるようになる。本ガイドでは、W3C 仕様に基づいた厳密な解説から、現場で即座に活用できるパターン集、さらにエッジケースやアンチパターンまでを網羅的にカバーする。

---

## 目次

1. [Box Model の詳細](#1-box-model-の詳細)
2. [Normal Flow とフォーマッティングコンテキスト](#2-normal-flow-とフォーマッティングコンテキスト)
3. [Flexbox の内部アルゴリズム](#3-flexbox-の内部アルゴリズム)
4. [CSS Grid の内部アルゴリズム](#4-css-grid-の内部アルゴリズム)
5. [Positioning と Stacking Context](#5-positioning-と-stacking-context)
6. [レイアウト計算アルゴリズムの全体像](#6-レイアウト計算アルゴリズムの全体像)
7. [パフォーマンスとレイアウトスラッシング](#7-パフォーマンスとレイアウトスラッシング)
8. [Flexbox vs Grid 使い分け徹底比較](#8-flexbox-vs-grid-使い分け徹底比較)
9. [実践コード例集](#9-実践コード例集)
10. [アンチパターン集](#10-アンチパターン集)
11. [エッジケース分析](#11-エッジケース分析)
12. [演習問題（3段階）](#12-演習問題3段階)
13. [FAQ](#13-faq)
14. [用語集](#14-用語集)
15. [参考文献](#15-参考文献)

---

## この章で学ぶこと

- [ ] Box Model の 2 つのモード（content-box / border-box）の計算差異を正確に理解する
- [ ] Normal Flow におけるブロックレベルとインラインレベルの配置規則を把握する
- [ ] BFC（Block Formatting Context）の生成条件と効果を理解する
- [ ] Flexbox の 6 段階レイアウトアルゴリズムを内部まで追跡できる
- [ ] CSS Grid のトラックサイジングアルゴリズムを理解する
- [ ] レイアウト計算がブラウザのレンダリングパイプラインのどこに位置するかを把握する
- [ ] パフォーマンスを考慮したレイアウト設計ができるようになる

---

## 前提知識

- レンダリングパイプラインの全体像 → 参照: [レンダリングパイプライン](./00-rendering-pipeline.md)
- CSSのボックスモデルとレイアウトモード（Flexbox, Grid）
- DOM/CSSOMツリーの構築 → 参照: [HTML/CSSパース](../00-browser-engine/02-parsing-html-css.md)

---

## 1. Box Model の詳細

### 1.1 CSS Box Model の構造

すべての HTML 要素は「ボックス」として描画される。各ボックスは 4 つの領域で構成される。

```
CSS Box Model の完全構造:

  ┌─────────────────────────────────────────────────────┐
  │                    margin-top                       │
  │  ┌──────────────────────────────────────────────┐   │
  │  │               border-top                     │   │
  │  │  ┌───────────────────────────────────────┐   │   │
  │  │  │            padding-top                │   │   │
  │  │  │  ┌─────────────────────────────────┐  │   │   │
  │  │  │  │                                 │  │   │   │
  │m │b │p │        content area             │p │b  │ m │
  │a │o │a │     (width x height)            │a │o  │ a │
  │r │r │d │                                 │d │r  │ r │
  │g │d │d │                                 │d │d  │ g │
  │i │e │i │                                 │i │e  │ i │
  │n │r │n │                                 │n │r  │ n │
  │  │  │g │                                 │g │   │   │
  │l │l │  │                                 │  │r  │ r │
  │e │e │l │                                 │r │i  │ i │
  │f │f │e │                                 │i │g  │ g │
  │t │t │f │                                 │g │h  │ h │
  │  │  │t │                                 │h │t  │ t │
  │  │  │  │                                 │t │   │   │
  │  │  │  └─────────────────────────────────┘  │   │   │
  │  │  │           padding-bottom              │   │   │
  │  │  └───────────────────────────────────────┘   │   │
  │  │              border-bottom                   │   │
  │  └──────────────────────────────────────────────┘   │
  │                   margin-bottom                     │
  └─────────────────────────────────────────────────────┘
```

各領域の役割:

| 領域 | 説明 | 負の値 | 背景適用 |
|------|------|--------|----------|
| content | テキストや子要素を配置する領域 | N/A | あり |
| padding | コンテンツとボーダー間の余白 | 不可 | あり |
| border | ボックスの境界線 | N/A | ボーダー自身 |
| margin | 他の要素との外側余白 | 可能 | なし（透明） |

### 1.2 box-sizing プロパティの詳細

`box-sizing` は、`width` と `height` が何を指すかを制御するプロパティである。

```css
/* content-box（デフォルト） */
.element-content-box {
  box-sizing: content-box;
  width: 300px;
  padding: 20px;
  border: 5px solid #333;
  /* 描画上の幅 = 300 + 20*2 + 5*2 = 350px */
  /* 描画上の幅を事前に計算する必要がある */
}

/* border-box（推奨） */
.element-border-box {
  box-sizing: border-box;
  width: 300px;
  padding: 20px;
  border: 5px solid #333;
  /* 描画上の幅 = 300px（指定したまま） */
  /* コンテンツ幅 = 300 - 20*2 - 5*2 = 250px */
}
```

計算の比較表:

| プロパティ | content-box | border-box |
|-----------|-------------|------------|
| 指定 width | 300px | 300px |
| padding | 20px x 2 = 40px | 20px x 2 = 40px |
| border | 5px x 2 = 10px | 5px x 2 = 10px |
| コンテンツ幅 | 300px | 250px |
| 描画上の幅 | 350px | 300px |
| margin 込みの占有幅 | 350px + margin | 300px + margin |

border-box を全要素に適用するリセット:

```css
/*
 * Universal box-sizing reset
 * 継承方式により、特定のコンポーネントで
 * content-box に戻すことも容易になる
 */
html {
  box-sizing: border-box;
}

*, *::before, *::after {
  box-sizing: inherit;
}

/* 特定コンポーネントだけ content-box に戻す場合 */
.legacy-component {
  box-sizing: content-box;
}
```

### 1.3 マージンの相殺（Margin Collapsing）

隣接するブロックレベル要素の垂直マージンは「相殺」される。これは CSS の重要な特性であり、多くの初学者が混乱するポイントでもある。

```
マージン相殺の基本:

  ケース1: 兄弟要素の隣接マージン
  ┌──────────────┐
  │   要素 A     │  margin-bottom: 30px
  └──────────────┘
         ↕ 30px（大きい方が採用される）
  ┌──────────────┐
  │   要素 B     │  margin-top: 20px
  └──────────────┘

  結果: 間隔は 30px（30 + 20 = 50px ではない）

  ケース2: 親と最初の子のマージン
  ┌──────────────────────┐  ← 親の margin-top と
  │ ┌──────────────────┐ │     子の margin-top が相殺
  │ │   子要素         │ │
  │ └──────────────────┘ │
  └──────────────────────┘

  ケース3: 空のブロック要素
  ┌──────────────┐
  │  要素 A      │
  └──────────────┘
                       ← 空の要素の margin-top と
  (空の <div>)            margin-bottom も相殺
                       ←
  ┌──────────────┐
  │  要素 B      │
  └──────────────┘
```

マージン相殺が発生しない条件:

```css
/* 以下の条件のいずれかに該当する場合、マージンは相殺されない */

/* 1. Flexbox / Grid の子要素 */
.flex-container { display: flex; flex-direction: column; }
.flex-container > * { /* margin は相殺されない */ }

/* 2. BFC を生成する要素 */
.bfc { overflow: hidden; }         /* BFC 境界でブロックされる */
.bfc { display: flow-root; }       /* より明示的な BFC 生成 */

/* 3. padding または border がある親要素 */
.parent { padding-top: 1px; }      /* 親子間の相殺を防ぐ */
.parent { border-top: 1px solid transparent; }

/* 4. float 要素 */
.floated { float: left; }

/* 5. position: absolute / fixed */
.positioned { position: absolute; }

/* 6. インライン要素が間に存在 */
/* テキストノードやインライン要素が兄弟マージンの間にある場合 */
```

### 1.4 負のマージン

マージンには負の値を設定できる。これは他のボックス領域（padding, border）にはない特性である。

```css
/* 負のマージンの効果 */
.pull-up {
  margin-top: -20px;
  /* 要素を上方向に 20px 引っ張る */
  /* 後続要素も一緒に引き上げられる */
}

.pull-left {
  margin-left: -20px;
  /* 要素を左方向に 20px 引っ張る */
}

/* 実用例: カード画像をコンテナからはみ出させる */
.card-image {
  margin-left: -16px;
  margin-right: -16px;
  margin-top: -16px;
  /* カードの padding 分だけ逆方向に拡張 */
}
```

---

## 2. Normal Flow とフォーマッティングコンテキスト

### 2.1 Normal Flow の基本規則

Normal Flow は CSS のデフォルトのレイアウトモードである。すべての要素は、特別なプロパティ（float, position, display: flex / grid 等）が適用されない限り、Normal Flow に従って配置される。

```
Normal Flow の 2 つのレベル:

  ブロックレベル要素 (display: block / list-item / table 等):
  ─────────────────────────────────────────
  ┌───────────────────────────────────────┐
  │ <div>  横幅は親の 100%              │
  └───────────────────────────────────────┘
  ┌───────────────────────────────────────┐
  │ <p>    新しい行から開始              │
  └───────────────────────────────────────┘
  ┌───────────────────────────────────────┐
  │ <h2>   縦に積み重なる               │
  └───────────────────────────────────────┘

  インラインレベル要素 (display: inline / inline-block 等):
  ─────────────────────────────────────────
  ここに│<span>│と│<a href>│と│<strong>│が
  横に並んで│<em>│配置される。行末で折り返す。
```

### 2.2 display プロパティの整理

CSS Display Level 3 仕様では、`display` プロパティは「外部表示型」と「内部表示型」の 2 つの値で構成される。

| 短縮形 | 外部表示型 | 内部表示型 | 説明 |
|--------|-----------|-----------|------|
| `block` | block | flow | ブロックコンテナ |
| `inline` | inline | flow | インラインボックス |
| `inline-block` | inline | flow-root | インラインレベルの BFC |
| `flex` | block | flex | ブロックレベルの Flex コンテナ |
| `inline-flex` | inline | flex | インラインレベルの Flex コンテナ |
| `grid` | block | grid | ブロックレベルの Grid コンテナ |
| `inline-grid` | inline | grid | インラインレベルの Grid コンテナ |
| `flow-root` | block | flow-root | BFC を生成するブロックコンテナ |
| `none` | - | - | レイアウトツリーから除外 |

### 2.3 BFC（Block Formatting Context）

BFC はブロックレベル要素の独立したレイアウト領域である。BFC 内部のレイアウトは外部に影響を与えず、外部のレイアウトも BFC 内部に影響しない。

BFC を生成する条件（主要なもの）:

```css
/* 1. ドキュメントのルート要素 */
/* <html> 要素は常に BFC を生成する */

/* 2. float 要素 */
.bfc-float { float: left; }  /* left / right どちらでも */

/* 3. position: absolute / fixed */
.bfc-positioned { position: absolute; }

/* 4. display: inline-block */
.bfc-inline-block { display: inline-block; }

/* 5. display: flow-root（最も明示的な方法） */
.bfc-flow-root { display: flow-root; }

/* 6. display: flex / grid のコンテナ */
.bfc-flex { display: flex; }

/* 7. overflow が visible 以外 */
.bfc-overflow { overflow: hidden; }  /* auto / scroll も可 */

/* 8. contain プロパティ */
.bfc-contain { contain: layout; }  /* content / strict も可 */
```

BFC の 3 つの主要効果:

```
効果1: フロートの包含
  ─────────────────────────────────────
  BFC なし:                BFC あり:
  ┌────────────────┐       ┌────────────────────┐
  │ 親              │       │ 親 (BFC)            │
  │ ┌──────┐       │       │ ┌──────┐            │
  │ │float │       │       │ │float │ テキスト   │
  │ └──────┘       │       │ └──────┘            │
  └────────────────┘       │                     │
  テキストがここに          └────────────────────┘
  はみ出す

効果2: マージン相殺の遮断
  ─────────────────────────────────────
  BFC 境界を跨ぐマージンは相殺されない

効果3: フロートとの重なり防止
  ─────────────────────────────────────
  BFC 要素は隣接するフロートと重ならない
```

### 2.4 インラインフォーマッティングコンテキスト（IFC）

IFC はインラインレベル要素のレイアウトコンテキストである。テキストの配置や `line-height` の計算はすべて IFC の中で行われる。

```css
/* IFC における重要な概念 */

/* line box: インライン要素を含む 1 行分の領域 */
.text-container {
  font-size: 16px;
  line-height: 1.5;
  /* line box の高さ = 16 * 1.5 = 24px */
}

/* vertical-align: line box 内の垂直配置 */
.icon {
  vertical-align: middle;
  /* baseline, top, bottom, text-top, text-bottom なども使用可能 */
}

/* inline-block の特性 */
.inline-block-element {
  display: inline-block;
  width: 100px;
  height: 50px;
  vertical-align: top;
  /* インラインの流れに参加しつつ、幅と高さを持てる */
}
```

---

## 3. Flexbox の内部アルゴリズム

### 3.1 Flexbox レイアウトの概念モデル

Flexbox は 1 次元レイアウトシステムであり、主軸（main axis）と交差軸（cross axis）の 2 つの軸で要素を配置する。

```
flex-direction: row（デフォルト）の軸:

  main-start                                    main-end
      │                                            │
      ▼                                            ▼
  ┌──────────────────────────────────────────────────┐ ← cross-start
  │  ┌─────────┐  ┌─────────┐  ┌─────────┐         │
  │  │ item 1  │  │ item 2  │  │ item 3  │         │
  │  │         │  │         │  │         │         │
  │  └─────────┘  └─────────┘  └─────────┘         │
  │                                                  │
  │  ──────── main axis（主軸）────────→             │
  │                                                  │
  └──────────────────────────────────────────────────┘ ← cross-end
                                                  │
                                           cross axis
                                           （交差軸）↓

flex-direction: column の場合:
  main axis が縦方向、cross axis が横方向に入れ替わる
```

### 3.2 Flex コンテナのプロパティ詳細

```css
.flex-container {
  display: flex;           /* flex コンテナを生成 */

  /* --- 主軸の方向 --- */
  flex-direction: row;           /* 左→右（デフォルト） */
  flex-direction: row-reverse;   /* 右→左 */
  flex-direction: column;        /* 上→下 */
  flex-direction: column-reverse;/* 下→上 */

  /* --- 折り返し --- */
  flex-wrap: nowrap;    /* 折り返さない（デフォルト） */
  flex-wrap: wrap;      /* 折り返す */
  flex-wrap: wrap-reverse; /* 逆方向に折り返す */

  /* --- 省略記法 --- */
  flex-flow: row wrap;  /* flex-direction + flex-wrap */

  /* --- 主軸上の配置 --- */
  justify-content: flex-start;    /* 先頭寄せ（デフォルト） */
  justify-content: flex-end;      /* 末尾寄せ */
  justify-content: center;        /* 中央寄せ */
  justify-content: space-between; /* 両端揃え・均等配置 */
  justify-content: space-around;  /* 各アイテム周囲に均等余白 */
  justify-content: space-evenly;  /* 完全均等余白 */

  /* --- 交差軸上の配置 --- */
  align-items: stretch;     /* 引き伸ばし（デフォルト） */
  align-items: flex-start;  /* 交差軸の先頭 */
  align-items: flex-end;    /* 交差軸の末尾 */
  align-items: center;      /* 交差軸の中央 */
  align-items: baseline;    /* テキストのベースライン揃え */

  /* --- 複数行の配置 --- */
  align-content: flex-start;    /* wrap 時に複数行を先頭寄せ */
  align-content: center;        /* wrap 時に複数行を中央寄せ */
  align-content: space-between; /* wrap 時に行間を均等配置 */

  /* --- 間隔 --- */
  gap: 16px;         /* 行・列共通 */
  row-gap: 16px;     /* 行間のみ */
  column-gap: 24px;  /* 列間のみ */
}
```

### 3.3 Flex アイテムのプロパティ詳細

```css
.flex-item {
  /* --- 伸縮の制御 --- */
  flex-grow: 0;     /* 余白の分配比率（デフォルト: 0 = 伸びない） */
  flex-shrink: 1;   /* 縮小の分配比率（デフォルト: 1 = 縮む） */
  flex-basis: auto; /* 基準サイズ（デフォルト: auto = コンテンツ依存） */

  /* --- flex 省略記法 --- */
  flex: initial;   /* = 0 1 auto → 縮小のみ */
  flex: auto;      /* = 1 1 auto → 伸縮両方 */
  flex: none;      /* = 0 0 auto → 固定サイズ */
  flex: 1;         /* = 1 1 0%   → 均等分配 */
  flex: 2;         /* = 2 1 0%   → 2 倍の比率で分配 */

  /* --- 個別の交差軸配置 --- */
  align-self: auto;       /* コンテナの align-items に従う */
  align-self: flex-start; /* このアイテムだけ先頭寄せ */
  align-self: center;     /* このアイテムだけ中央寄せ */

  /* --- 表示順序 --- */
  order: 0;   /* デフォルト。数値が小さいほど先 */
  order: -1;  /* 先頭に移動 */
  order: 1;   /* 末尾に移動 */
}
```

### 3.4 Flexbox レイアウト計算アルゴリズム（6 段階）

ブラウザのレイアウトエンジンが Flexbox をどのように計算するか、その内部アルゴリズムを段階的に解説する。

```
Flexbox レイアウト計算の 6 段階:

  ┌─────────────────────────────────────────────┐
  │ Stage 1: 利用可能空間の決定                 │
  │  コンテナの幅（または高さ）から padding     │
  │  と border を引いた値                        │
  └─────────────┬───────────────────────────────┘
                │
  ┌─────────────▼───────────────────────────────┐
  │ Stage 2: 各アイテムの仮サイズ決定           │
  │  flex-basis → width/height → コンテンツ     │
  │  の優先順位でベースサイズを決定             │
  └─────────────┬───────────────────────────────┘
                │
  ┌─────────────▼───────────────────────────────┐
  │ Stage 3: 余白（正 / 負）の計算              │
  │  利用可能空間 - 全アイテムの仮サイズ合計    │
  │  正 = 余白あり、負 = はみ出し               │
  └─────────────┬───────────────────────────────┘
                │
  ┌─────────────▼───────────────────────────────┐
  │ Stage 4: flex-grow / flex-shrink の適用     │
  │  余白が正 → flex-grow 比率で分配           │
  │  余白が負 → flex-shrink 比率で縮小         │
  └─────────────┬───────────────────────────────┘
                │
  ┌─────────────▼───────────────────────────────┐
  │ Stage 5: min/max 制約の適用                 │
  │  min-width / max-width の範囲にクランプ     │
  │  制約違反があれば Stage 4 を再実行          │
  └─────────────┬───────────────────────────────┘
                │
  ┌─────────────▼───────────────────────────────┐
  │ Stage 6: 配置の決定                         │
  │  justify-content → 主軸配置                │
  │  align-items → 交差軸配置                  │
  │  align-content → 複数行配置                │
  └─────────────────────────────────────────────┘
```

#### 計算例: flex-grow の分配

```css
.container {
  display: flex;
  width: 600px;  /* 利用可能空間: 600px */
}

.item-a { flex: 2 1 100px; }  /* basis: 100px, grow: 2 */
.item-b { flex: 1 1 150px; }  /* basis: 150px, grow: 1 */
.item-c { flex: 1 1 100px; }  /* basis: 100px, grow: 1 */
```

```
計算過程:

  Step 1: 仮サイズ合計 = 100 + 150 + 100 = 350px
  Step 2: 余白 = 600 - 350 = 250px（正の余白）
  Step 3: grow 合計 = 2 + 1 + 1 = 4
  Step 4: 各アイテムへの分配
    item-a: 100 + (250 * 2/4) = 100 + 125 = 225px
    item-b: 150 + (250 * 1/4) = 150 + 62.5 = 212.5px
    item-c: 100 + (250 * 1/4) = 100 + 62.5 = 162.5px

  検算: 225 + 212.5 + 162.5 = 600px ✓

  ┌────────────────────────────────────────────────────┐
  │ ┌──────────────┐ ┌─────────────┐ ┌──────────┐     │
  │ │   item-a     │ │   item-b    │ │  item-c  │     │
  │ │   225px      │ │   212.5px   │ │  162.5px │     │
  │ └──────────────┘ └─────────────┘ └──────────┘     │
  └────────────────────────────────────────────────────┘
                        600px
```

#### 計算例: flex-shrink の縮小

flex-shrink の計算は flex-grow よりも複雑である。縮小量はアイテムの flex-basis にも比例する。

```css
.container {
  display: flex;
  width: 400px;  /* 利用可能空間: 400px */
}

.item-a { flex: 0 2 200px; }  /* basis: 200px, shrink: 2 */
.item-b { flex: 0 1 300px; }  /* basis: 300px, shrink: 1 */
```

```
計算過程:

  Step 1: 仮サイズ合計 = 200 + 300 = 500px
  Step 2: 不足分 = 400 - 500 = -100px（負の余白）
  Step 3: 加重縮小係数の計算
    item-a の加重 = shrink * basis = 2 * 200 = 400
    item-b の加重 = shrink * basis = 1 * 300 = 300
    加重合計 = 400 + 300 = 700
  Step 4: 各アイテムの縮小量
    item-a: 200 - (100 * 400/700) = 200 - 57.14 ≈ 142.86px
    item-b: 300 - (100 * 300/700) = 300 - 42.86 ≈ 257.14px

  検算: 142.86 + 257.14 = 400px ✓

  ※ flex-shrink では basis の大きい要素ほど
    「同じ shrink 値でも多く縮む」ことに注意
```

### 3.5 flex-basis vs width

`flex-basis` と `width`（または `height`）の優先順位を正確に理解することは重要である。

```
flex-basis の解決優先順位:

  1. flex-basis が auto 以外 → flex-basis を使用
  2. flex-basis が auto かつ width 指定あり → width を使用
  3. flex-basis が auto かつ width なし → コンテンツサイズ
  4. flex-basis: 0 → コンテンツサイズを無視して 0 から開始

  注意: flex-basis: 0 と flex-basis: 0% は異なる場合がある
        コンテナに明示的なサイズがない場合、% はパーセント
        計算が不能になる
```

```css
/* よくある混乱: flex: 1 の内部動作 */
.item {
  flex: 1;
  /* これは flex: 1 1 0% と展開される */
  /* flex-basis: 0% → コンテンツサイズを考慮せず、 */
  /* 利用可能空間を grow 比率で完全均等に分配する */
}

.item-auto {
  flex: 1 1 auto;
  /* flex-basis: auto → まずコンテンツサイズを確保し、 */
  /* 余った空間を grow 比率で分配する */
  /* コンテンツの多いアイテムはより大きくなる */
}
```

---

## 4. CSS Grid の内部アルゴリズム

### 4.1 Grid レイアウトの概念モデル

CSS Grid は 2 次元レイアウトシステムであり、行（row）と列（column）の両方を同時に制御できる。Flexbox が「コンテンツに合わせて伸縮する」のに対し、Grid は「グリッド構造にコンテンツを配置する」アプローチをとる。

```
CSS Grid の基本構造:

  grid-template-columns: 200px 1fr 1fr;
  grid-template-rows: 80px 1fr 60px;
  gap: 12px;

       col 1      col 2       col 3
       200px       1fr         1fr
  ┌──────────┬───────────┬───────────┐
  │          │           │           │ row 1
  │  Cell    │  Cell     │  Cell     │ 80px
  │  (1,1)   │  (1,2)    │  (1,3)    │
  ├──────────┼───────────┼───────────┤ ← 12px gap
  │          │           │           │ row 2
  │  Cell    │  Cell     │  Cell     │ 1fr
  │  (2,1)   │  (2,2)    │  (2,3)    │
  │          │           │           │
  ├──────────┼───────────┼───────────┤ ← 12px gap
  │          │           │           │ row 3
  │  Cell    │  Cell     │  Cell     │ 60px
  │  (3,1)   │  (3,2)    │  (3,3)    │
  └──────────┴───────────┴───────────┘
       ↕          ↕
     12px gap   12px gap

  Grid Line（グリッド線）:
  │1       │2          │3          │4  ← 列のグリッド線
  ─1─────────────────────────────────  ← 行のグリッド線
  ─2─────────────────────────────────
  ─3─────────────────────────────────
  ─4─────────────────────────────────
```

### 4.2 Grid コンテナのプロパティ詳細

```css
.grid-container {
  display: grid;

  /* --- トラックの定義 --- */
  grid-template-columns: 200px 1fr 1fr;
  grid-template-rows: auto 1fr auto;

  /* repeat() 関数 */
  grid-template-columns: repeat(3, 1fr);          /* 3 等分 */
  grid-template-columns: repeat(4, 100px 200px);  /* パターン繰り返し */
  grid-template-columns: 200px repeat(3, 1fr);    /* 混合 */

  /* minmax() 関数 */
  grid-template-columns: minmax(200px, 1fr) 2fr;
  /* 最小 200px、最大で 1fr 相当 */

  /* auto-fill / auto-fit */
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));

  /* --- エリアテンプレート --- */
  grid-template-areas:
    "header  header  header"
    "sidebar content content"
    "footer  footer  footer";

  /* --- 間隔 --- */
  gap: 16px;
  row-gap: 16px;
  column-gap: 24px;

  /* --- 暗黙的トラック --- */
  grid-auto-rows: minmax(100px, auto);
  grid-auto-columns: 1fr;
  grid-auto-flow: row;     /* row | column | dense */

  /* --- 配置 --- */
  justify-items: stretch;   /* セル内の水平配置 */
  align-items: stretch;     /* セル内の垂直配置 */
  justify-content: start;   /* グリッド全体の水平配置 */
  align-content: start;     /* グリッド全体の垂直配置 */
}
```

### 4.3 Grid アイテムのプロパティ詳細

```css
.grid-item {
  /* --- 配置（ライン番号指定） --- */
  grid-column-start: 1;
  grid-column-end: 3;       /* 列1から列3まで（2カラム分） */
  grid-row-start: 1;
  grid-row-end: 2;

  /* --- 省略記法 --- */
  grid-column: 1 / 3;       /* start / end */
  grid-row: 1 / 2;
  grid-area: 1 / 1 / 2 / 3; /* row-start / col-start / row-end / col-end */

  /* --- span キーワード --- */
  grid-column: 1 / span 2;  /* 列1から2カラム分 */
  grid-column: span 2;      /* 自動配置で2カラム分 */

  /* --- エリア名指定 --- */
  grid-area: header;         /* grid-template-areas で定義した名前 */

  /* --- 個別配置 --- */
  justify-self: center;     /* セル内の水平配置（個別） */
  align-self: end;           /* セル内の垂直配置（個別） */
}
```

### 4.4 Grid トラックサイジングアルゴリズム

Grid のトラックサイジングは CSS 仕様の中でも最も複雑なアルゴリズムの一つである。以下にその概要を示す。

```
Grid トラックサイジングアルゴリズムの概要:

  ┌─────────────────────────────────────────────┐
  │ Phase 1: 固定サイズトラックの解決           │
  │  px, em 等の固定単位 → そのまま確定        │
  └─────────────┬───────────────────────────────┘
                │
  ┌─────────────▼───────────────────────────────┐
  │ Phase 2: コンテンツベーストラックの解決     │
  │  auto, min-content, max-content,            │
  │  fit-content() → コンテンツを測定して決定  │
  └─────────────┬───────────────────────────────┘
                │
  ┌─────────────▼───────────────────────────────┐
  │ Phase 3: fr 単位トラックの解決              │
  │  残りの利用可能空間を fr 比率で分配        │
  │  ※ minmax(auto, 1fr) の場合、            │
  │    最小値はコンテンツサイズ                 │
  └─────────────┬───────────────────────────────┘
                │
  ┌─────────────▼───────────────────────────────┐
  │ Phase 4: スパンアイテムの調整               │
  │  複数トラックにまたがるアイテムの          │
  │  サイズ要求をトラックに分配                │
  └─────────────────────────────────────────────┘
```

### 4.5 fr 単位の詳細な計算

```css
.container {
  display: grid;
  width: 900px;
  grid-template-columns: 200px 1fr 2fr;
  gap: 20px;
}
```

```
fr 単位の計算過程:

  Step 1: 利用可能空間の計算
    コンテナ幅: 900px
    gap の合計: 20px * 2 = 40px（3列の間に2つの gap）
    固定トラック: 200px
    fr に割り当て可能な空間 = 900 - 40 - 200 = 660px

  Step 2: fr の合計
    1fr + 2fr = 3fr

  Step 3: 1fr あたりの値
    660px / 3 = 220px

  Step 4: 各トラックの幅
    列1: 200px（固定）
    列2: 1fr = 220px
    列3: 2fr = 440px

  検算: 200 + 220 + 440 + 40(gap) = 900px ✓

  ┌──────────┬────────────────┬──────────────────────────────┐
  │  200px   │     220px      │           440px              │
  │  固定    │     1fr        │           2fr                │
  └──────────┴────────────────┴──────────────────────────────┘
       ↕ 20px gap     ↕ 20px gap
```

### 4.6 auto-fill vs auto-fit の違い

```css
/* auto-fill: 空のトラックも保持 */
.grid-auto-fill {
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
}

/* auto-fit: 空のトラックを折りたたむ */
.grid-auto-fit {
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
}
```

```
コンテナ幅: 600px、アイテム: 2個、minmax(150px, 1fr) の場合:

  auto-fill:
  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
  │ item1  │ │ item2  │ │ (empty)│ │ (empty)│
  │ 150px  │ │ 150px  │ │ 150px  │ │ 150px  │
  └────────┘ └────────┘ └────────┘ └────────┘
  → 4 トラック生成、空トラックも 150px 確保

  auto-fit:
  ┌──────────────────┐ ┌──────────────────┐
  │     item1        │ │     item2        │
  │     300px        │ │     300px        │
  └──────────────────┘ └──────────────────┘
  → 空トラックは 0px に折りたたまれ、
    残り空間を既存アイテムが 1fr で分配
```

### 4.7 grid-template-areas による名前付きレイアウト

```css
/* ダッシュボードレイアウトの例 */
.dashboard {
  display: grid;
  grid-template-columns: 250px 1fr 1fr;
  grid-template-rows: 60px 1fr 1fr 40px;
  grid-template-areas:
    "nav     header  header"
    "nav     main    aside"
    "nav     main    aside"
    "nav     footer  footer";
  gap: 8px;
  height: 100vh;
}

.nav    { grid-area: nav; }
.header { grid-area: header; }
.main   { grid-area: main; }
.aside  { grid-area: aside; }
.footer { grid-area: footer; }

/* "." でエリアを空白にすることも可能 */
.layout-with-gap {
  grid-template-areas:
    "header header header"
    "sidebar . content"
    "footer footer footer";
}
```

---

## 5. Positioning と Stacking Context

### 5.1 position プロパティの全モード

```css
/* static（デフォルト） */
.static {
  position: static;
  /* Normal Flow に従う */
  /* top / left / right / bottom は無効 */
}

/* relative: 元の位置から相対移動 */
.relative {
  position: relative;
  top: 10px;
  left: 20px;
  /* Normal Flow 上の元の位置から移動 */
  /* 元の位置には空白が残る（スペースを占有し続ける） */
}

/* absolute: 最寄りの positioned 祖先を基準に配置 */
.absolute {
  position: absolute;
  top: 0;
  right: 0;
  /* Normal Flow から完全に外れる */
  /* 基準: 最寄りの position: static 以外の祖先 */
  /* 祖先が全て static の場合は初期包含ブロック（viewport 相当） */
}

/* fixed: viewport を基準に固定配置 */
.fixed {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  /* スクロールしても位置が変わらない */
  /* ※ 祖先に transform, filter, perspective がある場合は */
  /*   その祖先が包含ブロックになる（重要なエッジケース） */
}

/* sticky: スクロール位置に応じて relative / fixed を切り替え */
.sticky {
  position: sticky;
  top: 0;
  /* スクロール前は relative と同じ */
  /* スクロールして top: 0 に達すると fixed のように振る舞う */
  /* ※ 最寄りのスクロール可能な祖先の中で動作する */
}
```

### 5.2 Stacking Context（重ね合わせコンテキスト）

```
Stacking Context の描画順序:

  背面                                          前面
  ─────────────────────────────────────────────────→

  1. 背景と    2. 負の     3. ブロック  4. float  5. インライン  6. z-index:0  7. 正の
     ボーダー    z-index     レベル       要素      レベル        / auto        z-index

  ┌──────────────────────────────────────────────────┐
  │ ┌─────────┐                                      │
  │ │z-index  │ ← 最前面 (7)                        │
  │ │  : 10   │                                      │
  │ └─────────┘                                      │
  │          ┌─────────┐                             │
  │          │z-index  │ ← (6)                       │
  │          │  : 0    │                              │
  │          └─────────┘                             │
  │ ┌─────────────────────┐                          │
  │ │ インラインテキスト  │ ← (5)                    │
  │ └─────────────────────┘                          │
  │    ┌─────────┐                                   │
  │    │ float   │ ← (4)                             │
  │    └─────────┘                                   │
  │ ┌────────────────────────────────────┐           │
  │ │ ブロックレベル子要素              │ ← (3)      │
  │ └────────────────────────────────────┘           │
  │  ┌──────────┐                                    │
  │  │z-index   │ ← (2)                             │
  │  │ : -1     │                                    │
  │  └──────────┘                                    │
  │ ████████████████████████████████████ ← (1) 背景  │
  └──────────────────────────────────────────────────┘
```

Stacking Context を生成する条件:

```css
/* Stacking Context を生成する主な条件 */

/* 1. ルート要素 (<html>) */

/* 2. position + z-index */
.sc-1 { position: relative; z-index: 1; }
.sc-2 { position: absolute; z-index: 0; }

/* 3. position: fixed / sticky は常に生成 */
.sc-3 { position: fixed; }
.sc-4 { position: sticky; top: 0; }

/* 4. opacity が 1 未満 */
.sc-5 { opacity: 0.99; }

/* 5. transform が none 以外 */
.sc-6 { transform: translateZ(0); }

/* 6. filter が none 以外 */
.sc-7 { filter: blur(0); }

/* 7. will-change が特定プロパティ */
.sc-8 { will-change: transform; }

/* 8. isolation: isolate */
.sc-9 { isolation: isolate; }

/* 9. mix-blend-mode が normal 以外 */
.sc-10 { mix-blend-mode: multiply; }

/* 10. contain: layout / paint / strict / content */
.sc-11 { contain: paint; }
```

### 5.3 包含ブロック（Containing Block）

要素のサイズ計算やパーセンテージ値の基準となるのが包含ブロックである。

```
包含ブロックの決定規則:

  position: static / relative
  → 最寄りのブロックコンテナ祖先のコンテンツ領域

  position: absolute
  → 最寄りの position: static 以外の祖先の
     パディング辺（padding edge）

  position: fixed
  → ビューポート（通常）
  → ※ transform/filter/perspective を持つ祖先がある場合は
     その祖先のパディング辺

  position: sticky
  → 最寄りのスクロール可能な祖先のコンテンツ領域
```

---

## 6. レイアウト計算アルゴリズムの全体像

### 6.1 ブラウザのレンダリングパイプラインにおける位置

```
レンダリングパイプライン全体図:

  HTML/CSS
    │
    ▼
  ┌──────────┐   ┌──────────┐   ┌──────────┐
  │  Parse   │──→│   DOM    │──→│  CSSOM   │
  │ (解析)   │   │  Tree    │   │  Tree    │
  └──────────┘   └──────────┘   └──────────┘
                       │              │
                       ▼              │
                 ┌──────────┐        │
                 │  Render  │←───────┘
                 │  Tree    │
                 └────┬─────┘
                      │
                      ▼
                ┌───────────┐
                │  Layout   │ ← ★ ここがレイアウト計算
                │ (Reflow)  │    要素の位置とサイズを決定
                └─────┬─────┘
                      │
                      ▼
                ┌───────────┐
                │  Paint    │
                │ (描画命令 │
                │  生成)    │
                └─────┬─────┘
                      │
                      ▼
                ┌───────────┐
                │ Composite │
                │ (合成)    │
                └───────────┘
                      │
                      ▼
                  画面表示
```

### 6.2 レイアウト計算の詳細フロー

```
Layout 計算の内部フロー:

  ┌──────────────────────────────────────────────┐
  │ 1. ルート要素から開始                        │
  │    初期包含ブロック = ビューポートサイズ      │
  └─────────────┬────────────────────────────────┘
                │
  ┌─────────────▼────────────────────────────────┐
  │ 2. 各要素の display / position を評価        │
  │    → レイアウトモードの決定                  │
  │    Normal Flow / Flex / Grid / Float / Abs   │
  └─────────────┬────────────────────────────────┘
                │
  ┌─────────────▼────────────────────────────────┐
  │ 3. ツリーを深さ優先で走査                    │
  │    親要素は「利用可能空間」を子に伝える      │
  │    子要素は「必要なサイズ」を親に返す        │
  └─────────────┬────────────────────────────────┘
                │
  ┌─────────────▼────────────────────────────────┐
  │ 4. 制約の解決                                │
  │    width / height / min / max / %            │
  │    利用可能空間と要素の固有サイズから決定    │
  └─────────────┬────────────────────────────────┘
                │
  ┌─────────────▼────────────────────────────────┐
  │ 5. 各レイアウトモード固有のアルゴリズム実行  │
  │    Block: 縦積み + マージン相殺             │
  │    Inline: 行ボックス生成 + 折り返し        │
  │    Flex: 6 段階アルゴリズム                  │
  │    Grid: トラックサイジング                  │
  └─────────────┬────────────────────────────────┘
                │
  ┌─────────────▼────────────────────────────────┐
  │ 6. 座標の確定                                │
  │    各要素の (x, y, width, height) を確定    │
  │    レイアウトツリーに書き込み                │
  └──────────────────────────────────────────────┘
```

### 6.3 パーセンテージ値の解決

パーセンテージ値は包含ブロックを基準に計算される。しかし、プロパティによって基準が異なる。

| プロパティ | パーセンテージの基準 |
|-----------|---------------------|
| width | 包含ブロックの幅 |
| height | 包含ブロックの高さ（※） |
| padding-top / padding-bottom | 包含ブロックの**幅**（高さではない） |
| margin-top / margin-bottom | 包含ブロックの**幅**（高さではない） |
| top / bottom | 包含ブロックの高さ |
| left / right | 包含ブロックの幅 |
| font-size | 親要素の font-size |
| line-height | 要素自身の font-size |

```css
/*
 * 重要: padding と margin の上下方向の % は
 * 「包含ブロックの幅」を基準にする
 *
 * この仕様は直感に反するが、循環参照を防ぐための設計である
 */
.aspect-ratio-hack {
  width: 100%;
  padding-top: 56.25%;  /* 16:9 のアスペクト比 */
  /* padding-top の % は幅を基準にするため、 */
  /* 幅の 56.25% = 9/16 の高さが得られる */
  height: 0;
}

/* 現代的なアスペクト比の指定 */
.modern-aspect-ratio {
  aspect-ratio: 16 / 9;
  width: 100%;
  /* aspect-ratio プロパティで直接指定可能 */
}
```

---

## 7. パフォーマンスとレイアウトスラッシング

### 7.1 レイアウトスラッシング（Layout Thrashing）

レイアウトスラッシングとは、JavaScript で DOM の読み取りと書き込みを交互に行うことで、ブラウザに同期的なレイアウト再計算を強制する問題である。

```javascript
// --- アンチパターン: レイアウトスラッシング ---
// 各ループでレイアウト再計算が発生する（O(n) のレイアウト計算）
const elements = document.querySelectorAll('.item');
for (const el of elements) {
  const height = el.offsetHeight;       // 読み取り → 同期レイアウト発生
  el.style.width = height * 2 + 'px';  // 書き込み → レイアウトが無効化
}

// --- 改善パターン: 読み取りと書き込みを分離 ---
const elements = document.querySelectorAll('.item');

// Phase 1: 全ての読み取りをまとめる
const heights = Array.from(elements).map(el => el.offsetHeight);

// Phase 2: 全ての書き込みをまとめる
elements.forEach((el, i) => {
  el.style.width = heights[i] * 2 + 'px';
});
```

### 7.2 レイアウトを発生させるプロパティ

以下のプロパティや API はレイアウト計算を発生させる。パフォーマンスに敏感な場面では注意が必要である。

```
レイアウトを発生（トリガー）させる操作:

  DOM プロパティの読み取り:
  ├── offsetTop, offsetLeft, offsetWidth, offsetHeight
  ├── scrollTop, scrollLeft, scrollWidth, scrollHeight
  ├── clientTop, clientLeft, clientWidth, clientHeight
  └── getComputedStyle() の一部プロパティ

  レイアウトのみ（Paint なし）:
  ├── width, height, min-*, max-*
  ├── padding, margin, border
  ├── display, position, float
  ├── top, left, right, bottom
  └── font-size, line-height, text-align

  Paint まで発生:
  ├── color, background, box-shadow
  ├── border-radius, border-style
  └── visibility

  Composite のみ（最軽量）:
  ├── transform
  ├── opacity
  └── will-change
```

### 7.3 contain プロパティによるレイアウト最適化

```css
/* contain: レイアウトの影響範囲を限定する */
.card {
  contain: layout;
  /* この要素内部のレイアウト変更は外部に影響しない */
  /* ブラウザはこの要素だけを再計算すればよい */
}

.widget {
  contain: strict;
  /* layout + paint + size + style を全て包含 */
  /* 最も強力だが、サイズがコンテンツに依存できない */
}

.article {
  contain: content;
  /* layout + paint + style を包含 */
  /* size は包含しないため、コンテンツに応じたサイズ変更は可能 */
}

/* content-visibility: 画面外の要素のレンダリングをスキップ */
.long-list-item {
  content-visibility: auto;
  contain-intrinsic-size: 0 200px;
  /* 画面外の場合、200px の仮サイズでレイアウトされ、 */
  /* 内部のレンダリングは完全にスキップされる */
}
```

---

## 8. Flexbox vs Grid 使い分け徹底比較

### 8.1 設計思想の違い

| 観点 | Flexbox | CSS Grid |
|------|---------|----------|
| 次元 | 1次元（行 **or** 列） | 2次元（行 **と** 列を同時に制御） |
| 設計アプローチ | コンテンツ主導（content-out） | レイアウト主導（layout-in） |
| サイズの決定 | コンテンツに基づいて伸縮 | トラック定義に基づいて配置 |
| 折り返し | flex-wrap で 1次元の延長 | 本質的に 2次元の構造 |
| アイテムの配置 | ソース順序に依存しやすい | grid-area で自由に配置可能 |
| 重なり | 標準では不可 | grid-area の重複で可能 |
| Gap サポート | あり | あり |
| サブグリッド | なし | subgrid で入れ子の整列が可能 |
| ブラウザ対応 | IE11 部分対応（-ms-） | IE11 非対応（旧仕様のみ） |
| 適切な用途 | ナビバー、カード内、ツールバー | ページレイアウト、ダッシュボード |

### 8.2 ユースケース別の推奨

```
ユースケース別推奨レイアウトモード:

  ┌────────────────────────────┬──────────┬──────────┐
  │ ユースケース               │ Flexbox  │  Grid    │
  ├────────────────────────────┼──────────┼──────────┤
  │ ナビゲーションバー         │  ★推奨  │  △可    │
  │ カードの内部レイアウト     │  ★推奨  │  △可    │
  │ ツールバーのボタン配置     │  ★推奨  │  ○可    │
  │ フォームの input + button  │  ★推奨  │  ○可    │
  │ 要素の中央揃え             │  ★推奨  │  ★推奨  │
  │ 等幅のカードグリッド       │  ○可    │  ★推奨  │
  │ ページ全体のレイアウト     │  △可    │  ★推奨  │
  │ ダッシュボード             │  ×不適  │  ★推奨  │
  │ 聖杯レイアウト             │  △可    │  ★推奨  │
  │ マガジン風レイアウト       │  ×不適  │  ★推奨  │
  │ 要素の重なり配置           │  ×不適  │  ★推奨  │
  │ レスポンシブカードリスト   │  ○可    │  ★推奨  │
  └────────────────────────────┴──────────┴──────────┘

  ★推奨: 最適な選択
  ○可:  使用可能だが最適ではない
  △可:  やや無理がある
  ×不適: 不適切
```

### 8.3 Flexbox と Grid の組み合わせ

実際のプロジェクトでは、Flexbox と Grid を適材適所で組み合わせることが最も効果的である。

```css
/* ページ全体: Grid */
.page-layout {
  display: grid;
  grid-template-columns: 250px 1fr;
  grid-template-rows: 60px 1fr 40px;
  grid-template-areas:
    "header header"
    "sidebar main"
    "footer footer";
  min-height: 100vh;
}

/* ヘッダー内部: Flexbox */
.header {
  grid-area: header;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
}

/* ナビゲーション: Flexbox */
.header-nav {
  display: flex;
  gap: 16px;
  align-items: center;
}

/* メインコンテンツ内のカードグリッド: Grid */
.main-content {
  grid-area: main;
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 24px;
  padding: 24px;
}

/* 各カード内部: Flexbox */
.card {
  display: flex;
  flex-direction: column;
}

.card-body {
  flex: 1;  /* カードの高さが揃う */
}

.card-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: auto;  /* フッターを底辺に固定 */
}
```

---

## 9. 実践コード例集

### 9.1 完全な中央揃え（5 つの方法）

```css
/* 方法1: Flexbox（最も汎用的） */
.center-flex {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 100vh;
}

/* 方法2: Grid（最も簡潔） */
.center-grid {
  display: grid;
  place-items: center;
  min-height: 100vh;
}

/* 方法3: Grid + margin: auto */
.center-grid-margin {
  display: grid;
  min-height: 100vh;
}
.center-grid-margin > .child {
  margin: auto;
}

/* 方法4: position + transform */
.center-position {
  position: relative;
  min-height: 100vh;
}
.center-position > .child {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
}

/* 方法5: position + inset + margin（モダン） */
.center-inset {
  position: relative;
  min-height: 100vh;
}
.center-inset > .child {
  position: absolute;
  inset: 0;
  margin: auto;
  width: fit-content;
  height: fit-content;
}
```

### 9.2 聖杯レイアウト（Holy Grail Layout）

```css
/* CSS Grid による聖杯レイアウト */
.holy-grail {
  display: grid;
  grid-template:
    "header header header" 60px
    "nav    main   aside"  1fr
    "footer footer footer" 40px
    / 200px 1fr    200px;
  min-height: 100vh;
  gap: 0;
}

.header { grid-area: header; background: #2d3748; color: white; }
.nav    { grid-area: nav;    background: #edf2f7; }
.main   { grid-area: main;   padding: 24px; }
.aside  { grid-area: aside;  background: #edf2f7; }
.footer { grid-area: footer; background: #2d3748; color: white; }

/* レスポンシブ対応 */
@media (max-width: 768px) {
  .holy-grail {
    grid-template:
      "header" 60px
      "nav"    auto
      "main"   1fr
      "aside"  auto
      "footer" 40px
      / 1fr;
  }
}
```

### 9.3 sticky フッター

```css
/* 方法1: Flexbox による sticky フッター */
.page-flex {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
}

.page-flex > main {
  flex: 1;
  /* メインコンテンツが少なくても、 */
  /* フッターはビューポートの底に配置される */
}

/* 方法2: Grid による sticky フッター */
.page-grid {
  display: grid;
  grid-template-rows: auto 1fr auto;
  min-height: 100vh;
}

/* 方法3: Grid + min-height（最も簡潔） */
body {
  display: grid;
  grid-template-rows: auto 1fr auto;
  min-height: 100dvh;  /* dvh: 動的ビューポート高さ */
}
```

### 9.4 レスポンシブカードグリッド

```css
/* 自動折り返しカードグリッド（メディアクエリ不要） */
.card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(min(100%, 300px), 1fr));
  gap: 24px;
  padding: 24px;
}

/*
 * min(100%, 300px) を使う理由:
 * - ビューポートが 300px 未満の場合、minmax(300px, 1fr) だと
 *   カードがコンテナをはみ出す
 * - min(100%, 300px) により、コンテナ幅が 300px 未満の場合は
 *   100% が適用され、はみ出しを防止する
 */

/* カードの均等な高さ揃え */
.card {
  display: flex;
  flex-direction: column;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  overflow: hidden;
}

.card-image {
  aspect-ratio: 16 / 9;
  object-fit: cover;
  width: 100%;
}

.card-content {
  flex: 1;
  padding: 16px;
  display: flex;
  flex-direction: column;
}

.card-title {
  font-size: 1.125rem;
  font-weight: 600;
  margin-bottom: 8px;
}

.card-description {
  flex: 1;
  color: #4a5568;
}

.card-action {
  margin-top: auto;
  padding-top: 16px;
}
```

### 9.5 サイドバー + メインの可変レイアウト

```css
/* サイドバーが常に一定幅、メインが残りを占める */
.sidebar-layout {
  display: flex;
  gap: 24px;
}

.sidebar {
  flex: 0 0 280px;  /* 固定幅 280px */
  /* flex-shrink: 0 で縮小を防止 */
}

.main-content {
  flex: 1;
  min-width: 0;
  /* min-width: 0 がないとテキストのオーバーフローが発生する */
  /* Flex アイテムのデフォルト min-width は auto（コンテンツ幅） */
}

/* レスポンシブ: 小さい画面では縦積み */
@media (max-width: 768px) {
  .sidebar-layout {
    flex-direction: column;
  }

  .sidebar {
    flex: none;  /* 固定幅を解除 */
    order: -1;   /* 必要に応じてサイドバーを上に */
  }
}
```

---

## 10. アンチパターン集

### 10.1 アンチパターン: min-width: 0 の未設定による Flex アイテムのオーバーフロー

```css
/* --- 問題のあるコード --- */
.container {
  display: flex;
}

.long-text-item {
  flex: 1;
  /* 長いテキストやURLがコンテナからはみ出す */
  /* Flex アイテムの min-width のデフォルトは auto であり、 */
  /* これは「コンテンツの最小幅より小さくならない」ことを意味する */
}

/* --- 修正後のコード --- */
.container {
  display: flex;
}

.long-text-item {
  flex: 1;
  min-width: 0;
  /* min-width: 0 により、コンテンツ幅未満への縮小を許可する */
  /* これで text-overflow: ellipsis なども正しく動作する */
}

.long-text-item p {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
```

```
問題の図解:

  min-width: auto（デフォルト）の場合:
  ┌──────────────────────────────────────────────────────┐
  │ ┌────────┐ ┌─────────────────────────────────────────┼──────┐
  │ │ sidebar│ │ この長いテキストがコンテナからはみ出す   │はみ出│
  │ └────────┘ └─────────────────────────────────────────┼──────┘
  └──────────────────────────────────────────────────────┘

  min-width: 0 を追加した場合:
  ┌──────────────────────────────────────────────────────┐
  │ ┌────────┐ ┌────────────────────────────────────────┐│
  │ │ sidebar│ │ この長いテキストがコンテナ内に収...     ││
  │ └────────┘ └────────────────────────────────────────┘│
  └──────────────────────────────────────────────────────┘
```

### 10.2 アンチパターン: height: 100% の連鎖忘れ

```css
/* --- 問題のあるコード --- */
.child {
  height: 100%;
  /* 親要素に明示的な高さが設定されていないと、 */
  /* height: 100% は無視される */
  /* ブラウザは「何の 100%?」を解決できない */
}

/* --- 修正後のコード --- */

/* 方法1: 高さの連鎖を確保する */
html, body {
  height: 100%;  /* まずルートから高さを確保 */
}
.parent {
  height: 100%;  /* 親にも明示的な高さが必要 */
}
.child {
  height: 100%;  /* これで正しく動作する */
}

/* 方法2: min-height + flex を使う（推奨） */
html {
  height: 100%;
}
body {
  min-height: 100%;
  display: flex;
  flex-direction: column;
}
.child {
  flex: 1;  /* 残りの空間を占める */
}

/* 方法3: dvh 単位を使う（最新の方法） */
.full-height {
  min-height: 100dvh;
  /* dvh = dynamic viewport height */
  /* モバイルでアドレスバーが出入りしても正確 */
}
```

### 10.3 アンチパターン: z-index のインフレーション

```css
/* --- 問題のあるコード --- */
.modal     { z-index: 99999; }
.tooltip   { z-index: 999999; }
.dropdown  { z-index: 9999; }
/* z-index の値が無秩序に増大し、管理不能になる */

/* --- 改善: z-index スケールの定義 --- */
:root {
  --z-dropdown:  100;
  --z-sticky:    200;
  --z-overlay:   300;
  --z-modal:     400;
  --z-popover:   500;
  --z-tooltip:   600;
  --z-toast:     700;
}

.dropdown { z-index: var(--z-dropdown); }
.modal    { z-index: var(--z-modal); }
.tooltip  { z-index: var(--z-tooltip); }

/*
 * さらに、isolation: isolate を使って
 * Stacking Context を明示的に区切ることで、
 * 各コンポーネント内の z-index が外部に漏れないようにする
 */
.component {
  isolation: isolate;
  /* このコンポーネント内の z-index は外部と独立する */
}
```

---

## 11. エッジケース分析

### 11.1 エッジケース: position: fixed と transform の相互作用

`position: fixed` の要素は通常ビューポートを基準に配置される。しかし、祖先要素に `transform`、`filter`、`perspective` のいずれかが設定されていると、その祖先要素が新しい包含ブロックとなり、fixed 配置が意図通りに動作しなくなる。

```css
/* 問題が発生するケース */
.animated-parent {
  transform: translateX(0);
  /* この transform により、新しい包含ブロックが生成される */
}

.modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.5);
  /* 期待: ビューポート全体をカバー */
  /* 現実: .animated-parent の領域内に制限される */
}
```

```
問題の図解:

  通常の fixed 配置:
  ┌── viewport ────────────────────────────┐
  │ ┌── fixed overlay ──────────────────┐  │
  │ │                                   │  │
  │ │  ビューポート全体をカバー         │  │
  │ │                                   │  │
  │ └───────────────────────────────────┘  │
  └────────────────────────────────────────┘

  祖先に transform がある場合:
  ┌── viewport ────────────────────────────┐
  │                                        │
  │  ┌── transform parent ──────┐          │
  │  │ ┌── fixed overlay ────┐  │          │
  │  │ │ 親の中に閉じ込め   │  │          │
  │  │ │ られてしまう       │  │          │
  │  │ └────────────────────┘  │          │
  │  └──────────────────────────┘          │
  │                                        │
  └────────────────────────────────────────┘
```

対策:

```css
/* 対策1: モーダルを DOM のトップレベルに配置する */
/* React の createPortal や Vue の Teleport を使用 */

/* 対策2: transform の代わりに will-change を使う（場合による） */
/* ※ will-change: transform も同様の問題を引き起こすため注意 */

/* 対策3: 祖先の transform を条件付きで適用する */
.parent {
  /* アニメーション中のみ transform を適用 */
  /* idle 状態では transform: none を維持 */
}

/* 対策4: CSS の @layer や :has() を使った条件付き transform */
.parent:not(:has(.modal-open)) {
  transform: translateX(var(--offset));
}
```

### 11.2 エッジケース: Flex アイテムの min-height と百分率の子要素

Flex アイテムに `min-height` を設定し、その子要素に `height: 100%` を指定すると、一部のブラウザで期待通りに動作しない場合がある。

```css
/* 問題が発生するケース */
.flex-container {
  display: flex;
  min-height: 500px;
}

.flex-item {
  /* flex アイテムは min-height: 500px のコンテナの中で */
  /* 引き伸ばされて 500px になる（align-items: stretch のため） */
}

.inner-child {
  height: 100%;
  /* 一部のブラウザ（古い Chrome 等）では、 */
  /* flex アイテムの「引き伸ばされた高さ」を */
  /* 百分率の基準として認識しない場合がある */
}

/* 対策 */
.flex-item-fixed {
  display: flex;
  flex-direction: column;
  /* flex アイテム自身も flex コンテナにすることで */
  /* 子要素に flex: 1 を使って高さを分配できる */
}

.inner-child-fixed {
  flex: 1;
  /* height: 100% の代わりに flex: 1 を使用 */
}
```

### 11.3 エッジケース: Grid の 1fr と min-content の関係

```css
/* 1fr は「minmax(auto, 1fr)」の省略形である */
/* auto = min-content のため、コンテンツが大きいと */
/* 1fr のトラックが均等にならない場合がある */

.grid-unequal {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  /* 各カラムのコンテンツ量が異なると、 */
  /* コンテンツの多いカラムが他より大きくなる場合がある */
}

/* 対策: minmax(0, 1fr) で最小値を 0 にする */
.grid-equal {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  /* 最小値が 0 なので、コンテンツに関係なく均等分配される */
  /* ※ コンテンツのオーバーフローが発生する可能性がある */
}
```

---

## 12. 演習問題（3段階）

### 12.1 基礎レベル（Beginner）

**演習 B-1: Box Model の計算**

以下の CSS が適用された要素の、画面上に描画される実際の幅と高さを答えよ。

```css
/* 問題 1 */
.box-a {
  box-sizing: content-box;
  width: 200px;
  height: 150px;
  padding: 15px;
  border: 3px solid black;
  margin: 20px;
}

/* 問題 2 */
.box-b {
  box-sizing: border-box;
  width: 200px;
  height: 150px;
  padding: 15px;
  border: 3px solid black;
  margin: 20px;
}
```

<details>
<summary>解答</summary>

問題 1（content-box）:
- 描画幅 = 200 + 15*2 + 3*2 = 236px
- 描画高さ = 150 + 15*2 + 3*2 = 186px
- 占有幅（margin 込み） = 236 + 20*2 = 276px
- 占有高さ（margin 込み） = 186 + 20*2 = 226px

問題 2（border-box）:
- 描画幅 = 200px（指定値そのまま）
- 描画高さ = 150px（指定値そのまま）
- コンテンツ幅 = 200 - 15*2 - 3*2 = 164px
- コンテンツ高さ = 150 - 15*2 - 3*2 = 114px
- 占有幅（margin 込み） = 200 + 20*2 = 240px
- 占有高さ（margin 込み） = 150 + 20*2 = 190px

</details>

**演習 B-2: マージン相殺**

以下のマークアップで、要素 A と要素 B の間の実際の間隔はいくらか。

```html
<div class="a" style="margin-bottom: 40px;">A</div>
<div class="b" style="margin-top: 25px;">B</div>
```

<details>
<summary>解答</summary>

マージン相殺により、大きい方の値が採用される。したがって間隔は **40px**（40px + 25px = 65px ではない）。

</details>

### 12.2 中級レベル（Intermediate）

**演習 I-1: Flexbox の計算**

以下のレイアウトで、各アイテムの最終的な幅を計算せよ。

```css
.container {
  display: flex;
  width: 800px;
}

.item-a { flex: 3 1 100px; }
.item-b { flex: 2 1 200px; }
.item-c { flex: 1 1 150px; }
```

<details>
<summary>解答</summary>

Step 1: 仮サイズ合計 = 100 + 200 + 150 = 450px
Step 2: 余白 = 800 - 450 = 350px（正の余白 → flex-grow 適用）
Step 3: grow 合計 = 3 + 2 + 1 = 6
Step 4: 分配
- item-a: 100 + (350 * 3/6) = 100 + 175 = **275px**
- item-b: 200 + (350 * 2/6) = 200 + 116.67 ≈ **316.67px**
- item-c: 150 + (350 * 1/6) = 150 + 58.33 ≈ **208.33px**

検算: 275 + 316.67 + 208.33 = 800px

</details>

**演習 I-2: Grid のトラックサイジング**

以下の Grid コンテナの各列の幅を計算せよ。

```css
.container {
  display: grid;
  width: 1200px;
  grid-template-columns: 300px 2fr 1fr;
  gap: 24px;
}
```

<details>
<summary>解答</summary>

Step 1: gap の合計 = 24px * 2 = 48px
Step 2: fr に割り当て可能な空間 = 1200 - 300 - 48 = 852px
Step 3: fr 合計 = 2 + 1 = 3
Step 4: 1fr = 852 / 3 = 284px
Step 5: 各列の幅
- 列1: **300px**（固定）
- 列2: 2fr = **568px**
- 列3: 1fr = **284px**

検算: 300 + 568 + 284 + 48(gap) = 1200px

</details>

### 12.3 上級レベル（Advanced）

**演習 A-1: flex-shrink の加重計算**

以下のレイアウトで、コンテナ幅が 300px の場合、各アイテムの最終的な幅を計算せよ。ただし、min-width の制約は考慮しないものとする。

```css
.container {
  display: flex;
  width: 300px;
}

.item-a { flex: 0 3 200px; }  /* shrink: 3, basis: 200px */
.item-b { flex: 0 2 150px; }  /* shrink: 2, basis: 150px */
.item-c { flex: 0 1 100px; }  /* shrink: 1, basis: 100px */
```

<details>
<summary>解答</summary>

Step 1: 仮サイズ合計 = 200 + 150 + 100 = 450px
Step 2: 不足分 = 300 - 450 = -150px（負の余白 → flex-shrink 適用）
Step 3: 加重縮小係数の計算
- item-a: shrink * basis = 3 * 200 = 600
- item-b: shrink * basis = 2 * 150 = 300
- item-c: shrink * basis = 1 * 100 = 100
- 加重合計 = 600 + 300 + 100 = 1000

Step 4: 各アイテムの縮小量
- item-a: 200 - (150 * 600/1000) = 200 - 90 = **110px**
- item-b: 150 - (150 * 300/1000) = 150 - 45 = **105px**
- item-c: 100 - (150 * 100/1000) = 100 - 15 = **85px**

検算: 110 + 105 + 85 = 300px

ポイント: flex-shrink では basis が大きい要素ほど多く縮む（加重方式）。これは flex-grow（単純な比率分配）とは異なる挙動である。

</details>

**演習 A-2: Stacking Context とレイアウトの総合問題**

以下のコードで、`.modal` がビューポート全体をカバーしない原因を特定し、修正案を 2 つ提示せよ。

```html
<div class="app" style="transform: scale(1);">
  <div class="content">
    <div class="modal" style="position: fixed; inset: 0; background: rgba(0,0,0,0.5);">
      モーダル
    </div>
  </div>
</div>
```

<details>
<summary>解答</summary>

原因: `.app` に `transform: scale(1)` が設定されているため、`.app` が `.modal`（position: fixed）の新しい包含ブロックとなる。結果として、fixed 配置はビューポートではなく `.app` を基準にしてしまう。

修正案 1: モーダルを `.app` の外側に移動する（DOM 構造の変更）。
```html
<div class="app" style="transform: scale(1);">
  <div class="content">...</div>
</div>
<div class="modal" style="position: fixed; inset: 0; background: rgba(0,0,0,0.5);">
  モーダル
</div>
```

修正案 2: React Portal や Vue Teleport を使い、モーダルを `<body>` 直下にレンダリングする。
```javascript
// React の例
createPortal(<Modal />, document.body);
```

</details>

---

## 13. FAQ

### Q1: Flexbox で子要素の高さを揃えるにはどうすればよいか

**A:** Flexbox のデフォルトの `align-items: stretch` により、同じ行のフレックスアイテムは自動的に最も高いアイテムに合わせて引き伸ばされる。特別な設定は不要である。

```css
.card-row {
  display: flex;
  gap: 16px;
}

.card {
  flex: 1;
  /* align-items: stretch がデフォルトなので */
  /* カードの高さは自動的に揃う */

  /* 内部のボタンを底辺に固定する場合 */
  display: flex;
  flex-direction: column;
}

.card-body { flex: 1; }
.card-button { margin-top: auto; }
```

ただし、`flex-wrap: wrap` を使用している場合、異なる行のアイテム同士の高さは揃わない。行をまたいだ高さの統一が必要な場合は、CSS Grid の使用を検討する。

```css
/* Grid なら行をまたいだ列の幅が揃う */
.card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 16px;
}
```

### Q2: Grid で auto-fill と auto-fit のどちらを使えばよいか

**A:** 以下の基準で選択する。

- **auto-fill**: アイテム数が少ない場合でもグリッドの列数を維持したい場合。空のセルが可視化されるデザイン（背景色やボーダーのあるグリッド）に適している。
- **auto-fit**: アイテム数が少ない場合に、既存のアイテムを引き伸ばして利用可能空間全体を埋めたい場合。カードリストやギャラリーなど、ほとんどのケースで auto-fit が適切である。

多くの場合、`auto-fit` が直感的な挙動をするため推奨される。

### Q3: なぜ padding-top のパーセンテージは幅基準なのか

**A:** CSS 仕様では、padding および margin のパーセンテージ値はすべて包含ブロックの「幅」を基準に計算される。上下方向であっても同様である。

この設計の理由は、高さ基準にすると循環参照が発生する可能性があるためである。例えば、要素の高さがコンテンツ量に依存し（auto）、そのコンテンツの padding-top が高さの 10% だとすると、高さ → padding → 高さ → ... と循環してしまう。幅はブロック要素では包含ブロックから確定的に決まるため、循環参照が発生しない。

```css
/* この仕様を利用したアスペクト比の維持テクニック */
.aspect-16-9 {
  width: 100%;
  padding-top: 56.25%; /* 9 / 16 * 100 = 56.25% */
  height: 0;
  position: relative;
}

.aspect-16-9 > * {
  position: absolute;
  inset: 0;
}

/* 現在は aspect-ratio プロパティが推奨される */
.modern-aspect {
  aspect-ratio: 16 / 9;
}
```

### Q4: position: sticky が動作しない場合の一般的な原因は何か

**A:** `position: sticky` が動作しない主な原因は以下の通りである。

1. **祖先要素に overflow: hidden / auto / scroll が設定されている**: sticky 要素は最寄りのスクロール可能な祖先の中で動作する。overflow が設定されていると、その要素がスクロールコンテナとなり、sticky の動作範囲がそのコンテナ内に限定される。

2. **sticky 要素の親が高さを持たない**: sticky 要素は親要素の範囲内でのみ「固定」される。親の高さが sticky 要素と同じ場合、スクロールしても動かないように見える。

3. **top / bottom / left / right が未指定**: sticky には必ずスクロール方向の閾値を指定する必要がある。

```css
/* 正しい sticky ヘッダーの実装 */
.sticky-header {
  position: sticky;
  top: 0;        /* 必須: 閾値の指定 */
  z-index: 10;   /* 推奨: 他の要素の上に表示 */
  background: white; /* 推奨: 背景色の設定 */
}

/* 祖先の overflow を確認する */
/* 以下のような祖先があると sticky が機能しない */
.problematic-ancestor {
  overflow: hidden; /* これが原因の場合が多い */
}
```

### Q5: CSS Grid の subgrid とは何か

**A:** `subgrid` は、Grid アイテムが親のグリッドトラックを継承する機能である。入れ子のグリッドコンテンツを親のグリッドラインに正確に揃えることができる。

```css
.parent-grid {
  display: grid;
  grid-template-columns: 1fr 2fr 1fr;
  gap: 16px;
}

.child-spanning {
  grid-column: 1 / -1;  /* 全列にまたがる */
  display: grid;
  grid-template-columns: subgrid;
  /* 親の 3 列のトラック定義をそのまま使用 */
  /* 子の列は親の列と正確にアラインされる */
}
```

subgrid がない場合は、入れ子のグリッドが親のトラック定義を参照できないため、ピクセル値の一致やカスタムプロパティの共有で回避する必要があった。subgrid により、カードリストのヘッダーや本文の位置を行をまたいで正確に揃えるといった表現が可能になる。

### Q6: Flexbox と Grid の使い分けの基準は何か

**A:** Flexbox と Grid の選択は、レイアウトの次元性と柔軟性の要求で決まる。

**Flexbox を選択すべき場合:**
- **1次元レイアウト**: 単一の行または列に沿ってアイテムを配置する場合（ナビゲーションバー、ツールバー、カードの内部レイアウト）
- **コンテンツ主導のサイズ**: アイテムのサイズがコンテンツ量に応じて自動調整されるべき場合
- **動的な並び**: アイテム数が動的に変化し、自動的に折り返したい場合（タグリスト、ボタングループ）

**Grid を選択すべき場合:**
- **2次元レイアウト**: 行と列の両方を同時に制御する必要がある場合（ページ全体のレイアウト、複雑なカードグリッド）
- **厳密な配置**: アイテムを特定のグリッドライン上に配置する必要がある場合
- **行をまたいだ整列**: 異なる行の要素を列方向で正確に揃える必要がある場合

**両方を組み合わせる場合:**
多くの実践的なレイアウトでは、Grid で大枠のレイアウトを定義し、各グリッドセル内部で Flexbox を使用するのが最も効果的である。

```css
/* Grid でページ全体の構造を定義 */
.page-layout {
  display: grid;
  grid-template-areas:
    "header header"
    "sidebar main"
    "footer footer";
  grid-template-columns: 250px 1fr;
  gap: 20px;
}

/* Flexbox でヘッダー内部のナビゲーションを配置 */
.header-nav {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 16px;
}
```

### Q7: レイアウトスラッシングの検出と対策方法は何か

**A:** レイアウトスラッシング（Layout Thrashing）は、JavaScriptによるDOMの読み取りと書き込みが交互に実行されることで、ブラウザが強制的に何度もレイアウト計算をやり直す現象である。

**検出方法:**

1. **Chrome DevTools Performance タブ**: 紫色の「Recalculate Style」と「Layout」のブロックが頻繁に出現する
2. **警告メッセージ**: "Forced reflow is a likely performance bottleneck" が Console に表示される
3. **パフォーマンス計測**: 同じ操作が他のブラウザやデバイスと比較して異常に遅い

**典型的なアンチパターン:**

```javascript
// ❌ レイアウトスラッシングを引き起こす
for (let i = 0; i < elements.length; i++) {
  const height = elements[i].offsetHeight; // 読み取り → Layout 発生
  elements[i].style.marginTop = height + 10 + 'px'; // 書き込み → 次の読み取りで再計算
}
```

**対策:**

1. **読み取りと書き込みを分離する:**

```javascript
// ✅ 読み取りをまとめて実行
const heights = elements.map(el => el.offsetHeight);

// ✅ 書き込みをまとめて実行
elements.forEach((el, i) => {
  el.style.marginTop = heights[i] + 10 + 'px';
});
```

2. **requestAnimationFrame を使用する:**

```javascript
// ✅ 読み取りと書き込みをフレームで分離
requestAnimationFrame(() => {
  const height = element.offsetHeight;
  requestAnimationFrame(() => {
    element.style.marginTop = height + 10 + 'px';
  });
});
```

3. **FastDOM などのライブラリを使用する:**

```javascript
// ✅ FastDOM がバッチ処理を自動化
fastdom.measure(() => {
  const height = element.offsetHeight;
  fastdom.mutate(() => {
    element.style.marginTop = height + 10 + 'px';
  });
});
```

4. **CSS で解決できる場合は CSS を優先する:**

```css
/* ✅ JavaScript を使わずに CSS で解決 */
.element {
  margin-top: calc(var(--element-height) + 10px);
}
```

### Q8: CSS Containment（contain プロパティ）の効果と使い方は何か

**A:** CSS Containment は、要素が文書の他の部分に与える影響を制限することで、ブラウザがレイアウト計算やペイント処理を最適化できるようにする機能である。

**contain プロパティの値:**

```css
/* レイアウトの影響範囲を限定 */
.container {
  contain: layout;
  /* この要素内のレイアウト変更は、外部に影響しない */
  /* ブラウザは外部の再計算をスキップできる */
}

/* ペイントの影響範囲を限定 */
.container {
  contain: paint;
  /* 子要素は親のボックス外に描画されない */
  /* overflow: hidden に似ているが、より効率的 */
}

/* サイズ計算を独立させる */
.container {
  contain: size;
  /* 子要素のサイズが親のサイズに影響しない */
  /* 明示的な width/height が必要 */
}

/* スタイル計算を限定（カウンターなど） */
.container {
  contain: style;
  /* CSS カウンターが外部に影響しない */
}

/* すべての containment を適用 */
.container {
  contain: strict; /* size layout paint style と同等 */
}

/* size 以外のすべてを適用（最も実用的） */
.container {
  contain: content; /* layout paint style と同等 */
}
```

**実用的な使用例:**

```css
/* 1. 大量のカードリスト（仮想スクロール） */
.card-item {
  contain: content;
  /* 各カードの変更が他のカードに影響しない */
  /* スクロールパフォーマンスが大幅に向上 */
}

/* 2. 独立したウィジェット */
.widget {
  contain: layout style paint;
  /* ウィジェット内部の変更が外部に影響しない */
}

/* 3. オフスクリーンレンダリングの最適化 */
.offscreen-content {
  content-visibility: auto;
  contain-intrinsic-size: 500px; /* 推定サイズ */
  /* 画面外のコンテンツはレンダリングされない */
}
```

**効果:**

- **レイアウト計算の削減**: 変更の影響範囲が限定されるため、ブラウザは不要な再計算をスキップできる
- **ペイント処理の最適化**: 描画領域が明確になるため、レイヤー分割が効率化される
- **メモリ使用量の削減**: 画面外のコンテンツをスキップできる（content-visibility と組み合わせた場合）

**注意点:**

- `contain: size` を使用する場合は、明示的な寸法指定が必須である
- 過度な使用は逆効果になる場合がある。パフォーマンス計測を行って効果を確認する
- `content-visibility: auto` と組み合わせることで、さらに大きな効果が得られる

---

## 14. 用語集

| 用語 | 説明 |
|------|------|
| Box Model | すべての HTML 要素を content、padding、border、margin の 4 領域で構成されるボックスとして扱うモデル |
| BFC | Block Formatting Context。ブロック要素の独立したレイアウトコンテキスト |
| IFC | Inline Formatting Context。インライン要素のレイアウトコンテキスト |
| Flex Container | `display: flex` が設定された要素。子要素は Flex アイテムになる |
| Flex Item | Flex コンテナの直接の子要素 |
| Main Axis | Flexbox の主軸。flex-direction により方向が決まる |
| Cross Axis | Flexbox の交差軸。主軸に垂直な方向 |
| Grid Track | Grid の 1 行または 1 列 |
| Grid Line | Grid のトラック間の境界線。番号で参照できる |
| Grid Area | Grid 内の矩形領域。名前を付けて参照できる |
| fr 単位 | Grid の利用可能空間を分数比率で分配する単位 |
| Stacking Context | z-index による重ね合わせの独立した評価コンテキスト |
| Containing Block | パーセンテージ値や absolute 配置の基準となる矩形領域 |
| Layout Thrashing | DOM の読み取りと書き込みの交互実行によるパフォーマンス低下 |
| Reflow | レイアウト計算のやり直し。DOM 変更によってトリガーされる |
| subgrid | Grid アイテムが親のグリッドトラック定義を継承する機能 |
| Normal Flow | CSS のデフォルトレイアウトモード。ブロック要素は縦積み、インライン要素は横並び |
| content-visibility | 画面外の要素のレンダリングを遅延させるプロパティ |

---


## FAQ

### Q1: このトピックを学ぶ上で最も重要なポイントは何ですか？

実践的な経験を積むことが最も重要です。理論だけでなく、実際にコードを書いて動作を確認することで理解が深まります。

### Q2: 初心者がよく陥る間違いは何ですか？

基礎を飛ばして応用に進むことです。このガイドで説明している基本概念をしっかり理解してから、次のステップに進むことをお勧めします。

### Q3: 実務ではどのように活用されていますか？

このトピックの知識は、日常的な開発業務で頻繁に活用されます。特にコードレビューやアーキテクチャ設計の際に重要になります。

---

## まとめ

| 概念 | 核心ポイント | 注意点 |
|------|-------------|--------|
| Box Model | border-box を全要素に適用する | margin は相殺される。padding/margin の % は幅基準 |
| Normal Flow | ブロックは縦積み、インラインは横並び | マージン相殺の条件を正確に把握する |
| BFC | display: flow-root で生成するのが最も明示的 | float の包含、マージン相殺の遮断に有効 |
| Flexbox | 1 次元レイアウト、flex: 1 で均等分配 | min-width: 0 を忘れるとオーバーフローする |
| Grid | 2 次元レイアウト、fr 単位で柔軟な分配 | 1fr は minmax(auto, 1fr) の省略形 |
| Positioning | fixed は transform のある祖先に影響される | sticky は overflow のある祖先で動作しない場合がある |
| パフォーマンス | contain で影響範囲を限定、読み書き分離を徹底 | Layout Thrashing は重大なパフォーマンス問題 |

---

## 次に読むべきガイド

- [Paint と Compositing](./02-paint-and-compositing.md) -- Paint と Compositing のパイプラインを理解し、レンダリングの後半工程を学ぶ
- CSS Animations と Transitions -- アニメーションとトランジションのパフォーマンス最適化を学ぶ

---

## 15. 参考文献

1. W3C. "CSS Box Model Module Level 3." W3C Working Draft. https://www.w3.org/TR/css-box-3/
2. W3C. "CSS Flexible Box Layout Module Level 1." W3C Candidate Recommendation. https://www.w3.org/TR/css-flexbox-1/
3. W3C. "CSS Grid Layout Module Level 2." W3C Candidate Recommendation. https://www.w3.org/TR/css-grid-2/
4. W3C. "CSS Containment Module Level 2." W3C Working Draft. https://www.w3.org/TR/css-contain-2/
5. W3C. "CSS Positioned Layout Module Level 3." W3C Working Draft. https://www.w3.org/TR/css-position-3/
6. MDN Web Docs. "CSS Layout." Mozilla Developer Network. https://developer.mozilla.org/en-US/docs/Learn/CSS/CSS_layout
7. Chromium Blog. "The Chromium Chronicle: Layout Performance." Google Chromium Team. https://developer.chrome.com/blog/
8. web.dev. "Avoid Large, Complex Layouts and Layout Thrashing." Google Chrome Developers. https://web.dev/avoid-large-complex-layouts-and-layout-thrashing/
9. Jen Simmons. "Designing Intrinsic Layouts." 2018. https://www.youtube.com/watch?v=AMPKmh98XLY
10. Rachel Andrew. "The New CSS Layout." A Book Apart, 2017.
11. Paul Irish. "What Forces Layout / Reflow." GitHub Gist. https://gist.github.com/paulirish/5d52fb081b3570c81e3a
12. web.dev. "content-visibility: the new CSS property that boosts your rendering performance." Google Chrome Developers. https://web.dev/content-visibility/
8. Google Developers. "Rendering Performance." Web Fundamentals. https://web.dev/rendering-performance/

### 追加 FAQ

### Q4: Flexbox と Grid を同一コンポーネント内で併用してもよいですか?
はい、Flexbox と Grid の併用は一般的なパターンです。例えば、ページ全体のレイアウト（ヘッダー・サイドバー・メイン・フッター）には Grid を使い、ナビゲーションバーやカードの内部レイアウトには Flexbox を使うのが典型的です。Grid は2次元配置、Flexbox は1次元配置に強いため、それぞれの特性を活かした使い分けが推奨されます。ネストしても性能上の問題はほとんどありません。

### Q5: position: sticky が効かない場合のよくある原因は何ですか?
最も多い原因は、sticky要素の祖先に `overflow: hidden`、`overflow: auto`、または `overflow: scroll` が設定されている場合です。sticky はスクロールコンテナを基準に動作するため、意図しない祖先がスクロールコンテナになっていると正しく機能しません。また、sticky 要素に `top`、`bottom`、`left`、`right` のいずれかの閾値が指定されていない場合も動作しません。DevTools の Computed パネルで `position` が `sticky` であることを確認し、祖先要素の `overflow` 値をチェックしてください。

### Q6: CSS Grid の subgrid はどのような場面で有効ですか?
subgrid は、親グリッドのトラック定義（行や列の幅・高さ）を子グリッドが継承できる機能です。カードリストで各カードのヘッダー・本文・フッターの高さを全カード間で揃えたい場合に特に有効です。subgrid がない場合は固定高さを指定するか JavaScript で高さを同期する必要がありましたが、subgrid により純粋な CSS で実現できます。2024年時点で主要ブラウザ（Chrome、Firefox、Safari）で対応済みです。

### 追加参考文献

13. Ahmad Shadeed. "Debugging CSS Grid and Flexbox Layouts." 2023. https://ishadeed.com/article/css-grid-debugging/
14. web.dev. "CSS subgrid." Google Chrome Developers. https://web.dev/articles/css-subgrid
15. W3C. "CSS Display Module Level 3." W3C Candidate Recommendation. https://www.w3.org/TR/css-display-3/

---

## 参考文献

- [MDN Web Docs](https://developer.mozilla.org/) - Web技術のリファレンス
- [Wikipedia](https://ja.wikipedia.org/) - 技術概念の概要
