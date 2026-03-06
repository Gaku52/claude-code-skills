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
