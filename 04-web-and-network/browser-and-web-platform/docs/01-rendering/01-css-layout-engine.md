# CSSレイアウトエンジン

> CSSレイアウトエンジンは要素の位置とサイズを決定する。Box Model、Normal Flow、Flexbox、Grid、positionの内部動作を理解し、意図通りのレイアウトを効率的に構築する。

## この章で学ぶこと

- [ ] Box Modelとレイアウトモードの違いを理解する
- [ ] FlexboxとGridの内部動作を把握する
- [ ] BFC（Block Formatting Context）の概念を学ぶ

---

## 1. Box Model

```
CSS Box Model:

  ┌─────────────────────────────────────────┐
  │ margin                                  │
  │  ┌──────────────────────────────────┐   │
  │  │ border                           │   │
  │  │  ┌───────────────────────────┐   │   │
  │  │  │ padding                   │   │   │
  │  │  │  ┌────────────────────┐   │   │   │
  │  │  │  │ content            │   │   │   │
  │  │  │  │ (width × height)  │   │   │   │
  │  │  │  └────────────────────┘   │   │   │
  │  │  └───────────────────────────┘   │   │
  │  └──────────────────────────────────┘   │
  └─────────────────────────────────────────┘

box-sizing:
  content-box（デフォルト）:
    width = コンテンツ幅のみ
    実際の幅 = width + padding + border

  border-box（推奨）:
    width = コンテンツ + padding + border
    実際の幅 = width（指定した値のまま）

  *, *::before, *::after {
    box-sizing: border-box;  /* 全要素に適用（推奨） */
  }
```

---

## 2. Normal Flow

```
Normal Flow = CSS のデフォルトレイアウト

  ブロック要素（div, p, h1等）:
  → 縦に積まれる
  → 親の幅いっぱいに広がる
  ┌──────────────────────┐
  │ <div>                │
  ├──────────────────────┤
  │ <p>                  │
  ├──────────────────────┤
  │ <div>                │
  └──────────────────────┘

  インライン要素（span, a, strong等）:
  → 横に並ぶ
  → コンテンツの幅だけ
  │<span>│<a>│<strong>│

マージンの相殺（Margin Collapsing）:
  上下のマージンが重なると、大きい方だけ適用

  ┌──────────┐
  │ div      │ margin-bottom: 20px
  └──────────┘
                ← 20px（30pxではない）
  ┌──────────┐
  │ div      │ margin-top: 10px
  └──────────┘

  相殺が起きない場合:
  → Flexbox / Grid の子要素
  → floatの要素
  → position: absolute / fixed
  → BFCを生成する要素（overflow: hidden等）
```

---

## 3. Flexbox

```
Flexbox の内部動作:

  コンテナ（display: flex）:
  ┌──────────────────────────────────────┐
  │ main axis（主軸）→                    │
  │ ┌─────┐ ┌─────┐ ┌─────┐             │
  │ │item1│ │item2│ │item3│             │ ↓ cross axis
  │ └─────┘ └─────┘ └─────┘             │   （交差軸）
  └──────────────────────────────────────┘

レイアウト計算の流れ:
  1. 利用可能空間の計算
  2. アイテムのベースサイズ決定（flex-basis）
  3. 余白の計算
  4. flex-grow で余白を分配
  5. flex-shrink で縮小を分配
  6. align-items / justify-content の適用

flex の省略記法:
  flex: 1;          → flex: 1 1 0%;    （均等分配）
  flex: auto;       → flex: 1 1 auto;  （コンテンツベース）
  flex: none;       → flex: 0 0 auto;  （固定サイズ）
  flex: 0 0 200px;  → 固定幅200px

よく使うパターン:
  /* 中央寄せ */
  display: flex;
  justify-content: center;
  align-items: center;

  /* サイドバーレイアウト */
  display: flex;
  .sidebar { flex: 0 0 250px; }
  .content { flex: 1; }

  /* 均等分配 */
  display: flex;
  gap: 16px;
  .item { flex: 1; }

  /* 最終要素を右端に */
  display: flex;
  .last { margin-left: auto; }
```

---

## 4. Grid

```
CSS Grid の内部動作:

  display: grid;
  grid-template-columns: 200px 1fr 1fr;
  grid-template-rows: auto 1fr auto;
  gap: 16px;

  ┌────────┬──────────┬──────────┐
  │ header │ header   │ header   │ auto
  ├────────┼──────────┼──────────┤
  │ sidebar│ content  │ content  │ 1fr
  │ 200px  │          │          │
  ├────────┼──────────┼──────────┤
  │ footer │ footer   │ footer   │ auto
  └────────┴──────────┴──────────┘
   200px     1fr        1fr

fr 単位:
  → 利用可能空間のフラクション（割合）
  → 1fr 1fr 1fr = 3等分
  → 200px 1fr 1fr = 200px固定 + 残りを2等分

grid-template-areas:
  grid-template-areas:
    "header header header"
    "sidebar content content"
    "footer footer footer";

  .header  { grid-area: header; }
  .sidebar { grid-area: sidebar; }
  .content { grid-area: content; }
  .footer  { grid-area: footer; }

auto-fill / auto-fit:
  /* レスポンシブカード */
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  → 250px以上のカードが入るだけ配置、残りは均等分配

  auto-fill: 空のトラックも保持
  auto-fit:  空のトラックを折りたたむ（要素を引き伸ばす）

Flexbox vs Grid:
  Flexbox: 1次元（行 or 列）
  Grid:    2次元（行 と 列）

  使い分け:
  Flexbox: ナビゲーション、ツールバー、カード内のレイアウト
  Grid:    ページ全体、カードグリッド、ダッシュボード
```

---

## 5. BFC（Block Formatting Context）

```
BFC = ブロック要素の独立したレイアウトコンテキスト

  BFC を生成する条件:
  → html ルート要素
  → float の要素
  → position: absolute / fixed
  → display: inline-block / flow-root / flex / grid
  → overflow: hidden / auto / scroll
  → contain: layout / content / strict

  BFC の効果:
  ① マージンの相殺を防ぐ
  ② フロートを含む（clearfix不要）
  ③ レイアウトの独立性

  .container {
    display: flow-root;  /* BFC生成の最もクリーンな方法 */
  }
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| Box Model | border-box推奨、margin相殺に注意 |
| Flexbox | 1次元レイアウト、flex: 1で均等分配 |
| Grid | 2次元レイアウト、fr単位、auto-fill |
| BFC | 独立したレイアウトコンテキスト |

---

## 次に読むべきガイド
→ [[02-paint-and-compositing.md]] — Paint と Compositing

---

## 参考文献
1. MDN Web Docs. "CSS Layout." Mozilla, 2024.
2. Jen Simmons. "Designing Intrinsic Layouts." 2018.
