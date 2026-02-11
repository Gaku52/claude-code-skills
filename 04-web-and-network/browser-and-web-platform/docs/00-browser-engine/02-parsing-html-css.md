# HTML/CSSパーシング

> HTMLとCSSのパース処理はレンダリングの出発点。HTMLパーサーによるDOM構築、CSSパーサーによるCSSOM構築、そしてレンダーツリーの生成プロセスを深く理解する。

## この章で学ぶこと

- [ ] HTMLパーサーのトークン化とツリー構築を理解する
- [ ] CSSOMの構築プロセスを把握する
- [ ] DOMとCSSOMの統合（レンダーツリー）を学ぶ

---

## 1. HTMLパーシング

```
HTMLパーサーの処理フロー:

  HTML文字列
    ↓
  トークナイザ（字句解析）
    ↓ トークン列
  ツリービルダー（構文解析）
    ↓
  DOM（Document Object Model）

トークンの種類:
  DOCTYPE:    <!DOCTYPE html>
  開始タグ:   <div class="container">
  終了タグ:   </div>
  テキスト:   "Hello World"
  コメント:   <!-- comment -->
  EOF:        ファイル終端

例:
  <html>
  <body>
    <h1>Hello</h1>
    <p>World</p>
  </body>
  </html>

  トークン列:
  [StartTag: html]
  [StartTag: body]
  [StartTag: h1]
  [Text: "Hello"]
  [EndTag: h1]
  [StartTag: p]
  [Text: "World"]
  [EndTag: p]
  [EndTag: body]
  [EndTag: html]

  DOM ツリー:
  Document
  └── html
      └── body
          ├── h1
          │   └── "Hello"
          └── p
              └── "World"

HTMLパーサーの特殊性:
  → HTML はエラーに寛容（不正なHTMLも処理する）
  → 閉じタグの省略を自動補完
  → <table>内の不正なテキストを外に移動（foster parenting）
  → 仕様: HTML Living Standard の「Parsing」セクション
```

---

## 2. DOMの構造

```
DOM（Document Object Model）:
  → HTMLをオブジェクトのツリー構造で表現
  → JavaScriptからアクセス・操作可能

ノードの種類:
  Document:    ルートノード
  Element:     HTML要素（div, p, span等）
  Text:        テキストコンテンツ
  Comment:     HTMLコメント
  Attribute:   要素の属性

DOMノードのプロパティ:
  node.nodeName        → タグ名（"DIV", "#text"等）
  node.nodeType        → 1(Element), 3(Text), 8(Comment)
  node.parentNode      → 親ノード
  node.childNodes      → 子ノード一覧
  node.firstChild      → 最初の子
  node.nextSibling     → 次の兄弟

Element固有:
  element.id           → id属性
  element.className    → class属性
  element.attributes   → 全属性
  element.innerHTML    → 内部HTML（文字列）
  element.children     → 子Element のみ（Textを除く）

DOMのメモリ上の表現:
  → C++オブジェクトのツリー（Blink/WebKit内部）
  → JSからのアクセスはV8のバインディング経由
  → DOM操作のコスト = C++オブジェクト操作 + JSバインディング
```

---

## 3. CSSパーシングとCSSOM

```
CSSパーサーの処理:

  CSS文字列
    ↓
  トークナイザ
    ↓ トークン列
  パーサー
    ↓
  CSSOM（CSS Object Model）

CSSOM ツリー:
  StyleSheetList
  └── StyleSheet
      └── CSSRuleList
          ├── CSSStyleRule: "body"
          │   └── color: black; font-size: 16px;
          ├── CSSStyleRule: ".container"
          │   └── max-width: 1200px; margin: 0 auto;
          └── CSSMediaRule: "@media (max-width: 768px)"
              └── CSSStyleRule: ".container"
                  └── max-width: 100%;

スタイルの計算（Computed Style）:
  各DOM要素に対して最終的なスタイルを決定

  計算の流れ:
  1. ブラウザデフォルトスタイル（User Agent Stylesheet）
  2. ユーザースタイル
  3. 作者スタイル（開発者のCSS）
  4. !important の処理
  5. 詳細度（Specificity）の計算
  6. カスケード順序の適用
  7. 継承プロパティの解決
  8. 相対値の解決（em → px, % → px）

詳細度の計算:
  ┌──────────────────────┬───────────────┐
  │ セレクタ              │ 詳細度        │
  ├──────────────────────┼───────────────┤
  │ *                    │ 0-0-0         │
  │ div                  │ 0-0-1         │
  │ .class               │ 0-1-0         │
  │ #id                  │ 1-0-0         │
  │ div.class            │ 0-1-1         │
  │ #id .class div       │ 1-1-1         │
  │ style=""（インライン）│ 最高          │
  │ !important           │ 全てに勝つ    │
  └──────────────────────┴───────────────┘
```

---

## 4. レンダーツリー

```
DOM + CSSOM → レンダーツリー:

  DOM:                    CSSOM:
  html                    body { color: black }
  └── body                .hidden { display: none }
      ├── h1.title        h1 { font-size: 2em }
      ├── p               p { margin: 1em 0 }
      ├── div.hidden
      └── img

  レンダーツリー:
  RenderView
  └── RenderBody (color: black)
      ├── RenderBlock h1 (font-size: 32px)
      ├── RenderBlock p (margin: 16px 0)
      └── RenderImage img

  注意: display: none の要素はレンダーツリーに含まれない
  → .hidden は DOM にはあるがレンダーツリーにはない
  → visibility: hidden は含まれる（スペースを占める）

レンダーツリー構築のコスト:
  → 全DOM要素 × 全CSSルール のマッチング
  → CSSセレクタは右から左に評価
  → .container > .item > a → まず a を全て見つける → 親が.item → 祖父が.container

CSSパフォーマンスのヒント:
  ✓ セレクタをシンプルに（.button ○, div.container > ul > li > a ✗）
  ✓ * セレクタを避ける
  ✓ ネストを深くしない（BEM等の命名規則推奨）
```

---

## 5. Incremental Parsing

```
増分パーシング:
  → HTMLは受信したバイトから順次パース（ストリーミング）
  → 全HTML受信を待たない

  ネットワーク: [chunk1][chunk2][chunk3]...
  パーサー:     [パース1][パース2][パース3]...
  → 受信しながらDOM構築

  利点:
  → First Paint が早まる
  → ユーザーは早くコンテンツを見られる

  パース中断のケース:
  → <script>（syncスクリプト）: パース停止 → DL → 実行 → 再開
  → document.write(): パーサーの状態を変更（非推奨）

Speculative Parsing（投機的パース）:
  → メインパーサーがブロック中も別のパーサーが先読み
  → <link>, <script>, <img> を事前発見
  → ネットワークリクエストを先行開始
  → ただしDOM操作には反映しない（リソース発見のみ）
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| HTMLパース | トークン化 → ツリー構築 → DOM |
| CSSOM | CSSルール → 詳細度計算 → Computed Style |
| レンダーツリー | DOM + CSSOM の統合、display:noneは除外 |
| 増分パース | 受信しながら順次パース |

---

## 次に読むべきガイド
→ [[03-browser-security-model.md]] — ブラウザセキュリティモデル

---

## 参考文献
1. HTML Living Standard. "Parsing HTML documents." WHATWG, 2024.
2. Grigore, T. "How Browsers Work." web.dev, 2011.
