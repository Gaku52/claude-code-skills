# HTML/CSSパーシング

> HTMLとCSSのパース処理はレンダリングの出発点。HTMLパーサーによるDOM構築、CSSパーサーによるCSSOM構築、そしてレンダーツリーの生成プロセスを深く理解する。ブラウザがバイト列を受け取ってから画面に描画可能なデータ構造を生成するまでの全工程を、仕様レベルで解説する。

## この章で学ぶこと

- [ ] HTMLパーサーのトークン化とツリー構築を仕様レベルで理解する
- [ ] CSS字句解析・構文解析のアルゴリズムを把握する
- [ ] CSSOMの構築プロセスとスタイル計算の全体像を学ぶ
- [ ] DOMとCSSOMの統合（レンダーツリー）を深く理解する
- [ ] エラー回復・投機的パースなどブラウザ固有の最適化を知る
- [ ] パフォーマンスに影響するアンチパターンを見抜けるようになる

---

## 1. パーシングの全体像

ブラウザがHTMLドキュメントを受信してから描画可能な状態に到達するまでには、複数のパース工程が直列・並列に動作する。まず全体像を俯瞰する。

### 1.1 バイト列からレンダーツリーまでの処理フロー

```
  ネットワークからバイト列を受信
       │
       ▼
  ┌──────────────────────────────────────────────────────┐
  │  1. 文字エンコーディング検出                          │
  │     HTTP Content-Type ヘッダ                         │
  │     BOM (Byte Order Mark)                            │
  │     <meta charset="UTF-8">                           │
  │     → バイト列を Unicode 文字列に変換                 │
  └──────────────┬───────────────────────────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────────────────────────┐
  │  2. HTML トークナイザ (Tokenizer / 字句解析)          │
  │     文字列 → トークン列                              │
  │     DOCTYPE, StartTag, EndTag, Comment, Character,   │
  │     EndOfFile                                        │
  └──────────────┬───────────────────────────────────────┘
                 │ トークンを1つずつ発行
                 ▼
  ┌──────────────────────────────────────────────────────┐
  │  3. HTML ツリービルダ (Tree Construction / 構文解析)   │
  │     トークン列 → DOM ツリー                          │
  │     挿入モード (Insertion Mode) による状態遷移        │
  │     エラー回復・暗黙の要素補完                        │
  └──────────────┬───────────────────────────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────────────────────────┐
  │  4. DOM (Document Object Model)                      │
  │     メモリ上のオブジェクトツリー                      │
  │     JavaScript からアクセス可能                      │
  └──────────────┬───────────────────────────────────────┘
                 │                    ┌───────────────────────────────────┐
                 │                    │  5. CSS パーサー                   │
                 │                    │     CSS文字列 → トークン列         │
                 │                    │     トークン列 → CSSルール群       │
                 │                    │     → CSSOM 構築                  │
                 │                    └──────────┬────────────────────────┘
                 │                               │
                 ▼                               ▼
  ┌──────────────────────────────────────────────────────┐
  │  6. スタイル計算 (Style Resolution)                   │
  │     DOM の各ノード × CSSOM の全ルールをマッチング     │
  │     → Computed Style の決定                          │
  └──────────────┬───────────────────────────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────────────────────────┐
  │  7. レンダーツリー (Render Tree / Layout Tree)        │
  │     表示対象の要素 + 確定スタイル                     │
  │     display: none は除外                             │
  └──────────────────────────────────────────────────────┘
```

この処理フローにおいて、HTML パースと CSS パースは部分的に並列で進行する。HTML パーサーが `<link rel="stylesheet">` や `<style>` タグを検出すると、CSS パーサーが起動して CSSOM の構築を開始する。ただし CSS の読み込み完了を待たずに HTML のパース自体は継続される点が重要である。

### 1.2 文字エンコーディング検出の詳細

ブラウザがバイト列を文字列として解釈するためには、まず文字エンコーディングを確定する必要がある。HTML Living Standard では以下の優先順位でエンコーディングを決定する。

```
エンコーディング決定の優先順位:

  1. BOM (Byte Order Mark)
     UTF-8:    EF BB BF
     UTF-16 BE: FE FF
     UTF-16 LE: FF FE
     → BOMが存在すれば最優先で採用

  2. HTTP Content-Type ヘッダ
     Content-Type: text/html; charset=UTF-8
     → サーバーが明示的に指定

  3. <meta> タグによる宣言
     <meta charset="UTF-8">
     <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
     → HTML の先頭 1024 バイト以内で検出する必要がある

  4. Prescan アルゴリズム
     → パーサーが HTML の先頭部分をスキャンし、
       meta タグの charset 属性を探す
     → 1024 バイトまでしかスキャンしない

  5. 親ドキュメントのエンコーディング
     → iframe の場合、親のエンコーディングを参考にする

  6. ブラウザのデフォルト
     → 地域設定に応じたフォールバック
     → 多くのモダンブラウザでは UTF-8 がデフォルト
```

**コード例 1: エンコーディング指定のベストプラクティス**

```html
<!DOCTYPE html>
<html lang="ja">
<head>
  <!-- charset宣言は <head> 内の最初の要素として配置する -->
  <!-- 先頭1024バイト以内に含まれることが保証されるため -->
  <meta charset="UTF-8">
  <title>エンコーディング指定の例</title>
</head>
<body>
  <p>日本語を含むページでは UTF-8 を明示的に宣言する</p>
</body>
</html>
```

サーバー側のHTTPヘッダでもエンコーディングを指定するのが理想的である。

```
HTTP/1.1 200 OK
Content-Type: text/html; charset=UTF-8
```

もし HTTP ヘッダと `<meta>` タグで異なるエンコーディングが指定された場合、HTTP ヘッダが優先される（BOM がある場合を除く）。

---

## 2. HTMLトークナイザ（字句解析）

### 2.1 トークナイザのステートマシン

HTML トークナイザは**有限状態機械 (Finite State Machine)** として実装される。HTML Living Standard では 80 以上の状態が定義されており、入力文字に応じて状態遷移しながらトークンを生成する。

```
HTMLトークナイザの主要な状態遷移図:

  ┌─────────────┐    '<'     ┌──────────────┐
  │  Data State │──────────→│  Tag Open    │
  │  (初期状態)  │           │  State       │
  └──────┬──────┘           └──────┬───────┘
         │                         │
    文字  │ トークン生成              │ 文字種による分岐
         ▼                         │
  ┌─────────────┐           ┌──────┴───────┐
  │ Character   │           │ 英字         │
  │ Token 発行  │           │  → Tag Name  │
  └─────────────┘           │    State     │
                            │              │
                            │ '/'          │
                            │  → End Tag   │
                            │    Open      │
                            │              │
                            │ '!'          │
                            │  → Markup    │
                            │    Decl.     │
                            │    Open      │
                            │              │
                            │ '?'          │
                            │  → Bogus     │
                            │    Comment   │
                            └──────────────┘

  Tag Name State での遷移:
  ┌──────────────┐    空白    ┌───────────────────┐
  │  Tag Name    │──────────→│ Before Attribute  │
  │  State       │           │ Name State        │
  └──────┬───────┘           └─────────┬─────────┘
         │                             │
    '>'  │ トークン発行           英字  │
         ▼                             ▼
  ┌─────────────┐           ┌───────────────────┐
  │ Data State  │           │ Attribute Name    │
  │ へ戻る      │           │ State             │
  └─────────────┘           └─────────┬─────────┘
                                      │
                                 '='  │
                                      ▼
                            ┌───────────────────┐
                            │ Before Attr Value │
                            │ State             │
                            └─────────┬─────────┘
                                      │
                              '"' or '│' or 文字
                                      ▼
                            ┌───────────────────┐
                            │ Attribute Value   │
                            │ State             │
                            └───────────────────┘
```

### 2.2 トークンの種類と構造

HTMLトークナイザが生成するトークンは以下の6種類である。

```
┌────────────────┬─────────────────────────────────────────────────┐
│ トークン種別    │ 説明と例                                         │
├────────────────┼─────────────────────────────────────────────────┤
│ DOCTYPE        │ <!DOCTYPE html>                                 │
│                │ 属性: name, publicId, systemId, forceQuirks     │
├────────────────┼─────────────────────────────────────────────────┤
│ StartTag       │ <div class="main" id="content">                │
│                │ 属性: tagName, attributes[], selfClosing        │
├────────────────┼─────────────────────────────────────────────────┤
│ EndTag         │ </div>                                          │
│                │ 属性: tagName                                   │
├────────────────┼─────────────────────────────────────────────────┤
│ Comment        │ <!-- コメント本文 -->                             │
│                │ 属性: data                                      │
├────────────────┼─────────────────────────────────────────────────┤
│ Character      │ テキストノード用の文字                            │
│                │ 属性: data (1文字ずつ or バッファリング)           │
├────────────────┼─────────────────────────────────────────────────┤
│ EndOfFile      │ 入力の終端を示す特殊トークン                      │
│                │ パース完了のシグナル                              │
└────────────────┴─────────────────────────────────────────────────┘
```

**コード例 2: トークン化の具体的な流れ**

以下の HTML 断片がどのようにトークン化されるかを追跡する。

```html
<p class="intro">Hello, <em>world</em>!</p>
```

```
トークン化の過程:

入力文字列: <p class="intro">Hello, <em>world</em>!</p>

位置 0:  '<'    → Data → Tag Open State
位置 1:  'p'    → Tag Open → Tag Name State (tagName = "p")
位置 2:  ' '    → Tag Name → Before Attribute Name State
                   StartTag トークン生成開始 {tagName: "p"}
位置 3:  'c'    → Attribute Name State (attrName = "c")
位置 4:  'l'    → attrName = "cl"
位置 5:  'a'    → attrName = "cla"
位置 6:  's'    → attrName = "clas"
位置 7:  's'    → attrName = "class"
位置 8:  '='    → Before Attribute Value State
位置 9:  '"'    → Attribute Value (Double-Quoted) State
位置 10: 'i'    → attrValue = "i"
...
位置 14: 'o'    → attrValue = "intro"
位置 15: '"'    → After Attribute Value (Quoted) State
位置 16: '>'    → Data State
                   ★ StartTag トークン発行: {tagName: "p", attrs: [{name:"class", value:"intro"}]}

位置 17: 'H'    → Character トークン蓄積
...
位置 23: ' '    → Character トークン蓄積
                   ★ Character トークン発行: "Hello, "

位置 24: '<'    → Data → Tag Open State
位置 25: 'e'    → Tag Name State (tagName = "e")
位置 26: 'm'    → tagName = "em"
位置 27: '>'    → Data State
                   ★ StartTag トークン発行: {tagName: "em", attrs: []}

位置 28: 'w'    → Character トークン蓄積
...
位置 32: 'd'    → Character トークン蓄積
                   ★ Character トークン発行: "world"

位置 33: '<'    → Tag Open State
位置 34: '/'    → End Tag Open State
位置 35: 'e'    → Tag Name State (tagName = "e")
位置 36: 'm'    → tagName = "em"
位置 37: '>'    → Data State
                   ★ EndTag トークン発行: {tagName: "em"}

位置 38: '!'    → Character トークン発行: "!"

位置 39: '<'    → Tag Open State
位置 40: '/'    → End Tag Open State
位置 41: 'p'    → Tag Name State (tagName = "p")
位置 42: '>'    → Data State
                   ★ EndTag トークン発行: {tagName: "p"}

発行されたトークン列:
  [StartTag: p (class="intro")]
  [Character: "Hello, "]
  [StartTag: em]
  [Character: "world"]
  [EndTag: em]
  [Character: "!"]
  [EndTag: p]
```

### 2.3 文字参照 (Character Reference) の処理

トークナイザは文字参照（HTML エンティティ）も処理する。

```
文字参照の種類:

  1. 名前付き文字参照
     &amp;   → &
     &lt;    → <
     &gt;    → >
     &quot;  → "
     &apos;  → '
     &nbsp;  → U+00A0 (Non-Breaking Space)

  2. 10進数文字参照
     &#65;   → A  (ASCIIコード 65)
     &#8364; → €  (Unicodeコードポイント)

  3. 16進数文字参照
     &#x41;  → A
     &#x20AC; → €

処理フロー:
  Data State で '&' を検出
    → Character Reference State へ遷移
    → '#' なら数値参照
    → 英字なら名前付き参照
    → 名前テーブルからマッチ検索
    → 解決した文字を Character トークンとして発行
```

### 2.4 スクリプトタグ内のトークン化

`<script>` タグ内部は通常の HTML とは異なるトークン化ルールが適用される。

```
<script> タグの特殊処理:

  通常の Data State:
    '<' を検出 → Tag Open State → タグとして処理

  Script Data State:
    '<' を検出 → Script Data Less-Than Sign State
    → '</script>' にマッチするかチェック
    → マッチしなければ全てテキストとして扱う

  これにより以下のコードが正しく処理される:

  <script>
    var html = "<div>これはHTMLではなくJSの文字列</div>";
    if (a < b && c > d) { /* < と > はタグではない */ }
  </script>

  注意: </script> はスクリプトの終了を示す
  → スクリプト内で "</script>" という文字列リテラルを
    直接書くと意図しない終了が起きる

  回避策:
  <script>
    // NG: var s = "</script>";
    // OK: var s = "<\/script>";
    // OK: var s = "<" + "/script>";
  </script>
```

---

## 3. HTMLツリービルダ（構文解析・DOM構築）

### 3.1 挿入モード (Insertion Mode) による状態管理

HTML ツリービルダは、トークナイザから受け取ったトークンを DOM ツリーに変換する。ツリービルダもステートマシンとして実装されており、「挿入モード (Insertion Mode)」と呼ばれる状態を持つ。

HTML Living Standard では以下の挿入モードが定義されている。

```
主要な挿入モード一覧:

  ┌─ 初期状態 ─────────────────────────────────┐
  │  initial                                    │
  │    → DOCTYPE トークンを処理                  │
  │    → before html へ遷移                     │
  ├─────────────────────────────────────────────┤
  │  before html                                │
  │    → <html> StartTag を処理                 │
  │    → before head へ遷移                     │
  ├─────────────────────────────────────────────┤
  │  before head                                │
  │    → <head> StartTag を処理                 │
  │    → in head へ遷移                         │
  ├─────────────────────────────────────────────┤
  │  in head                                    │
  │    → <meta>, <title>, <link>, <style>,      │
  │      <script> 等を処理                      │
  │    → </head> で after head へ遷移           │
  ├─────────────────────────────────────────────┤
  │  in head noscript                           │
  │    → <noscript> 内部の処理                   │
  ├─────────────────────────────────────────────┤
  │  after head                                 │
  │    → <body> StartTag を処理                 │
  │    → in body へ遷移                         │
  ├─────────────────────────────────────────────┤
  │  in body                                    │
  │    → 本文中の全要素を処理                    │
  │    → 最も複雑なモード                        │
  ├─────────────────────────────────────────────┤
  │  in table                                   │
  │    → <table> 内部の処理                     │
  │    → foster parenting が発生するモード       │
  ├─────────────────────────────────────────────┤
  │  in row / in cell / in caption              │
  │    → テーブル内の各部位の処理                │
  ├─────────────────────────────────────────────┤
  │  in select                                  │
  │    → <select> 内部の処理                    │
  ├─────────────────────────────────────────────┤
  │  after body                                 │
  │    → </body> 後の処理                       │
  │    → after after body へ遷移                │
  ├─────────────────────────────────────────────┤
  │  in frameset / after frameset               │
  │    → フレームセットの処理（レガシー）         │
  ├─────────────────────────────────────────────┤
  │  after after body                           │
  │    → </html> 後の処理                       │
  │    → EOF で完了                             │
  └─────────────────────────────────────────────┘
```

### 3.2 オープン要素スタック (Stack of Open Elements)

ツリービルダは「オープン要素スタック」を管理する。このスタックはネスト構造を追跡するためのデータ構造である。

**コード例 3: オープン要素スタックの変化を追跡する**

```html
<!DOCTYPE html>
<html>
<head><title>Test</title></head>
<body>
  <div>
    <p>Hello <strong>World</strong></p>
  </div>
</body>
</html>
```

```
オープン要素スタックの変化:

  トークン                  スタック状態           挿入モード
  ─────────────────────────────────────────────────────────────
  DOCTYPE html              []                    initial
                            []                    before html
  <html>                    [html]                before head
  <head>                    [html, head]          in head
  <title>                   [html, head, title]   text
  "Test"                    [html, head, title]   text
  </title>                  [html, head]          in head
  </head>                   [html]                after head
  <body>                    [html, body]          in body
  <div>                     [html, body, div]     in body
  <p>                       [html, body, div, p]  in body
  "Hello "                  [html, body, div, p]  in body
  <strong>                  [html, body, div, p,  in body
                             strong]
  "World"                   [html, body, div, p,  in body
                             strong]
  </strong>                 [html, body, div, p]  in body
  </p>                      [html, body, div]     in body
  </div>                    [html, body]          in body
  </body>                   [html]                after body
  </html>                   []                    after after body
  EOF                       (完了)

  生成される DOM ツリー:
  Document
  ├── DOCTYPE: html
  └── html
      ├── head
      │   └── title
      │       └── "Test"
      └── body
          └── div
              └── p
                  ├── "Hello "
                  └── strong
                      └── "World"
```

### 3.3 エラー回復と暗黙の要素補完

HTML パーサーの最大の特徴は、**不正な HTML に対してもエラーを投げずに回復する**ことである。これは HTML Living Standard で詳細に仕様化されている。

**コード例 4: エラー回復の動作例**

```html
<!-- 入力 (不正なHTML) -->
<p>First
<p>Second
<div><span></div>
<table><td>Cell</table>
```

```
パーサーによるエラー回復処理:

  入力: <p>First<p>Second
  ──────────────────────
  1. <p>First を処理: p 要素を生成、"First" テキストを追加
  2. 2つ目の <p> を検出:
     → 現在の p はまだ閉じられていない
     → 仕様: "in body" モードで <p> の StartTag を受信し、
       スタック上に p がある場合は暗黙的に閉じる
     → 暗黙の </p> を挿入
     → 新しい p 要素を生成
  3. 結果:
     <p>First</p>        ← 暗黙の終了タグ
     <p>Second</p>       ← 暗黙の終了タグ

  入力: <div><span></div>
  ──────────────────────
  1. <div> を生成しスタックに積む
  2. <span> を生成しスタックに積む: [html, body, div, span]
  3. </div> を受信:
     → スタック上の span は閉じられていない
     → 仕様: </div> はスタックを div まで巻き戻す
     → span を暗黙的に閉じる
  4. 結果:
     <div><span></span></div>

  入力: <table><td>Cell</table>
  ──────────────────────────────
  1. <table> を生成、挿入モード "in table" へ
  2. <td> を受信:
     → <td> は <tr> 内にあるべき
     → 仕様: 暗黙の <tbody> と <tr> を生成
  3. 結果:
     <table>
       <tbody>          ← 暗黙生成
         <tr>           ← 暗黙生成
           <td>Cell</td>
         </tr>
       </tbody>
     </table>
```

### 3.4 Foster Parenting（里親処理）

テーブル要素内に不正な要素やテキストが出現した場合、ブラウザは「foster parenting」と呼ばれる特殊な処理を行う。

```
Foster Parenting の動作:

  入力:
  <table>
    <tr>
      <td>正しい位置</td>
    </tr>
    テーブル外のテキスト
    <div>テーブル外の要素</div>
  </table>

  問題:
  → テキストや <div> は <table> の直接の子になれない
  → <table> 内で許可されるのは <thead>, <tbody>, <tfoot>,
    <tr>, <caption>, <colgroup>, <col> のみ

  Foster Parenting の結果:
  不正な要素はテーブルの「前」に移動される

  DOM 上の結果:
  テーブル外のテキスト          ← table の前に移動
  <div>テーブル外の要素</div>  ← table の前に移動
  <table>
    <tbody>
      <tr>
        <td>正しい位置</td>
      </tr>
    </tbody>
  </table>

  → DevTools で確認すると、テキストや div が
    table タグの前に移動していることが分かる
```

### 3.5 アクティブフォーマッティング要素リスト (Active Formatting Elements)

HTML パーサーは `<b>`, `<i>`, `<em>`, `<strong>`, `<a>`, `<font>` などのフォーマッティング要素に対して、特別な「再構築 (Reconstruction)」処理を行う。

```
Adoption Agency Algorithm:

  入力: <p>Normal <b>Bold <i>Both</b> Italic?</i></p>

  問題:
  → <b> と <i> が交差してネストされている
  → 正しいツリー構造に変換する必要がある

  Adoption Agency Algorithm の結果:
  <p>
    Normal
    <b>Bold <i>Both</i></b>
    <i> Italic?</i>
  </p>

  → <b> の終了で <i> を一旦閉じ、
    <b> を閉じた後に <i> を再度開く
  → ブラウザ間で統一的な挙動（仕様で定義済み）
```

---

## 4. DOMの構造と内部表現

### 4.1 DOMノードの分類

DOM (Document Object Model) はHTMLドキュメントをオブジェクトのツリー構造として表現するプログラミングインターフェイスである。

```
DOM ノード階層:

  Node (抽象基底クラス)
  ├── Document           nodeType = 9   ルートノード
  ├── DocumentType       nodeType = 10  <!DOCTYPE html>
  ├── DocumentFragment   nodeType = 11  仮想コンテナ
  ├── Element            nodeType = 1   HTML要素
  │   ├── HTMLElement
  │   │   ├── HTMLDivElement
  │   │   ├── HTMLParagraphElement
  │   │   ├── HTMLInputElement
  │   │   ├── HTMLAnchorElement
  │   │   └── ... (各HTML要素に対応するクラス)
  │   └── SVGElement
  │       ├── SVGSVGElement
  │       └── ...
  ├── Attr               nodeType = 2   属性ノード
  ├── Text               nodeType = 3   テキストノード
  ├── Comment            nodeType = 8   コメントノード
  └── CDATASection       nodeType = 4   CDATA（XMLのみ）
```

### 4.2 DOMノードの主要プロパティとメソッド

```
ノード間のナビゲーション:

  parentNode                  ← 親ノード
  childNodes                  ← 子ノード一覧 (NodeList)
  firstChild / lastChild      ← 最初/最後の子
  previousSibling / nextSibling  ← 前後の兄弟
  children                    ← 子要素のみ (HTMLCollection)
  firstElementChild           ← 最初の子要素
  parentElement               ← 親要素

  ┌────────────────────────────────────────────────────┐
  │  プロパティ           │ 全ノード含む │ 要素のみ    │
  ├──────────────────────┼─────────────┼────────────┤
  │  子ノード一覧         │ childNodes  │ children   │
  │  最初の子             │ firstChild  │ firstElem. │
  │  最後の子             │ lastChild   │ lastElem.  │
  │  次の兄弟             │ nextSibling │ nextElem.  │
  │  前の兄弟             │ prevSibling │ prevElem.  │
  └──────────────────────┴─────────────┴────────────┘

  ※ childNodes はテキストノードやコメントも含む
  ※ children は Element ノードのみ
```

### 4.3 ブラウザエンジン内部でのDOM表現

DOM はブラウザエンジン内部では C++ のオブジェクトとして実装される。JavaScript からの DOM アクセスはバインディングレイヤーを経由する。

```
Blink (Chrome) での DOM 内部表現:

  C++ 側:
  ┌──────────────────────────────────┐
  │  blink::Node                     │
  │  ├── parent_: Node*              │
  │  ├── previous_: Node*            │
  │  ├── next_: Node*                │
  │  ├── first_child_: Node*         │
  │  ├── tree_scope_: TreeScope*     │
  │  └── node_flags_: unsigned       │
  └──────────────────────────────────┘
  ┌──────────────────────────────────┐
  │  blink::Element : Node           │
  │  ├── tag_name_: AtomicString     │
  │  ├── attributes_: AttributeMap   │
  │  ├── computed_style_: ComputedStyle* │
  │  └── class_list_: DOMTokenList*  │
  └──────────────────────────────────┘

  JavaScript 側 (V8 バインディング):
  ┌──────────────────────────────────┐
  │  v8::Object (JS オブジェクト)     │
  │  └── internal_field_ ──→ blink::Node* │
  └──────────────────────────────────┘

  JS から DOM にアクセスするコスト:
  1. V8 の JS オブジェクトを参照
  2. internal field から C++ ポインタを取得
  3. C++ オブジェクトのメソッドを呼び出し
  4. 戻り値を V8 の JS 値に変換
  → この往復コストが DOM 操作のオーバーヘッドとなる
```

---

## 5. CSSパーシングとCSSOM構築

### 5.1 CSS字句解析（トークン化）

CSSパーサーは HTML パーサーとは異なり、**文脈自由文法 (Context-Free Grammar)** に基づいて動作する。CSS Syntax Module Level 3 で定義されるトークン化アルゴリズムにより、CSS テキストはトークン列に変換される。

```
CSSトークンの種類:

  ┌────────────────────┬──────────────────────────────────────┐
  │ トークン種別        │ 例                                   │
  ├────────────────────┼──────────────────────────────────────┤
  │ <ident-token>      │ color, margin, div, .class           │
  │ <function-token>   │ rgb(, calc(, var(                    │
  │ <at-keyword-token> │ @media, @import, @keyframes          │
  │ <hash-token>       │ #id, #ff0000                         │
  │ <string-token>     │ "hello", 'world'                     │
  │ <number-token>     │ 42, 3.14, -1                         │
  │ <percentage-token> │ 50%, 100%                            │
  │ <dimension-token>  │ 16px, 2em, 100vh, 300ms              │
  │ <url-token>        │ url(image.png)                       │
  │ <delim-token>      │ ., >, +, ~, *, |                     │
  │ <colon-token>      │ :                                    │
  │ <semicolon-token>  │ ;                                    │
  │ <comma-token>      │ ,                                    │
  │ <{-token>          │ {                                    │
  │ <}-token>          │ }                                    │
  │ <(-token>          │ (                                    │
  │ <)-token>          │ )                                    │
  │ <[-token>          │ [                                    │
  │ <]-token>          │ ]                                    │
  │ <whitespace-token> │ スペース、タブ、改行                   │
  │ <CDC-token>        │ -->                                  │
  │ <CDO-token>        │ <!--                                 │
  │ <EOF-token>        │ 入力終端                              │
  └────────────────────┴──────────────────────────────────────┘
```

**コード例 5: CSSトークン化の具体例**

```css
.container > .item {
  color: rgba(255, 0, 0, 0.5);
  font-size: calc(16px + 2vw);
  --custom-prop: #333;
}
```

```
トークン化結果:

  <delim-token: .>
  <ident-token: container>
  <whitespace-token>
  <delim-token: >>
  <whitespace-token>
  <delim-token: .>
  <ident-token: item>
  <whitespace-token>
  <{-token>
  <whitespace-token>
  <ident-token: color>
  <colon-token>
  <whitespace-token>
  <function-token: rgba>
  <number-token: 255>
  <comma-token>
  <whitespace-token>
  <number-token: 0>
  <comma-token>
  <whitespace-token>
  <number-token: 0>
  <comma-token>
  <whitespace-token>
  <number-token: 0.5>
  <)-token>
  <semicolon-token>
  <whitespace-token>
  <ident-token: font-size>
  <colon-token>
  <whitespace-token>
  <function-token: calc>
  <dimension-token: 16px>
  <whitespace-token>
  <delim-token: +>
  <whitespace-token>
  <dimension-token: 2vw>
  <)-token>
  <semicolon-token>
  <whitespace-token>
  <ident-token: --custom-prop>
  <colon-token>
  <whitespace-token>
  <hash-token: 333>
  <semicolon-token>
  <whitespace-token>
  <}-token>
  <EOF-token>
```

### 5.2 CSS構文解析（パーシング）

トークン列は CSS 構文解析器によって構造化されたルール群に変換される。CSS の文法は以下の構造で定義される。

```
CSS の文法構造 (BNF風表記):

  stylesheet  ::= rule*
  rule        ::= at-rule | qualified-rule
  at-rule     ::= '@' IDENT component-value* ('{' rule* '}' | ';')
  qualified-rule ::= component-value* '{' declaration-list '}'
  declaration-list ::= declaration (';' declaration)* ';'?
  declaration ::= IDENT ':' component-value+ ('!' 'important')?

セレクタの文法:
  selector-list    ::= complex-selector (',' complex-selector)*
  complex-selector ::= compound-selector (combinator compound-selector)*
  compound-selector ::= type-selector? (class-selector | id-selector |
                         attr-selector | pseudo-class)* pseudo-element?
  combinator       ::= '>' | '+' | '~' | ' ' (子孫)

  例: div.container > ul.menu li.active a:hover::before
  分解:
  ├── compound: div.container
  ├── combinator: > (子)
  ├── compound: ul.menu
  ├── combinator: ' ' (子孫)
  ├── compound: li.active
  ├── combinator: ' ' (子孫)
  └── compound: a:hover::before
```

### 5.3 CSSOMの構造

CSSOM (CSS Object Model) は CSS をプログラムから操作するためのオブジェクトモデルである。

```
CSSOM ツリーの構造:

  document.styleSheets (StyleSheetList)
  ├── StyleSheet[0] (CSSStyleSheet)
  │   │  href: null (inline <style>)
  │   │  media: MediaList
  │   │  ownerNode: <style> element
  │   │  disabled: false
  │   │
  │   └── cssRules (CSSRuleList)
  │       ├── CSSStyleRule[0]
  │       │   │  selectorText: "body"
  │       │   │  style.cssText: "margin: 0; font-family: sans-serif;"
  │       │   └── style (CSSStyleDeclaration)
  │       │       ├── margin: "0"
  │       │       └── fontFamily: "sans-serif"
  │       │
  │       ├── CSSStyleRule[1]
  │       │   │  selectorText: ".container"
  │       │   └── style (CSSStyleDeclaration)
  │       │       ├── maxWidth: "1200px"
  │       │       └── margin: "0 auto"
  │       │
  │       └── CSSMediaRule[2]
  │           │  conditionText: "(max-width: 768px)"
  │           │  media: MediaList ["(max-width: 768px)"]
  │           └── cssRules (CSSRuleList)
  │               └── CSSStyleRule[0]
  │                   │  selectorText: ".container"
  │                   └── style
  │                       └── maxWidth: "100%"
  │
  └── StyleSheet[1] (CSSStyleSheet)
      │  href: "styles.css" (external)
      │  ownerNode: <link> element
      └── cssRules (CSSRuleList)
          └── ...

CSSRule の種類:
  ┌──────────────────────────┬──────┬───────────────────────┐
  │ ルール型                  │ type │ 説明                   │
  ├──────────────────────────┼──────┼───────────────────────┤
  │ CSSStyleRule             │ 1    │ 通常のスタイルルール    │
  │ CSSImportRule            │ 3    │ @import                │
  │ CSSMediaRule             │ 4    │ @media                 │
  │ CSSFontFaceRule          │ 5    │ @font-face             │
  │ CSSKeyframesRule         │ 7    │ @keyframes             │
  │ CSSSupportsRule          │ 12   │ @supports              │
  │ CSSLayerBlockRule        │ --   │ @layer                 │
  │ CSSContainerRule         │ --   │ @container             │
  └──────────────────────────┴──────┴───────────────────────┘
```

### 5.4 スタイルの計算（Style Resolution）

DOM ツリーと CSSOM が構築された後、ブラウザは各 DOM 要素に対して最終的なスタイル（Computed Style）を計算する。このプロセスは以下のステップで進行する。

```
スタイル計算の全体フロー:

  ステップ 1: スタイルソースの収集
  ┌─────────────────────────────────────────────────┐
  │  User Agent Stylesheet (ブラウザデフォルト)       │
  │  ↓                                              │
  │  User Stylesheet (ユーザー設定)                  │
  │  ↓                                              │
  │  Author Stylesheet (開発者のCSS)                 │
  │    - 外部CSS (<link>)                            │
  │    - 内部CSS (<style>)                           │
  │    - インラインCSS (style="...")                  │
  │  ↓                                              │
  │  CSS Animations / Transitions                    │
  └─────────────────────────────────────────────────┘

  ステップ 2: セレクタマッチング
  → 各 DOM 要素に対して、全 CSS ルールのセレクタを評価
  → マッチするルールの宣言を収集

  ステップ 3: カスケード (Cascade)
  → マッチした宣言を優先度順にソート

  カスケード順序（優先度の低い順）:
  ┌─────────────────────────────────────────────────┐
  │  1. Normal User Agent declarations               │
  │  2. Normal User declarations                     │
  │  3. Normal Author declarations                   │
  │  4. CSS Animations                               │
  │  5. !important Author declarations               │
  │  6. !important User declarations                 │
  │  7. !important User Agent declarations           │
  │  8. CSS Transitions                              │
  └─────────────────────────────────────────────────┘

  ※ CSS Cascade Layers (@layer) が追加された場合、
    同一オリジン内でさらに細かい優先度制御が可能

  ステップ 4: 詳細度 (Specificity) の計算
  → 同一カスケードレベル内で競合する場合に使用

  詳細度の計算式: (A, B, C)
  ┌──────────────────────────────┬──────────┬──────┐
  │ セレクタ                     │ (A,B,C)  │ 値   │
  ├──────────────────────────────┼──────────┼──────┤
  │ *                            │ (0,0,0)  │ 0    │
  │ li                           │ (0,0,1)  │ 1    │
  │ ul li                        │ (0,0,2)  │ 2    │
  │ .active                      │ (0,1,0)  │ 10   │
  │ li.active                    │ (0,1,1)  │ 11   │
  │ #nav                         │ (1,0,0)  │ 100  │
  │ #nav .active                 │ (1,1,0)  │ 110  │
  │ #nav ul li.active a          │ (1,1,3)  │ 113  │
  │ :is(#nav) .item              │ (1,1,0)  │ 110  │
  │ :where(#nav) .item           │ (0,1,0)  │ 10   │
  │ style="" (インライン)         │ 最高      │ --   │
  └──────────────────────────────┴──────────┴──────┘

  注意: :is() は引数の最大詳細度を採用
        :where() は常に詳細度 0
        :not() は引数の詳細度を採用

  ステップ 5: 宣言値 (Declared Value) の決定
  → カスケード + 詳細度 + ソース順で最終的な宣言値を決定

  ステップ 6: 指定値 (Specified Value) の決定
  → 宣言値がない場合: 継承 or 初期値
  → inherit, initial, unset, revert の解決

  継承プロパティと非継承プロパティ:
  ┌───────────────────────┬────────────────────────┐
  │ 継承する               │ 継承しない              │
  ├───────────────────────┼────────────────────────┤
  │ color                 │ margin                 │
  │ font-family           │ padding                │
  │ font-size             │ border                 │
  │ line-height           │ width / height         │
  │ text-align            │ display                │
  │ visibility            │ position               │
  │ cursor                │ background             │
  │ list-style            │ overflow               │
  │ letter-spacing        │ flex / grid 関連        │
  └───────────────────────┴────────────────────────┘

  ステップ 7: 計算値 (Computed Value) の算出
  → 相対値を絶対値に変換
  → em, rem → px
  → percentage → px (一部を除く)
  → currentColor → 実際の色値
  → inherit → 親の計算値

  ステップ 8: 使用値 (Used Value) の算出
  → レイアウト計算に必要な最終値
  → auto → 実際の px 値
  → percentage (width等) → 実際の px 値

  ステップ 9: 実際値 (Actual Value) の算出
  → デバイスの制約に合わせた最終調整
  → サブピクセル丸め
  → 利用不可能なフォントのフォールバック
```

### 5.5 セレクタマッチングの最適化

ブラウザは全DOM要素 x 全CSSルールのマッチングを効率化するために、複数の最適化手法を使用する。

```
セレクタの右から左への評価:

  セレクタ: #main .content p a.link

  素朴な方法（左から右）:
  1. #main を探す
  2. その子孫で .content を探す
  3. その子孫で p を探す
  4. その子孫で a.link を探す
  → 多くの候補が生まれ、途中で失敗するケースが多い

  実際のブラウザ（右から左）:
  1. a.link を全て探す（キーセレクタ）
  2. 各 a.link の祖先に p があるか
  3. その祖先に .content があるか
  4. その祖先に #main があるか
  → キーセレクタで候補を絞り込み、
    祖先チェーンを辿って検証する方が効率的

Bloom Filter による高速化:
  → DOM 要素の祖先チェーンに含まれる
    id, class, tag name を Bloom Filter に記録
  → セレクタの祖先要素が Bloom Filter にないなら
    確実にマッチしない（高速な否定判定）
  → False positive はあるが False negative はない

スタイル共有 (Style Sharing):
  → 兄弟要素で同じクラス・属性を持つ場合、
    Computed Style を共有して計算コストを削減
  → 条件: 同一タグ名、同一クラス、同一属性、
    同一親要素のスタイルから同一セレクタにマッチ
```

### 5.6 CSS パーサーのエラー処理

CSS パーサーもエラーに対して寛容であり、認識できないプロパティや値はスキップして処理を継続する。

```
CSS エラー回復の例:

  /* 未知のプロパティ → スキップ */
  .box {
    color: red;         /* OK: 適用 */
    colr: blue;         /* NG: スキップ（タイポ） */
    font-size: 16px;    /* OK: 適用 */
  }

  /* 不正な値 → その宣言のみスキップ */
  .box {
    width: 100px;       /* OK: 適用 */
    width: abc;         /* NG: スキップ */
    height: 50px;       /* OK: 適用 */
  }

  /* 不正なセレクタ → ルール全体をスキップ */
  .valid { color: red; }           /* OK: 適用 */
  .invalid[[ { color: blue; }      /* NG: ルール全体スキップ */
  .also-valid { color: green; }    /* OK: 適用 */

  /* 中括弧の不一致 → 回復を試みる */
  .box { color: red;
    /* '}' が欠落 → 次の '}' まで読み飛ばす */
  .next { color: blue; }

この「フォワード互換性」は CSS の設計哲学の核心であり、
古いブラウザでも新しい CSS 構文を含むスタイルシートを
（未知部分をスキップして）処理できる。
```

---

## 6. レンダーツリーの構築

### 6.1 DOMとCSSOMの統合

レンダーツリー（Layout Tree とも呼ばれる）は、DOM ツリーと CSSOM を統合して構築される。表示対象の各要素に対して、確定したスタイル情報が付与される。

```
DOM + CSSOM → レンダーツリー の詳細:

  DOM ツリー:                    CSSOM ルール:
  Document                      body { font: 16px/1.5 sans-serif; }
  └── html                      h1 { font-size: 2em; color: #333; }
      ├── head                   p { margin: 1em 0; }
      │   ├── title              .hidden { display: none; }
      │   ├── style              .invisible { visibility: hidden; }
      │   └── link               img { max-width: 100%; }
      └── body                   ::before { content: "★"; }
          ├── h1
          ├── p
          ├── div.hidden
          ├── div.invisible
          ├── img
          └── script

  レンダーツリー (構築結果):
  RenderView (viewport)
  └── RenderBody
      │  font: 16px/1.5 sans-serif
      ├── RenderBlock (h1)
      │   │  font-size: 32px; color: #333
      │   ├── RenderInline (::before pseudo)
      │   │   └── "★"
      │   └── RenderText: タイトルテキスト
      ├── RenderBlock (p)
      │   │  margin: 16px 0
      │   └── RenderText: 段落テキスト
      ├── RenderBlock (div.invisible)    ← visibility:hidden は含まれる
      │   │  visibility: hidden
      │   └── (子要素...)
      ├── RenderImage (img)
      │   └── max-width: 100%
      │
      │  ※ head 要素は含まれない (display: none が UA スタイルで設定)
      │  ※ div.hidden は含まれない (display: none)
      │  ※ script 要素は含まれない (display: none が UA スタイルで設定)
      └── (以上)

  レンダーツリーに含まれない要素:
  ┌────────────────────────┬──────────────────────────────┐
  │ 要素                    │ 理由                          │
  ├────────────────────────┼──────────────────────────────┤
  │ <head> とその子要素     │ UA スタイルで display: none    │
  │ <script>               │ UA スタイルで display: none    │
  │ display: none の要素    │ 明示的に非表示                │
  │ <meta>, <link>         │ UA スタイルで display: none    │
  └────────────────────────┴──────────────────────────────┘

  レンダーツリーに含まれるが見えない要素:
  ┌────────────────────────┬──────────────────────────────┐
  │ 要素                    │ 理由                          │
  ├────────────────────────┼──────────────────────────────┤
  │ visibility: hidden     │ スペースを占めるが透明         │
  │ opacity: 0             │ 完全に透明だがスペースを占める  │
  │ position: absolute +   │ 画面外に配置されている         │
  │   left: -9999px        │                              │
  │ clip-path: inset(100%) │ クリップで完全に切り取られる   │
  └────────────────────────┴──────────────────────────────┘
```

### 6.2 擬似要素のレンダーツリーへの挿入

`::before` と `::after` 擬似要素は DOM には存在しないが、レンダーツリーには含まれる。

```
擬似要素の扱い:

  CSS:
  .quote::before {
    content: "「";
    color: gray;
  }
  .quote::after {
    content: "」";
    color: gray;
  }

  DOM:
  <p class="quote">重要な言葉</p>

  DOM ツリー（擬似要素は含まれない）:
  p.quote
  └── "重要な言葉"

  レンダーツリー（擬似要素が含まれる）:
  RenderBlock (p.quote)
  ├── RenderInline (::before)
  │   └── RenderText: "「"
  ├── RenderText: "重要な言葉"
  └── RenderInline (::after)
      └── RenderText: "」"

  → 擬似要素は DOM API からはアクセスできない
  → querySelectorAll('::before') は動作しない
  → getComputedStyle(el, '::before') でスタイルのみ取得可能
```

### 6.3 Anonymous Box の生成

レンダーツリーでは、CSS の視覚フォーマットモデルに従って「匿名ボックス (Anonymous Box)」が自動生成される場合がある。

```
Anonymous Box の例:

  DOM:
  <div>
    テキスト1
    <p>段落</p>
    テキスト2
  </div>

  CSS:
  div { display: block; }
  p { display: block; }

  レンダーツリー:
  RenderBlock (div)
  ├── RenderBlock (anonymous)    ← 匿名ブロックボックス
  │   └── RenderText: "テキスト1"
  ├── RenderBlock (p)
  │   └── RenderText: "段落"
  └── RenderBlock (anonymous)    ← 匿名ブロックボックス
      └── RenderText: "テキスト2"

  理由:
  → ブロック要素 (div) の直接の子にテキストとブロック要素が
    混在する場合、テキストは匿名ブロックボックスで包まれる
  → CSS の規則: ブロックコンテナはブロックレベルの子のみ、
    またはインラインレベルの子のみを持つべき
  → 混在する場合は匿名ボックスで包んで統一する
```

---

## 7. Incremental Parsing と Speculative Parsing

### 7.1 ストリーミングパース（増分パーシング）

HTML パーサーはネットワークからのデータ受信を待つことなく、受け取ったチャンクから順次パースを行う。

```
増分パーシングの動作:

  ネットワーク受信:
  ──────────────────────────────────────────────────→ 時間
  │chunk1│      │chunk2│      │chunk3│      │chunk4│
  │<html>│      │<body>│      │<div> │      │</div>│
  │<head>│      │  <h1>│      │  <p> │      │</body│
  │...   │      │  ... │      │  ... │      │</html│

  パーサー動作:
  ──────────────────────────────────────────────────→ 時間
  │parse1│      │parse2│      │parse3│      │parse4│
  │DOM構築│     │DOM追加│     │DOM追加│     │DOM完成│
        ↓             ↓             ↓
      DOMContentLoaded前のDOM部分木がどんどん成長

  利点:
  → First Contentful Paint が早まる
  → ユーザーは全 HTML のダウンロード完了前にコンテンツを見られる
  → HTML全体のサイズに関わらず応答性が向上する

  制約:
  → パーサーは未受信部分の構造を予測できない
  → <script> でパースがブロックされる場合がある
```

### 7.2 パーサーブロッキングとその回避

同期 `<script>` タグは HTML パーサーをブロックする。これはスクリプトが `document.write()` を使用してパーサーの入力を変更する可能性があるためである。

```
パーサーブロッキングの種類:

  1. 同期スクリプト（パーサーブロッキング）
  ──────────────────────────────────────────────────
  <script src="app.js"></script>

  パーサー: [パース]→[停止......DL......実行]→[再開]
                    ↑                        ↑
               スクリプト発見           実行完了後に再開

  2. async スクリプト（非パーサーブロッキング）
  ──────────────────────────────────────────────────
  <script src="app.js" async></script>

  パーサー: [パース]→[パース継続]→[パース継続]→[完了]
  スクリプト:       [DL........]→[実行]
                     ↑ パースと並行してDL、DL完了次第実行

  3. defer スクリプト（非パーサーブロッキング）
  ──────────────────────────────────────────────────
  <script src="app.js" defer></script>

  パーサー: [パース]→[パース継続]→[完了]→[実行]
  スクリプト:       [DL...........]     ↑
                                  DOMContentLoaded 前に
                                  ソース順で実行

  4. module スクリプト（defer と同等）
  ──────────────────────────────────────────────────
  <script type="module" src="app.mjs"></script>

  → デフォルトで defer と同じ動作
  → async 属性を付けると async 動作に変更可能

  比較表:
  ┌──────────┬────────────┬──────────┬───────────────┐
  │ 属性      │ パース     │ 実行タイミング │ 実行順序    │
  │          │ ブロック   │              │             │
  ├──────────┼────────────┼──────────────┼─────────────┤
  │ なし      │ する       │ DL直後       │ ソース順    │
  │ async    │ しない     │ DL直後       │ 不定        │
  │ defer    │ しない     │ DOM構築後    │ ソース順    │
  │ module   │ しない     │ DOM構築後    │ ソース順    │
  │ module   │ しない     │ DL直後       │ 不定        │
  │ +async   │            │              │             │
  └──────────┴────────────┴──────────────┴─────────────┘
```

### 7.3 Speculative Parsing（投機的パース / Preload Scanner）

メインパーサーがスクリプト実行でブロックされている間、ブラウザは「Preload Scanner」と呼ばれる軽量パーサーを並行して動作させる。

```
Speculative Parsing の動作:

  メインパーサー:
  [パース]→[ブロック(script DL+実行)]→[再開]→[パース]
                  ↓ 同時に
  Preload Scanner:
           [先読みスキャン..................]
           発見: <link rel="stylesheet" href="styles.css">
           発見: <script src="other.js">
           発見: <img src="hero.jpg">
                  ↓
  ネットワーク:
           [styles.css DL開始]
           [other.js DL開始]
           [hero.jpg DL開始]

  Preload Scanner が検出するリソース:
  → <link rel="stylesheet" href="...">
  → <script src="...">
  → <img src="...">
  → <video src="..."> / <source src="...">
  → <link rel="preload" href="...">

  Preload Scanner が行わないこと:
  → DOM ツリーの構築
  → CSS の解析
  → JavaScript の実行
  → レイアウト計算
  → あくまでリソース URL の発見とネットワークリクエスト発行のみ

  パフォーマンスへの影響:
  → Preload Scanner が効果を発揮する条件:
    同期 <script> の後に多くのリソース参照がある場合
  → 効果がない場合:
    全リソースが <head> 内の <script> より前に宣言されている場合
```

---

## 8. CSS がレンダリングに与えるブロッキング効果

### 8.1 CSS のレンダーブロッキング

CSS はパーサーブロッキングではないが、**レンダーブロッキング**である。つまり、CSS の読み込みが完了するまで画面の描画が開始されない。

```
CSS レンダーブロッキングの動作:

  HTML パーサー:
  [パース開始]→[<link> 発見]→[パース継続]→[DOM 構築完了]
                    ↓
  CSS ダウンロード:
               [DL.................]→[CSSOM 構築]
                                             ↓
  レンダリング:                        [待機........]→[レンダーツリー構築]→[描画]
                                      ↑
                                 CSSOM 構築完了まで
                                 レンダリングは開始されない

  理由:
  → CSSOM なしでレンダリングすると FOUC (Flash of Unstyled Content) が発生
  → スタイルなしの瞬間的な表示はユーザー体験を大きく損なう
  → そのためブラウザは CSSOM 構築完了を待つ

  CSS が JavaScript もブロックするケース:
  → <link rel="stylesheet"> の後に <script> がある場合
  → CSS の読み込みが完了するまでスクリプトの実行も遅延する
  → スクリプトが Computed Style を参照する可能性があるため

  <link rel="stylesheet" href="styles.css">
  <script>
    // styles.css の読み込み完了まで実行されない
    // getComputedStyle() が正しい値を返すことを保証するため
    const style = getComputedStyle(document.body);
  </script>
```

### 8.2 Critical CSS とリソースヒント

レンダーブロッキングの影響を最小化するための手法を解説する。

**コード例 6: Critical CSS のインライン化**

```html
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">

  <!-- Critical CSS: ATF (Above The Fold) に必要な最小限のスタイルをインライン化 -->
  <style>
    /* ファーストビューに必要なスタイルのみ */
    body { margin: 0; font-family: sans-serif; }
    .header { background: #333; color: white; padding: 1rem; }
    .hero { padding: 2rem; text-align: center; }
    .hero h1 { font-size: 2.5rem; margin: 0; }
  </style>

  <!-- 非クリティカル CSS は非同期で読み込む -->
  <link rel="preload" href="styles.css" as="style"
        onload="this.onload=null;this.rel='stylesheet'">
  <noscript><link rel="stylesheet" href="styles.css"></noscript>

  <!-- リソースヒント -->
  <link rel="dns-prefetch" href="//fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link rel="preload" href="hero-image.webp" as="image">
</head>
<body>
  <header class="header">サイト名</header>
  <section class="hero">
    <h1>ようこそ</h1>
  </section>
  <!-- 以下のコンテンツは非同期 CSS 読み込み完了後にスタイル適用 -->
</body>
</html>
```

```
リソースヒントの種類と効果:

  ┌────────────────────┬───────────────────────────────────┐
  │ ヒント              │ 効果                              │
  ├────────────────────┼───────────────────────────────────┤
  │ dns-prefetch       │ DNS 解決のみ先行実行               │
  │ preconnect         │ DNS + TCP + TLS を先行実行         │
  │ preload            │ リソースを高優先度で先行取得        │
  │ prefetch           │ 次のナビゲーションで必要な          │
  │                    │ リソースを低優先度で先行取得        │
  │ modulepreload      │ ES Module を先行取得・解析         │
  └────────────────────┴───────────────────────────────────┘
```

---

## 9. アンチパターンと対策

### 9.1 アンチパターン 1: document.write() の使用

`document.write()` はパーサーの入力ストリームに直接テキストを挿入するAPIであり、多くの問題を引き起こす。

```
document.write() のアンチパターン:

  問題のあるコード:
  <script>
    document.write('<link rel="stylesheet" href="dynamic.css">');
    document.write('<script src="analytics.js"><\/script>');
  </script>

  問題点:
  1. パーサーブロッキング
     → document.write() 内のスクリプトもパーサーをブロック
     → ネストされたブロッキングが発生

  2. Speculative Parser の無効化
     → document.write() はパーサーの入力を変更する
     → Preload Scanner が発見したリソースが無効になる可能性
     → ブラウザの最適化が機能しなくなる

  3. 遅い接続での自動ブロック
     → Chrome は 2G 接続で document.write() による
       外部スクリプトの読み込みをブロックする（Intervention）

  4. 非同期スクリプトからの呼び出しで文書が破壊される
     → DOMContentLoaded 後に document.write() を呼ぶと
       ドキュメント全体が置き換えられる

  代替策:
  // NG: document.write()
  document.write('<script src="analytics.js"><\/script>');

  // OK: DOM API を使用
  const script = document.createElement('script');
  script.src = 'analytics.js';
  script.async = true;
  document.head.appendChild(script);

  // OK: insertAdjacentHTML を使用
  document.body.insertAdjacentHTML('beforeend',
    '<div class="dynamic-content">動的コンテンツ</div>');
```

### 9.2 アンチパターン 2: 過度に深いセレクタネスト

```
過度に深いセレクタのアンチパターン:

  NG: 深いネストのセレクタ（パフォーマンスが悪い）
  ────────────────────────────────────────────
  #app > .main-content > .sidebar > .widget-area >
  .widget > .widget-header > h3 > span.icon {
    color: blue;
  }

  問題点:
  1. セレクタマッチングのコスト増大
     → 右から左に評価するため、まず全ての span.icon を検索
     → 各候補について 7 階層の祖先チェーンを辿る
     → DOM の深さに比例してマッチングコストが増大

  2. 詳細度の過剰な上昇
     → (1, 1, 4) という高い詳細度
     → オーバーライドに !important が必要になる悪循環

  3. HTML 構造への強い依存
     → HTML の構造を変更するとスタイルが崩壊する
     → 保守性が著しく低下する

  OK: BEM 命名規則によるフラットなセレクタ
  ────────────────────────────────────────────
  .widget__header-icon {
    color: blue;
  }

  → 詳細度 (0, 1, 0)
  → DOM 構造に依存しない
  → マッチングコストが最小
  → オーバーライドも容易

  OK: CSS Custom Properties + コンポーネント設計
  ────────────────────────────────────────────
  .widget {
    --icon-color: blue;
  }
  .widget .icon {
    color: var(--icon-color);
  }

  → 最大2階層で済む
  → カスタムプロパティでテーマ化も容易
```

### 9.3 アンチパターン 3: @import によるCSS読み込みの連鎖

```
@import チェーンのアンチパターン:

  styles.css:
    @import url('reset.css');
    @import url('layout.css');
    @import url('components.css');

  components.css:
    @import url('buttons.css');
    @import url('forms.css');

  問題点:
  → @import は直列ダウンロードを引き起こす
  → styles.css DL完了 → reset.css, layout.css, components.css DL開始
  → components.css DL完了 → buttons.css, forms.css DL開始
  → ウォーターフォール型のリクエストチェーンが発生

  ┌──────────────────────────────────────────────────┐
  │ <link>                     │ @import              │
  ├────────────────────────────┼──────────────────────┤
  │ 並列ダウンロード            │ 直列ダウンロード      │
  │ Preload Scanner が検出可能  │ CSS パース後に発見    │
  │ 高速                       │ 低速                 │
  └────────────────────────────┴──────────────────────┘

  対策:
  <!-- NG: @import チェーン -->
  <link rel="stylesheet" href="styles.css">

  <!-- OK: 全てを <link> で並列読み込み -->
  <link rel="stylesheet" href="reset.css">
  <link rel="stylesheet" href="layout.css">
  <link rel="stylesheet" href="buttons.css">
  <link rel="stylesheet" href="forms.css">

  <!-- さらに良い: ビルドツールで1ファイルに結合 -->
  <link rel="stylesheet" href="bundle.css">
```

---

## 10. エッジケース分析

### 10.1 エッジケース 1: エンコーディング誤判定とパース失敗

```
エンコーディング誤判定のシナリオ:

  状況:
  → サーバーが Content-Type: text/html (charsetなし) を返す
  → HTML に <meta charset> がない
  → HTML 内に Shift_JIS の日本語が含まれている
  → ブラウザが UTF-8 と判定

  発生する問題:
  1. マルチバイト文字の途中でタグ区切り文字 '<' に相当する
     バイトが出現する可能性
  2. 属性値の中で引用符 '"' に相当するバイトが出現する可能性
  3. パーサーが意図しないタグやコメントを検出

  例:
  Shift_JIS での「表」= 0x95 0x5C
  → 0x5C は ASCII の '\' (バックスラッシュ)
  → UTF-8 として解釈すると不正なバイト列
  → パース結果が文字化けするだけでなく、
    DOM 構造自体が壊れる可能性がある

  Shift_JIS での「ソ」= 0x83 0x5C
  → 同様に 0x5C を含む
  → CSS の url() 内でパス区切りと誤認される場合がある

  対策:
  → 必ず UTF-8 を使用し、HTTP ヘッダと meta タグの両方で宣言
  → Content-Type: text/html; charset=UTF-8
  → <meta charset="UTF-8">（head 内の最初の要素として配置）
  → BOM の付与は推奨されないが、最終手段として有効
```

### 10.2 エッジケース 2: 超巨大DOMとパフォーマンス劣化

```
巨大 DOM のパフォーマンス影響:

  問題の発生条件:
  → DOM ノード数が数万〜数十万に達するページ
  → 例: 無限スクロールで全データを DOM に追加し続ける
  → 例: 大量の行を持つテーブル（<tr> が 10,000 行以上）

  影響を受ける処理:
  ┌──────────────────────────┬──────────────────────────────┐
  │ 処理                     │ 影響                          │
  ├──────────────────────────┼──────────────────────────────┤
  │ 初期パース                │ DOM 構築時間の線形的増大       │
  │ スタイル計算              │ 全要素×全ルールのマッチング    │
  │                          │ O(n * m) のコスト増大          │
  │ レイアウト                │ ボックスモデル計算の増大       │
  │ メモリ使用量              │ 各ノードが C++ オブジェクト    │
  │                          │ として存在するため増大          │
  │ querySelector            │ サブツリー全体を走査           │
  │ DOM 操作                 │ リフロー範囲の拡大             │
  │ ガベージコレクション      │ 大量オブジェクトの管理コスト   │
  └──────────────────────────┴──────────────────────────────┘

  推奨される DOM ノード数の目安:
  → 合計ノード数: 1,500 以下が理想
  → 最大の深さ: 32 レベル以下
  → 親ノードあたりの子ノード: 60 以下
  → Lighthouse は 1,400 ノード超で警告を出す

  対策:
  → 仮想スクロール (Virtual Scrolling) の導入
    → 表示領域内の要素のみ DOM に配置
    → スクロールに応じて DOM を動的に入れ替え
  → コンテンツの遅延読み込み (Lazy Loading)
  → content-visibility: auto の活用
    → 画面外の要素のレンダリングをスキップ
    → DOM には存在するがスタイル計算・レイアウトを省略
```

### 10.3 エッジケース 3: テンプレートタグと Shadow DOM のパース

```
<template> タグの特殊なパース処理:

  <template id="card-template">
    <div class="card">
      <h2 class="card-title"></h2>
      <p class="card-body"></p>
    </div>
  </template>

  パーサーの動作:
  1. <template> StartTag を検出
  2. 挿入モードを保存し、"in template" モードに切り替え
  3. テンプレートの内容は別の DocumentFragment に構築
     → メインの DOM ツリーには接続されない
  4. </template> で保存したモードに復帰

  結果:
  DOM:
  template#card-template
  └── #document-fragment (template.content)
      └── div.card
          ├── h2.card-title
          └── p.card-body

  → template.content が DocumentFragment
  → template 要素自体の childNodes は空
  → レンダーツリーには含まれない（表示されない）

Shadow DOM のパース:
  → JavaScript で attachShadow() を使用して作成
  → HTML パーサーが直接 Shadow DOM を構築するわけではない
  → ただし Declarative Shadow DOM (<template shadowrootmode="open">)
    は HTML パーサーが処理する

  <div id="host">
    <template shadowrootmode="open">
      <style>:host { display: block; border: 1px solid; }</style>
      <slot></slot>
    </template>
    <span>Light DOM content</span>
  </div>

  → パーサーが <template shadowrootmode="open"> を検出
  → Shadow Root を作成し、テンプレート内容を Shadow Tree に配置
  → <slot> を通じて Light DOM の子要素が配置される
```

---

## 11. 演習問題

### 演習 1: 基礎レベル - トークン化とDOM構築の追跡

以下の HTML を手動でトークン化し、生成される DOM ツリーを描画せよ。

```html
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>演習</title>
</head>
<body>
  <main>
    <article>
      <h1 class="title" id="top">見出し</h1>
      <p>本文 <a href="#">リンク</a> 続き</p>
    </article>
  </main>
</body>
</html>
```

```
解答のガイド:

  ステップ 1: トークン列を列挙する
  → DOCTYPE トークン (name: "html")
  → StartTag: html (lang="ja")
  → Character: 改行+空白
  → StartTag: head
  → Character: 改行+空白
  → StartTag: meta (charset="UTF-8") [self-closing]
  → Character: 改行+空白
  → StartTag: title
  → Character: "演習"
  → EndTag: title
  → ... (以下省略、全てのトークンを列挙する)

  ステップ 2: DOM ツリーを構築する
  Document
  ├── DOCTYPE: html
  └── html [lang="ja"]
      ├── head
      │   ├── meta [charset="UTF-8"]
      │   └── title
      │       └── "演習"
      └── body
          └── main
              └── article
                  ├── h1 [class="title", id="top"]
                  │   └── "見出し"
                  └── p
                      ├── "本文 "
                      ├── a [href="#"]
                      │   └── "リンク"
                      └── " 続き"

  ステップ 3: オープン要素スタックの変化を記録する
  → 各トークン受信時のスタック状態と挿入モードを追跡
```

### 演習 2: 中級レベル - エラー回復の予測

以下の不正な HTML がブラウザでどのようにパースされるかを予測せよ。DevTools の Elements パネルで結果を確認し、予測と比較せよ。

```html
<div>
  <p>段落1
  <p>段落2
  <table>
    <tr>
      <td>セル
      不正なテキスト
    </tr>
  </table>
  <b><i>交差するタグ</b></i>
  <ul>
    <li>リスト1
    <li>リスト2
  </ul>
  <form>
    <form>ネストされたform</form>
  </form>
</div>
```

```
解答のポイント:

  1. <p> タグの暗黙的な閉じ
     → <p>段落1 の後に <p> が来ると暗黙の </p> が挿入される
     → 結果: <p>段落1</p><p>段落2</p>

  2. <table> 内の不正なテキスト
     → "不正なテキスト" は foster parenting により
       テーブルの前に移動される

  3. <b><i> の交差
     → Adoption Agency Algorithm により:
       <b><i>交差するタグ</i></b><i></i>
       と再構成される

  4. <li> の暗黙的な閉じ
     → <li>リスト1 の後に <li> が来ると暗黙の </li> が挿入される

  5. ネストされた <form>
     → HTML 仕様では <form> のネストは禁止
     → 内側の <form> タグは無視される
     → 結果: <form>ネストされたform</form>
```

### 演習 3: 上級レベル - パフォーマンス最適化の設計

以下の要件を満たすHTMLドキュメントのリソース読み込み戦略を設計せよ。

```
要件:
  → ファーストビューに3つのCSSファイルが必要
     - reset.css (2KB)
     - layout.css (5KB)
     - hero.css (3KB)
  → スクロール後に必要なCSS
     - components.css (15KB)
     - animations.css (8KB)
  → JavaScript
     - app.js (50KB) - メインアプリケーション
     - analytics.js (10KB) - 分析（非同期で可）
     - widget.js (20KB) - ページ下部のウィジェット
  → 画像
     - hero.webp (100KB) - ファーストビューのヒーロー画像
     - icon-sprite.svg (5KB) - アイコン群
  → フォント
     - custom-font.woff2 (30KB)

設計のガイドライン:
  1. Critical CSS のインライン化を検討する
     → reset.css + layout.css + hero.css = 合計 10KB
     → インライン化すれば外部CSSのDL待ちが不要
     → ただし HTML サイズが増加するトレードオフ

  2. 非クリティカル CSS の遅延読み込み
     → components.css, animations.css は
       media="print" + onload で非同期化
     → または rel="preload" + as="style" で先行取得

  3. JavaScript の読み込み戦略
     → app.js: defer（DOM構築後にソース順で実行）
     → analytics.js: async（DL次第実行、順序不問）
     → widget.js: defer + ページ下部に配置

  4. リソースヒントの活用
     → preload: hero.webp, custom-font.woff2
     → preconnect: フォント配信サーバー
     → dns-prefetch: 分析サーバー

  5. 最終的な <head> の構成例を書き出す
```

**コード例 7: 最適化されたリソース読み込み**

```html
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">

  <!-- リソースヒント: 先行接続 -->
  <link rel="preconnect" href="https://fonts.example.com" crossorigin>
  <link rel="dns-prefetch" href="//analytics.example.com">

  <!-- Critical リソースの先行取得 -->
  <link rel="preload" href="hero.webp" as="image" type="image/webp">
  <link rel="preload" href="custom-font.woff2" as="font"
        type="font/woff2" crossorigin>

  <!-- Critical CSS をインライン化 -->
  <style>
    /* reset.css + layout.css + hero.css の内容 (合計約10KB) */
    *, *::before, *::after { box-sizing: border-box; margin: 0; }
    body { font-family: 'CustomFont', sans-serif; line-height: 1.6; }
    .header { /* ... */ }
    .hero { /* ... */ }
    @font-face {
      font-family: 'CustomFont';
      src: url('custom-font.woff2') format('woff2');
      font-display: swap;
    }
  </style>

  <!-- 非クリティカル CSS の非同期読み込み -->
  <link rel="preload" href="components.css" as="style"
        onload="this.onload=null;this.rel='stylesheet'">
  <link rel="preload" href="animations.css" as="style"
        onload="this.onload=null;this.rel='stylesheet'">
  <noscript>
    <link rel="stylesheet" href="components.css">
    <link rel="stylesheet" href="animations.css">
  </noscript>

  <!-- JavaScript: defer でパーサーブロッキング回避 -->
  <script src="app.js" defer></script>
  <script src="widget.js" defer></script>
  <!-- Analytics: async で独立実行 -->
  <script src="analytics.js" async></script>

  <title>最適化されたページ</title>
</head>
<body>
  <header class="header">
    <img src="icon-sprite.svg" alt="" width="24" height="24"
         loading="eager">
  </header>
  <section class="hero">
    <img src="hero.webp" alt="ヒーロー画像"
         width="1200" height="600"
         fetchpriority="high">
  </section>
  <!-- 以下は遅延読み込み対象 -->
</body>
</html>
```

---

## 12. HTMLパーサーとCSSパーサーの比較

```
┌────────────────────┬──────────────────────┬──────────────────────┐
│ 観点                │ HTML パーサー          │ CSS パーサー          │
├────────────────────┼──────────────────────┼──────────────────────┤
│ 文法の種類          │ 文脈依存             │ 文脈自由              │
│                    │ (非正規、非CFG)       │ (CFG)                │
├────────────────────┼──────────────────────┼──────────────────────┤
│ エラー処理          │ 仕様で詳細に         │ 不正な宣言を          │
│                    │ 回復手順が定義        │ スキップ              │
├────────────────────┼──────────────────────┼──────────────────────┤
│ 出力                │ DOM ツリー            │ CSSOM                │
├────────────────────┼──────────────────────┼──────────────────────┤
│ 増分パース          │ 対応                 │ 通常は全体を          │
│                    │ (ストリーミング)       │ 一括パース            │
├────────────────────┼──────────────────────┼──────────────────────┤
│ スクリプトとの       │ ブロッキングされる     │ スタイルシート読込で   │
│ 相互作用            │ (同期スクリプト)       │ JS 実行をブロック     │
├────────────────────┼──────────────────────┼──────────────────────┤
│ 仕様の定義場所      │ HTML Living Standard  │ CSS Syntax Module    │
│                    │ "Parsing" セクション   │ Level 3              │
├────────────────────┼──────────────────────┼──────────────────────┤
│ 状態数              │ 80+ 状態              │ トークナイザ状態は     │
│                    │ (トークナイザのみ)     │ 比較的少数            │
├────────────────────┼──────────────────────┼──────────────────────┤
│ ツール表現          │ ステートマシン +       │ 再帰下降パーサー      │
│                    │ スタックベース構文解析  │ (多くの実装で)        │
├────────────────────┼──────────────────────┼──────────────────────┤
│ フォワード          │ 未知のタグは           │ 未知のプロパティは    │
│ 互換性              │ HTMLUnknownElement    │ スキップ              │
│                    │ として処理             │                      │
└────────────────────┴──────────────────────┴──────────────────────┘
```

---

## 13. パフォーマンス計測とデバッグ手法

### 13.1 DevTools を使ったパース状況の確認

```
Chrome DevTools でのパフォーマンス分析:

  Performance パネル:
  1. "Record" を押してページ読み込みを記録
  2. "Main" セクションでパース処理を確認

  確認できるイベント:
  ┌──────────────────────────┬──────────────────────────┐
  │ イベント名                │ 意味                      │
  ├──────────────────────────┼──────────────────────────┤
  │ Parse HTML               │ HTML パース処理時間        │
  │ Parse Stylesheet         │ CSS パース処理時間         │
  │ Recalculate Style        │ スタイル再計算             │
  │ Layout                   │ レイアウト計算             │
  │ Evaluate Script          │ JS 実行                   │
  │ DOMContentLoaded         │ DOM 構築完了               │
  │ First Paint              │ 最初の描画                │
  │ First Contentful Paint   │ 最初のコンテンツ描画       │
  │ Largest Contentful Paint │ 最大コンテンツ描画         │
  └──────────────────────────┴──────────────────────────┘

  Network パネル:
  → CSS ファイルのウォーターフォールチャートで
    読み込み順序とブロッキングを確認
  → "Disable cache" にチェックしてキャッシュなしの
    本来の読み込み時間を確認

  Coverage パネル:
  → 使用されていない CSS/JS の割合を表示
  → 赤色の部分が未使用コード
  → Critical CSS の特定に活用できる

  Elements パネル:
  → Computed タブで最終的な Computed Style を確認
  → 各プロパティがどのルールから来ているかを追跡
  → Styles パネルで適用されるルールの優先順位を確認
```

### 13.2 PerformanceObserver による計測

**コード例 8: パース関連のパフォーマンスメトリクスを取得する**

```javascript
// PerformanceObserver で LCP を計測
const lcpObserver = new PerformanceObserver((list) => {
  const entries = list.getEntries();
  const lastEntry = entries[entries.length - 1];
  console.log('LCP:', lastEntry.startTime, 'ms');
  console.log('LCP element:', lastEntry.element);
});
lcpObserver.observe({ type: 'largest-contentful-paint', buffered: true });

// Resource Timing で CSS ファイルの読み込み時間を計測
const resourceObserver = new PerformanceObserver((list) => {
  for (const entry of list.getEntries()) {
    if (entry.initiatorType === 'link' || entry.initiatorType === 'css') {
      console.log(`CSS: ${entry.name}`);
      console.log(`  DNS: ${entry.domainLookupEnd - entry.domainLookupStart}ms`);
      console.log(`  TCP: ${entry.connectEnd - entry.connectStart}ms`);
      console.log(`  DL:  ${entry.responseEnd - entry.responseStart}ms`);
      console.log(`  Total: ${entry.duration}ms`);
    }
  }
});
resourceObserver.observe({ type: 'resource', buffered: true });

// Navigation Timing で DOM パース完了時刻を取得
window.addEventListener('load', () => {
  const nav = performance.getEntriesByType('navigation')[0];
  console.log('DOM Interactive:', nav.domInteractive, 'ms');
  console.log('DOM Content Loaded:', nav.domContentLoadedEventEnd, 'ms');
  console.log('DOM Complete:', nav.domComplete, 'ms');
});
```

---

## FAQ（よくある質問）

### Q1: innerHTML と DOM API のどちらがパフォーマンスに優れるか？

```
innerHTML vs DOM API:

  innerHTML を使用する場合:
  element.innerHTML = '<div class="card"><h2>Title</h2><p>Body</p></div>';

  → ブラウザ内部で HTML パーサーが起動される
  → 文字列をトークン化 → ツリー構築 → DOM ノード生成
  → 既存の子ノードは全て破棄される（GC 対象）

  DOM API を使用する場合:
  const card = document.createElement('div');
  card.className = 'card';
  const h2 = document.createElement('h2');
  h2.textContent = 'Title';
  const p = document.createElement('p');
  p.textContent = 'Body';
  card.appendChild(h2);
  card.appendChild(p);
  element.appendChild(card);

  → パーサーを経由しない直接的な DOM 操作
  → 各操作が個別の DOM ミューテーション

  一般的な傾向:
  → 少数の要素: DOM API が高速（パーサーのオーバーヘッドなし）
  → 大量の要素: innerHTML が高速（文字列結合のほうが軽い）
  → 最適解: DocumentFragment + DOM API、
    または requestAnimationFrame でバッチ化

  セキュリティの観点:
  → innerHTML はXSSの脆弱性を生むリスクがある
  → ユーザー入力を含む場合は textContent を使用
  → 信頼できないHTMLを挿入する場合は DOMPurify 等でサニタイズ
```

### Q2: なぜ CSS セレクタは右から左に評価されるのか？

```
右から左の評価が効率的な理由:

  セレクタ: .sidebar .widget h3

  左から右の場合:
  1. .sidebar を探す → 数個見つかる
  2. 各 .sidebar の子孫で .widget を探す → 多数の子孫を走査
  3. 各 .widget の子孫で h3 を探す → さらに走査
  → 各段階で候補が「扇状に」広がる可能性
  → 失敗するパスも最後まで走査しないと分からない

  右から左の場合:
  1. h3 を全て探す → ページ上の h3 は比較的少数
  2. 各 h3 の祖先に .widget があるか → 祖先チェーンを辿る
  3. .widget の祖先に .sidebar があるか → さらに辿る
  → 最初のステップで候補が大きく絞り込まれる
  → 祖先チェーンは1本道なので走査コストが低い
  → 失敗判定が早い段階で行える

  定量的な比較:
  → DOM ノード数 N、セレクタの深さ D、マッチ数 M とすると
  → 左から右: O(N * D) の平均ケース
  → 右から左: O(M * D) の平均ケース
  → 通常 M << N なので右から左が効率的
```

### Q3: display: none と visibility: hidden のパース・レンダリングへの違いは何か？

```
display: none vs visibility: hidden:

  ┌──────────────────┬──────────────────┬──────────────────┐
  │ 項目              │ display: none    │ visibility:hidden│
  ├──────────────────┼──────────────────┼──────────────────┤
  │ DOM に存在        │ はい              │ はい              │
  │ レンダーツリーに   │ いいえ            │ はい              │
  │ 含まれるか        │                  │                  │
  │ レイアウト計算    │ されない          │ される            │
  │ スペースを占める  │ 占めない          │ 占める            │
  │ 子要素への影響    │ 子も全て非表示    │ 子で visible に   │
  │                  │ (解除不可)        │ 戻せる            │
  │ 再表示のコスト    │ レンダーツリー     │ 再描画のみ        │
  │                  │ 再構築が必要      │ (リフロー不要)    │
  │ アクセシビリティ  │ 読み上げ対象外    │ 読み上げ対象外    │
  │ イベント受信      │ 受信しない        │ 受信しない        │
  │ トランジション    │ 適用不可          │ 適用可能          │
  └──────────────────┴──────────────────┴──────────────────┘

  content-visibility: auto との違い:
  → content-visibility: auto は DOM にもレンダーツリーにも存在
  → ただし画面外の場合、子要素のレンダリングをスキップ
  → contain-intrinsic-size でサイズのヒントを与えることで
    レイアウトシフトを防止
```

### Q4: HTML パーサーはなぜ文脈自由文法で定義できないのか？

```
HTML が文脈自由文法で定義できない理由:

  1. エラー回復が文脈依存
     → 同じトークンでも現在のオープン要素スタックの状態によって
       異なる処理が必要
     → 例: <p> 内で <p> が来たら暗黙の </p> を挿入するが、
       <div> 内で <p> が来たら通常の開始タグとして処理

  2. スクリプトによるパーサー状態の変更
     → <script> 内で document.write() が呼ばれると
       パーサーの入力ストリームが変更される
     → これは通常の文法定義では表現できない

  3. 挿入モード（23種類）による文脈依存処理
     → 同じタグでも挿入モードによって全く異なる動作
     → "in table" モードでの <td> と
       "in body" モードでの <td> は異なる処理

  4. Foster Parenting
     → テーブル内の不正な要素を別の位置に移動する処理
     → 文法規則だけでは表現できない

  対して CSS は:
  → 文脈自由文法で十分に定義可能
  → 不正な入力はスキップするだけ（回復手順が単純）
  → パース中にルールの出力が他のルールに影響しない
```

### Q5: ブラウザのパースにおいて Web Worker は使われるか？

```
Web Worker とパーシングの関係:

  HTML パーサー:
  → メインスレッドで動作する（DOMはメインスレッド専用）
  → Web Worker からは DOM にアクセスできない
  → そのため HTML パースは並列化できない

  CSS パーサー:
  → 一部のブラウザではスタイル計算の一部を並列化
  → ただしパース自体は通常メインスレッドで実行

  Off-Main-Thread の取り組み:
  → Chrome は "Off-Main-Thread CSS" を研究中
  → CSS のパースとスタイルマッチングの一部を
    ワーカースレッドに移譲する試み
  → Servo (Rust製エンジン) はスタイル計算を並列化

  開発者が活用できるパターン:
  → Web Worker 内で DOMParser を使うことはできない
  → ただし Worker 内で文字列としてHTMLを処理し、
    結果の構造化データをメインスレッドに送ることは可能
  → 例: マークダウンを Worker でパースし、
    生成された HTML 文字列をメインスレッドで innerHTML に設定
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| 文字エンコーディング検出 | BOM > HTTP ヘッダ > meta charset の優先順位 |
| HTMLトークナイザ | 80+状態のステートマシン、6種のトークンを生成 |
| HTMLツリービルダ | 挿入モード + オープン要素スタックでDOM構築 |
| エラー回復 | 暗黙の要素補完、Foster Parenting、Adoption Agency |
| DOM | C++オブジェクトツリー、JSバインディング経由でアクセス |
| CSSトークナイザ | 20+種のトークンを生成、文脈自由文法 |
| CSSOM | StyleSheetList > CSSStyleSheet > CSSRuleList の階層 |
| スタイル計算 | カスケード → 詳細度 → 継承 → 値の解決 (9ステップ) |
| セレクタマッチング | 右から左評価、Bloom Filter、スタイル共有で最適化 |
| レンダーツリー | DOM + CSSOM の統合、display:none は除外 |
| 増分パース | ストリーミング処理、Preload Scanner による最適化 |
| CSS ブロッキング | レンダーブロッキング（パーサーブロッキングではない） |

---

## 次に読むべきガイド
→ [[03-browser-security-model.md]] — ブラウザセキュリティモデル

---

## 参考文献
1. WHATWG. "HTML Living Standard - Parsing HTML documents." https://html.spec.whatwg.org/multipage/parsing.html
2. W3C. "CSS Syntax Module Level 3." https://www.w3.org/TR/css-syntax-3/
3. Garsiel, T. and Irish, P. "How Browsers Work: Behind the scenes of modern web browsers." web.dev, 2011. https://web.dev/articles/howbrowserswork
4. W3C. "CSS Cascading and Inheritance Level 5." https://www.w3.org/TR/css-cascade-5/
5. Google. "Render-tree Construction, Layout, and Paint." web.dev. https://web.dev/articles/critical-rendering-path/render-tree-construction
6. Mozilla. "How CSS is structured." MDN Web Docs. https://developer.mozilla.org/en-US/docs/Learn/CSS/First_steps/How_CSS_is_structured

