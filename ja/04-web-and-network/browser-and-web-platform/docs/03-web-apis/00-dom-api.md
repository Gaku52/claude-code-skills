# DOM API

> DOMはHTMLをJavaScriptで操作するためのAPIである。DOMツリーの構造理解、効率的なノード操作、イベントモデル、MutationObserver、Shadow DOM、Virtual DOMとの比較まで網羅的に学び、パフォーマンスを意識した堅牢な実装を目指す。

## この章で学ぶこと

- [ ] DOMツリーの構造とノードの種類を正確に理解する
- [ ] 要素の取得・作成・挿入・削除を効率的に行う
- [ ] Layout Thrashing を回避するバッチ処理パターンを習得する
- [ ] イベントモデル（キャプチャ / バブリング / 委任）を使い分ける
- [ ] MutationObserver で DOM 変更を非同期監視する設計を理解する
- [ ] Shadow DOM によるスタイル・DOM の隔離を体験する
- [ ] Virtual DOM との設計思想の違いを比較し、選択基準を持つ

## 前提知識

本章を学習する前に、以下の知識を習得しておくことを推奨する。

- **ブラウザのHTML/CSSパース**: ブラウザがHTMLとCSSをどのように解析し、DOMツリーとCSSOMツリーを構築するかを理解することで、DOM APIの操作がレンダリングにどう影響するかを把握できる。詳細は [../00-browser-engine/02-parsing-html-css.md](../00-browser-engine/02-parsing-html-css.md) を参照。

- **JavaScriptの基本構文**: 変数、関数、オブジェクト、配列などの基本的な構文に加え、`const`/`let`のスコープ、アロー関数、テンプレートリテラルなどのES2015以降の機能を理解していることが前提となる。

- **イベント駆動プログラミングの概念**: DOMはイベント駆動モデルを採用しており、ユーザーのインタラクション（クリック、キー入力など）やブラウザの状態変化（DOM読み込み完了、リサイズなど）に応じて処理を実行する。コールバック関数、非同期処理の基礎を理解しておくと、イベントリスナーの実装がスムーズになる。

---

## 1. DOMツリーの基礎構造

### 1.1 DOMとは何か

DOM（Document Object Model）は、HTMLやXML文書をプログラムから操作するための標準インターフェースである。ブラウザはHTMLを受け取ると、まずパーサがトークン化と構文解析を行い、その結果をツリー構造のオブジェクトモデルとしてメモリ上に構築する。このツリーがDOMツリーであり、JavaScriptはこのDOMツリーのAPIを通じて文書の構造・スタイル・内容を読み書きする。

DOMの仕様はWHATWGが管理する DOM Living Standard として継続的に更新されている。歴史的には DOM Level 1（1998年）から始まり、Level 2、Level 3 と段階的に拡張されてきたが、現在は「レベル」による区分は廃止され、単一の Living Standard として運用されている。

### 1.2 ノードの種類とツリー構造

DOMツリーは多種のノードで構成される。主要なノード型を以下に示す。

```
Node (nodeType)
├── Document (9)         ... 文書全体のルート
├── DocumentType (10)    ... <!DOCTYPE html>
├── Element (1)          ... <div>, <p>, <span> など
├── Attr (2)             ... 属性ノード（現在は直接アクセス非推奨）
├── Text (3)             ... テキストコンテンツ
├── Comment (8)          ... <!-- コメント -->
├── DocumentFragment (11)... メモリ上の仮想コンテナ
└── ProcessingInstruction (7) ... <?xml ... ?>（XMLのみ）
```

典型的なHTML文書のDOMツリーを ASCII 図で表す。

```
                        Document
                           |
                      DocumentType
                       <!DOCTYPE html>
                           |
                     Element <html>
                    /                \
            Element <head>       Element <body>
               |                    |
          Element <title>      Element <div#app>
               |                /        \
          Text "My Page"   Element <h1>   Element <ul>
                               |          /    |    \
                          Text "Title" <li>  <li>  <li>
                                        |     |     |
                                     Text   Text   Text
                                     "A"    "B"    "C"
```

### 1.3 ノード間のナビゲーション

各ノードは親・子・兄弟への参照を持ち、ツリーを自由に走査できる。ただし、全ノード用プロパティとElement専用プロパティが存在する点に注意が必要である。

| 関係 | 全ノード用 | Element専用 |
|------|-----------|------------|
| 親 | `parentNode` | `parentElement` |
| 子（先頭） | `firstChild` | `firstElementChild` |
| 子（末尾） | `lastChild` | `lastElementChild` |
| 前の兄弟 | `previousSibling` | `previousElementSibling` |
| 次の兄弟 | `nextSibling` | `nextElementSibling` |
| 子リスト | `childNodes`（NodeList） | `children`（HTMLCollection） |

全ノード用プロパティはテキストノードやコメントノードも含む。例えば HTML 中の改行やインデントに対応するテキストノードも `childNodes` には含まれる。Element のみを走査したい場合は Element 専用プロパティを使う。

```javascript
// 全ノード走査 vs Element 走査の違い
const body = document.body;

// childNodes はテキストノード（改行/空白）も含む
console.log(body.childNodes.length);    // 例: 7（テキスト3 + 要素3 + テキスト1）

// children は Element のみ
console.log(body.children.length);      // 例: 3（要素のみ）

// 再帰的なツリー走査
function walkDOM(node, callback, depth = 0) {
  callback(node, depth);
  let child = node.firstChild;
  while (child) {
    walkDOM(child, callback, depth + 1);
    child = child.nextSibling;
  }
}

walkDOM(document.body, (node, depth) => {
  const indent = '  '.repeat(depth);
  const info = node.nodeType === 1
    ? `Element <${node.tagName.toLowerCase()}>`
    : node.nodeType === 3
      ? `Text "${node.textContent.trim() || '(whitespace)'}"`
      : `Node type=${node.nodeType}`;
  console.log(`${indent}${info}`);
});
```

---

## 2. 要素の取得

### 2.1 取得メソッドの一覧と特性

要素を取得するメソッドは大きく分けて2系統ある。`querySelector` 系（静的スナップショット）と `getElementsBy` 系（ライブコレクション）である。

```javascript
// ---- querySelector 系（静的 NodeList） ----
const el  = document.querySelector('#app');          // 最初の1つ
const els = document.querySelectorAll('.card');       // 全て

// ---- getElementsBy 系（ライブ HTMLCollection） ----
const byId    = document.getElementById('app');               // 単一要素
const byClass = document.getElementsByClassName('card');       // ライブ
const byTag   = document.getElementsByTagName('div');          // ライブ
const byName  = document.getElementsByName('email');           // ライブ NodeList

// ---- 特殊な取得 ----
const closest = element.closest('.container');  // 祖先方向に最も近い一致要素
const matches = element.matches('.active');     // セレクタに一致するか判定
```

### 2.2 静的 NodeList vs ライブ HTMLCollection

この違いは実務でバグの原因になりやすい。比較表で整理する。

| 特性 | `querySelectorAll` | `getElementsByClassName` |
|------|-------------------|------------------------|
| 返却型 | 静的 `NodeList` | ライブ `HTMLCollection` |
| DOM変更の反映 | されない（取得時点のスナップショット） | リアルタイムに反映される |
| `forEach` 対応 | あり | なし（`Array.from()` が必要） |
| セレクタの柔軟性 | CSSセレクタ全般 | クラス名のみ |
| パフォーマンス | やや遅い（セレクタ解析あり） | 高速（単純なインデックス参照） |
| ループ中の追加/削除 | 安全（スナップショットのため） | 危険（コレクションが変化する） |

```javascript
// ライブコレクションの落とし穴
const items = document.getElementsByClassName('item');
console.log(items.length);  // 3

// ループ中に class を除去すると、インデックスがずれる
for (let i = 0; i < items.length; i++) {
  items[i].classList.remove('item');  // 除去した瞬間に items から消える
  // i=0 で除去 → items.length が 2 に → i=1 は元の3番目の要素
}
// 結果: 1つおきにしか処理されない

// 安全な方法1: querySelectorAll（静的）
document.querySelectorAll('.item').forEach(el => {
  el.classList.remove('item');  // 安全
});

// 安全な方法2: 逆順ループ
for (let i = items.length - 1; i >= 0; i--) {
  items[i].classList.remove('item');  // 後ろから処理すればインデックスが崩れない
}
```

---

## 3. 要素の作成・挿入・削除

### 3.1 基本的な CRUD 操作

```javascript
// ---- Create ----
const div = document.createElement('div');
div.className = 'card';
div.id = 'card-1';
div.setAttribute('data-category', 'tech');
div.textContent = 'Hello, DOM!';

// テンプレートから作成（複雑な構造向き）
const template = document.getElementById('card-template');
const clone = template.content.cloneNode(true);  // deep clone

// ---- Insert ----
parent.appendChild(child);                 // 末尾に追加
parent.insertBefore(newChild, refChild);   // refChild の前に挿入
parent.replaceChild(newChild, oldChild);   // 置換

// モダン API（IE 非対応だが現在は問題なし）
parent.append(child1, child2, 'text');     // 末尾に複数追加（テキストも可）
parent.prepend(child);                     // 先頭に追加
refChild.before(newChild);                 // refChild の前に
refChild.after(newChild);                  // refChild の後に
oldChild.replaceWith(newChild);            // 自身を置換

// ---- Delete ----
parent.removeChild(child);                 // 従来の方法
child.remove();                            // モダン API

// ---- Read / Update ----
element.getAttribute('href');
element.setAttribute('href', '/new-path');
element.removeAttribute('disabled');
element.hasAttribute('hidden');
element.toggleAttribute('hidden');         // あれば削除、なければ追加
```

### 3.2 DocumentFragment によるバッチ挿入

DOM に要素を1つずつ追加すると、追加のたびにレイアウト再計算が発生する可能性がある。`DocumentFragment` を使うと、メモリ上で仮想的にツリーを構築し、最後に1回だけ DOM に反映できる。

```javascript
// DocumentFragment を使った効率的な大量挿入
function createList(items) {
  const fragment = document.createDocumentFragment();

  items.forEach((item, index) => {
    const li = document.createElement('li');
    li.className = 'list-item';
    li.dataset.index = index;

    const span = document.createElement('span');
    span.textContent = item.name;
    li.appendChild(span);

    const badge = document.createElement('span');
    badge.className = 'badge';
    badge.textContent = item.count;
    li.appendChild(badge);

    fragment.appendChild(li);
  });

  return fragment;
}

// 1,000 件のデータを一括挿入
const data = Array.from({ length: 1000 }, (_, i) => ({
  name: `Item ${i}`,
  count: Math.floor(Math.random() * 100),
}));

const ul = document.querySelector('#list');
ul.appendChild(createList(data));  // DOM操作は1回だけ
```

### 3.3 insertAdjacentHTML / insertAdjacentElement

`innerHTML` は対象要素の全子孫を破棄して再構築するが、`insertAdjacentHTML` は既存の DOM を保持したまま指定位置に HTML を挿入する。

```
insertAdjacentHTML の4つのポジション:

  <!-- 'beforebegin' -->
  <div id="target">
    <!-- 'afterbegin' -->
    <p>既存の内容</p>
    <!-- 'beforeend' -->
  </div>
  <!-- 'afterend' -->
```

```javascript
const target = document.getElementById('target');

// 末尾に追加（既存の内容を壊さない）
target.insertAdjacentHTML('beforeend', '<p class="new">追加コンテンツ</p>');

// 要素の前に挿入
target.insertAdjacentHTML('beforebegin', '<h2>見出し</h2>');

// insertAdjacentElement: Element オブジェクトを挿入
const newEl = document.createElement('div');
newEl.textContent = '新しい要素';
target.insertAdjacentElement('afterend', newEl);

// insertAdjacentText: テキストノードを挿入
target.insertAdjacentText('afterbegin', 'テキスト先頭追加: ');
```

---

## 4. DOM 操作とレンダリングパイプライン

### 4.1 ブラウザのレンダリングフロー

DOM 操作がなぜパフォーマンスに影響するかを理解するには、ブラウザのレンダリングパイプラインを知る必要がある。

```
┌─────────┐   ┌──────────┐   ┌────────┐   ┌─────────┐   ┌──────────┐
│  Parse  │──▶│  Style   │──▶│ Layout │──▶│  Paint  │──▶│Composite │
│ HTML/CSS│   │ Compute  │   │(Reflow)│   │(Repaint)│   │ (Layers) │
└─────────┘   └──────────┘   └────────┘   └─────────┘   └──────────┘
     DOM          CSSOM        位置/寸法     ピクセル描画    GPU合成
   ツリー構築     スタイル計算     計算
```

各段階のコスト:

| 段階 | トリガーとなる操作 | コスト |
|------|-------------------|--------|
| Style | クラス追加・削除、スタイル変更 | 中 |
| Layout (Reflow) | 幅・高さ変更、要素追加/削除、`offsetHeight` 読み取り | 高 |
| Paint (Repaint) | 背景色・影の変更、`visibility` 変更 | 中～高 |
| Composite | `transform`、`opacity` の変更 | 低 |

### 4.2 Layout Thrashing（レイアウト スラッシング）

Layout Thrashing は、レイアウト情報の読み取りと書き込みを交互に行うことで、フレームごとに何度もレイアウト再計算が発生する現象である。

```javascript
// ---- アンチパターン: Layout Thrashing ----
// offsetHeight の読み取りごとに強制的な同期レイアウトが発生する
function badResize(elements) {
  elements.forEach(el => {
    const height = el.offsetHeight;          // 読み → 強制レイアウト
    el.style.height = (height * 2) + 'px';   // 書き → レイアウト無効化
    // 次の反復で再び offsetHeight → 再度強制レイアウト ...
  });
}

// ---- 推奨パターン: 読み書き分離 ----
function goodResize(elements) {
  // Phase 1: 全ての読み取りをまとめる
  const heights = elements.map(el => el.offsetHeight);

  // Phase 2: 全ての書き込みをまとめる
  elements.forEach((el, i) => {
    el.style.height = (heights[i] * 2) + 'px';
  });
}

// ---- 推奨パターン: requestAnimationFrame で書き込みを遅延 ----
function rafResize(elements) {
  const heights = elements.map(el => el.offsetHeight);  // 読み取り

  requestAnimationFrame(() => {
    elements.forEach((el, i) => {
      el.style.height = (heights[i] * 2) + 'px';       // 書き込み
    });
  });
}
```

レイアウトを強制するプロパティ・メソッドの代表例:

- `offsetTop`, `offsetLeft`, `offsetWidth`, `offsetHeight`
- `scrollTop`, `scrollLeft`, `scrollWidth`, `scrollHeight`
- `clientTop`, `clientLeft`, `clientWidth`, `clientHeight`
- `getComputedStyle()`
- `getBoundingClientRect()`

### 4.3 効率的な DOM 操作のベストプラクティス

```javascript
// 1. classList API でクラスを操作（className 直接操作より安全）
element.classList.add('active', 'highlight');
element.classList.remove('active');
element.classList.toggle('visible');
element.classList.contains('active');    // boolean
element.classList.replace('old', 'new');

// 2. dataset API でカスタムデータ属性を操作
// HTML: <div data-user-id="42" data-is-admin="true">
element.dataset.userId;      // "42"（camelCase に変換される）
element.dataset.isAdmin;     // "true"（文字列であることに注意）
delete element.dataset.userId;

// 3. style プロパティの一括設定
// 悪い例: 複数回の style 書き込み
element.style.width = '100px';
element.style.height = '200px';
element.style.background = 'red';

// 良い例: cssText で一括設定
element.style.cssText = 'width: 100px; height: 200px; background: red;';

// さらに良い例: クラスの付け替え（スタイルは CSS に定義）
element.classList.add('card--expanded');

// 4. display: none で DOM から一時的に切り離し、操作後に復帰
element.style.display = 'none';  // レイアウトツリーから除外
// ... 複数の DOM 操作 ...
element.style.display = '';       // 復帰（1回だけ Reflow）
```

---

## 5. イベントモデル

### 5.1 イベント伝播の3フェーズ

DOM イベントは、ルートからターゲットへ降りていく「キャプチャフェーズ」、ターゲットでの「ターゲットフェーズ」、ターゲットからルートへ昇っていく「バブリングフェーズ」の3段階で伝播する。

```
イベント伝播の流れ（クリックイベントの例）:

  Window
    │  ↓ キャプチャ          ↑ バブリング
  Document
    │  ↓                    ↑
  <html>
    │  ↓                    ↑
  <body>
    │  ↓                    ↑
  <div#container>
    │  ↓                    ↑
  <button#target>  ← ターゲットフェーズ（ここでイベント発火）
```

```javascript
const container = document.getElementById('container');
const button = document.getElementById('target');

// キャプチャフェーズで処理（第3引数 true または { capture: true }）
container.addEventListener('click', (e) => {
  console.log('1. container キャプチャ');
}, true);

// バブリングフェーズで処理（デフォルト）
container.addEventListener('click', (e) => {
  console.log('3. container バブリング');
});

button.addEventListener('click', (e) => {
  console.log('2. button ターゲット');
});

// ボタンクリック時の出力順:
// 1. container キャプチャ
// 2. button ターゲット
// 3. container バブリング
```

### 5.2 イベントの制御メソッド

```javascript
element.addEventListener('click', (e) => {
  // イベントの伝播を停止（後続のリスナーは実行される）
  e.stopPropagation();

  // 同一要素の残りのリスナーも含めて停止
  e.stopImmediatePropagation();

  // デフォルト動作をキャンセル（リンク遷移、フォーム送信など）
  e.preventDefault();

  // デフォルト動作がキャンセル可能か確認
  console.log(e.cancelable);   // true or false

  // イベント発生源の情報
  console.log(e.target);       // 実際にクリックされた要素
  console.log(e.currentTarget); // リスナーが登録された要素
  console.log(e.eventPhase);   // 1=キャプチャ, 2=ターゲット, 3=バブリング
});
```

### 5.3 イベント委任（Event Delegation）

個々の子要素にリスナーを登録する代わりに、共通の親要素に1つだけリスナーを登録し、`event.target` で発火元を特定するパターンをイベント委任と呼ぶ。動的に追加される要素にも対応できるため、SPA などで頻用される。

```javascript
// ---- イベント委任の実装例 ----

// 1,000 件のリストアイテムに個別リスナーを登録するのは非効率
// 親の <ul> に1つだけ登録する

const todoList = document.getElementById('todo-list');

todoList.addEventListener('click', (e) => {
  // closest で最も近い li を探す（クリックが li 内の span や icon でも対応）
  const item = e.target.closest('li.todo-item');
  if (!item) return;  // li 以外のクリックは無視

  // data-action で処理を分岐
  const action = e.target.closest('[data-action]')?.dataset.action;

  switch (action) {
    case 'toggle':
      item.classList.toggle('completed');
      break;
    case 'delete':
      item.remove();
      break;
    case 'edit':
      startEditing(item);
      break;
  }
});

// HTML構造:
// <ul id="todo-list">
//   <li class="todo-item">
//     <span data-action="toggle">Buy milk</span>
//     <button data-action="edit">Edit</button>
//     <button data-action="delete">Delete</button>
//   </li>
//   ... 動的に追加されるアイテムにも自動対応 ...
// </ul>
```

### 5.4 addEventListener のオプション

```javascript
element.addEventListener('scroll', handler, {
  capture: false,    // キャプチャフェーズで発火するか（デフォルト: false）
  once: true,        // 1回だけ実行し自動的に解除（デフォルト: false）
  passive: true,     // preventDefault() を呼ばないことを宣言
  signal: controller.signal,  // AbortSignal でリスナーを解除
});

// passive: true のメリット
// scroll/touchmove イベントで passive: true を指定すると
// ブラウザは preventDefault がないことを保証できるため
// スクロールを即座に開始し、スムーズなスクロールが実現する

// AbortController によるリスナー解除
const controller = new AbortController();

element.addEventListener('click', handler, { signal: controller.signal });
element.addEventListener('keyup', handler2, { signal: controller.signal });
element.addEventListener('scroll', handler3, { signal: controller.signal });

// 3つのリスナーをまとめて解除
controller.abort();
```

### 5.5 カスタムイベント

```javascript
// CustomEvent で独自イベントを発火
const event = new CustomEvent('user:login', {
  detail: { userId: 42, username: 'alice' },
  bubbles: true,      // バブリングさせるか
  cancelable: true,    // preventDefault 可能にするか
  composed: true,      // Shadow DOM 境界を越えるか
});

element.dispatchEvent(event);

// 受け取り側
document.addEventListener('user:login', (e) => {
  console.log(e.detail.username);  // "alice"
});
```

---

## 6. MutationObserver

### 6.1 概要と用途

`MutationObserver` は DOM の変更をバッチで非同期に通知するAPIである。従来の `Mutation Events`（DOMNodeInserted 等）はイベントごとに同期的に発火しパフォーマンスが極端に悪かったため、その代替として設計された。

主な用途:

- サードパーティスクリプトによる DOM 変更の検知と対処
- 動的コンテンツのロード完了検知（広告、埋め込みウィジェットなど）
- WYSIWYG エディタでのコンテンツ変更追跡
- アクセシビリティツールでの動的コンテンツ監視
- ブラウザ拡張機能でのページ変更の検知

### 6.2 基本的な使い方

```javascript
// MutationObserver の基本パターン

// 1. コールバック関数を定義
const callback = (mutationList, observer) => {
  for (const mutation of mutationList) {
    switch (mutation.type) {
      case 'childList':
        // 子要素の追加
        mutation.addedNodes.forEach(node => {
          if (node.nodeType === Node.ELEMENT_NODE) {
            console.log('追加された要素:', node.tagName, node.className);
          }
        });
        // 子要素の削除
        mutation.removedNodes.forEach(node => {
          if (node.nodeType === Node.ELEMENT_NODE) {
            console.log('削除された要素:', node.tagName);
          }
        });
        break;

      case 'attributes':
        console.log(
          `属性変更: ${mutation.attributeName}`,
          `旧値: ${mutation.oldValue}`,
          `新値: ${mutation.target.getAttribute(mutation.attributeName)}`
        );
        break;

      case 'characterData':
        console.log(
          'テキスト変更:',
          `旧値: ${mutation.oldValue}`,
          `新値: ${mutation.target.textContent}`
        );
        break;
    }
  }
};

// 2. オブザーバーを作成
const observer = new MutationObserver(callback);

// 3. 監視を開始（オプションで対象を絞る）
observer.observe(document.getElementById('app'), {
  childList: true,           // 子要素の追加/削除を監視
  attributes: true,          // 属性の変更を監視
  characterData: true,       // テキストコンテンツの変更を監視
  subtree: true,             // 子孫要素も含めて監視
  attributeOldValue: true,   // 変更前の属性値を記録
  characterDataOldValue: true, // 変更前のテキストを記録
  attributeFilter: ['class', 'style', 'data-state'], // 監視する属性を限定
});

// 4. 未処理の変更を即時取得
const pendingMutations = observer.takeRecords();

// 5. 監視を停止
observer.disconnect();
```

### 6.3 実用例: 要素の出現を待つユーティリティ

```javascript
/**
 * 指定セレクタに一致する要素が DOM に追加されるまで待つ
 * @param {string} selector - CSSセレクタ
 * @param {Element} root - 監視対象のルート要素
 * @param {number} timeout - タイムアウト（ms）
 * @returns {Promise<Element>}
 */
function waitForElement(selector, root = document.body, timeout = 10000) {
  return new Promise((resolve, reject) => {
    // 既に存在するか確認
    const existing = root.querySelector(selector);
    if (existing) {
      resolve(existing);
      return;
    }

    const timeoutId = setTimeout(() => {
      observer.disconnect();
      reject(new Error(`Element "${selector}" not found within ${timeout}ms`));
    }, timeout);

    const observer = new MutationObserver((mutations) => {
      const element = root.querySelector(selector);
      if (element) {
        clearTimeout(timeoutId);
        observer.disconnect();
        resolve(element);
      }
    });

    observer.observe(root, {
      childList: true,
      subtree: true,
    });
  });
}

// 使用例
try {
  const modal = await waitForElement('.modal-dialog');
  console.log('モーダルが表示された:', modal);
} catch (e) {
  console.error('モーダルが表示されなかった:', e.message);
}
```

### 6.4 MutationObserver のパフォーマンス考慮点

MutationObserver はマイクロタスクとして処理されるため、同期的な DOM 変更が全て完了した後にまとめて通知される。これにより Mutation Events よりも大幅にパフォーマンスが改善されているが、以下の点に注意が必要である。

- `subtree: true` で広範囲を監視すると、大量の Mutation レコードが生成される
- コールバック内で DOM を変更すると、再帰的に通知が発生する可能性がある
- `attributeFilter` で監視対象属性を絞ることで、不要な通知を減らす
- 不要になったら必ず `disconnect()` を呼ぶ（メモリリーク防止）

---

## 7. Shadow DOM

### 7.1 Shadow DOM の概念

Shadow DOM は DOM のサブツリーをカプセル化する仕組みである。Shadow DOM 内のスタイルと DOM 構造は外部から隔離され、外部のスタイルも Shadow DOM 内に影響しない。これにより、コンポーネントの再利用性と堅牢性が大幅に向上する。

Shadow DOM の構造を ASCII 図で示す。

```
<my-card>                          ← Host Element
  ├── #shadow-root (open)          ← Shadow Root
  │     ├── <style>                ← スコープ付きスタイル（外部に影響しない）
  │     │     .title { color: blue; }
  │     ├── <div class="title">
  │     │     └── <slot name="title">  ← 名前付きスロット
  │     │           └── (fallback: "Default Title")
  │     └── <div class="content">
  │           └── <slot>           ← デフォルトスロット
  │                 └── (fallback: なし)
  │
  └── Light DOM (子要素)
        ├── <span slot="title">カスタムタイトル</span>  → name="title" のスロットへ
        └── <p>カード本文</p>                           → デフォルトスロットへ
```

### 7.2 Shadow DOM を使った Web Component の実装

```javascript
// ---- 完全な Web Component の例 ----

class AccordionItem extends HTMLElement {
  // 監視する属性を宣言
  static get observedAttributes() {
    return ['open', 'disabled'];
  }

  constructor() {
    super();
    this._shadow = this.attachShadow({ mode: 'open' });
    this._shadow.innerHTML = `
      <style>
        :host {
          display: block;
          border: 1px solid #e2e8f0;
          border-radius: 8px;
          overflow: hidden;
          margin-bottom: 8px;
        }

        :host([disabled]) {
          opacity: 0.5;
          pointer-events: none;
        }

        :host(:not([open])) .panel {
          display: none;
        }

        .header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 12px 16px;
          cursor: pointer;
          background: #f7fafc;
          user-select: none;
        }

        .header:hover {
          background: #edf2f7;
        }

        .arrow {
          transition: transform 0.2s;
        }

        :host([open]) .arrow {
          transform: rotate(90deg);
        }

        .panel {
          padding: 16px;
          border-top: 1px solid #e2e8f0;
        }
      </style>

      <div class="header" part="header">
        <slot name="title">Untitled</slot>
        <span class="arrow" part="arrow">&#9654;</span>
      </div>
      <div class="panel" part="panel">
        <slot></slot>
      </div>
    `;

    // イベントバインド
    this._shadow.querySelector('.header').addEventListener('click', () => {
      if (!this.hasAttribute('disabled')) {
        this.toggleAttribute('open');
      }
    });
  }

  // ライフサイクルコールバック
  connectedCallback() {
    // DOM に追加された時
    this.setAttribute('role', 'region');
  }

  disconnectedCallback() {
    // DOM から削除された時（クリーンアップ）
  }

  attributeChangedCallback(name, oldValue, newValue) {
    // 監視対象属性が変更された時
    if (name === 'open') {
      this.dispatchEvent(new CustomEvent('toggle', {
        detail: { open: this.hasAttribute('open') },
        bubbles: true,
      }));
    }
  }
}

customElements.define('accordion-item', AccordionItem);

// HTML での使用:
// <accordion-item open>
//   <span slot="title">セクション1</span>
//   <p>コンテンツ...</p>
// </accordion-item>
```

### 7.3 Shadow DOM のスタイル境界

Shadow DOM のスタイル隔離に関するルールを整理する。

| スタイルの方向 | 動作 | 回避策 |
|---------------|------|--------|
| 外部 CSS → Shadow DOM 内 | 適用されない | `::part()` 疑似要素で公開 |
| Shadow DOM 内 CSS → 外部 | 漏れない | 意図通りの動作 |
| 継承プロパティ（color, font 等） | Shadow DOM 境界を越えて継承される | `all: initial` でリセット可能 |
| CSS カスタムプロパティ（変数） | Shadow DOM 境界を越える | テーマ設定に活用可能 |

```javascript
// CSS カスタムプロパティによるテーマ設定
// 外部 CSS:
//   accordion-item {
//     --accordion-bg: #f0f0f0;
//     --accordion-color: #333;
//   }

// Shadow DOM 内 CSS:
//   .header {
//     background: var(--accordion-bg, #f7fafc);
//     color: var(--accordion-color, inherit);
//   }

// ::part() による外部からのスタイリング
// 外部 CSS:
//   accordion-item::part(header) {
//     background: navy;
//     color: white;
//   }
```

### 7.4 open vs closed モード

```javascript
// open モード: shadowRoot プロパティで外部からアクセス可能
const openEl = document.createElement('div');
const openShadow = openEl.attachShadow({ mode: 'open' });
console.log(openEl.shadowRoot === openShadow);  // true

// closed モード: shadowRoot は null を返す
const closedEl = document.createElement('div');
const closedShadow = closedEl.attachShadow({ mode: 'closed' });
console.log(closedEl.shadowRoot);  // null
// closedShadow への参照を保持していれば操作は可能
// 完全なセキュリティ境界ではないことに注意
```

実務では `open` モードが推奨される。理由は以下の通り:

- DevTools でのデバッグが容易
- テストフレームワークからアクセス可能
- `closed` は完全なセキュリティ境界を提供しない（WeakMap 等で迂回可能）
- ブラウザ内部コンポーネント（`<input type="date">` 等）は `closed` を使用

---

## 8. Virtual DOM との比較

### 8.1 Virtual DOM とは

Virtual DOM は React が普及させた概念で、実 DOM のツリー構造を JavaScript オブジェクトとしてメモリ上に保持し、状態変更時に新旧の仮想ツリーを比較（差分検出 = Reconciliation / Diffing）して、最小限の実 DOM 操作のみを行うアーキテクチャである。

Virtual DOM は「DOM 操作が遅い」という前提に基づいている。JavaScript のオブジェクト操作は DOM 操作より桁違いに高速であるため、差分計算を JavaScript 側で行い、実 DOM への書き込みを最小化するという戦略を取る。

### 8.2 Virtual DOM の動作原理

```
Virtual DOM の更新サイクル:

  ┌──────────────┐    状態変更     ┌──────────────┐
  │  旧 Virtual  │ ──────────────▶ │  新 Virtual  │
  │     DOM      │                 │     DOM      │
  │   (v-node)   │                 │   (v-node)   │
  └──────┬───────┘                 └──────┬───────┘
         │                                │
         └────────────┬───────────────────┘
                      │
                   Diffing
                   (差分検出)
                      │
                      ▼
              ┌───────────────┐
              │  最小限の      │
              │  DOM パッチ    │
              │  (実DOM更新)   │
              └───────────────┘
```

```javascript
// Virtual DOM ノードの概念的な構造（React の場合）
// JSX: <div className="card"><h1>Title</h1><p>Body</p></div>
// ↓ トランスパイル後
const vnode = {
  type: 'div',
  props: { className: 'card' },
  children: [
    {
      type: 'h1',
      props: {},
      children: ['Title'],
    },
    {
      type: 'p',
      props: {},
      children: ['Body'],
    },
  ],
};

// 状態変更により新しい vnode が生成される
const newVnode = {
  type: 'div',
  props: { className: 'card active' },  // className 変更
  children: [
    {
      type: 'h1',
      props: {},
      children: ['New Title'],            // テキスト変更
    },
    {
      type: 'p',
      props: {},
      children: ['Body'],                 // 変更なし
    },
  ],
};

// Diff 結果:
// 1. div の className を 'card' → 'card active' に変更
// 2. h1 のテキストを 'Title' → 'New Title' に変更
// 3. p は変更なし → 何もしない
```

### 8.3 Virtual DOM vs 直接 DOM 操作 vs Shadow DOM

| 比較項目 | Virtual DOM (React等) | 直接 DOM 操作 | Shadow DOM |
|---------|----------------------|--------------|------------|
| 目的 | 宣言的UIと効率的な更新 | DOM の直接制御 | DOM/CSSの隔離 |
| 抽象化レベル | 高い（JSX → vnode → DOM） | 低い（DOM API 直接） | 中間（ネイティブAPI） |
| パフォーマンス | 中（diff コストあり） | 最高（最適化次第） | 高（ネイティブ） |
| メモリ使用量 | 多い（仮想ツリー保持） | 少ない | 中程度 |
| 学習コスト | 中～高（フレームワーク依存） | 低～中 | 中 |
| CSS隔離 | なし（CSS Modules等で別途対応） | なし | あり（ネイティブ） |
| コンポーネント化 | フレームワーク提供 | 自作が必要 | Web Components |
| SSR対応 | フレームワークが対応 | N/A | 限定的 |
| ブラウザ互換性 | フレームワーク依存 | 最高 | モダンブラウザのみ |
| 適用シナリオ | 複雑な状態管理を持つSPA | シンプルなインタラクション | 再利用可能なUIパーツ |

### 8.4 各アプローチの使い分け指針

```
                 アプローチ選択フローチャート:

                    UIの複雑さは?
                   /            \
              単純               複雑
              /                    \
     頻繁な更新あり?          状態管理が必要?
      /         \              /          \
    Yes          No          Yes           No
    /             \           /              \
 直接DOM操作    直接DOM操作  Virtual DOM     Shadow DOM +
 (バッチ処理)  (シンプル)   (React/Vue等)   Web Components
```

- **直接 DOM 操作**: フォームバリデーション、簡易アニメーション、jQuery 的な操作
- **Virtual DOM**: 複雑な状態管理を持つ SPA、頻繁な再レンダリングが必要なUI
- **Shadow DOM**: デザインシステム、埋め込みウィジェット、マイクロフロントエンド

### 8.5 Incremental DOM と Svelte のアプローチ

Virtual DOM の代替として注目される2つのアプローチがある。

**Incremental DOM（Angular Ivy）**: 仮想ツリーを保持せず、実 DOM を直接インクリメンタルに走査・更新する。メモリ効率が高い。

**Svelte のコンパイル時アプローチ**: ビルド時にコンポーネントを効率的な命令型 DOM 操作コードにコンパイルする。ランタイムに仮想 DOM の diff エンジンを持たないため、バンドルサイズが小さく、実行時パフォーマンスも高い。

```javascript
// Svelte のコンパイル結果の概念イメージ
// 入力（.svelte ファイル）:
//   <script>
//     let count = 0;
//     function increment() { count += 1; }
//   </script>
//   <button on:click={increment}>{count}</button>

// コンパイル出力（概念的）:
function create_fragment(ctx) {
  let button;
  let t;

  return {
    c() {  // create
      button = document.createElement('button');
      t = document.createTextNode(ctx[0]);  // count
      button.appendChild(t);
    },
    m(target) {  // mount
      target.appendChild(button);
      button.addEventListener('click', ctx[1]);  // increment
    },
    p(ctx) {  // update（差分のみ）
      t.data = ctx[0];  // テキストノードを直接更新（diff不要）
    },
    d(detaching) {  // destroy
      if (detaching) button.remove();
    },
  };
}
```

---

## 9. DOM 操作の高度なパターン

### 9.1 Range API によるテキスト操作

`Range` API は DOM ツリー内の任意の範囲を表現し、テキスト選択やリッチテキストエディタの実装に不可欠である。

```javascript
// Range の基本操作
const range = document.createRange();

// 要素の内容全体を選択
range.selectNodeContents(element);

// 特定のテキストノードの一部を選択
const textNode = element.firstChild;  // テキストノード
range.setStart(textNode, 5);   // 5文字目から
range.setEnd(textNode, 10);    // 10文字目まで

// 選択範囲の情報取得
console.log(range.toString());           // 選択されたテキスト
console.log(range.getBoundingClientRect()); // 選択範囲の座標

// 選択範囲を操作
range.deleteContents();                   // 選択範囲を削除
range.insertNode(document.createElement('mark')); // ノード挿入

// ユーザの選択範囲を取得
const selection = window.getSelection();
if (selection.rangeCount > 0) {
  const userRange = selection.getRangeAt(0);
  console.log('選択テキスト:', userRange.toString());

  // 選択範囲をマーカーで囲む
  const mark = document.createElement('mark');
  mark.style.backgroundColor = '#ffeb3b';
  userRange.surroundContents(mark);
}
```

### 9.2 TreeWalker によるツリー走査

`TreeWalker` は DOM ツリーを効率的に走査するためのイテレータである。フィルタリング機能を持ち、特定のノード型のみを走査できる。

```javascript
// テキストノードのみを走査
const walker = document.createTreeWalker(
  document.body,          // ルート
  NodeFilter.SHOW_TEXT,   // テキストノードのみ
  {
    acceptNode(node) {
      // 空白のみのテキストノードを除外
      return node.textContent.trim()
        ? NodeFilter.FILTER_ACCEPT
        : NodeFilter.FILTER_REJECT;
    }
  }
);

const textNodes = [];
let current;
while ((current = walker.nextNode())) {
  textNodes.push(current);
}

// テキスト検索と置換
function findAndHighlight(root, searchText) {
  const walker = document.createTreeWalker(
    root,
    NodeFilter.SHOW_TEXT,
    null
  );

  const matches = [];
  let node;
  while ((node = walker.nextNode())) {
    if (node.textContent.includes(searchText)) {
      matches.push(node);
    }
  }

  matches.forEach(textNode => {
    const parts = textNode.textContent.split(searchText);
    const fragment = document.createDocumentFragment();

    parts.forEach((part, i) => {
      fragment.appendChild(document.createTextNode(part));
      if (i < parts.length - 1) {
        const mark = document.createElement('mark');
        mark.textContent = searchText;
        fragment.appendChild(mark);
      }
    });

    textNode.parentNode.replaceChild(fragment, textNode);
  });
}
```

### 9.3 IntersectionObserver との連携

`IntersectionObserver` は要素のビューポート内への出入りを監視する API で、DOM 操作と組み合わせて遅延ロードやアニメーション制御に使える。

```javascript
// 遅延ロード + DOM 操作の組み合わせ
function setupLazyLoading() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const img = entry.target;
        const src = img.dataset.src;

        // 実際の src を設定
        img.src = src;
        img.removeAttribute('data-src');
        img.classList.add('loaded');

        // 監視を解除
        observer.unobserve(img);
      }
    });
  }, {
    rootMargin: '200px 0px',  // ビューポートの200px手前から読み込み開始
    threshold: 0.01,
  });

  // data-src 属性を持つ全画像を監視
  document.querySelectorAll('img[data-src]').forEach(img => {
    observer.observe(img);
  });
}

// 無限スクロールの実装
function setupInfiniteScroll(container, loadMore) {
  const sentinel = document.createElement('div');
  sentinel.className = 'scroll-sentinel';
  sentinel.style.height = '1px';
  container.appendChild(sentinel);

  let isLoading = false;

  const observer = new IntersectionObserver(async (entries) => {
    if (entries[0].isIntersecting && !isLoading) {
      isLoading = true;

      const newItems = await loadMore();

      const fragment = document.createDocumentFragment();
      newItems.forEach(item => {
        const el = createItemElement(item);
        fragment.appendChild(el);
      });

      // sentinel の前に挿入（sentinel は常に末尾）
      container.insertBefore(fragment, sentinel);
      isLoading = false;
    }
  }, { threshold: 0.1 });

  observer.observe(sentinel);
  return () => observer.disconnect();
}
```

### 9.4 ResizeObserver と DOM レイアウト変更

```javascript
// 要素のサイズ変更を検知してレイアウトを調整
const resizeObserver = new ResizeObserver((entries) => {
  for (const entry of entries) {
    const { width, height } = entry.contentRect;

    // コンテナ幅に応じたレスポンシブレイアウト（CSS Container Queries の代替）
    const container = entry.target;
    container.classList.toggle('compact', width < 400);
    container.classList.toggle('medium', width >= 400 && width < 800);
    container.classList.toggle('wide', width >= 800);

    // グリッドの列数を動的に調整
    const columns = Math.max(1, Math.floor(width / 250));
    container.style.setProperty('--columns', columns);
  }
});

resizeObserver.observe(document.querySelector('.grid-container'));
```

---

## 10. Template 要素と Declarative Shadow DOM

### 10.1 `<template>` 要素

`<template>` 要素はレンダリングされないが、JavaScript からクローンして利用できる HTML テンプレートを定義する。`innerHTML` による文字列パースと比べ、テンプレートはパース済みの DOM フラグメントを提供するため効率的である。

```javascript
// HTML:
// <template id="card-template">
//   <div class="card">
//     <h3 class="card-title"></h3>
//     <p class="card-body"></p>
//     <button class="card-action">詳細</button>
//   </div>
// </template>

function createCard(title, body) {
  const template = document.getElementById('card-template');
  const clone = template.content.cloneNode(true);  // DocumentFragment

  clone.querySelector('.card-title').textContent = title;
  clone.querySelector('.card-body').textContent = body;
  clone.querySelector('.card-action').addEventListener('click', () => {
    console.log(`${title} の詳細を表示`);
  });

  return clone;
}

// テンプレートを使った大量生成
const container = document.getElementById('card-list');
const fragment = document.createDocumentFragment();

for (const item of dataList) {
  fragment.appendChild(createCard(item.title, item.body));
}
container.appendChild(fragment);
```

### 10.2 Declarative Shadow DOM (DSD)

Declarative Shadow DOM は、HTML 内で直接 Shadow DOM を宣言できる機能である。JavaScript なしで Shadow DOM を構築でき、サーバーサイドレンダリング（SSR）との互換性が向上する。

```html
<!-- Declarative Shadow DOM -->
<my-card>
  <template shadowrootmode="open">
    <style>
      :host { display: block; border: 1px solid #ccc; padding: 16px; }
      .title { font-weight: bold; font-size: 1.2em; }
    </style>
    <div class="title">
      <slot name="title">Default Title</slot>
    </div>
    <div class="content">
      <slot></slot>
    </div>
  </template>
  <span slot="title">宣言的 Shadow DOM</span>
  <p>JavaScript なしで Shadow DOM が構築される</p>
</my-card>
```

DSD の利点:

- SSR でレンダリングした HTML に Shadow DOM を含められる
- JavaScript の読み込み前にコンポーネントの構造とスタイルが適用される
- FOUC（Flash of Unstyled Content）を防止できる
- ストリーミング SSR との相性が良い

---

## 11. アンチパターンと対策

### 11.1 アンチパターン1: innerHTML による XSS 脆弱性

`innerHTML` にユーザ入力を直接代入することは、クロスサイトスクリプティング（XSS）の典型的な原因となる。

```javascript
// ---- 危険: innerHTML にユーザ入力を直接代入 ----
const userInput = '<img src=x onerror="alert(document.cookie)">';
element.innerHTML = userInput;  // XSS! スクリプトが実行される

// ---- 安全策1: textContent を使う ----
element.textContent = userInput;  // テキストとして表示される（HTMLとして解釈されない）

// ---- 安全策2: DOMPurify でサニタイズ ----
// import DOMPurify from 'dompurify';
element.innerHTML = DOMPurify.sanitize(userInput);

// ---- 安全策3: Sanitizer API（ブラウザネイティブ、Chrome 105+） ----
const sanitizer = new Sanitizer({
  allowElements: ['b', 'i', 'em', 'strong', 'a'],
  allowAttributes: { 'href': ['a'] },
  blockElements: ['script', 'style'],
});
element.setHTML(userInput, { sanitizer });

// ---- 安全策4: DOM API で要素を構築 ----
function safeRender(data) {
  const div = document.createElement('div');
  const heading = document.createElement('h2');
  heading.textContent = data.title;  // 常にテキストとして扱われる
  div.appendChild(heading);

  const link = document.createElement('a');
  link.textContent = data.linkText;
  link.href = data.url;

  // href の検証（javascript: プロトコル対策）
  if (!/^https?:\/\//i.test(data.url)) {
    link.href = '#';  // 不正なURLを無効化
  }

  div.appendChild(link);
  return div;
}
```

### 11.2 アンチパターン2: DOM 操作によるメモリリーク

イベントリスナーの登録解除漏れや、循環参照によるメモリリークは長時間稼働するSPAで深刻な問題となる。

```javascript
// ---- 危険: リスナーの解除漏れ ----
class BadComponent {
  constructor(element) {
    this.element = element;
    this.data = new Array(10000).fill('large data');

    // グローバルリスナーを登録したが解除を忘れる
    window.addEventListener('resize', this.onResize.bind(this));
    document.addEventListener('scroll', this.onScroll.bind(this));
  }

  onResize() { /* ... */ }
  onScroll() { /* ... */ }

  destroy() {
    this.element.remove();
    // リスナーが残ったまま → this への参照が保持 → GC されない
    // this.data の 10,000 要素分のメモリがリークする
  }
}

// ---- 安全: AbortController でリスナーを一括管理 ----
class GoodComponent {
  constructor(element) {
    this.element = element;
    this.data = new Array(10000).fill('large data');
    this.controller = new AbortController();
    const { signal } = this.controller;

    window.addEventListener('resize', this.onResize.bind(this), { signal });
    document.addEventListener('scroll', this.onScroll.bind(this), { signal });
    element.addEventListener('click', this.onClick.bind(this), { signal });
  }

  onResize() { /* ... */ }
  onScroll() { /* ... */ }
  onClick() { /* ... */ }

  destroy() {
    this.controller.abort();  // 全リスナーを一括解除
    this.element.remove();
    this.data = null;         // 大きなデータへの参照を明示的に解放
  }
}

// ---- WeakRef / FinalizationRegistry による参照管理 ----
const cache = new Map();
const registry = new FinalizationRegistry((key) => {
  // 要素がGCされたらキャッシュからも削除
  cache.delete(key);
  console.log(`Element with key "${key}" was garbage collected`);
});

function cacheElement(key, element) {
  const weakRef = new WeakRef(element);
  cache.set(key, weakRef);
  registry.register(element, key);
}

function getCachedElement(key) {
  const weakRef = cache.get(key);
  if (!weakRef) return null;

  const element = weakRef.deref();
  if (!element) {
    cache.delete(key);
    return null;
  }
  return element;
}
```

### 11.3 アンチパターン3: 同期的な大量 DOM 更新

大量のデータを一度に DOM に反映すると、メインスレッドをブロックしてフレームドロップが発生する。

```javascript
// ---- 危険: 10,000 件を一度に DOM に追加 ----
function badRender(items) {
  const container = document.getElementById('list');
  container.innerHTML = '';  // 全削除（内部イベントリスナーもリーク）

  items.forEach(item => {
    const div = document.createElement('div');
    div.textContent = item.name;
    container.appendChild(div);  // 10,000回のDOM操作
  });
}

// ---- 安全策1: DocumentFragment + requestAnimationFrame でチャンク処理 ----
function chunkedRender(items, chunkSize = 100) {
  const container = document.getElementById('list');
  let index = 0;

  function renderChunk() {
    const fragment = document.createDocumentFragment();
    const end = Math.min(index + chunkSize, items.length);

    for (; index < end; index++) {
      const div = document.createElement('div');
      div.textContent = items[index].name;
      fragment.appendChild(div);
    }

    container.appendChild(fragment);

    if (index < items.length) {
      requestAnimationFrame(renderChunk);
    }
  }

  requestAnimationFrame(renderChunk);
}

// ---- 安全策2: requestIdleCallback で空き時間に処理 ----
function idleRender(items) {
  const container = document.getElementById('list');
  let index = 0;

  function renderBatch(deadline) {
    const fragment = document.createDocumentFragment();

    while (index < items.length && deadline.timeRemaining() > 2) {
      const div = document.createElement('div');
      div.textContent = items[index].name;
      fragment.appendChild(div);
      index++;
    }

    container.appendChild(fragment);

    if (index < items.length) {
      requestIdleCallback(renderBatch);
    }
  }

  requestIdleCallback(renderBatch);
}
```

---

## 12. エッジケース分析

### 12.1 エッジケース1: disconnected な要素への操作

DOM から取り外された要素（disconnected element）に対する操作は、エラーにはならないが意図しない結果を招くことがある。

```javascript
// ---- エッジケース: 取り外された要素への操作 ----

const div = document.createElement('div');
div.textContent = 'Hello';
document.body.appendChild(div);

// 参照を保持したまま DOM から削除
div.remove();

// 以下の操作はエラーにならないが、画面に反映されない
div.textContent = 'Updated';           // 成功するが画面に見えない
div.classList.add('active');           // 成功するが効果なし
div.style.backgroundColor = 'red';   // 成功するが効果なし

// offsetHeight 等のレイアウト情報は 0 を返す
console.log(div.offsetHeight);  // 0（DOMツリーに属していないため）
console.log(div.offsetWidth);   // 0

// isConnected プロパティで確認可能
console.log(div.isConnected);  // false

// ---- 安全な実装パターン ----
function updateElement(el, updates) {
  if (!el.isConnected) {
    console.warn('Element is not connected to the DOM');
    return false;
  }

  if (updates.text !== undefined) {
    el.textContent = updates.text;
  }
  if (updates.className !== undefined) {
    el.className = updates.className;
  }
  return true;
}

// MutationObserver のコールバック内での注意点
const observer = new MutationObserver((mutations) => {
  for (const mutation of mutations) {
    // removedNodes 内の要素は既に disconnected
    mutation.removedNodes.forEach(node => {
      if (node.nodeType === Node.ELEMENT_NODE) {
        // クリーンアップ処理（リスナー解除、タイマー停止など）
        cleanupElement(node);
      }
    });
  }
});
```

### 12.2 エッジケース2: ライブコレクションの反復中の変更

前述のライブ HTMLCollection の問題をさらに深掘りする。削除だけでなく、追加も危険である。

```javascript
// ---- エッジケース: ライブコレクションへの要素追加 ----

const container = document.getElementById('container');
const children = container.getElementsByTagName('div');

// 反復中に新しい div を追加すると無限ループになる
// 以下は意図的に示す危険なコードである（実行してはならない）
/*
for (let i = 0; i < children.length; i++) {
  const newDiv = document.createElement('div');
  newDiv.textContent = 'clone';
  container.appendChild(newDiv);
  // children.length が毎回増加 → 無限ループ
}
*/

// ---- 安全策: スプレッド構文で静的配列に変換 ----
const staticChildren = [...container.getElementsByTagName('div')];
staticChildren.forEach(child => {
  const clone = child.cloneNode(true);
  container.appendChild(clone);  // 安全：staticChildren は変化しない
});

// ---- 安全策: querySelectorAll（静的 NodeList） ----
container.querySelectorAll('div').forEach(child => {
  const clone = child.cloneNode(true);
  container.appendChild(clone);  // 安全
});

// ---- エッジケース: NodeList の forEach 可否 ----
// querySelectorAll の NodeList → forEach あり
// childNodes の NodeList → forEach あり（モダンブラウザ）
// getElementsBy の HTMLCollection → forEach なし
// 安全のため Array.from() を経由するのが確実
Array.from(document.getElementsByClassName('item')).forEach(el => {
  // 安全に処理
});
```

### 12.3 エッジケース3: Shadow DOM 境界とイベント retargeting

Shadow DOM 境界を越えるイベントでは、`event.target` がリターゲットされる。

```javascript
// Shadow DOM 内のボタンをクリックした場合
class MyWidget extends HTMLElement {
  constructor() {
    super();
    const shadow = this.attachShadow({ mode: 'open' });
    shadow.innerHTML = '<button id="inner-btn">Click me</button>';
  }
}
customElements.define('my-widget', MyWidget);

// 外部からリスナーを登録
document.addEventListener('click', (e) => {
  // Shadow DOM 内のボタンがクリックされても
  // event.target は <my-widget> ホスト要素になる（リターゲット）
  console.log(e.target);       // <my-widget> （内部のボタンではない）
  console.log(e.composedPath()); // [button#inner-btn, #shadow-root, my-widget, body, html, document, Window]
});

// composed: false のイベントは Shadow DOM 境界を越えない
// composed: true のイベント（click, focus, input 等）は境界を越える

// composedPath() で実際のイベント経路を確認できる
document.addEventListener('click', (e) => {
  const path = e.composedPath();
  // path[0] が実際にクリックされた要素（Shadow DOM 内部含む）
  console.log('実際のターゲット:', path[0]);
});
```

---

## 13. 演習問題

### 演習1（初級）: TODO リストの CRUD 実装

以下の仕様を満たす TODO リストを、フレームワークを使わず素の DOM API のみで実装せよ。

**要件:**
- テキスト入力欄と「追加」ボタンがある
- Enter キーでも追加できる
- 各 TODO に「完了」トグルボタンと「削除」ボタンがある
- 完了した TODO には取り消し線が表示される
- 空文字の TODO は追加できない（バリデーション）
- イベント委任を使って `<ul>` に1つだけリスナーを登録する

```javascript
// ---- 演習1の解答例 ----

function createTodoApp(rootSelector) {
  const root = document.querySelector(rootSelector);

  // DOM 構造の構築
  root.innerHTML = '';
  const form = document.createElement('form');
  form.innerHTML = `
    <input type="text" class="todo-input" placeholder="TODOを入力..." />
    <button type="submit">追加</button>
  `;

  const list = document.createElement('ul');
  list.className = 'todo-list';

  const stats = document.createElement('div');
  stats.className = 'todo-stats';

  root.append(form, list, stats);

  // 状態管理
  let todos = [];
  let nextId = 1;

  function updateStats() {
    const total = todos.length;
    const completed = todos.filter(t => t.done).length;
    stats.textContent = `全 ${total} 件 / 完了 ${completed} 件 / 残り ${total - completed} 件`;
  }

  function renderTodo(todo) {
    const li = document.createElement('li');
    li.dataset.id = todo.id;
    li.className = todo.done ? 'todo-item completed' : 'todo-item';

    li.innerHTML = `
      <span class="todo-text">${escapeHtml(todo.text)}</span>
      <button data-action="toggle">${todo.done ? '戻す' : '完了'}</button>
      <button data-action="delete">削除</button>
    `;

    return li;
  }

  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  // フォーム送信
  form.addEventListener('submit', (e) => {
    e.preventDefault();
    const input = form.querySelector('.todo-input');
    const text = input.value.trim();

    if (!text) {
      input.classList.add('error');
      return;
    }

    input.classList.remove('error');
    const todo = { id: nextId++, text, done: false };
    todos.push(todo);

    list.appendChild(renderTodo(todo));
    updateStats();
    input.value = '';
    input.focus();
  });

  // イベント委任で TODO の操作を処理
  list.addEventListener('click', (e) => {
    const actionBtn = e.target.closest('[data-action]');
    if (!actionBtn) return;

    const li = actionBtn.closest('li[data-id]');
    if (!li) return;

    const id = Number(li.dataset.id);
    const action = actionBtn.dataset.action;

    if (action === 'toggle') {
      const todo = todos.find(t => t.id === id);
      if (todo) {
        todo.done = !todo.done;
        li.classList.toggle('completed');
        li.querySelector('.todo-text').style.textDecoration =
          todo.done ? 'line-through' : 'none';
        actionBtn.textContent = todo.done ? '戻す' : '完了';
        updateStats();
      }
    }

    if (action === 'delete') {
      todos = todos.filter(t => t.id !== id);
      li.remove();
      updateStats();
    }
  });

  updateStats();
}

// 使用: createTodoApp('#app');
```

### 演習2（中級）: MutationObserver を使った DOM 変更ログ

外部スクリプトが DOM を変更する状況をシミュレートし、MutationObserver で変更履歴を記録・表示する仕組みを実装せよ。

**要件:**
- 監視対象の要素と、変更ログの表示エリアがある
- 子要素の追加・削除、属性の変更、テキストの変更を検知する
- 各変更のタイプ、タイムスタンプ、詳細情報をログに表示する
- 「監視開始」「監視停止」のトグルボタンがある
- ログのクリアボタンがある

```javascript
// ---- 演習2の解答例 ----

function createDOMMutationLogger(targetSelector, logSelector) {
  const target = document.querySelector(targetSelector);
  const logContainer = document.querySelector(logSelector);
  let observer = null;
  let isObserving = false;
  let logEntries = [];

  function formatTime() {
    return new Date().toISOString().split('T')[1].split('.')[0];
  }

  function addLogEntry(type, detail) {
    const entry = { time: formatTime(), type, detail };
    logEntries.push(entry);

    const div = document.createElement('div');
    div.className = `log-entry log-${type}`;
    div.innerHTML = `
      <span class="log-time">[${entry.time}]</span>
      <span class="log-type">${type}</span>
      <span class="log-detail">${escapeHtml(detail)}</span>
    `;
    logContainer.appendChild(div);
    logContainer.scrollTop = logContainer.scrollHeight;
  }

  function escapeHtml(str) {
    const el = document.createElement('span');
    el.textContent = str;
    return el.innerHTML;
  }

  function startObserving() {
    if (isObserving) return;

    observer = new MutationObserver((mutations) => {
      for (const mutation of mutations) {
        switch (mutation.type) {
          case 'childList':
            mutation.addedNodes.forEach(node => {
              if (node.nodeType === Node.ELEMENT_NODE) {
                addLogEntry('childList',
                  `追加: <${node.tagName.toLowerCase()}> → ${getPath(mutation.target)}`);
              }
            });
            mutation.removedNodes.forEach(node => {
              if (node.nodeType === Node.ELEMENT_NODE) {
                addLogEntry('childList',
                  `削除: <${node.tagName.toLowerCase()}> from ${getPath(mutation.target)}`);
              }
            });
            break;
          case 'attributes':
            addLogEntry('attributes',
              `属性変更: ${mutation.attributeName} on ${getPath(mutation.target)}`);
            break;
          case 'characterData':
            addLogEntry('characterData',
              `テキスト変更: "${mutation.oldValue?.substring(0, 30)}..." → "${mutation.target.textContent.substring(0, 30)}..."`);
            break;
        }
      }
    });

    observer.observe(target, {
      childList: true,
      attributes: true,
      characterData: true,
      subtree: true,
      attributeOldValue: true,
      characterDataOldValue: true,
    });

    isObserving = true;
    addLogEntry('system', '監視を開始しました');
  }

  function stopObserving() {
    if (!isObserving || !observer) return;
    observer.disconnect();
    isObserving = false;
    addLogEntry('system', '監視を停止しました');
  }

  function getPath(el) {
    const parts = [];
    while (el && el !== document.body) {
      let selector = el.tagName.toLowerCase();
      if (el.id) selector += `#${el.id}`;
      parts.unshift(selector);
      el = el.parentElement;
    }
    return parts.join(' > ');
  }

  function clearLog() {
    logEntries = [];
    logContainer.innerHTML = '';
  }

  return { startObserving, stopObserving, clearLog };
}
```

### 演習3（上級）: Shadow DOM を使った再利用可能なモーダルコンポーネント

Web Components と Shadow DOM を使い、以下の仕様を満たすモーダルダイアログを実装せよ。

**要件:**
- `<modal-dialog>` カスタム要素として登録する
- `open` 属性でモーダルの表示/非表示を制御する
- `title` スロットと `default` スロットでコンテンツを注入する
- `footer` スロットにアクションボタンを配置できる
- ESC キーで閉じる、背景クリックで閉じる
- フォーカストラップ（Tab キーがモーダル外に出ない）
- CSS カスタムプロパティでテーマカスタマイズ可能
- `open` / `close` カスタムイベントを発火する

```javascript
// ---- 演習3の解答例 ----

class ModalDialog extends HTMLElement {
  static get observedAttributes() {
    return ['open'];
  }

  constructor() {
    super();
    this._shadow = this.attachShadow({ mode: 'open' });
    this._shadow.innerHTML = `
      <style>
        :host {
          display: none;
          position: fixed;
          inset: 0;
          z-index: var(--modal-z-index, 1000);
        }

        :host([open]) {
          display: flex;
          justify-content: center;
          align-items: center;
        }

        .backdrop {
          position: fixed;
          inset: 0;
          background: var(--modal-backdrop-color, rgba(0, 0, 0, 0.5));
          backdrop-filter: blur(var(--modal-backdrop-blur, 2px));
        }

        .dialog {
          position: relative;
          background: var(--modal-bg, #ffffff);
          border-radius: var(--modal-radius, 12px);
          box-shadow: var(--modal-shadow, 0 20px 60px rgba(0, 0, 0, 0.3));
          max-width: var(--modal-max-width, 500px);
          width: 90%;
          max-height: 85vh;
          overflow: auto;
          animation: modal-enter 0.2s ease-out;
        }

        @keyframes modal-enter {
          from {
            opacity: 0;
            transform: translateY(-20px) scale(0.95);
          }
          to {
            opacity: 1;
            transform: translateY(0) scale(1);
          }
        }

        .header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 16px 20px;
          border-bottom: 1px solid var(--modal-border-color, #e2e8f0);
        }

        .close-btn {
          background: none;
          border: none;
          font-size: 1.5em;
          cursor: pointer;
          padding: 4px 8px;
          border-radius: 4px;
          color: var(--modal-close-color, #718096);
        }

        .close-btn:hover {
          background: var(--modal-close-hover-bg, #f7fafc);
        }

        .body {
          padding: 20px;
        }

        .footer {
          padding: 12px 20px;
          border-top: 1px solid var(--modal-border-color, #e2e8f0);
          display: flex;
          justify-content: flex-end;
          gap: 8px;
        }

        .footer:empty {
          display: none;
        }
      </style>

      <div class="backdrop" part="backdrop"></div>
      <div class="dialog" role="dialog" aria-modal="true" part="dialog">
        <div class="header" part="header">
          <slot name="title"><span>ダイアログ</span></slot>
          <button class="close-btn" aria-label="閉じる" part="close-btn">&times;</button>
        </div>
        <div class="body" part="body">
          <slot></slot>
        </div>
        <div class="footer" part="footer">
          <slot name="footer"></slot>
        </div>
      </div>
    `;

    // イベントバインド
    this._shadow.querySelector('.backdrop').addEventListener('click', () => {
      this.close();
    });

    this._shadow.querySelector('.close-btn').addEventListener('click', () => {
      this.close();
    });

    this._onKeyDown = this._handleKeyDown.bind(this);
  }

  connectedCallback() {
    document.addEventListener('keydown', this._onKeyDown);
  }

  disconnectedCallback() {
    document.removeEventListener('keydown', this._onKeyDown);
  }

  attributeChangedCallback(name, oldValue, newValue) {
    if (name === 'open') {
      if (newValue !== null) {
        this._onOpen();
      } else {
        this._onClose();
      }
    }
  }

  open() {
    this.setAttribute('open', '');
  }

  close() {
    this.removeAttribute('open');
  }

  _onOpen() {
    // フォーカスをダイアログ内に移動
    const dialog = this._shadow.querySelector('.dialog');
    const firstFocusable = dialog.querySelector('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])');
    if (firstFocusable) {
      requestAnimationFrame(() => firstFocusable.focus());
    }

    this.dispatchEvent(new CustomEvent('modal-open', { bubbles: true, composed: true }));
  }

  _onClose() {
    this.dispatchEvent(new CustomEvent('modal-close', { bubbles: true, composed: true }));
  }

  _handleKeyDown(e) {
    if (!this.hasAttribute('open')) return;

    if (e.key === 'Escape') {
      this.close();
      return;
    }

    // フォーカストラップ
    if (e.key === 'Tab') {
      const dialog = this._shadow.querySelector('.dialog');
      const focusables = [
        ...dialog.querySelectorAll('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'),
        ...this.querySelectorAll('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'),
      ];

      if (focusables.length === 0) return;

      const first = focusables[0];
      const last = focusables[focusables.length - 1];

      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    }
  }
}

customElements.define('modal-dialog', ModalDialog);

// HTML での使用:
// <modal-dialog id="my-modal">
//   <h2 slot="title">確認</h2>
//   <p>この操作を実行しますか?</p>
//   <div slot="footer">
//     <button onclick="document.getElementById('my-modal').close()">キャンセル</button>
//     <button onclick="confirm()">OK</button>
//   </div>
// </modal-dialog>
```

---

## 14. DOM API のブラウザ間差異と Polyfill

### 14.1 モダン API のブラウザサポート状況

| API | Chrome | Firefox | Safari | Edge |
|-----|--------|---------|--------|------|
| `element.remove()` | 23+ | 23+ | 7+ | 12+ |
| `element.closest()` | 41+ | 35+ | 6+ | 15+ |
| `element.matches()` | 33+ | 34+ | 7+ | 15+ |
| `element.toggleAttribute()` | 69+ | 63+ | 12+ | 79+ |
| `element.append()/prepend()` | 54+ | 49+ | 10+ | 17+ |
| `MutationObserver` | 26+ | 14+ | 7+ | 12+ |
| `IntersectionObserver` | 51+ | 55+ | 12.1+ | 15+ |
| `ResizeObserver` | 64+ | 69+ | 13.1+ | 79+ |
| Shadow DOM v1 | 53+ | 63+ | 10+ | 79+ |
| Declarative Shadow DOM | 111+ | 123+ | 16.4+ | 111+ |
| `element.isConnected` | 51+ | 49+ | 10+ | 79+ |

### 14.2 Safari 固有の注意点

Safari は他のブラウザに比べ、一部の DOM API の実装が遅れることがある。特に以下の点に注意が必要である。

- `adoptedStyleSheets` のサポートが遅れた（Safari 16.4+ で対応）
- Form-associated custom elements の対応が遅い
- Declarative Shadow DOM の `shadowrootmode` が Safari 16.4+ で対応
- `:focus-visible` 疑似クラスの挙動が微妙に異なる場合がある

---

## 15. パフォーマンス計測

### 15.1 DOM 操作のパフォーマンス計測手法

```javascript
// Performance API でDOM操作の所要時間を計測
function measureDOMOperation(label, operation) {
  performance.mark(`${label}-start`);

  operation();

  performance.mark(`${label}-end`);
  performance.measure(label, `${label}-start`, `${label}-end`);

  const measure = performance.getEntriesByName(label)[0];
  console.log(`${label}: ${measure.duration.toFixed(2)}ms`);

  // クリーンアップ
  performance.clearMarks(`${label}-start`);
  performance.clearMarks(`${label}-end`);
  performance.clearMeasures(label);

  return measure.duration;
}

// 使用例: innerHTML vs DocumentFragment の比較
const container = document.getElementById('test');
const items = Array.from({ length: 5000 }, (_, i) => `Item ${i}`);

// innerHTML
measureDOMOperation('innerHTML', () => {
  container.innerHTML = items.map(item => `<div class="item">${item}</div>`).join('');
});

// DocumentFragment
measureDOMOperation('DocumentFragment', () => {
  container.innerHTML = '';
  const fragment = document.createDocumentFragment();
  items.forEach(item => {
    const div = document.createElement('div');
    div.className = 'item';
    div.textContent = item;
    fragment.appendChild(div);
  });
  container.appendChild(fragment);
});

// PerformanceObserver でレイアウトシフトを監視
const layoutShiftObserver = new PerformanceObserver((list) => {
  for (const entry of list.getEntries()) {
    if (entry.entryType === 'layout-shift' && !entry.hadRecentInput) {
      console.warn('Layout shift detected:', entry.value, entry.sources);
    }
  }
});
layoutShiftObserver.observe({ type: 'layout-shift', buffered: true });
```

---

## 16. FAQ

### Q1: querySelector と getElementById はどちらを使うべきか?

**回答:** 一般的には `querySelector` / `querySelectorAll` を推奨する。CSS セレクタの柔軟性があり、静的な NodeList を返すためループ中の安全性が高い。ただし、ID による単一要素の取得で最高のパフォーマンスが求められる場面（例えばアニメーションフレーム内で毎フレーム呼ばれるなど）では `getElementById` がわずかに高速である。日常的な開発ではこの差は無視できるため、コードの一貫性を重視して `querySelector` 系に統一するチームが多い。

### Q2: MutationObserver のコールバック内で DOM を変更するとどうなるか?

**回答:** MutationObserver のコールバック内で DOM を変更すると、その変更は次のマイクロタスクキューで再度 MutationObserver に通知される。直接的な無限ループにはならない（同期的に再帰しないため）が、コールバック → DOM 変更 → コールバック → DOM 変更 ... という連鎖が起こりうる。これを防ぐには、コールバック内で変更を行う前に `observer.disconnect()` し、変更後に再度 `observer.observe()` するか、フラグ変数で再帰を防止する。

```javascript
let isUpdating = false;

const observer = new MutationObserver((mutations) => {
  if (isUpdating) return;  // 自身の変更による通知を無視

  isUpdating = true;
  // DOM 変更処理
  element.setAttribute('data-count', String(mutations.length));
  isUpdating = false;
});
```

### Q3: Shadow DOM を使うと SEO に影響はあるか?

**回答:** 検索エンジンのクローラ（特に Googlebot）は JavaScript を実行し、Shadow DOM のコンテンツもインデックスする能力を持つ。ただし、Light DOM のコンテンツ（スロットに挿入されるコンテンツ）の方が確実にインデックスされるため、SEO 上重要なテキストは Light DOM 側に配置し、Shadow DOM 内では構造とスタイリングのみを担当させるのが安全な設計である。Declarative Shadow DOM を使えば、HTML のソース上に Shadow DOM の構造が存在するため、JavaScript 実行を待たずにコンテンツが利用可能になり、SSR との組み合わせで SEO への影響を最小化できる。

### Q4: Virtual DOM は本当に「速い」のか?

**回答:** 正確には「手動の DOM 操作より速い」わけではなく、「最適化されていない DOM 操作より安全に速い」という表現が適切である。手動で最小限の DOM 操作を正確に行えるなら、それが最速である。しかし、複雑な UI の状態遷移において、開発者が毎回最小限の差分を計算して DOM 操作を行うのは困難であり、バグの温床になる。Virtual DOM は「宣言的に UI を記述でき、かつ十分に高速」という開発体験とパフォーマンスのバランスを実現する。Svelte のようなコンパイル時アプローチが Virtual DOM のオーバーヘッドを回避しつつ宣言的な開発体験を提供する点も、近年注目されている。

### Q5: Web Components は React や Vue の代替になるか?

**回答:** Web Components と React/Vue は競合ではなく、レイヤーが異なる。Web Components はブラウザネイティブのコンポーネント機構であり、フレームワーク非依存の再利用可能な UI パーツを作るのに適している。一方、React/Vue は状態管理、ルーティング、エコシステム全体を含む包括的なフレームワークである。デザインシステムの基盤を Web Components で構築し、アプリケーション層を React/Vue で構築するハイブリッドアプローチも有効である。Lit や Stencil など、Web Components の開発体験を向上させるライブラリも成熟してきている。

### Q6: innerHTML と textContent の使い分けは?

**回答:** `innerHTML` はHTML文字列をパースして要素として挿入するため、タグを含むコンテンツを設定する場合に使用する。一方、`textContent` は文字列をそのままテキストノードとして挿入するため、HTMLタグはエスケープされて表示される。**セキュリティ上の理由から、ユーザー入力をそのまま表示する場合は必ず `textContent` を使用する。** `innerHTML` でユーザー入力を扱うとXSS（クロスサイトスクリプティング）攻撃のリスクがある。また、`textContent` は単純なテキスト挿入のため、`innerHTML` より高速である。HTML構造を動的に生成する必要がある場合は `createElement` + `appendChild` または `insertAdjacentHTML` を検討する。パフォーマンス面では、大量のテキストを設定する際は `textContent` が最も効率的である。

### Q7: 仮想DOMとリアルDOMの違いとは?

**回答:** リアルDOM（Real DOM）はブラウザが実際にメモリ上に保持する文書構造であり、DOM APIを通じて直接操作する。仮想DOM（Virtual DOM）はReactやVueなどのフレームワークが採用するメモリ上のJavaScriptオブジェクトで、リアルDOMの軽量な表現である。仮想DOMのメリットは、宣言的なUI記述と効率的な差分更新（diff/patch）の両立にある。開発者は「最終的なUIの状態」だけを記述し、フレームワークが前回の仮想DOMと比較して最小限のリアルDOM操作を自動生成する。これにより、手動で差分計算を行う複雑さから解放され、バグを減らしつつパフォーマンスを維持できる。ただし、仮想DOMの比較処理自体にもコストがあるため、極めて単純なDOM操作では直接操作の方が高速な場合もある。Svelteのようなコンパイル時アプローチは、仮想DOMのオーバーヘッドを回避しつつ宣言的な開発体験を提供する。

### Q8: DOM操作のパフォーマンス最適化のポイントは?

**回答:** DOM操作の最適化には以下のポイントがある。**(1) Layout Thrashingの回避**: 読み取り（offsetHeight等）と書き込み（style変更等）を交互に行うと、ブラウザが毎回レイアウト再計算を強制される。`fastdom` ライブラリや手動でのバッチ処理により、読み取りをまとめて行い、その後書き込みをまとめて行う。**(2) DocumentFragmentの活用**: 複数要素を挿入する際は、DocumentFragmentに一度追加してから一括挿入することで、レンダリング回数を削減できる。**(3) `requestAnimationFrame` の使用**: アニメーションやスクロール連動処理では、ブラウザの再描画タイミングに合わせて処理を実行することで、滑らかな描画を実現できる。**(4) イベント委任**: 多数の子要素にイベントリスナーを個別に設定せず、親要素で一括処理することでメモリ使用量とイベント登録コストを削減できる。**(5) `will-change` CSSプロパティ**: 頻繁に変更されるプロパティを事前にブラウザに通知することで、最適化のヒントを与えられる（ただし過度な使用はメモリを圧迫する）。

---

## 17. まとめ

### 要点の整理

| 概念 | 重要ポイント | 典型的な使用場面 |
|------|-------------|----------------|
| DOM ツリー構造 | ノード型の理解、Element 専用ナビゲーション | ツリー走査、構造分析 |
| 要素の取得 | 静的 NodeList vs ライブ HTMLCollection の区別 | 要素検索、フィルタリング |
| CRUD 操作 | DocumentFragment、insertAdjacentHTML | 大量挿入、部分更新 |
| レンダリング | Layout Thrashing 回避、読み書き分離 | パフォーマンス最適化 |
| イベントモデル | キャプチャ / バブリング / 委任 / passive | ユーザインタラクション |
| MutationObserver | 非同期バッチ通知、attributeFilter | DOM 変更監視 |
| Shadow DOM | スタイル隔離、スロット、ライフサイクル | Web Components |
| Virtual DOM | diff/patch、宣言的 UI | React/Vue 等のSPA |
| テンプレート | `<template>` 要素、Declarative Shadow DOM | 効率的な要素生成 |

### キーポイント

1. **DOM操作はレンダリングパフォーマンスに直結する**: Layout Thrashingを避けるため、読み取り操作と書き込み操作を分離し、DocumentFragmentやrequestAnimationFrameを活用してバッチ処理を行う。直接DOM操作が最速だが、複雑なUIでは仮想DOMによる宣言的アプローチが開発効率とパフォーマンスのバランスを取る。

2. **イベント処理の効率化が重要**: イベント委任により親要素で一括処理することでメモリとCPUコストを削減し、`passive: true` オプションでスクロールパフォーマンスを向上させる。MutationObserverを使えばDOM変更を非同期バッチで監視でき、ポーリングによるCPU浪費を避けられる。

3. **Shadow DOMとWeb Componentsでカプセル化を実現**: Shadow DOMによりスタイルとDOMを隔離し、グローバルCSSの競合を防ぐ。スロット機構でコンテンツ投影を行い、Custom ElementsとShadow DOMを組み合わせることでフレームワーク非依存の再利用可能なコンポーネントを構築できる。

### チェックリスト

- [ ] `querySelector` と `getElementsBy` の違い（静的 vs ライブ）を説明できる
- [ ] DocumentFragment を使ったバッチ挿入を実装できる
- [ ] Layout Thrashing とその回避策を説明できる
- [ ] イベントキャプチャ / バブリング / 委任の使い分けができる
- [ ] `passive` オプションの意義を説明できる
- [ ] MutationObserver で DOM 変更を監視する処理を書ける
- [ ] Shadow DOM の隔離境界とスタイルの挙動を説明できる
- [ ] Virtual DOM と直接 DOM 操作の長所短所を比較できる
- [ ] メモリリークの原因と AbortController による対策を実装できる
- [ ] Web Components のライフサイクルコールバックを列挙できる

---

## FAQ

### Q1: innerHTMLとtextContentとinnerTextの違いは何ですか?
`innerHTML` はHTML文字列としてノードの内容を取得・設定します。HTMLタグが解釈されるため、XSS脆弱性の原因になりえます。`textContent` はノードとその子孫の全テキストコンテンツを取得・設定し、HTMLタグは純粋なテキストとして扱われます。非表示要素（`display: none`）のテキストも含まれ、レイアウトの再計算を引き起こしません。`innerText` は「表示されているテキスト」を返し、CSSを考慮してレンダリング結果に基づくテキストを取得するため、`display: none` の要素は除外され、レイアウト再計算が発生します。パフォーマンスを重視する場合は `textContent` を使用してください。

### Q2: addEventListener の第三引数にはどのような値を指定できますか?
第三引数には真偽値（キャプチャフェーズで処理するか）またはオプションオブジェクトを指定できます。オプションオブジェクトでは `capture`（キャプチャフェーズで処理）、`once`（一度だけ実行して自動解除）、`passive`（preventDefault()を呼ばないことを宣言しスクロールパフォーマンスを向上）、`signal`（AbortSignal でリスナーを一括解除）を指定できます。特に `passive: true` はタッチイベントやスクロールイベントで重要で、ブラウザはスクロールの阻止がないことを前提に最適化を行えます。

### Q3: Virtual DOMと直接DOM操作のどちらを選ぶべきですか?
UIの複雑さとチームの規模によって判断します。状態変化に応じてUIの多くの部分が連動して更新されるアプリケーション（ダッシュボード、チャット、フォームウィザードなど）では、Virtual DOM を採用するフレームワーク（React、Vue）が開発効率とバグの少なさの面で優れます。一方、軽量なウィジェット、パフォーマンスが極めて重要な部分、フレームワーク非依存のライブラリでは、直接DOM操作の方がオーバーヘッドが少なく高速です。両者は排他的ではなく、React内でも `ref` を使って直接DOM操作を行う場面があります。

---

## まとめ

このガイドでは以下を学びました:

- DOMツリーの構造とノード型の分類、及びDOMとCSSOMの関係
- 要素の取得・作成・挿入・削除の効率的なパターンとDocumentFragmentによるバッチ処理
- Layout Thrashingの原因と回避策（読み書き分離、requestAnimationFrameの活用）
- イベントモデル（キャプチャ/バブリング/委任）とpassiveオプションによるパフォーマンス最適化
- Shadow DOMによるスタイル・DOMの隔離とWeb Componentsの設計パターン

---

## 次に読むべきガイド

- [01-fetch-and-streams.md](./01-fetch-and-streams.md) -- Fetch と Streams
- [02-intersection-resize-observer.md](./02-intersection-resize-observer.md) -- Observer API (Intersection, Resize, Mutation)
- [../04-storage-and-caching/00-web-storage.md](../04-storage-and-caching/00-web-storage.md) -- Web Storage API

---

## 参考文献

1. WHATWG. "DOM Living Standard." https://dom.spec.whatwg.org/ , 2024.
2. MDN Web Docs. "Document Object Model (DOM)." Mozilla, https://developer.mozilla.org/en-US/docs/Web/API/Document_Object_Model , 2024.
3. Google Developers. "Rendering Performance." https://web.dev/rendering-performance/ , 2024.
4. MDN Web Docs. "Web Components." Mozilla, https://developer.mozilla.org/en-US/docs/Web/API/Web_components , 2024.
5. WHATWG. "HTML Living Standard - Shadow DOM." https://html.spec.whatwg.org/multipage/scripting.html , 2024.
6. Wilcox, Jason. "Virtual DOM is pure overhead." Svelte Blog, https://svelte.dev/blog/virtual-dom-is-pure-overhead , 2018.

