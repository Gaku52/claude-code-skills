# DOM API

> DOMはHTMLをJavaScriptで操作するためのAPI。効率的なDOM操作、MutationObserver、Shadow DOMを理解し、パフォーマンスを意識した実装を行う。

## この章で学ぶこと

- [ ] 効率的なDOM操作の方法を理解する
- [ ] MutationObserverの使い方を把握する
- [ ] Shadow DOMの概念と用途を学ぶ

---

## 1. DOM操作の基本

```javascript
// 要素の取得
document.getElementById('app');           // ID
document.querySelector('.card');          // CSSセレクタ（最初の1つ）
document.querySelectorAll('.card');       // CSSセレクタ（全て）
document.getElementsByClassName('card');  // クラス名（ライブコレクション）

// querySelector vs getElement:
// querySelector:       静的 NodeList（呼び出し時点のスナップショット）
// getElementsBy...:    ライブ HTMLCollection（DOMの変更が反映される）

// 要素の作成と追加
const div = document.createElement('div');
div.className = 'card';
div.textContent = 'Hello';
parent.appendChild(div);

// 効率的な複数要素の追加
const fragment = document.createDocumentFragment();
for (let i = 0; i < 100; i++) {
  const li = document.createElement('li');
  li.textContent = `Item ${i}`;
  fragment.appendChild(li);
}
list.appendChild(fragment);  // 1回のDOM操作

// innerHTML vs textContent vs innerText:
element.innerHTML = '<b>HTML</b>';     // HTMLとしてパース（XSSリスク）
element.textContent = '<b>安全</b>';   // テキストとして（安全）
element.innerText = 'テキスト';        // 表示テキスト（レイアウト考慮、遅い）
```

---

## 2. 効率的なDOM操作

```javascript
// バッチ処理（Layout Thrashing を避ける）

// 悪い例: 読み書き交互
elements.forEach(el => {
  const height = el.offsetHeight;      // 読み（Layout強制）
  el.style.height = height * 2 + 'px'; // 書き
});

// 良い例: 読みをまとめてから書く
const heights = elements.map(el => el.offsetHeight);
elements.forEach((el, i) => {
  el.style.height = heights[i] * 2 + 'px';
});

// insertAdjacentHTML — innerHTML より効率的な挿入
element.insertAdjacentHTML('beforeend', '<div>new content</div>');
// 'beforebegin': 要素の前
// 'afterbegin': 要素の最初の子として
// 'beforeend': 要素の最後の子として
// 'afterend': 要素の後

// classList API
element.classList.add('active');
element.classList.remove('active');
element.classList.toggle('active');
element.classList.contains('active');
element.classList.replace('old', 'new');

// dataset API
// <div data-user-id="123" data-role="admin">
element.dataset.userId;   // "123"
element.dataset.role;     // "admin"
```

---

## 3. MutationObserver

```javascript
// MutationObserver = DOM変更の監視

const observer = new MutationObserver((mutations) => {
  for (const mutation of mutations) {
    switch (mutation.type) {
      case 'childList':
        console.log('子要素が変更:', mutation.addedNodes, mutation.removedNodes);
        break;
      case 'attributes':
        console.log('属性が変更:', mutation.attributeName, mutation.target);
        break;
      case 'characterData':
        console.log('テキストが変更:', mutation.target.textContent);
        break;
    }
  }
});

observer.observe(targetElement, {
  childList: true,      // 子要素の追加/削除
  attributes: true,     // 属性の変更
  characterData: true,  // テキストの変更
  subtree: true,        // 子孫要素も監視
  attributeFilter: ['class', 'style'], // 監視する属性を限定
});

// 監視停止
observer.disconnect();

// 用途:
// → サードパーティスクリプトのDOM変更を検知
// → 動的コンテンツの変更に応じた処理
// → Virtual DOM のないフレームワークでの差分検知
```

---

## 4. Shadow DOM

```javascript
// Shadow DOM = DOMの隔離されたスコープ

// Shadow DOM の構造:
// <host-element>
//   #shadow-root
//     <style>...</style>  ← 外部CSSの影響を受けない
//     <slot></slot>       ← Light DOMの子が入る
//   (end shadow-root)
//   <span>Light DOM content</span>  ← slotに入る
// </host-element>

// Custom Element + Shadow DOM
class MyCard extends HTMLElement {
  constructor() {
    super();
    const shadow = this.attachShadow({ mode: 'open' });
    shadow.innerHTML = `
      <style>
        :host {
          display: block;
          border: 1px solid #ccc;
          border-radius: 8px;
          padding: 16px;
        }
        .title {
          font-size: 1.5em;
          font-weight: bold;
        }
      </style>
      <div class="title">
        <slot name="title">Default Title</slot>
      </div>
      <div class="content">
        <slot></slot>
      </div>
    `;
  }
}

customElements.define('my-card', MyCard);

// 使用:
// <my-card>
//   <span slot="title">Custom Title</span>
//   <p>Card content here</p>
// </my-card>

// Shadow DOM のメリット:
// ✓ CSSの完全な隔離（外部スタイルの影響なし）
// ✓ DOMの隔離（querySelector で外からアクセス不可）
// ✓ Web Components の基盤
// ✓ 再利用可能なコンポーネント

// mode: 'open' vs 'closed':
// open:   element.shadowRoot でアクセス可能
// closed: アクセス不可（完全な隠蔽）
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| DOM操作 | DocumentFragment でバッチ、読み書き分離 |
| MutationObserver | DOM変更の非同期監視 |
| Shadow DOM | CSS/DOMの隔離、Web Components基盤 |
| Custom Elements | HTMLタグの自作 |

---

## 次に読むべきガイド
→ [[01-fetch-and-streams.md]] — Fetch と Streams

---

## 参考文献
1. MDN Web Docs. "Document Object Model (DOM)." Mozilla, 2024.
2. HTML Living Standard. "Shadow DOM." WHATWG, 2024.
