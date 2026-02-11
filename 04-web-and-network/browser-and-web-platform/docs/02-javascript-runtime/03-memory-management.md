# メモリ管理

> ブラウザのメモリ管理を理解し、メモリリークを検出・修正する。JavaScriptのガベージコレクション、メモリプロファイリング、よくあるリークパターンと対策を学ぶ。

## この章で学ぶこと

- [ ] JavaScriptのメモリモデルを理解する
- [ ] メモリリークのパターンと検出方法を把握する
- [ ] Chrome DevToolsでのプロファイリングを学ぶ

---

## 1. メモリモデル

```
JavaScriptのメモリ:

  スタック:                   ヒープ:
  ┌────────────────┐        ┌────────────────────────┐
  │ プリミティブ値  │        │ オブジェクト            │
  │ 参照（ポインタ）│───→    │ 配列                    │
  │ 関数コール情報  │        │ 関数（クロージャ）      │
  └────────────────┘        │ Map, Set                │
                             │ DOM ノード              │
                             └────────────────────────┘

  GCの対象判定:
  → ルート（グローバル変数、コールスタック）から到達可能か
  → 到達不能なオブジェクトがGC対象

  ルート:
  ├── window / globalThis
  ├── 現在のコールスタック上の変数
  ├── DOM ツリー
  └── アクティブな setTimeout / setInterval
```

---

## 2. メモリリークのパターン

```
① グローバル変数の意図しない作成:
  function leak() {
    leakedVar = 'oops';  // var/let/const なし → グローバル
  }
  対策: 'use strict' を使用

② タイマーのクリア忘れ:
  // リーク
  setInterval(() => {
    const data = fetchData();
    updateUI(data);
  }, 1000);
  // コンポーネント破棄後も動き続ける

  // 修正
  const id = setInterval(callback, 1000);
  // クリーンアップ時
  clearInterval(id);

③ イベントリスナーの解除忘れ:
  // リーク
  element.addEventListener('click', handler);
  // element が DOM から削除されても handler が参照を保持

  // 修正
  element.addEventListener('click', handler);
  // クリーンアップ
  element.removeEventListener('click', handler);
  // または { once: true } を使用

④ クロージャによるリーク:
  function createLeak() {
    const largeData = new Array(1000000);
    return function() {
      // largeData を参照していなくても、
      // 同じスコープの変数は保持される場合がある
      console.log('callback');
    };
  }

⑤ DOM 参照の保持:
  const elements = [];
  function addElement() {
    const div = document.createElement('div');
    document.body.appendChild(div);
    elements.push(div);  // 配列が参照を保持
  }
  function removeElement() {
    document.body.removeChild(elements[0]);
    // elements 配列にはまだ参照が残っている → リーク
    elements.shift();  // これも忘れずに
  }

⑥ React でのリーク:
  useEffect(() => {
    const controller = new AbortController();
    fetch('/api/data', { signal: controller.signal })
      .then(res => res.json())
      .then(data => setState(data));  // アンマウント後に setState

    return () => controller.abort();  // クリーンアップ
  }, []);
```

---

## 3. WeakRef と WeakMap

```
WeakRef / WeakMap = GCを妨げない参照

  // 通常の参照（GCを妨げる）
  const cache = new Map();
  cache.set(key, largeObject);  // largeObject はGCされない

  // WeakMap（GCを妨げない）
  const cache = new WeakMap();
  cache.set(key, largeObject);  // key がGCされると entry も消える

  // WeakRef（弱い参照）
  const ref = new WeakRef(largeObject);
  // 後で参照を取得（GC済みなら undefined）
  const obj = ref.deref();
  if (obj) {
    // まだ生きている
  }

FinalizationRegistry:
  // オブジェクトがGCされた時にコールバック
  const registry = new FinalizationRegistry((value) => {
    console.log(`${value} was garbage collected`);
  });

  registry.register(someObject, 'myObject');
```

---

## 4. DevTools でのプロファイリング

```
Chrome DevTools Memory パネル:

  ① Heap Snapshot:
     → ある時点のメモリの全オブジェクトを記録
     → オブジェクトの保持者（retainer）を確認

     手順:
     1. DevTools > Memory > Heap snapshot
     2. 「Take snapshot」ボタン
     3. Summary / Comparison / Containment ビューで分析

     Comparison:
     → 2つのスナップショットの差分
     → リーク特定に最も有効
     → 操作前と操作後で比較

  ② Allocation Timeline:
     → 時間軸でメモリ割り当てを記録
     → どの操作でメモリが増えたか特定
     → 青いバー = 生存、灰色バー = GC済み

  ③ Allocation Sampling:
     → サンプリングベース（低オーバーヘッド）
     → 長時間のプロファイリングに適する

リーク検出の手順:
  1. 初期状態でHeap Snapshot（Snapshot 1）
  2. リークが疑われる操作を実行
  3. GCを強制実行（DevToolsのゴミ箱アイコン）
  4. 再度Heap Snapshot（Snapshot 2）
  5. Comparison ビューで Snapshot 1 と比較
  6. #New（新規オブジェクト数）が多いものを調査
  7. Retainers で何がオブジェクトを保持しているか確認

Performance.memory API:
  console.log({
    usedJSHeapSize: performance.memory.usedJSHeapSize / 1024 / 1024 + 'MB',
    totalJSHeapSize: performance.memory.totalJSHeapSize / 1024 / 1024 + 'MB',
    jsHeapSizeLimit: performance.memory.jsHeapSizeLimit / 1024 / 1024 + 'MB',
  });
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| GC | ルートから到達不能なオブジェクトを解放 |
| リークパターン | タイマー、リスナー、クロージャ、DOM参照 |
| WeakMap/WeakRef | GCを妨げない参照 |
| プロファイリング | Heap Snapshot の Comparison ビュー |

---

## 次に読むべきガイド
→ [[../03-web-apis/00-dom-api.md]] — DOM API

---

## 参考文献
1. Chrome DevTools. "Fix memory problems." Google, 2024.
2. Addy Osmani. "JavaScript Memory Management." 2012.
