# Webストレージ

> ブラウザに備わる複数のストレージ機構を理解する。Cookie、localStorage、sessionStorage、IndexedDB、Cache APIの特性を比較し、用途に応じた最適な選択を行う。

## この章で学ぶこと

- [ ] 各ストレージの特性と容量制限を理解する
- [ ] 用途に応じたストレージの選択基準を把握する
- [ ] IndexedDBの基本操作を学ぶ

---

## 1. ストレージの比較

```
┌──────────────┬─────────┬───────────┬──────────┬────────────┐
│              │ Cookie  │ localStorage│ session  │ IndexedDB  │
├──────────────┼─────────┼───────────┼──────────┼────────────┤
│ 容量         │ 4KB     │ 5-10MB    │ 5-10MB   │ 制限なし(※)│
│ 有効期間     │ 設定可能│ 永続      │ タブ閉じまで│ 永続      │
│ サーバー送信 │ 自動    │ なし      │ なし     │ なし       │
│ API          │ 文字列  │ 文字列    │ 文字列   │ 非同期     │
│ Worker利用   │ ✗      │ ✗        │ ✗       │ ✓         │
│ 検索         │ ✗      │ ✗        │ ✗       │ ✓(インデックス)│
│ トランザクション│ ✗   │ ✗        │ ✗       │ ✓         │
└──────────────┴─────────┴───────────┴──────────┴────────────┘
※ IndexedDB はブラウザの空き容量に依存（通常数百MB〜GB）

用途の選択:
  認証トークン:    Cookie（HttpOnly, Secure）
  ユーザー設定:    localStorage
  フォーム一時保存: sessionStorage
  大量データ:      IndexedDB
  オフラインキャッシュ: Cache API
```

---

## 2. localStorage / sessionStorage

```javascript
// localStorage — 永続的なキーバリューストア
localStorage.setItem('theme', 'dark');
const theme = localStorage.getItem('theme');  // 'dark'
localStorage.removeItem('theme');
localStorage.clear();

// オブジェクトの保存（JSONシリアライズ）
const user = { name: 'Taro', age: 25 };
localStorage.setItem('user', JSON.stringify(user));
const saved = JSON.parse(localStorage.getItem('user'));

// sessionStorage — タブ単位で独立、タブ閉じで消える
sessionStorage.setItem('formDraft', JSON.stringify(formData));

// storage イベント（他のタブからの変更を検知）
window.addEventListener('storage', (event) => {
  console.log(`${event.key}: ${event.oldValue} → ${event.newValue}`);
  // 他のタブでの変更を同期
});

// 注意点:
// ✗ 同期API（大量データでメインスレッドをブロック）
// ✗ 文字列のみ（オブジェクトはJSON変換が必要）
// ✗ 容量制限（5-10MB）
// ✗ セキュリティ: XSSで全データ読み取り可能
// → 機密データは保存しない
```

---

## 3. IndexedDB

```javascript
// IndexedDB — 大容量の構造化データストア

// データベースを開く
function openDB(name, version) {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(name, version);

    request.onupgradeneeded = (event) => {
      const db = event.target.result;
      if (!db.objectStoreNames.contains('users')) {
        const store = db.createObjectStore('users', { keyPath: 'id' });
        store.createIndex('email', 'email', { unique: true });
        store.createIndex('age', 'age');
      }
    };

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

// CRUD操作
async function addUser(user) {
  const db = await openDB('myApp', 1);
  const tx = db.transaction('users', 'readwrite');
  tx.objectStore('users').add(user);
  return tx.complete;
}

async function getUser(id) {
  const db = await openDB('myApp', 1);
  const tx = db.transaction('users', 'readonly');
  return new Promise((resolve) => {
    const request = tx.objectStore('users').get(id);
    request.onsuccess = () => resolve(request.result);
  });
}

// ライブラリ推奨: idb（Promise ラッパー）
import { openDB } from 'idb';

const db = await openDB('myApp', 1, {
  upgrade(db) {
    db.createObjectStore('users', { keyPath: 'id' });
  },
});

await db.add('users', { id: 1, name: 'Taro', email: 'taro@example.com' });
const user = await db.get('users', 1);
const allUsers = await db.getAll('users');
```

---

## 4. Cookie

```javascript
// Cookie — サーバーと自動送信される小さなデータ

// 読み取り
const cookies = document.cookie; // "name=value; name2=value2"

// 書き込み
document.cookie = "theme=dark; max-age=86400; path=/; SameSite=Lax";

// 削除（max-age=0）
document.cookie = "theme=; max-age=0; path=/";

// 推奨: Cookie は認証にのみ使用し、サーバー側で設定
// Set-Cookie: session=abc; HttpOnly; Secure; SameSite=Lax; Path=/

// js-cookie ライブラリ（便利なラッパー）
import Cookies from 'js-cookie';
Cookies.set('theme', 'dark', { expires: 7 });
const theme = Cookies.get('theme');
Cookies.remove('theme');
```

---

## 5. ストレージ容量の確認

```javascript
// Storage API（容量の確認）
if (navigator.storage) {
  const estimate = await navigator.storage.estimate();
  console.log({
    usage: (estimate.usage / 1024 / 1024).toFixed(2) + 'MB',
    quota: (estimate.quota / 1024 / 1024).toFixed(2) + 'MB',
    percent: (estimate.usage / estimate.quota * 100).toFixed(1) + '%',
  });
}

// 永続ストレージのリクエスト
const persisted = await navigator.storage.persist();
// true: ブラウザが許可（データが自動削除されない）
// false: ブラウザが拒否（容量圧迫時に削除される可能性）

// ストレージの退去（Eviction）:
// ブラウザはストレージ圧迫時に以下の順で削除:
// 1. 最も長く使われていないオリジン（LRU）
// 2. Service Worker キャッシュ
// 3. IndexedDB
// → persist() で保護可能
```

---

## まとめ

| ストレージ | 容量 | 用途 |
|-----------|------|------|
| Cookie | 4KB | 認証（HttpOnly） |
| localStorage | 5-10MB | ユーザー設定、テーマ |
| sessionStorage | 5-10MB | フォーム一時保存 |
| IndexedDB | GB級 | 大量データ、オフラインDB |
| Cache API | GB級 | HTTP レスポンスキャッシュ |

---

## 次に読むべきガイド
→ [[01-service-worker-cache.md]] — Service Worker

---

## 参考文献
1. MDN Web Docs. "Web Storage API." Mozilla, 2024.
2. MDN Web Docs. "IndexedDB API." Mozilla, 2024.
