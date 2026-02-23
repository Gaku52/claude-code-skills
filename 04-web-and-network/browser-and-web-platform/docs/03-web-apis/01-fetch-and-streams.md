# Fetch と Streams

> Fetch APIはXMLHttpRequestの後継として策定されたモダンなHTTPクライアントAPI。Streams APIと組み合わせることで、大きなレスポンスの段階的処理、進捗表示、AbortControllerによるキャンセルを実現する。本章ではFetch APIの基礎から高度なパターン、Streams APIによるストリーミング処理、実務でのベストプラクティスまでを包括的に解説する。

## この章で学ぶこと

- [ ] Fetch APIの基本と高度な使い方を理解する
- [ ] Request / Response オブジェクトの詳細を把握する
- [ ] Streams APIでのストリーミング処理を実装できるようになる
- [ ] AbortControllerによるリクエストキャンセルとタイムアウトを学ぶ
- [ ] 実務レベルのfetchラッパーとエラーハンドリングを構築する
- [ ] Server-Sent Events / NDJSON / チャンク転送のストリーム処理を理解する
- [ ] テスト戦略とモック手法を把握する

---

## 1. Fetch APIの基礎

### 1.1 XMLHttpRequestからFetch APIへの進化

XMLHttpRequest（XHR）はAjaxの基盤として長年使われてきたが、コールバックベースのAPIは複雑になりやすく、ストリーミング処理のサポートも限定的だった。Fetch APIはこれらの課題を解決するために設計された。

```javascript
// XMLHttpRequest（旧来のパターン）
function fetchDataXHR(url, callback) {
  const xhr = new XMLHttpRequest();
  xhr.open('GET', url);
  xhr.onreadystatechange = function () {
    if (xhr.readyState === 4) {
      if (xhr.status === 200) {
        callback(null, JSON.parse(xhr.responseText));
      } else {
        callback(new Error(`HTTP ${xhr.status}`));
      }
    }
  };
  xhr.onerror = function () {
    callback(new Error('Network error'));
  };
  xhr.send();
}

// Fetch API（モダンなパターン）
async function fetchData(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }
  return response.json();
}
```

Fetch APIの主な利点を以下にまとめる。

| 特徴 | XMLHttpRequest | Fetch API |
|------|---------------|-----------|
| 非同期モデル | コールバック | Promise |
| ストリーミング | 限定的 | ReadableStream |
| リクエストキャンセル | xhr.abort() | AbortController |
| CORS制御 | 限定的 | mode オプション |
| キャッシュ制御 | 手動ヘッダー | cache オプション |
| Service Worker連携 | 不可 | FetchEvent で統一 |
| 構文の簡潔さ | 冗長 | シンプル |

### 1.2 基本的なGETリクエスト

```javascript
// 最もシンプルなGET
const response = await fetch('/api/users');
const users = await response.json();
console.log(users);

// URLSearchParams によるクエリパラメータの構築
const params = new URLSearchParams({
  page: '1',
  limit: '20',
  sort: 'created_at',
  order: 'desc',
});
const response = await fetch(`/api/users?${params}`);
const data = await response.json();

// 配列パラメータの追加
const params = new URLSearchParams();
params.append('tag', 'javascript');
params.append('tag', 'typescript');
params.append('tag', 'react');
// → tag=javascript&tag=typescript&tag=react

// URL オブジェクトとの組み合わせ
const url = new URL('/api/search', 'https://api.example.com');
url.searchParams.set('q', 'fetch api');
url.searchParams.set('lang', 'ja');
const response = await fetch(url);
// → https://api.example.com/api/search?q=fetch+api&lang=ja
```

### 1.3 POSTリクエスト

```javascript
// JSON送信
const response = await fetch('/api/users', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    name: '田中太郎',
    email: 'taro@example.com',
    role: 'admin',
  }),
});

if (!response.ok) {
  throw new Error(`HTTP ${response.status}: ${response.statusText}`);
}

const created = await response.json();
console.log('Created user:', created.id);

// FormData送信（ファイルアップロード含む）
const formData = new FormData();
formData.append('name', '田中太郎');
formData.append('avatar', fileInput.files[0]);
formData.append('documents', file1);
formData.append('documents', file2);

// FormDataの場合、Content-Typeは自動設定される（boundary含む）
const response = await fetch('/api/users', {
  method: 'POST',
  body: formData,
  // headers: { 'Content-Type': 'multipart/form-data' } は設定しない！
});

// URLSearchParams送信（application/x-www-form-urlencoded）
const body = new URLSearchParams({
  username: 'taro',
  password: 'secret123',
  grant_type: 'password',
});

const response = await fetch('/oauth/token', {
  method: 'POST',
  body, // Content-Typeは自動でapplication/x-www-form-urlencodedになる
});
```

### 1.4 PUT / PATCH / DELETEリクエスト

```javascript
// PUTリクエスト（リソース全体の置換）
const response = await fetch(`/api/users/${userId}`, {
  method: 'PUT',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    name: '田中太郎',
    email: 'taro@example.com',
    role: 'editor',
    active: true,
  }),
});

// PATCHリクエスト（部分更新）
const response = await fetch(`/api/users/${userId}`, {
  method: 'PATCH',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    role: 'admin',
  }),
});

// JSON Patch形式（RFC 6902）
const response = await fetch(`/api/users/${userId}`, {
  method: 'PATCH',
  headers: { 'Content-Type': 'application/json-patch+json' },
  body: JSON.stringify([
    { op: 'replace', path: '/role', value: 'admin' },
    { op: 'add', path: '/permissions/-', value: 'manage_users' },
    { op: 'remove', path: '/temporaryFlag' },
  ]),
});

// DELETEリクエスト
const response = await fetch(`/api/users/${userId}`, {
  method: 'DELETE',
});

if (response.status === 204) {
  console.log('Successfully deleted (no content)');
} else if (response.ok) {
  const result = await response.json();
  console.log('Deleted:', result);
}

// DELETEリクエストにボディを含める場合
const response = await fetch('/api/users/batch', {
  method: 'DELETE',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    ids: [1, 2, 3, 4, 5],
    reason: 'Account cleanup',
  }),
});
```

### 1.5 fetchの重要な注意点

```javascript
// ★ 注意1: fetchはネットワークエラー時のみrejectする
// → 404や500はrejectされない！
try {
  const response = await fetch('/api/nonexistent');
  // response.status === 404 だが、catchには入らない
  console.log(response.ok); // false
  console.log(response.status); // 404
} catch (err) {
  // ネットワーク切断・DNS解決失敗・CORSエラー等の場合のみ
  console.error('Network error:', err);
}

// ★ 注意2: レスポンスボディは1回しか読めない
const response = await fetch('/api/data');
const json = await response.json();
// const text = await response.text(); // エラー！bodyは消費済み

// 複数回読みたい場合はcloneを使う
const response = await fetch('/api/data');
const clone = response.clone();
const json = await response.json();
const text = await clone.text(); // これはOK

// ★ 注意3: cookieのデフォルト送信挙動
// same-originリクエストではcookieが送信される（credentials: 'same-origin'がデフォルト）
// cross-originリクエストではcookieは送信されない
// cross-originでcookieを送信するにはcredentials: 'include'が必要
const response = await fetch('https://other-domain.com/api/data', {
  credentials: 'include', // クロスオリジンでcookieを送信
});

// ★ 注意4: リダイレクトの処理
const response = await fetch('/api/redirect', {
  redirect: 'follow',  // デフォルト: リダイレクトを自動追跡
  // redirect: 'error', // リダイレクト時にエラー
  // redirect: 'manual', // リダイレクトを手動処理
});

// manualの場合、opaqueredirect レスポンスが返る
if (response.type === 'opaqueredirect') {
  const redirectUrl = response.url;
  console.log('Redirected to:', redirectUrl);
}
```

---

## 2. Request / Response オブジェクト

### 2.1 Request オブジェクト

Fetch APIのfetch()関数は内部でRequestオブジェクトを生成する。明示的にRequestオブジェクトを作成することで、リクエストの再利用やService Workerでの操作が可能になる。

```javascript
// Requestオブジェクトの明示的な生成
const request = new Request('/api/users', {
  method: 'GET',
  headers: new Headers({
    'Accept': 'application/json',
    'Authorization': 'Bearer eyJhbGciOiJIUzI1NiJ9...',
  }),
  mode: 'cors',
  credentials: 'same-origin',
  cache: 'default',
  redirect: 'follow',
  referrer: 'about:client',
  referrerPolicy: 'strict-origin-when-cross-origin',
  integrity: 'sha256-abc123...', // Subresource Integrity
});

// Requestオブジェクトをfetchに渡す
const response = await fetch(request);

// Requestのクローン（Service Workerで頻用）
const clonedRequest = request.clone();

// Requestの主要プロパティ
console.log(request.url);        // 完全なURL
console.log(request.method);     // GET, POST, etc.
console.log(request.headers);    // Headers オブジェクト
console.log(request.body);       // ReadableStream | null
console.log(request.mode);       // cors, no-cors, same-origin
console.log(request.credentials);// include, same-origin, omit
console.log(request.cache);      // default, no-store, reload, etc.
console.log(request.redirect);   // follow, error, manual
console.log(request.signal);     // AbortSignal

// 既存Requestを基にオプションを上書き
const authenticatedRequest = new Request(request, {
  headers: new Headers({
    ...Object.fromEntries(request.headers.entries()),
    'Authorization': `Bearer ${newToken}`,
  }),
});
```

### 2.2 Headers オブジェクト

```javascript
// Headersの作成と操作
const headers = new Headers();
headers.append('Content-Type', 'application/json');
headers.append('Accept', 'application/json');
headers.append('X-Custom-Header', 'value1');
headers.append('X-Custom-Header', 'value2'); // 複数値の追加

// set は上書き、append は追加
headers.set('X-Custom-Header', 'single-value'); // 上書き

// 値の取得
console.log(headers.get('Content-Type'));      // 'application/json'
console.log(headers.has('Authorization'));       // false
console.log(headers.get('X-Custom-Header'));    // 'single-value'

// ヘッダーの削除
headers.delete('X-Custom-Header');

// イテレーション
for (const [name, value] of headers) {
  console.log(`${name}: ${value}`);
}

// オブジェクトからの初期化
const headers = new Headers({
  'Content-Type': 'application/json',
  'Authorization': 'Bearer token123',
  'Accept-Language': 'ja,en;q=0.9',
});

// レスポンスヘッダーの読み取り
const response = await fetch('/api/data');
console.log(response.headers.get('Content-Type'));
console.log(response.headers.get('X-Request-Id'));
console.log(response.headers.get('X-RateLimit-Remaining'));

// ★ CORSではサーバーがAccess-Control-Expose-Headersで
//    公開していないヘッダーは読み取れない
// サーバー側: Access-Control-Expose-Headers: X-Request-Id, X-RateLimit-Remaining

// Headersをオブジェクトに変換
const headerObj = Object.fromEntries(headers.entries());
```

### 2.3 Response オブジェクト

```javascript
// Responseの主要プロパティ
const response = await fetch('/api/users');

console.log(response.ok);         // true（status 200-299）
console.log(response.status);     // 200
console.log(response.statusText); // 'OK'
console.log(response.url);        // リクエストの最終URL
console.log(response.redirected); // リダイレクトが発生したか
console.log(response.type);       // 'basic', 'cors', 'opaque', etc.
console.log(response.headers);    // Headers オブジェクト
console.log(response.body);       // ReadableStream

// レスポンスボディの読み取りメソッド
const json = await response.json();        // JSON → Object
const text = await response.text();        // テキスト
const blob = await response.blob();        // Blob（バイナリデータ）
const buffer = await response.arrayBuffer(); // ArrayBuffer
const formData = await response.formData(); // FormData

// カスタムResponseの生成（Service Workerで活用）
const customResponse = new Response(
  JSON.stringify({ message: 'Hello from cache' }),
  {
    status: 200,
    statusText: 'OK',
    headers: {
      'Content-Type': 'application/json',
      'X-Source': 'service-worker-cache',
    },
  }
);

// 静的メソッド
const redirectResponse = Response.redirect('https://example.com/new-url', 301);
const errorResponse = Response.error(); // ネットワークエラーレスポンス
const jsonResponse = Response.json({ ok: true }); // JSON レスポンス（新API）
```

---

## 3. AbortController 詳解

### 3.1 基本的なキャンセル

```javascript
// AbortControllerの基本
const controller = new AbortController();
const { signal } = controller;

// signalをfetchに渡す
const fetchPromise = fetch('/api/large-data', { signal });

// 何らかの条件でキャンセル
document.getElementById('cancelBtn').addEventListener('click', () => {
  controller.abort();
});

try {
  const response = await fetchPromise;
  const data = await response.json();
  console.log(data);
} catch (err) {
  if (err.name === 'AbortError') {
    console.log('Fetch was cancelled by user');
  } else {
    console.error('Fetch failed:', err);
  }
}
```

### 3.2 タイムアウトの実装

```javascript
// 方法1: setTimeout + AbortController
function fetchWithTimeout(url, options = {}, timeout = 5000) {
  const controller = new AbortController();
  const { signal } = controller;

  // 既存のsignalがある場合はany()で合成
  const combinedSignal = options.signal
    ? AbortSignal.any([signal, options.signal])
    : signal;

  const timeoutId = setTimeout(() => {
    controller.abort(new DOMException('Request timed out', 'TimeoutError'));
  }, timeout);

  return fetch(url, {
    ...options,
    signal: combinedSignal,
  }).finally(() => {
    clearTimeout(timeoutId);
  });
}

// 使用例
try {
  const response = await fetchWithTimeout('/api/slow-endpoint', {}, 3000);
  const data = await response.json();
} catch (err) {
  if (err.name === 'TimeoutError') {
    console.error('Request timed out after 3 seconds');
  } else if (err.name === 'AbortError') {
    console.error('Request was manually cancelled');
  }
}

// 方法2: AbortSignal.timeout()（推奨・ブラウザサポート確認が必要）
const response = await fetch('/api/data', {
  signal: AbortSignal.timeout(5000),
});

// 方法3: 複数シグナルの合成
const userController = new AbortController();
const combinedSignal = AbortSignal.any([
  userController.signal,
  AbortSignal.timeout(10000),
]);

const response = await fetch('/api/data', { signal: combinedSignal });

// ユーザーがキャンセルボタンを押した場合
cancelButton.onclick = () => userController.abort();
```

### 3.3 Reactでのキャンセルパターン

```typescript
import { useEffect, useState, useCallback } from 'react';

// パターン1: useEffectでのクリーンアップ
function UserList() {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const controller = new AbortController();

    async function loadUsers() {
      try {
        setLoading(true);
        const response = await fetch('/api/users', {
          signal: controller.signal,
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        setUsers(data);
        setError(null);
      } catch (err) {
        if (err.name !== 'AbortError') {
          setError(err.message);
        }
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      }
    }

    loadUsers();
    return () => controller.abort(); // アンマウント時にキャンセル
  }, []);

  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error}</div>;
  return <ul>{users.map(u => <li key={u.id}>{u.name}</li>)}</ul>;
}

// パターン2: カスタムフック
function useFetch<T>(url: string, options?: RequestInit) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    let isMounted = true;

    async function fetchData() {
      try {
        setLoading(true);
        setError(null);

        const response = await fetch(url, {
          ...options,
          signal: controller.signal,
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const json = await response.json();

        if (isMounted) {
          setData(json);
        }
      } catch (err) {
        if (isMounted && err.name !== 'AbortError') {
          setError(err instanceof Error ? err : new Error(String(err)));
        }
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    }

    fetchData();

    return () => {
      isMounted = false;
      controller.abort();
    };
  }, [url]);

  return { data, loading, error };
}

// カスタムフックの使用
function UserProfile({ userId }: { userId: string }) {
  const { data, loading, error } = useFetch<User>(
    `/api/users/${userId}`
  );

  if (loading) return <Spinner />;
  if (error) return <ErrorMessage error={error} />;
  if (!data) return null;

  return <div>{data.name}</div>;
}

// パターン3: 検索のデバウンスとキャンセル
function SearchComponent() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const controllerRef = useRef<AbortController | null>(null);

  const search = useCallback(async (searchQuery: string) => {
    // 前のリクエストをキャンセル
    if (controllerRef.current) {
      controllerRef.current.abort();
    }

    if (!searchQuery.trim()) {
      setResults([]);
      return;
    }

    const controller = new AbortController();
    controllerRef.current = controller;

    try {
      const params = new URLSearchParams({ q: searchQuery });
      const response = await fetch(`/api/search?${params}`, {
        signal: controller.signal,
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      setResults(data.results);
    } catch (err) {
      if (err.name !== 'AbortError') {
        console.error('Search failed:', err);
      }
    }
  }, []);

  // デバウンス処理
  useEffect(() => {
    const timeoutId = setTimeout(() => search(query), 300);
    return () => clearTimeout(timeoutId);
  }, [query, search]);

  return (
    <div>
      <input
        value={query}
        onChange={e => setQuery(e.target.value)}
        placeholder="Search..."
      />
      <ul>
        {results.map(r => (
          <li key={r.id}>{r.title}</li>
        ))}
      </ul>
    </div>
  );
}
```

### 3.4 AbortControllerの応用

```javascript
// fetch以外でのAbortController活用
// EventListenerのキャンセル
const controller = new AbortController();

document.addEventListener('click', handleClick, { signal: controller.signal });
document.addEventListener('keydown', handleKey, { signal: controller.signal });
document.addEventListener('scroll', handleScroll, { signal: controller.signal });

// まとめてリスナーを削除
controller.abort();

// カスタムの非同期処理でのキャンセル対応
async function processItems(items, signal) {
  const results = [];

  for (const item of items) {
    // 各イテレーションでキャンセルをチェック
    if (signal?.aborted) {
      throw new DOMException('Processing cancelled', 'AbortError');
    }

    const result = await processItem(item);
    results.push(result);
  }

  return results;
}

// signalのイベントリスナー
const controller = new AbortController();

controller.signal.addEventListener('abort', () => {
  console.log('Abort reason:', controller.signal.reason);
  // クリーンアップ処理
});

// abort理由を指定
controller.abort(new Error('User navigated away'));
console.log(controller.signal.reason); // Error: User navigated away
```

---

## 4. Streams API 詳解

### 4.1 ReadableStream

ReadableStreamは非同期的にデータを読み取るためのインターフェース。fetch()のresponse.bodyはReadableStreamを返す。

```javascript
// ReadableStreamの基本構造
const stream = new ReadableStream({
  start(controller) {
    // ストリーム初期化時に呼ばれる
    controller.enqueue('Hello');
    controller.enqueue(' ');
    controller.enqueue('World');
    controller.close();
  },

  pull(controller) {
    // コンシューマーがデータを要求した時に呼ばれる
    // 非同期データソースからの読み取りに適している
  },

  cancel(reason) {
    // ストリームがキャンセルされた時に呼ばれる
    console.log('Stream cancelled:', reason);
  },
});

// Readerを使った読み取り
const reader = stream.getReader();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  console.log(value);
}

reader.releaseLock(); // ロックを解放

// カウンティングストラテジー（バックプレッシャー制御）
const stream = new ReadableStream(
  {
    start(controller) {
      // データをエンキュー
    },
    pull(controller) {
      // desiredSizeが0以下ならバッファが満杯
      console.log('Desired size:', controller.desiredSize);
    },
  },
  new CountQueuingStrategy({ highWaterMark: 10 }) // 最大10チャンク
);

// ByteLengthQueuingStrategy
const stream = new ReadableStream(
  {
    // ...
  },
  new ByteLengthQueuingStrategy({ highWaterMark: 1024 * 64 }) // 64KB
);
```

### 4.2 ダウンロード進捗表示

```javascript
// 実務で使えるダウンロード進捗コンポーネント
async function downloadWithProgress(url, onProgress) {
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }

  // Content-Lengthヘッダーからファイルサイズを取得
  const contentLength = response.headers.get('Content-Length');
  const total = contentLength ? parseInt(contentLength, 10) : null;

  if (!response.body) {
    // body が null の場合（通常は発生しない）
    return response.blob();
  }

  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  const startTime = Date.now();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    chunks.push(value);
    received += value.length;

    const elapsed = (Date.now() - startTime) / 1000;
    const speed = received / elapsed; // bytes/sec

    onProgress({
      loaded: received,
      total,
      percentage: total ? Math.round((received / total) * 100) : null,
      speed, // bytes/sec
      eta: total ? Math.round((total - received) / speed) : null, // 残り秒数
    });
  }

  // チャンクを結合してBlobを生成
  const blob = new Blob(chunks);
  return blob;
}

// Reactでの使用例
function DownloadButton({ url, filename }) {
  const [progress, setProgress] = useState(null);
  const [downloading, setDownloading] = useState(false);

  const handleDownload = async () => {
    setDownloading(true);
    try {
      const blob = await downloadWithProgress(url, setProgress);

      // ダウンロードリンクを生成
      const objectUrl = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = objectUrl;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(objectUrl);
    } catch (err) {
      console.error('Download failed:', err);
    } finally {
      setDownloading(false);
      setProgress(null);
    }
  };

  return (
    <div>
      <button onClick={handleDownload} disabled={downloading}>
        {downloading ? 'Downloading...' : 'Download'}
      </button>
      {progress && (
        <div>
          <progress
            value={progress.loaded}
            max={progress.total || undefined}
          />
          <span>
            {progress.percentage !== null
              ? `${progress.percentage}%`
              : `${(progress.loaded / 1024 / 1024).toFixed(1)} MB`
            }
            {progress.speed && ` (${formatSpeed(progress.speed)})`}
            {progress.eta !== null && ` - 残り ${progress.eta}秒`}
          </span>
        </div>
      )}
    </div>
  );
}

function formatSpeed(bytesPerSec) {
  if (bytesPerSec > 1024 * 1024) {
    return `${(bytesPerSec / 1024 / 1024).toFixed(1)} MB/s`;
  }
  return `${(bytesPerSec / 1024).toFixed(1)} KB/s`;
}
```

### 4.3 アップロード進捗（XMLHttpRequestとの併用）

```javascript
// Fetch APIではアップロード進捗を直接取得できない（2024年時点）
// XMLHttpRequestのupload.onprogressを使用する

function uploadWithProgress(url, file, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();

    xhr.upload.addEventListener('progress', (event) => {
      if (event.lengthComputable) {
        onProgress({
          loaded: event.loaded,
          total: event.total,
          percentage: Math.round((event.loaded / event.total) * 100),
        });
      }
    });

    xhr.addEventListener('load', () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(JSON.parse(xhr.responseText));
      } else {
        reject(new Error(`HTTP ${xhr.status}: ${xhr.statusText}`));
      }
    });

    xhr.addEventListener('error', () => reject(new Error('Upload failed')));
    xhr.addEventListener('abort', () => reject(new Error('Upload cancelled')));

    const formData = new FormData();
    formData.append('file', file);

    xhr.open('POST', url);
    xhr.setRequestHeader('Authorization', `Bearer ${getToken()}`);
    xhr.send(formData);
  });
}

// チャンクアップロード（大きなファイルの分割送信）
async function chunkedUpload(url, file, chunkSize = 5 * 1024 * 1024) {
  const totalChunks = Math.ceil(file.size / chunkSize);
  const uploadId = crypto.randomUUID();

  for (let i = 0; i < totalChunks; i++) {
    const start = i * chunkSize;
    const end = Math.min(start + chunkSize, file.size);
    const chunk = file.slice(start, end);

    const formData = new FormData();
    formData.append('chunk', chunk);
    formData.append('uploadId', uploadId);
    formData.append('chunkIndex', String(i));
    formData.append('totalChunks', String(totalChunks));

    const response = await fetch(url, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Chunk ${i} upload failed: HTTP ${response.status}`);
    }

    console.log(`Uploaded chunk ${i + 1}/${totalChunks}`);
  }

  // 全チャンクのアップロード完了を通知
  const response = await fetch(`${url}/complete`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ uploadId, totalChunks }),
  });

  return response.json();
}
```

### 4.4 TransformStream

```javascript
// TransformStreamの基本
const uppercaseTransform = new TransformStream({
  transform(chunk, controller) {
    controller.enqueue(chunk.toUpperCase());
  },
});

// JSONラインパーサー（NDJSON対応）
function createNDJSONParser() {
  let buffer = '';

  return new TransformStream({
    transform(chunk, controller) {
      buffer += chunk;
      const lines = buffer.split('\n');
      buffer = lines.pop(); // 不完全な最後の行をバッファに残す

      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed) {
          try {
            const parsed = JSON.parse(trimmed);
            controller.enqueue(parsed);
          } catch (e) {
            console.warn('Invalid JSON line:', trimmed);
          }
        }
      }
    },

    flush(controller) {
      // ストリーム終了時にバッファの残りを処理
      const trimmed = buffer.trim();
      if (trimmed) {
        try {
          controller.enqueue(JSON.parse(trimmed));
        } catch (e) {
          console.warn('Invalid final JSON line:', trimmed);
        }
      }
    },
  });
}

// 使用例
async function* streamNDJSON(url) {
  const response = await fetch(url);
  const reader = response.body
    .pipeThrough(new TextDecoderStream())
    .pipeThrough(createNDJSONParser())
    .getReader();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    yield value;
  }
}

// データフィルタリング TransformStream
function createFilterStream(predicate) {
  return new TransformStream({
    transform(chunk, controller) {
      if (predicate(chunk)) {
        controller.enqueue(chunk);
      }
    },
  });
}

// バッチング TransformStream
function createBatchStream(batchSize) {
  let batch = [];

  return new TransformStream({
    transform(chunk, controller) {
      batch.push(chunk);
      if (batch.length >= batchSize) {
        controller.enqueue(batch);
        batch = [];
      }
    },
    flush(controller) {
      if (batch.length > 0) {
        controller.enqueue(batch);
      }
    },
  });
}

// パイプライン構築
const response = await fetch('/api/events');
const reader = response.body
  .pipeThrough(new TextDecoderStream())
  .pipeThrough(createNDJSONParser())
  .pipeThrough(createFilterStream(event => event.type === 'message'))
  .pipeThrough(createBatchStream(10))
  .getReader();

while (true) {
  const { done, value: batch } = await reader.read();
  if (done) break;
  await processBatch(batch); // 10件ずつバッチ処理
}
```

### 4.5 WritableStream

```javascript
// WritableStreamの基本
const writableStream = new WritableStream({
  start(controller) {
    console.log('Stream started');
  },

  write(chunk, controller) {
    console.log('Writing chunk:', chunk);
    // 非同期処理も可能
    return processChunk(chunk);
  },

  close() {
    console.log('Stream closed');
  },

  abort(reason) {
    console.log('Stream aborted:', reason);
  },
});

// WriterでWritableStreamに書き込む
const writer = writableStream.getWriter();
await writer.write('Hello');
await writer.write(' World');
await writer.close();

// ReadableStreamからWritableStreamへのパイプ
const response = await fetch('/api/large-data');
await response.body.pipeTo(writableStream);

// ファイルへの書き込み（File System Access API）
async function saveStreamToFile(readableStream) {
  const fileHandle = await window.showSaveFilePicker({
    suggestedName: 'download.txt',
    types: [
      {
        description: 'Text files',
        accept: { 'text/plain': ['.txt'] },
      },
    ],
  });

  const writable = await fileHandle.createWritable();

  await readableStream
    .pipeThrough(new TextEncoderStream())
    .pipeTo(writable);
}

// DOM への段階的書き込み
function createDOMWritableStream(container) {
  return new WritableStream({
    write(chunk) {
      const element = document.createElement('div');
      element.textContent = typeof chunk === 'string' ? chunk : JSON.stringify(chunk);
      container.appendChild(element);
    },
  });
}

const container = document.getElementById('results');
const response = await fetch('/api/events');
await response.body
  .pipeThrough(new TextDecoderStream())
  .pipeThrough(createNDJSONParser())
  .pipeTo(createDOMWritableStream(container));
```

---

## 5. Server-Sent Events (SSE) とストリーミング

### 5.1 EventSource API

```javascript
// EventSource による SSE の受信
const eventSource = new EventSource('/api/events');

eventSource.onopen = () => {
  console.log('Connection opened');
};

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};

eventSource.onerror = (event) => {
  console.error('EventSource error:', event);
  if (eventSource.readyState === EventSource.CLOSED) {
    console.log('Connection closed');
  }
};

// 名前付きイベントの受信
eventSource.addEventListener('user-update', (event) => {
  const user = JSON.parse(event.data);
  console.log('User updated:', user);
});

eventSource.addEventListener('notification', (event) => {
  const notification = JSON.parse(event.data);
  showNotification(notification);
});

// 接続を閉じる
eventSource.close();

// EventSourceの制限:
// - GETリクエストのみ
// - カスタムヘッダーを設定できない
// - 認証トークンの送信にはCookieかURLパラメータが必要
```

### 5.2 Fetch APIによるSSE処理

```javascript
// Fetch APIを使ったSSE（カスタムヘッダー対応）
async function fetchSSE(url, options = {}) {
  const { onMessage, onError, signal, headers = {} } = options;

  const response = await fetch(url, {
    headers: {
      'Accept': 'text/event-stream',
      ...headers,
    },
    signal,
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }

  const reader = response.body
    .pipeThrough(new TextDecoderStream())
    .getReader();

  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += value;
    const events = buffer.split('\n\n');
    buffer = events.pop(); // 不完全なイベントをバッファに残す

    for (const eventStr of events) {
      if (!eventStr.trim()) continue;

      const event = parseSSEEvent(eventStr);
      if (event) {
        onMessage?.(event);
      }
    }
  }
}

function parseSSEEvent(eventStr) {
  const lines = eventStr.split('\n');
  const event = { data: '', type: 'message', id: null, retry: null };

  for (const line of lines) {
    if (line.startsWith('data:')) {
      event.data += (event.data ? '\n' : '') + line.slice(5).trim();
    } else if (line.startsWith('event:')) {
      event.type = line.slice(6).trim();
    } else if (line.startsWith('id:')) {
      event.id = line.slice(3).trim();
    } else if (line.startsWith('retry:')) {
      event.retry = parseInt(line.slice(6).trim(), 10);
    }
  }

  return event.data ? event : null;
}

// 使用例: ChatGPT風のストリーミングレスポンス
async function streamChatResponse(prompt) {
  const controller = new AbortController();

  await fetchSSE('/api/chat/stream', {
    signal: controller.signal,
    headers: {
      'Authorization': `Bearer ${getToken()}`,
      'Content-Type': 'application/json',
    },
    onMessage(event) {
      if (event.data === '[DONE]') {
        console.log('Stream complete');
        return;
      }

      try {
        const data = JSON.parse(event.data);
        appendToChat(data.content);
      } catch (e) {
        console.warn('Failed to parse event:', event.data);
      }
    },
  });

  return controller;
}
```

### 5.3 AI/LLM APIのストリーミング応答処理

```typescript
// OpenAI互換APIのストリーミング処理
interface ChatChunk {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    delta: {
      role?: string;
      content?: string;
    };
    finish_reason: string | null;
  }>;
}

async function* streamChatCompletion(
  messages: Array<{ role: string; content: string }>,
  options: { model?: string; temperature?: number; signal?: AbortSignal } = {}
): AsyncGenerator<string> {
  const response = await fetch('/api/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${process.env.API_KEY}`,
    },
    body: JSON.stringify({
      model: options.model || 'gpt-4',
      messages,
      temperature: options.temperature ?? 0.7,
      stream: true,
    }),
    signal: options.signal,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(`API error ${response.status}: ${error.message || response.statusText}`);
  }

  const reader = response.body!
    .pipeThrough(new TextDecoderStream())
    .getReader();

  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += value;
    const lines = buffer.split('\n');
    buffer = lines.pop()!;

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed || trimmed === 'data: [DONE]') continue;
      if (!trimmed.startsWith('data: ')) continue;

      try {
        const chunk: ChatChunk = JSON.parse(trimmed.slice(6));
        const content = chunk.choices[0]?.delta?.content;
        if (content) {
          yield content;
        }
      } catch (e) {
        // パースエラーは無視
      }
    }
  }
}

// React コンポーネントでの使用
function ChatStream({ messages }: { messages: Message[] }) {
  const [response, setResponse] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const controllerRef = useRef<AbortController | null>(null);

  const startStream = async () => {
    controllerRef.current = new AbortController();
    setResponse('');
    setIsStreaming(true);

    try {
      let fullResponse = '';
      for await (const chunk of streamChatCompletion(messages, {
        signal: controllerRef.current.signal,
      })) {
        fullResponse += chunk;
        setResponse(fullResponse);
      }
    } catch (err) {
      if (err.name !== 'AbortError') {
        console.error('Stream error:', err);
      }
    } finally {
      setIsStreaming(false);
    }
  };

  const stopStream = () => {
    controllerRef.current?.abort();
  };

  return (
    <div>
      <div className="response">{response}</div>
      {isStreaming ? (
        <button onClick={stopStream}>Stop</button>
      ) : (
        <button onClick={startStream}>Send</button>
      )}
    </div>
  );
}
```

---

## 6. 高度なFetchパターン

### 6.1 リトライ戦略

```typescript
// 指数バックオフ付きリトライ
interface RetryOptions {
  maxRetries?: number;
  baseDelay?: number;
  maxDelay?: number;
  retryableStatuses?: number[];
  onRetry?: (attempt: number, error: Error) => void;
}

async function fetchWithRetry(
  url: string,
  options: RequestInit = {},
  retryOptions: RetryOptions = {}
): Promise<Response> {
  const {
    maxRetries = 3,
    baseDelay = 1000,
    maxDelay = 30000,
    retryableStatuses = [408, 429, 500, 502, 503, 504],
    onRetry,
  } = retryOptions;

  let lastError: Error | null = null;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const response = await fetch(url, options);

      // リトライ可能なステータスコードの場合
      if (retryableStatuses.includes(response.status) && attempt < maxRetries) {
        // Retry-Afterヘッダーの確認
        const retryAfter = response.headers.get('Retry-After');
        let delay: number;

        if (retryAfter) {
          // Retry-After は秒数またはHTTP日付形式
          const retrySeconds = parseInt(retryAfter, 10);
          if (!isNaN(retrySeconds)) {
            delay = retrySeconds * 1000;
          } else {
            delay = new Date(retryAfter).getTime() - Date.now();
          }
        } else {
          // 指数バックオフ + ジッター
          delay = Math.min(
            baseDelay * Math.pow(2, attempt) + Math.random() * 1000,
            maxDelay
          );
        }

        const error = new Error(`HTTP ${response.status}`);
        onRetry?.(attempt + 1, error);
        await new Promise(resolve => setTimeout(resolve, delay));
        continue;
      }

      return response;
    } catch (err) {
      lastError = err instanceof Error ? err : new Error(String(err));

      if (attempt < maxRetries) {
        const delay = Math.min(
          baseDelay * Math.pow(2, attempt) + Math.random() * 1000,
          maxDelay
        );
        onRetry?.(attempt + 1, lastError);
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
  }

  throw lastError || new Error('Max retries reached');
}

// 使用例
const response = await fetchWithRetry('/api/unreliable', {}, {
  maxRetries: 5,
  baseDelay: 500,
  onRetry(attempt, error) {
    console.warn(`Retry ${attempt}: ${error.message}`);
  },
});
```

### 6.2 並行リクエストと制御

```typescript
// Promise.all による並行リクエスト
async function fetchMultiple(urls: string[]) {
  const responses = await Promise.all(
    urls.map(url => fetch(url).then(r => {
      if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
      return r.json();
    }))
  );
  return responses;
}

// Promise.allSettled でエラー耐性のある並行リクエスト
async function fetchMultipleSafe(urls: string[]) {
  const results = await Promise.allSettled(
    urls.map(url => fetch(url).then(r => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json();
    }))
  );

  return results.map((result, i) => ({
    url: urls[i],
    status: result.status,
    data: result.status === 'fulfilled' ? result.value : null,
    error: result.status === 'rejected' ? result.reason : null,
  }));
}

// 並行数制限（コンカレンシー制御）
async function fetchWithConcurrencyLimit<T>(
  urls: string[],
  concurrency: number,
  fetcher: (url: string) => Promise<T>
): Promise<T[]> {
  const results: T[] = new Array(urls.length);
  let index = 0;

  async function worker() {
    while (index < urls.length) {
      const currentIndex = index++;
      results[currentIndex] = await fetcher(urls[currentIndex]);
    }
  }

  const workers = Array.from(
    { length: Math.min(concurrency, urls.length) },
    () => worker()
  );

  await Promise.all(workers);
  return results;
}

// 使用例: 最大5並行でAPI呼び出し
const userIds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
const users = await fetchWithConcurrencyLimit(
  userIds.map(id => `/api/users/${id}`),
  5,
  async (url) => {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  }
);

// Promise.race による最速レスポンス取得
async function fetchFastest(urls: string[]) {
  const controller = new AbortController();

  try {
    const result = await Promise.race(
      urls.map(async (url) => {
        const response = await fetch(url, { signal: controller.signal });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json();
      })
    );

    // 最初のレスポンスを受信したら他のリクエストをキャンセル
    controller.abort();
    return result;
  } catch (err) {
    controller.abort();
    throw err;
  }
}
```

### 6.3 リクエストのキューイング

```typescript
// リクエストキュー（順番に実行・レート制限対応）
class RequestQueue {
  private queue: Array<() => Promise<void>> = [];
  private running = 0;
  private concurrency: number;
  private delayMs: number;

  constructor(concurrency = 1, delayMs = 0) {
    this.concurrency = concurrency;
    this.delayMs = delayMs;
  }

  async add<T>(fn: () => Promise<T>): Promise<T> {
    return new Promise((resolve, reject) => {
      this.queue.push(async () => {
        try {
          const result = await fn();
          resolve(result);
        } catch (err) {
          reject(err);
        }
      });
      this.process();
    });
  }

  private async process() {
    if (this.running >= this.concurrency || this.queue.length === 0) {
      return;
    }

    this.running++;
    const task = this.queue.shift()!;

    try {
      await task();
    } finally {
      if (this.delayMs > 0) {
        await new Promise(resolve => setTimeout(resolve, this.delayMs));
      }
      this.running--;
      this.process();
    }
  }
}

// 使用例: APIレート制限（1秒に1リクエスト）
const queue = new RequestQueue(1, 1000);

const results = await Promise.all(
  userIds.map(id =>
    queue.add(() => fetch(`/api/users/${id}`).then(r => r.json()))
  )
);
```

### 6.4 キャッシュ戦略

```typescript
// メモリキャッシュ付きfetch
class FetchCache {
  private cache = new Map<string, {
    data: any;
    timestamp: number;
    etag?: string;
    lastModified?: string;
  }>();
  private ttl: number;

  constructor(ttlMs = 5 * 60 * 1000) {
    this.ttl = ttlMs;
  }

  async fetch<T>(url: string, options?: RequestInit): Promise<T> {
    const cached = this.cache.get(url);
    const now = Date.now();

    // キャッシュが有効な場合
    if (cached && now - cached.timestamp < this.ttl) {
      return cached.data;
    }

    // 条件付きリクエスト（ETag / Last-Modified）
    const headers = new Headers(options?.headers);
    if (cached?.etag) {
      headers.set('If-None-Match', cached.etag);
    }
    if (cached?.lastModified) {
      headers.set('If-Modified-Since', cached.lastModified);
    }

    const response = await fetch(url, { ...options, headers });

    // 304 Not Modified
    if (response.status === 304 && cached) {
      cached.timestamp = now;
      return cached.data;
    }

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();

    this.cache.set(url, {
      data,
      timestamp: now,
      etag: response.headers.get('ETag') || undefined,
      lastModified: response.headers.get('Last-Modified') || undefined,
    });

    return data;
  }

  invalidate(url: string) {
    this.cache.delete(url);
  }

  invalidateAll() {
    this.cache.clear();
  }

  // パターンにマッチするエントリを無効化
  invalidatePattern(pattern: RegExp) {
    for (const key of this.cache.keys()) {
      if (pattern.test(key)) {
        this.cache.delete(key);
      }
    }
  }
}

// 使用例
const apiCache = new FetchCache(60 * 1000); // 1分TTL

// 同じURLへの複数リクエストを集約（デデュプリケーション）
class RequestDeduplicator {
  private pending = new Map<string, Promise<any>>();

  async fetch<T>(url: string, options?: RequestInit): Promise<T> {
    const key = `${options?.method || 'GET'}:${url}`;

    if (this.pending.has(key)) {
      return this.pending.get(key)!;
    }

    const promise = fetch(url, options)
      .then(response => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json();
      })
      .finally(() => {
        this.pending.delete(key);
      });

    this.pending.set(key, promise);
    return promise;
  }
}

const dedup = new RequestDeduplicator();

// 同時に呼ばれても実際のfetchは1回だけ
const [users1, users2, users3] = await Promise.all([
  dedup.fetch('/api/users'),
  dedup.fetch('/api/users'),
  dedup.fetch('/api/users'),
]);
```

---

## 7. 実務レベルのfetchラッパー

### 7.1 型安全なAPIクライアント

```typescript
// エラークラス階層
class ApiError extends Error {
  constructor(
    public status: number,
    public statusText: string,
    public body: unknown,
    public requestUrl: string,
    public requestMethod: string
  ) {
    super(`${requestMethod} ${requestUrl}: HTTP ${status} ${statusText}`);
    this.name = 'ApiError';
  }

  get isClientError() { return this.status >= 400 && this.status < 500; }
  get isServerError() { return this.status >= 500; }
  get isUnauthorized() { return this.status === 401; }
  get isForbidden() { return this.status === 403; }
  get isNotFound() { return this.status === 404; }
  get isConflict() { return this.status === 409; }
  get isRateLimited() { return this.status === 429; }
}

class NetworkError extends Error {
  constructor(public originalError: Error) {
    super(`Network error: ${originalError.message}`);
    this.name = 'NetworkError';
  }
}

class TimeoutError extends Error {
  constructor(public timeoutMs: number) {
    super(`Request timed out after ${timeoutMs}ms`);
    this.name = 'TimeoutError';
  }
}

// APIクライアント設定
interface ApiClientConfig {
  baseUrl: string;
  timeout?: number;
  headers?: Record<string, string>;
  getAuthToken?: () => string | null | Promise<string | null>;
  onUnauthorized?: () => void;
  onError?: (error: ApiError | NetworkError | TimeoutError) => void;
  retryOptions?: RetryOptions;
}

// 本格的なAPIクライアント
class ApiClient {
  private config: Required<Omit<ApiClientConfig, 'getAuthToken' | 'onUnauthorized' | 'onError'>> & Partial<Pick<ApiClientConfig, 'getAuthToken' | 'onUnauthorized' | 'onError'>>;

  constructor(config: ApiClientConfig) {
    this.config = {
      timeout: 30000,
      headers: {},
      retryOptions: { maxRetries: 0 },
      ...config,
    };
  }

  private async request<T>(
    method: string,
    path: string,
    options: {
      body?: unknown;
      query?: Record<string, string | number | boolean | undefined>;
      headers?: Record<string, string>;
      signal?: AbortSignal;
      timeout?: number;
    } = {}
  ): Promise<T> {
    // URL構築
    const url = new URL(path, this.config.baseUrl);
    if (options.query) {
      for (const [key, value] of Object.entries(options.query)) {
        if (value !== undefined) {
          url.searchParams.set(key, String(value));
        }
      }
    }

    // ヘッダー構築
    const headers = new Headers({
      'Accept': 'application/json',
      ...this.config.headers,
      ...options.headers,
    });

    // 認証トークン
    if (this.config.getAuthToken) {
      const token = await this.config.getAuthToken();
      if (token) {
        headers.set('Authorization', `Bearer ${token}`);
      }
    }

    // ボディの処理
    let body: BodyInit | undefined;
    if (options.body !== undefined) {
      if (options.body instanceof FormData) {
        body = options.body;
        // FormDataの場合はContent-Typeを設定しない（ブラウザが自動設定）
      } else {
        headers.set('Content-Type', 'application/json');
        body = JSON.stringify(options.body);
      }
    }

    // タイムアウト設定
    const timeout = options.timeout ?? this.config.timeout;
    const timeoutSignal = AbortSignal.timeout(timeout);
    const combinedSignal = options.signal
      ? AbortSignal.any([options.signal, timeoutSignal])
      : timeoutSignal;

    try {
      const response = await fetch(url.toString(), {
        method,
        headers,
        body,
        signal: combinedSignal,
        credentials: 'same-origin',
      });

      if (!response.ok) {
        let errorBody: unknown;
        try {
          errorBody = await response.json();
        } catch {
          errorBody = await response.text().catch(() => null);
        }

        const apiError = new ApiError(
          response.status,
          response.statusText,
          errorBody,
          url.toString(),
          method
        );

        // 401の特別処理
        if (apiError.isUnauthorized) {
          this.config.onUnauthorized?.();
        }

        this.config.onError?.(apiError);
        throw apiError;
      }

      // 204 No Content
      if (response.status === 204) {
        return undefined as T;
      }

      // Content-Typeに応じたレスポンスの解析
      const contentType = response.headers.get('Content-Type') || '';
      if (contentType.includes('application/json')) {
        return response.json();
      } else if (contentType.includes('text/')) {
        return response.text() as Promise<T>;
      } else {
        return response.blob() as Promise<T>;
      }
    } catch (err) {
      if (err instanceof ApiError) throw err;

      if (err instanceof DOMException) {
        if (err.name === 'TimeoutError') {
          const timeoutErr = new TimeoutError(timeout);
          this.config.onError?.(timeoutErr);
          throw timeoutErr;
        }
        if (err.name === 'AbortError') {
          throw err; // ユーザーによるキャンセルはそのまま
        }
      }

      const networkErr = new NetworkError(
        err instanceof Error ? err : new Error(String(err))
      );
      this.config.onError?.(networkErr);
      throw networkErr;
    }
  }

  // HTTPメソッドのショートカット
  get<T>(path: string, query?: Record<string, string | number | boolean | undefined>, options?: { signal?: AbortSignal }) {
    return this.request<T>('GET', path, { query, ...options });
  }

  post<T>(path: string, body?: unknown, options?: { signal?: AbortSignal }) {
    return this.request<T>('POST', path, { body, ...options });
  }

  put<T>(path: string, body?: unknown, options?: { signal?: AbortSignal }) {
    return this.request<T>('PUT', path, { body, ...options });
  }

  patch<T>(path: string, body?: unknown, options?: { signal?: AbortSignal }) {
    return this.request<T>('PATCH', path, { body, ...options });
  }

  delete<T>(path: string, options?: { signal?: AbortSignal }) {
    return this.request<T>('DELETE', path, options);
  }

  // ストリーミングリクエスト
  async *stream<T>(
    path: string,
    options: {
      method?: string;
      body?: unknown;
      signal?: AbortSignal;
    } = {}
  ): AsyncGenerator<T> {
    const url = new URL(path, this.config.baseUrl);
    const headers = new Headers({
      'Accept': 'text/event-stream',
      ...this.config.headers,
    });

    if (this.config.getAuthToken) {
      const token = await this.config.getAuthToken();
      if (token) headers.set('Authorization', `Bearer ${token}`);
    }

    let body: string | undefined;
    if (options.body) {
      headers.set('Content-Type', 'application/json');
      body = JSON.stringify(options.body);
    }

    const response = await fetch(url.toString(), {
      method: options.method || 'POST',
      headers,
      body,
      signal: options.signal,
    });

    if (!response.ok) {
      throw new ApiError(
        response.status, response.statusText, null,
        url.toString(), options.method || 'POST'
      );
    }

    const reader = response.body!
      .pipeThrough(new TextDecoderStream())
      .getReader();

    let buffer = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += value;
      const lines = buffer.split('\n');
      buffer = lines.pop()!;

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || !trimmed.startsWith('data: ')) continue;
        const data = trimmed.slice(6);
        if (data === '[DONE]') return;

        try {
          yield JSON.parse(data) as T;
        } catch {
          // パースエラーは無視
        }
      }
    }
  }
}

// 使用例
const api = new ApiClient({
  baseUrl: 'https://api.example.com',
  timeout: 15000,
  getAuthToken: () => localStorage.getItem('access_token'),
  onUnauthorized: () => {
    // トークンリフレッシュまたはログイン画面へリダイレクト
    window.location.href = '/login';
  },
  onError: (error) => {
    // エラー監視サービスに送信
    errorTracker.capture(error);
  },
});

// 型安全なAPI呼び出し
interface User {
  id: number;
  name: string;
  email: string;
  role: string;
}

interface PaginatedResponse<T> {
  data: T[];
  total: number;
  page: number;
  limit: number;
}

// GET
const users = await api.get<PaginatedResponse<User>>('/users', {
  page: 1,
  limit: 20,
  role: 'admin',
});

// POST
const newUser = await api.post<User>('/users', {
  name: '田中太郎',
  email: 'taro@example.com',
  role: 'editor',
});

// PATCH
const updated = await api.patch<User>(`/users/${userId}`, {
  role: 'admin',
});

// DELETE
await api.delete(`/users/${userId}`);

// ストリーミング
for await (const chunk of api.stream<{ content: string }>('/chat', {
  body: { message: 'Hello' },
})) {
  console.log(chunk.content);
}
```

### 7.2 インターセプターパターン

```typescript
// リクエスト/レスポンスインターセプター
type RequestInterceptor = (
  url: string,
  options: RequestInit
) => Promise<[string, RequestInit]> | [string, RequestInit];

type ResponseInterceptor = (
  response: Response,
  url: string,
  options: RequestInit
) => Promise<Response> | Response;

class InterceptableFetch {
  private requestInterceptors: RequestInterceptor[] = [];
  private responseInterceptors: ResponseInterceptor[] = [];

  addRequestInterceptor(interceptor: RequestInterceptor) {
    this.requestInterceptors.push(interceptor);
    return () => {
      const index = this.requestInterceptors.indexOf(interceptor);
      if (index > -1) this.requestInterceptors.splice(index, 1);
    };
  }

  addResponseInterceptor(interceptor: ResponseInterceptor) {
    this.responseInterceptors.push(interceptor);
    return () => {
      const index = this.responseInterceptors.indexOf(interceptor);
      if (index > -1) this.responseInterceptors.splice(index, 1);
    };
  }

  async fetch(url: string, options: RequestInit = {}): Promise<Response> {
    // リクエストインターセプターを順番に適用
    let currentUrl = url;
    let currentOptions = { ...options };

    for (const interceptor of this.requestInterceptors) {
      [currentUrl, currentOptions] = await interceptor(currentUrl, currentOptions);
    }

    let response = await fetch(currentUrl, currentOptions);

    // レスポンスインターセプターを順番に適用
    for (const interceptor of this.responseInterceptors) {
      response = await interceptor(response, currentUrl, currentOptions);
    }

    return response;
  }
}

// 使用例
const client = new InterceptableFetch();

// ロギングインターセプター
client.addRequestInterceptor(async (url, options) => {
  console.log(`[API] ${options.method || 'GET'} ${url}`);
  const startTime = performance.now();
  (options as any).__startTime = startTime;
  return [url, options];
});

client.addResponseInterceptor(async (response, url, options) => {
  const duration = performance.now() - (options as any).__startTime;
  console.log(`[API] ${response.status} ${url} (${duration.toFixed(0)}ms)`);
  return response;
});

// 認証インターセプター
client.addRequestInterceptor(async (url, options) => {
  const token = await getAccessToken();
  const headers = new Headers(options.headers);
  if (token) {
    headers.set('Authorization', `Bearer ${token}`);
  }
  return [url, { ...options, headers }];
});

// トークンリフレッシュインターセプター
client.addResponseInterceptor(async (response, url, options) => {
  if (response.status === 401) {
    const newToken = await refreshToken();
    if (newToken) {
      const headers = new Headers(options.headers);
      headers.set('Authorization', `Bearer ${newToken}`);
      return fetch(url, { ...options, headers });
    }
  }
  return response;
});
```

---

## 8. CORS（Cross-Origin Resource Sharing）

### 8.1 CORSの基本

```javascript
// Simple Request（プリフライト不要）
// 条件: GET/HEAD/POST, 特定のヘッダーのみ, 特定のContent-Typeのみ
const response = await fetch('https://api.example.com/data', {
  method: 'GET',
  mode: 'cors', // デフォルト
});

// Preflight が必要なリクエスト
// カスタムヘッダーやContent-Type: application/jsonを使う場合
const response = await fetch('https://api.example.com/data', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json', // preflightトリガー
    'X-Custom-Header': 'value',          // preflightトリガー
  },
  body: JSON.stringify({ key: 'value' }),
  mode: 'cors',
});

// サーバー側の設定例（Express.js）
// app.use(cors({
//   origin: ['https://example.com', 'https://app.example.com'],
//   methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH'],
//   allowedHeaders: ['Content-Type', 'Authorization', 'X-Custom-Header'],
//   exposedHeaders: ['X-Request-Id', 'X-RateLimit-Remaining'],
//   credentials: true,
//   maxAge: 86400, // プリフライト結果のキャッシュ（秒）
// }));
```

### 8.2 CORSのトラブルシューティング

```javascript
// よくあるCORSエラーと対処法

// エラー1: No 'Access-Control-Allow-Origin' header
// → サーバー側でAccess-Control-Allow-Originヘッダーを設定

// エラー2: credentials flagがtrueだがAccess-Control-Allow-Origin が *
// → credentials: 'include' を使う場合、サーバーは具体的なオリジンを返す必要がある
// → Access-Control-Allow-Origin: https://app.example.com（* は不可）

// エラー3: Method not allowed
// → サーバーのAccess-Control-Allow-MethodsにHTTPメソッドを追加

// no-corsモード（レスポンスは読めないが、リクエストは送信される）
const response = await fetch('https://third-party.com/api', {
  mode: 'no-cors', // opaque response（ステータスやボディにアクセス不可）
});
// response.type === 'opaque'
// response.status === 0
// response.body は null

// プロキシ経由でCORSを回避（開発環境）
// vite.config.ts
// export default defineConfig({
//   server: {
//     proxy: {
//       '/api': {
//         target: 'https://api.example.com',
//         changeOrigin: true,
//         rewrite: (path) => path.replace(/^\/api/, ''),
//       },
//     },
//   },
// });
```

---

## 9. テスト戦略

### 9.1 MSW（Mock Service Worker）によるモック

```typescript
// msw v2 のセットアップ
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';

// ハンドラー定義
const handlers = [
  http.get('/api/users', () => {
    return HttpResponse.json([
      { id: 1, name: '田中太郎', email: 'taro@example.com' },
      { id: 2, name: '鈴木花子', email: 'hanako@example.com' },
    ]);
  }),

  http.get('/api/users/:id', ({ params }) => {
    const { id } = params;
    if (id === '999') {
      return new HttpResponse(null, { status: 404 });
    }
    return HttpResponse.json({
      id: Number(id),
      name: '田中太郎',
      email: 'taro@example.com',
    });
  }),

  http.post('/api/users', async ({ request }) => {
    const body = await request.json();
    return HttpResponse.json(
      { id: 3, ...body },
      { status: 201 }
    );
  }),

  // ストリーミングレスポンスのモック
  http.get('/api/events', () => {
    const stream = new ReadableStream({
      async start(controller) {
        const encoder = new TextEncoder();
        controller.enqueue(encoder.encode('data: {"type":"hello"}\n\n'));
        await new Promise(r => setTimeout(r, 100));
        controller.enqueue(encoder.encode('data: {"type":"update","value":42}\n\n'));
        await new Promise(r => setTimeout(r, 100));
        controller.enqueue(encoder.encode('data: [DONE]\n\n'));
        controller.close();
      },
    });

    return new HttpResponse(stream, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
      },
    });
  }),

  // エラーレスポンス
  http.get('/api/error', () => {
    return HttpResponse.json(
      { error: 'Internal Server Error' },
      { status: 500 }
    );
  }),

  // ネットワークエラー
  http.get('/api/network-error', () => {
    return HttpResponse.error();
  }),

  // 遅延レスポンス
  http.get('/api/slow', async () => {
    await new Promise(r => setTimeout(r, 5000));
    return HttpResponse.json({ data: 'slow response' });
  }),
];

const server = setupServer(...handlers);

// テストセットアップ
beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

// テスト例
describe('API Client', () => {
  test('ユーザー一覧を取得できる', async () => {
    const users = await api.get('/api/users');
    expect(users).toHaveLength(2);
    expect(users[0].name).toBe('田中太郎');
  });

  test('404エラーを適切に処理する', async () => {
    await expect(api.get('/api/users/999')).rejects.toThrow(ApiError);
    await expect(api.get('/api/users/999')).rejects.toMatchObject({
      status: 404,
    });
  });

  test('ネットワークエラーを適切に処理する', async () => {
    await expect(api.get('/api/network-error')).rejects.toThrow(NetworkError);
  });

  test('タイムアウトを適切に処理する', async () => {
    const clientWithShortTimeout = new ApiClient({
      baseUrl: '',
      timeout: 100,
    });

    await expect(
      clientWithShortTimeout.get('/api/slow')
    ).rejects.toThrow(TimeoutError);
  });

  test('リクエストがキャンセルできる', async () => {
    const controller = new AbortController();

    const promise = api.get('/api/slow', undefined, {
      signal: controller.signal,
    });

    controller.abort();

    await expect(promise).rejects.toThrow();
  });

  // テスト内でハンドラーを上書き
  test('サーバーエラー時にリトライする', async () => {
    let attempts = 0;

    server.use(
      http.get('/api/data', () => {
        attempts++;
        if (attempts <= 2) {
          return HttpResponse.json(null, { status: 503 });
        }
        return HttpResponse.json({ success: true });
      })
    );

    const result = await fetchWithRetry('/api/data', {}, { maxRetries: 3 });
    const data = await result.json();
    expect(data.success).toBe(true);
    expect(attempts).toBe(3);
  });
});
```

### 9.2 ユニットテストでのfetchモック

```typescript
// グローバルfetchのモック（Vitest）
import { vi, describe, test, expect, beforeEach } from 'vitest';

describe('fetchData', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  test('正常レスポンスを処理する', async () => {
    const mockData = { id: 1, name: 'Test' };

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: () => Promise.resolve(mockData),
      headers: new Headers({ 'Content-Type': 'application/json' }),
    });

    const result = await fetchData('/api/test');
    expect(result).toEqual(mockData);
    expect(fetch).toHaveBeenCalledWith('/api/test', expect.any(Object));
  });

  test('HTTPエラーを処理する', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      statusText: 'Internal Server Error',
      json: () => Promise.resolve({ message: 'Server error' }),
    });

    await expect(fetchData('/api/test')).rejects.toThrow('HTTP 500');
  });

  test('AbortControllerが正しく使われる', async () => {
    global.fetch = vi.fn().mockImplementation((url, options) => {
      // signalが渡されていることを確認
      expect(options.signal).toBeInstanceOf(AbortSignal);
      return Promise.resolve({
        ok: true,
        status: 200,
        json: () => Promise.resolve({}),
      });
    });

    await fetchData('/api/test');
    expect(fetch).toHaveBeenCalled();
  });
});

// ReadableStreamのモック
function createMockReadableStream(chunks: string[]) {
  let index = 0;
  return new ReadableStream({
    pull(controller) {
      if (index < chunks.length) {
        controller.enqueue(new TextEncoder().encode(chunks[index]));
        index++;
      } else {
        controller.close();
      }
    },
  });
}

test('ストリーミングレスポンスを処理する', async () => {
  const chunks = [
    'data: {"content":"Hello"}\n\n',
    'data: {"content":" World"}\n\n',
    'data: [DONE]\n\n',
  ];

  global.fetch = vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    body: createMockReadableStream(chunks),
    headers: new Headers({ 'Content-Type': 'text/event-stream' }),
  });

  const results: string[] = [];
  for await (const chunk of streamResponse('/api/stream')) {
    results.push(chunk.content);
  }

  expect(results).toEqual(['Hello', ' World']);
});
```

---

## 10. パフォーマンス最適化

### 10.1 接続の最適化

```javascript
// DNS プリフェッチ
// <link rel="dns-prefetch" href="https://api.example.com">

// プリコネクト（DNS + TCP + TLS）
// <link rel="preconnect" href="https://api.example.com">

// プリフェッチ（リソースの先読み）
// <link rel="prefetch" href="/api/next-page-data">

// プリロード（高優先度リソース）
// <link rel="preload" href="/api/critical-data" as="fetch" crossorigin>

// fetch の priority ヒント
const response = await fetch('/api/critical-data', {
  priority: 'high', // 'high', 'low', 'auto'
});

const response = await fetch('/api/analytics', {
  priority: 'low',
  keepalive: true, // ページ遷移後もリクエストを維持
});

// keepalive でページ離脱時にデータを送信
window.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'hidden') {
    fetch('/api/analytics/page-exit', {
      method: 'POST',
      body: JSON.stringify({
        page: window.location.pathname,
        duration: performance.now(),
      }),
      keepalive: true, // ページが閉じても送信を維持
    });
  }
});

// navigator.sendBeacon（keepaliveの代替）
window.addEventListener('unload', () => {
  navigator.sendBeacon('/api/analytics/page-exit', JSON.stringify({
    page: window.location.pathname,
    duration: performance.now(),
  }));
});
```

### 10.2 レスポンスのキャッシュ

```javascript
// Cache APIの直接利用
const cacheName = 'api-cache-v1';

async function fetchWithCache(url, options = {}) {
  const cache = await caches.open(cacheName);

  // キャッシュから検索
  const cachedResponse = await cache.match(url);
  if (cachedResponse) {
    // キャッシュのAge確認
    const cachedDate = new Date(cachedResponse.headers.get('Date') || 0);
    const age = Date.now() - cachedDate.getTime();

    if (age < 5 * 60 * 1000) { // 5分以内
      return cachedResponse;
    }
  }

  // ネットワークからフェッチ
  const response = await fetch(url, options);

  if (response.ok) {
    // レスポンスをキャッシュに保存（cloneが必要）
    cache.put(url, response.clone());
  }

  return response;
}

// Stale-While-Revalidate パターン
async function staleWhileRevalidate(url, options = {}) {
  const cache = await caches.open(cacheName);
  const cachedResponse = await cache.match(url);

  // バックグラウンドで更新
  const fetchPromise = fetch(url, options).then(response => {
    if (response.ok) {
      cache.put(url, response.clone());
    }
    return response;
  });

  // キャッシュがあればすぐ返す（同時にバックグラウンド更新）
  return cachedResponse || fetchPromise;
}
```

### 10.3 バンドルサイズの考慮

```javascript
// fetchのポリフィル（レガシーブラウザ対応は不要な場合がほとんど）
// ES2017+をサポートするブラウザは全てfetchを実装している
// Safari 10.1+, Chrome 42+, Firefox 39+, Edge 14+

// ★ whatwg-fetch ポリフィルは新規プロジェクトでは不要
// ★ isomorphic-fetch も不要（Node.js 18以降はネイティブfetch対応）

// Node.js でのfetch
// Node.js 18+: ネイティブfetchが利用可能
// Node.js 16-17: undici パッケージを使用
// import { fetch } from 'undici';

// Denoでのfetch: ネイティブサポート
// Bun: ネイティブサポート
```

---

## 11. セキュリティ考慮事項

### 11.1 XSS対策

```javascript
// APIレスポンスの安全な処理

// ★ レスポンスデータを直接DOMに挿入しない
const user = await fetch('/api/users/1').then(r => r.json());

// 危険: XSS脆弱性
// element.innerHTML = user.bio;

// 安全: textContentを使う
element.textContent = user.bio;

// ReactではデフォルトでXSS対策済み
// <div>{user.bio}</div> → 自動エスケープ

// ★ dangerouslySetInnerHTML は信頼できるデータのみ
// <div dangerouslySetInnerHTML={{ __html: sanitizedHtml }} />

// DOMPurifyによるサニタイゼーション
import DOMPurify from 'dompurify';
const clean = DOMPurify.sanitize(user.richBio);
element.innerHTML = clean;
```

### 11.2 CSRF対策

```javascript
// CSRFトークンの送信
async function fetchWithCSRF(url, options = {}) {
  const csrfToken = document.querySelector('meta[name="csrf-token"]')?.content
    || getCookie('XSRF-TOKEN');

  const headers = new Headers(options.headers);
  if (csrfToken) {
    headers.set('X-CSRF-Token', csrfToken);
  }

  return fetch(url, { ...options, headers, credentials: 'same-origin' });
}

// SameSite Cookie（サーバー側の設定）
// Set-Cookie: session=abc123; SameSite=Lax; Secure; HttpOnly

// Double Submit Cookie パターン
// 1. サーバーがCSRFトークンをCookieとレスポンスボディの両方で送信
// 2. クライアントはリクエスト時にCookieのトークンをヘッダーに含める
// 3. サーバーはCookieとヘッダーのトークンが一致することを確認
```

### 11.3 機密情報の保護

```javascript
// ★ アクセストークンをURLに含めない
// 悪い例: fetch(`/api/data?token=${accessToken}`)
// → URLはログに記録される、Refererヘッダーで漏洩する

// 良い例: Authorizationヘッダーを使用
fetch('/api/data', {
  headers: { 'Authorization': `Bearer ${accessToken}` },
});

// ★ レスポンスのキャッシュに注意
// 機密データにはキャッシュ制御ヘッダーを設定
// Cache-Control: no-store, no-cache, must-revalidate
// Pragma: no-cache

// ★ エラーメッセージに機密情報を含めない
// 悪い例: throw new Error(`API key ${apiKey} is invalid`);
// 良い例: throw new Error('Authentication failed');

// Content-Security-Policy でfetchの宛先を制限
// Content-Security-Policy: connect-src 'self' https://api.example.com
```

---

## 12. 実務パターン集

### 12.1 ページネーション

```typescript
// オフセットベースのページネーション
async function fetchPaginated<T>(
  url: string,
  page: number,
  limit: number
): Promise<{ data: T[]; total: number; hasMore: boolean }> {
  const params = new URLSearchParams({
    page: String(page),
    limit: String(limit),
  });

  const response = await fetch(`${url}?${params}`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);

  const result = await response.json();
  return {
    data: result.data,
    total: result.total,
    hasMore: page * limit < result.total,
  };
}

// カーソルベースのページネーション
async function* fetchAllPages<T>(
  url: string,
  limit = 100
): AsyncGenerator<T[]> {
  let cursor: string | null = null;

  while (true) {
    const params = new URLSearchParams({ limit: String(limit) });
    if (cursor) params.set('cursor', cursor);

    const response = await fetch(`${url}?${params}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const result = await response.json();
    yield result.data;

    cursor = result.nextCursor;
    if (!cursor || result.data.length === 0) break;
  }
}

// 使用例: 全ページのデータを収集
async function fetchAllUsers() {
  const allUsers: User[] = [];

  for await (const page of fetchAllPages<User>('/api/users', 50)) {
    allUsers.push(...page);
    console.log(`Loaded ${allUsers.length} users so far...`);
  }

  return allUsers;
}

// 無限スクロールの実装（React）
function InfiniteScrollList() {
  const [items, setItems] = useState<Item[]>([]);
  const [cursor, setCursor] = useState<string | null>(null);
  const [hasMore, setHasMore] = useState(true);
  const [loading, setLoading] = useState(false);
  const observerRef = useRef<IntersectionObserver | null>(null);
  const sentinelRef = useRef<HTMLDivElement | null>(null);

  const loadMore = useCallback(async () => {
    if (loading || !hasMore) return;
    setLoading(true);

    try {
      const params = new URLSearchParams({ limit: '20' });
      if (cursor) params.set('cursor', cursor);

      const response = await fetch(`/api/items?${params}`);
      const result = await response.json();

      setItems(prev => [...prev, ...result.data]);
      setCursor(result.nextCursor);
      setHasMore(!!result.nextCursor);
    } catch (err) {
      console.error('Load more failed:', err);
    } finally {
      setLoading(false);
    }
  }, [cursor, hasMore, loading]);

  useEffect(() => {
    observerRef.current = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          loadMore();
        }
      },
      { threshold: 0.1 }
    );

    if (sentinelRef.current) {
      observerRef.current.observe(sentinelRef.current);
    }

    return () => observerRef.current?.disconnect();
  }, [loadMore]);

  return (
    <div>
      {items.map(item => (
        <ItemCard key={item.id} item={item} />
      ))}
      {hasMore && <div ref={sentinelRef}>{loading ? 'Loading...' : ''}</div>}
    </div>
  );
}
```

### 12.2 楽観的更新（Optimistic Updates）

```typescript
// 楽観的更新パターン
async function optimisticUpdate<T>(
  currentState: T,
  optimisticState: T,
  setState: (state: T) => void,
  apiCall: () => Promise<T>
): Promise<T> {
  // 1. 即座にUIを更新
  setState(optimisticState);

  try {
    // 2. APIコール
    const serverState = await apiCall();
    // 3. サーバーの結果で上書き
    setState(serverState);
    return serverState;
  } catch (err) {
    // 4. エラー時はロールバック
    setState(currentState);
    throw err;
  }
}

// React での使用例（いいねボタン）
function LikeButton({ postId, initialLiked, initialCount }) {
  const [liked, setLiked] = useState(initialLiked);
  const [count, setCount] = useState(initialCount);

  const toggleLike = async () => {
    const previousLiked = liked;
    const previousCount = count;

    // 楽観的更新
    setLiked(!liked);
    setCount(liked ? count - 1 : count + 1);

    try {
      const result = await fetch(`/api/posts/${postId}/like`, {
        method: liked ? 'DELETE' : 'POST',
      });

      if (!result.ok) throw new Error('Failed');

      const data = await result.json();
      setCount(data.likeCount);
    } catch (err) {
      // ロールバック
      setLiked(previousLiked);
      setCount(previousCount);
      toast.error('操作に失敗しました');
    }
  };

  return (
    <button onClick={toggleLike} className={liked ? 'liked' : ''}>
      {liked ? '❤' : '♡'} {count}
    </button>
  );
}
```

### 12.3 ポーリングとWebSocket

```typescript
// ロングポーリング
async function longPoll(url: string, onMessage: (data: any) => void) {
  while (true) {
    try {
      const response = await fetch(url, {
        signal: AbortSignal.timeout(60000), // 60秒タイムアウト
      });

      if (response.ok) {
        const data = await response.json();
        onMessage(data);
      }
    } catch (err) {
      if (err.name === 'TimeoutError') {
        // タイムアウトは正常（再接続）
        continue;
      }
      // エラー時は少し待ってからリトライ
      await new Promise(r => setTimeout(r, 5000));
    }
  }
}

// インターバルポーリング（指数バックオフ付き）
class Poller {
  private timer: ReturnType<typeof setTimeout> | null = null;
  private interval: number;
  private maxInterval: number;
  private currentInterval: number;

  constructor(
    private url: string,
    private onData: (data: any) => void,
    options: { interval?: number; maxInterval?: number } = {}
  ) {
    this.interval = options.interval || 5000;
    this.maxInterval = options.maxInterval || 60000;
    this.currentInterval = this.interval;
  }

  start() {
    this.poll();
  }

  stop() {
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }
  }

  private async poll() {
    try {
      const response = await fetch(this.url);
      if (response.ok) {
        const data = await response.json();
        this.onData(data);
        this.currentInterval = this.interval; // 成功時はリセット
      }
    } catch (err) {
      // エラー時はバックオフ
      this.currentInterval = Math.min(
        this.currentInterval * 2,
        this.maxInterval
      );
    }

    this.timer = setTimeout(() => this.poll(), this.currentInterval);
  }
}

// 使用例
const poller = new Poller('/api/notifications', (data) => {
  updateNotifications(data);
}, { interval: 10000 });

poller.start();
// poller.stop();
```

---

## 13. Node.js / Edge Runtime でのFetch

### 13.1 Node.jsでのFetch API

```javascript
// Node.js 18+ でのネイティブfetch
const response = await fetch('https://api.example.com/data');
const data = await response.json();

// Node.js固有の設定
// ★ keepalive はNode.jsではデフォルトでfalse
const response = await fetch('https://api.example.com/data', {
  keepalive: true,
});

// ★ Node.jsではHTTPSの証明書検証をカスタマイズ可能（undici使用時）
import { Agent, fetch } from 'undici';

const agent = new Agent({
  connect: {
    rejectUnauthorized: false, // 開発環境のみ
  },
});

const response = await fetch('https://self-signed.example.com/api', {
  dispatcher: agent,
});

// プロキシの設定（undici使用時）
import { ProxyAgent, fetch } from 'undici';

const proxyAgent = new ProxyAgent('http://proxy.example.com:8080');
const response = await fetch('https://api.example.com/data', {
  dispatcher: proxyAgent,
});
```

### 13.2 Next.js のfetch拡張

```typescript
// Next.js App Router のfetch拡張
// サーバーコンポーネントでのデータ取得

// 静的レンダリング（ビルド時に実行、キャッシュ）
const data = await fetch('https://api.example.com/posts', {
  cache: 'force-cache', // デフォルト（Next.js 14以前）
});

// 動的レンダリング（リクエスト毎に実行）
const data = await fetch('https://api.example.com/posts', {
  cache: 'no-store',
});

// ISR（Incremental Static Regeneration）
const data = await fetch('https://api.example.com/posts', {
  next: {
    revalidate: 60, // 60秒ごとに再検証
  },
});

// タグベースの再検証
const data = await fetch('https://api.example.com/posts', {
  next: {
    tags: ['posts'], // revalidateTag('posts') で無効化
  },
});

// Server Action からの再検証
'use server';
import { revalidateTag, revalidatePath } from 'next/cache';

async function createPost(formData: FormData) {
  await fetch('https://api.example.com/posts', {
    method: 'POST',
    body: JSON.stringify(Object.fromEntries(formData)),
  });

  revalidateTag('posts');
  revalidatePath('/posts');
}
```

---

## 14. デバッグとトラブルシューティング

### 14.1 DevToolsでの調査

```javascript
// DevTools の Network タブで確認できる情報
// - リクエスト/レスポンスヘッダー
// - リクエストボディ
// - レスポンスボディ
// - タイミング（DNS, TCP, TLS, TTFB, コンテンツダウンロード）
// - CORSヘッダー（プリフライトリクエスト含む）

// コンソールでのfetchデバッグ
// 全てのfetchリクエストをインターセプト
const originalFetch = window.fetch;
window.fetch = async function (...args) {
  const [url, options] = args;
  console.group(`fetch: ${options?.method || 'GET'} ${url}`);
  console.log('Options:', options);

  const startTime = performance.now();

  try {
    const response = await originalFetch.apply(this, args);
    const duration = performance.now() - startTime;

    console.log(`Status: ${response.status} ${response.statusText}`);
    console.log(`Duration: ${duration.toFixed(0)}ms`);
    console.log('Headers:', Object.fromEntries(response.headers.entries()));
    console.groupEnd();

    return response;
  } catch (err) {
    const duration = performance.now() - startTime;
    console.error(`Error after ${duration.toFixed(0)}ms:`, err);
    console.groupEnd();
    throw err;
  }
};

// Resource Timing API でパフォーマンス測定
const entries = performance.getEntriesByType('resource');
const fetchEntries = entries.filter(e => e.initiatorType === 'fetch');

for (const entry of fetchEntries) {
  console.log({
    name: entry.name,
    duration: entry.duration,
    transferSize: entry.transferSize,
    dnsLookup: entry.domainLookupEnd - entry.domainLookupStart,
    tcpConnect: entry.connectEnd - entry.connectStart,
    ttfb: entry.responseStart - entry.requestStart,
    download: entry.responseEnd - entry.responseStart,
  });
}
```

### 14.2 よくある問題と解決策

```javascript
// 問題1: JSONのパースエラー
// → レスポンスがJSONでない場合（HTML、エラーページ等）
try {
  const data = await response.json();
} catch (err) {
  if (err instanceof SyntaxError) {
    const text = await response.clone().text();
    console.error('Invalid JSON response:', text.substring(0, 200));
  }
}

// 問題2: メモリリーク（レスポンスの未消費）
// → レスポンスボディを読まないとメモリに残り続ける
const response = await fetch('/api/data');
if (!response.ok) {
  // ★ エラー時もボディを消費する
  await response.text(); // または response.body?.cancel()
  throw new Error(`HTTP ${response.status}`);
}

// 問題3: 同時リクエスト制限
// → ブラウザはドメインあたり6-8並行接続まで
// → 多数のリクエストを送る場合は並行数を制限する

// 問題4: fetchが完了しない
// → タイムアウトを必ず設定する
// → AbortSignal.timeout() を使用する

// 問題5: Service Worker内でのfetch
// → 無限ループに注意（fetchイベント内でfetchを呼ぶ）
self.addEventListener('fetch', (event) => {
  // ★ 同じURLへのfetchを避ける
  if (event.request.url.includes('/api/')) {
    event.respondWith(
      caches.match(event.request).then(cached => {
        return cached || fetch(event.request); // Service Workerのfetchは別コンテキスト
      })
    );
  }
});
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| Fetch API | XMLHttpRequestの後継、Promiseベース、response.okの確認必須 |
| Request / Response | イミュータブル、clone()で複製、1回のみボディ読み取り可能 |
| AbortController | リクエストキャンセル、タイムアウト、AbortSignal.any()で合成 |
| ReadableStream | レスポンスの段階的処理、バックプレッシャー制御 |
| TransformStream | ストリームデータの変換パイプライン |
| WritableStream | データの書き込み先、pipeTo()で接続 |
| SSE | Server-Sent Events、EventSourceまたはFetch+Streamsで処理 |
| リトライ | 指数バックオフ、Retry-Afterヘッダー、ジッター |
| キャッシュ | ETag / Last-Modified条件付きリクエスト、Cache API |
| CORS | プリフライト、credentials、mode設定 |
| セキュリティ | CSRF対策、XSS防止、トークンの安全な送信 |
| テスト | MSWによるモック、インテグレーションテスト |

---

## 次に読むべきガイド

- [[02-intersection-resize-observer.md]] -- Observer API（IntersectionObserver, ResizeObserver, MutationObserver）
- [[../04-storage-and-caching/00-web-storage.md]] -- Web Storage API（localStorage, sessionStorage, IndexedDB）
- [[../04-storage-and-caching/01-service-worker-cache.md]] -- Service Worker と Cache API

---

## 参考文献

1. Fetch Living Standard. WHATWG, 2024. https://fetch.spec.whatwg.org/
2. MDN Web Docs. "Fetch API." Mozilla, 2024. https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API
3. MDN Web Docs. "Streams API." Mozilla, 2024. https://developer.mozilla.org/en-US/docs/Web/API/Streams_API
4. MDN Web Docs. "AbortController." Mozilla, 2024. https://developer.mozilla.org/en-US/docs/Web/API/AbortController
5. MDN Web Docs. "ReadableStream." Mozilla, 2024. https://developer.mozilla.org/en-US/docs/Web/API/ReadableStream
6. MDN Web Docs. "TransformStream." Mozilla, 2024. https://developer.mozilla.org/en-US/docs/Web/API/TransformStream
7. Jake Archibald. "Streams — The Definitive Guide." web.dev, 2023.
8. Web.dev. "Fetch API." Google, 2024. https://web.dev/articles/introduction-to-fetch
9. MSW Documentation. "Mock Service Worker." 2024. https://mswjs.io/
10. Undici Documentation. "Node.js HTTP Client." 2024. https://undici.nodejs.org/
