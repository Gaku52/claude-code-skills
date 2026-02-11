# Fetch と Streams

> Fetch APIはXMLHttpRequestの後継。Streams APIと組み合わせることで、大きなレスポンスの段階的処理、進捗表示、AbortControllerによるキャンセルを実現する。

## この章で学ぶこと

- [ ] Fetch APIの基本と高度な使い方を理解する
- [ ] Streams APIでのストリーミング処理を把握する
- [ ] AbortControllerによるリクエストキャンセルを学ぶ

---

## 1. Fetch API

```javascript
// 基本的なGET
const response = await fetch('/api/users');
const data = await response.json();

// POSTリクエスト
const response = await fetch('/api/users', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ name: 'Taro', email: 'taro@example.com' }),
});

// fetchの注意点:
// → 404や500でもrejectしない（response.ok で確認）
if (!response.ok) {
  throw new Error(`HTTP ${response.status}: ${response.statusText}`);
}

// レスポンスの読み取り（1回のみ）
response.json();    // JSON
response.text();    // テキスト
response.blob();    // バイナリ（画像等）
response.arrayBuffer(); // ArrayBuffer
response.formData(); // FormData

// リクエストオプション
fetch(url, {
  method: 'POST',
  headers: { 'Authorization': 'Bearer token' },
  body: JSON.stringify(data),
  mode: 'cors',           // cors, no-cors, same-origin
  credentials: 'include', // include, same-origin, omit
  cache: 'no-cache',      // default, no-store, reload, no-cache
  redirect: 'follow',     // follow, error, manual
  signal: abortController.signal,
});
```

---

## 2. AbortController

```javascript
// リクエストのキャンセル
const controller = new AbortController();

fetch('/api/data', { signal: controller.signal })
  .then(res => res.json())
  .catch(err => {
    if (err.name === 'AbortError') {
      console.log('Request was cancelled');
    }
  });

// キャンセル実行
controller.abort();

// タイムアウト
const controller = new AbortController();
const timeoutId = setTimeout(() => controller.abort(), 5000);

try {
  const response = await fetch('/api/data', { signal: controller.signal });
  clearTimeout(timeoutId);
  return response.json();
} catch (err) {
  if (err.name === 'AbortError') {
    throw new Error('Request timed out');
  }
  throw err;
}

// AbortSignal.timeout()（新しいAPI）
const response = await fetch('/api/data', {
  signal: AbortSignal.timeout(5000),
});

// 複数のシグナルを結合
const controller = new AbortController();
const signal = AbortSignal.any([
  controller.signal,
  AbortSignal.timeout(5000),
]);

// React での使用パターン
useEffect(() => {
  const controller = new AbortController();

  fetch('/api/data', { signal: controller.signal })
    .then(res => res.json())
    .then(data => setData(data))
    .catch(err => {
      if (err.name !== 'AbortError') setError(err);
    });

  return () => controller.abort();  // クリーンアップ
}, []);
```

---

## 3. Streams API

```javascript
// ReadableStream — レスポンスのストリーミング読み取り

// ダウンロード進捗表示
async function fetchWithProgress(url, onProgress) {
  const response = await fetch(url);
  const total = parseInt(response.headers.get('content-length'), 10);
  const reader = response.body.getReader();
  let received = 0;
  const chunks = [];

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    chunks.push(value);
    received += value.length;
    onProgress(received / total);
  }

  const blob = new Blob(chunks);
  return blob;
}

// ストリーミングJSONパース（大きなJSONの段階的処理）
async function streamJSON(url) {
  const response = await fetch(url);
  const reader = response.body
    .pipeThrough(new TextDecoderStream())
    .getReader();

  let buffer = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += value;
    // 行ごとに処理（NDJSON形式の場合）
    const lines = buffer.split('\n');
    buffer = lines.pop();
    for (const line of lines) {
      if (line.trim()) {
        const item = JSON.parse(line);
        processItem(item);
      }
    }
  }
}

// TransformStream — データの変換
const uppercaseStream = new TransformStream({
  transform(chunk, controller) {
    controller.enqueue(chunk.toUpperCase());
  },
});

response.body
  .pipeThrough(new TextDecoderStream())
  .pipeThrough(uppercaseStream)
  .pipeTo(writableStream);
```

---

## 4. fetch ラッパー

```typescript
// プロダクションレベルの fetch ラッパー
async function api<T>(
  url: string,
  options: RequestInit = {},
): Promise<T> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 10000);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new ApiError(response.status, error.message || response.statusText);
    }

    return response.json();
  } finally {
    clearTimeout(timeoutId);
  }
}

class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

// 使用
const users = await api<User[]>('/api/users');
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| Fetch | XMLHttpRequest の後継、Promise ベース |
| AbortController | リクエストキャンセル、タイムアウト |
| ReadableStream | レスポンスの段階的処理 |
| TransformStream | ストリームデータの変換 |

---

## 次に読むべきガイド
→ [[02-intersection-resize-observer.md]] — Observer API

---

## 参考文献
1. Fetch Living Standard. WHATWG, 2024.
2. MDN Web Docs. "Streams API." Mozilla, 2024.
