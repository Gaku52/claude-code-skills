# ブラウザセキュリティモデル

> ブラウザはユーザーとWeb上のコンテンツの間に立つセキュリティの要である。サンドボックス、同一オリジンポリシー、CSP、サイトアイソレーション、Cookie のセキュリティ属性など、多層防御の仕組みを体系的に理解することが、安全なWebアプリケーション開発の基盤となる。

## この章で学ぶこと

- [ ] ブラウザのサンドボックスモデルの多層構造を理解する
- [ ] Same-Origin Policy（同一オリジンポリシー）の原理と例外を把握する
- [ ] CSP（Content Security Policy）の設計思想と実践的な設定方法を習得する
- [ ] サイトアイソレーションのアーキテクチャと Spectre 対策を学ぶ
- [ ] Cookie のセキュリティ属性を正しく設定できるようになる
- [ ] CORS、SRI、Trusted Types 等の補助的セキュリティ機構を活用できる
- [ ] セキュリティヘッダーの組み合わせによる多層防御を実装できる

## 前提知識

- ブラウザのアーキテクチャ → 参照: [ブラウザアーキテクチャ](./00-browser-architecture.md)
- HTMLパースとDOM構築 → 参照: [HTML/CSSパース](./02-parsing-html-css.md)
- Webセキュリティの基礎（XSS, CSRF等） → 参照: セキュリティ基礎

---

## 0. ブラウザセキュリティの全体像

### 0.1 なぜブラウザセキュリティが重要なのか

ブラウザは現代のコンピューティングにおいて最も広く使われるアプリケーションの一つであり、ユーザーは日常的に銀行取引、個人情報の入力、機密文書の閲覧などをブラウザ上で行う。しかしブラウザは同時に、信頼できないコンテンツ（任意のWebサイトのHTML、CSS、JavaScript）を実行する環境でもある。この「信頼できないコードを安全に実行する」という根本的な課題に対処するため、ブラウザは複数のセキュリティ層を組み合わせた多層防御（Defense in Depth）アーキテクチャを採用している。

### 0.2 多層防御の概念図

```
+=====================================================================+
|                        ユーザーの操作環境                              |
+=====================================================================+
|  Layer 5: UI レベルの保護                                            |
|  ┌─────────────────────────────────────────────────────────────┐    |
|  │ ・アドレスバーの表示（フィッシング対策）                        │    |
|  │ ・権限プロンプト（カメラ、位置情報など）                        │    |
|  │ ・混合コンテンツ警告                                          │    |
|  │ ・証明書エラー表示                                            │    |
|  └─────────────────────────────────────────────────────────────┘    |
|  Layer 4: ネットワークレベルの保護                                    |
|  ┌─────────────────────────────────────────────────────────────┐    |
|  │ ・HTTPS/TLS による通信暗号化                                   │    |
|  │ ・HSTS（HTTP Strict Transport Security）                      │    |
|  │ ・Certificate Transparency                                    │    |
|  │ ・DNS over HTTPS (DoH)                                        │    |
|  └─────────────────────────────────────────────────────────────┘    |
|  Layer 3: コンテンツレベルの保護                                      |
|  ┌─────────────────────────────────────────────────────────────┐    |
|  │ ・CSP（Content Security Policy）                               │    |
|  │ ・CORS（Cross-Origin Resource Sharing）                        │    |
|  │ ・SRI（Subresource Integrity）                                 │    |
|  │ ・Trusted Types                                                │    |
|  │ ・Referrer Policy                                              │    |
|  └─────────────────────────────────────────────────────────────┘    |
|  Layer 2: オリジンレベルの保護                                        |
|  ┌─────────────────────────────────────────────────────────────┐    |
|  │ ・Same-Origin Policy（同一オリジンポリシー）                    │    |
|  │ ・Cookie の SameSite 属性                                      │    |
|  │ ・オリジン単位のストレージ隔離                                  │    |
|  └─────────────────────────────────────────────────────────────┘    |
|  Layer 1: プロセスレベルの保護                                        |
|  ┌─────────────────────────────────────────────────────────────┐    |
|  │ ・サンドボックス（OS レベルの権限制限）                         │    |
|  │ ・サイトアイソレーション（プロセス分離）                        │    |
|  │ ・V8 エンジンのメモリ安全性                                    │    |
|  └─────────────────────────────────────────────────────────────┘    |
|  Layer 0: OS レベルの保護                                             |
|  ┌─────────────────────────────────────────────────────────────┐    |
|  │ ・ASLR（Address Space Layout Randomization）                   │    |
|  │ ・DEP/NX（Data Execution Prevention）                          │    |
|  │ ・seccomp-bpf (Linux) / Seatbelt (macOS) / LPAC (Windows)     │    |
|  └─────────────────────────────────────────────────────────────┘    |
+=====================================================================+
```

この図に示すとおり、ブラウザセキュリティは単一の仕組みではなく、OSレベルからUIレベルまでの6つの層が連携して動作する。どの層が突破されても、他の層が被害を最小化する設計となっている。

### 0.3 主要な攻撃ベクトルとそれに対応する防御層

| 攻撃手法 | 概要 | 主な防御層 |
|----------|------|-----------|
| XSS（Cross-Site Scripting） | 悪意あるスクリプトの注入と実行 | CSP, Trusted Types, Same-Origin Policy |
| CSRF（Cross-Site Request Forgery） | ユーザーの認証情報を悪用した不正リクエスト | SameSite Cookie, CSRF Token, Origin ヘッダー検証 |
| Clickjacking | 透明なiframeで意図しない操作を誘導 | X-Frame-Options, CSP frame-ancestors |
| MITM（Man-in-the-Middle） | 通信の傍受・改ざん | HTTPS/TLS, HSTS, Certificate Pinning |
| Spectre/Meltdown | CPUの投機的実行を悪用したメモリ読み取り | サイトアイソレーション, Cross-Origin Isolation |
| Drive-by Download | 脆弱性を突いたマルウェアの自動ダウンロード | サンドボックス, Safe Browsing API |
| Supply Chain Attack | CDN等の第三者リソースの改ざん | SRI, CSP |
| DNS Rebinding | DNS応答を操作してオリジン制限を回避 | Same-Origin Policy, DNS Pinning |

---

## 1. サンドボックス

### 1.1 サンドボックスの基本概念

サンドボックスとは、プログラムの実行環境を隔離し、そのプログラムがアクセスできるリソースを厳しく制限する仕組みである。ブラウザにおけるサンドボックスは、Webコンテンツ（HTML/CSS/JavaScript）を処理するレンダラープロセスに対して適用され、たとえレンダラープロセスが攻撃者に乗っ取られたとしても、ユーザーのシステムへの影響を最小限に抑えることを目的とする。

### 1.2 Chromium のマルチプロセスアーキテクチャ

```
+-------------------------------------------------------------------+
|                     Chromium プロセスモデル                          |
+-------------------------------------------------------------------+
|                                                                   |
|  ┌──────────────────────────────────────────┐                     |
|  │          ブラウザプロセス (Browser)         │  ← 高権限          |
|  │  ・UI管理（タブ、アドレスバー）              │                     |
|  │  ・ネットワークI/O                          │                     |
|  │  ・ファイルシステムアクセス                   │                     |
|  │  ・子プロセスの生成と管理                    │                     |
|  │  ・権限管理                                │                     |
|  └──────────┬────────────┬──────────────────┘                     |
|             │ IPC (Mojo) │                                        |
|     ┌───────┴────┐  ┌────┴───────┐  ┌────────────┐               |
|     │ レンダラー  │  │ レンダラー  │  │ レンダラー  │  ← 低権限     |
|     │ プロセスA   │  │ プロセスB   │  │ プロセスC   │  (サンドボックス)|
|     │            │  │            │  │            │               |
|     │ site-a.com │  │ site-b.com │  │ site-c.com │               |
|     └────────────┘  └────────────┘  └────────────┘               |
|                                                                   |
|     ┌────────────┐  ┌────────────┐  ┌────────────┐               |
|     │  GPU       │  │ Network    │  │ Storage    │               |
|     │ プロセス    │  │ サービス    │  │ サービス    │               |
|     └────────────┘  └────────────┘  └────────────┘               |
+-------------------------------------------------------------------+
```

Chromium では、各サイトのコンテンツは独立したレンダラープロセスで実行される。レンダラープロセスはサンドボックス内で動作し、以下の制限が課される。

### 1.3 OS 別のサンドボックス実装

サンドボックスの具体的な実装はOSごとに異なる。各OSが提供するセキュリティ機構を活用して、レンダラープロセスの権限を最小化する。

| OS | サンドボックス技術 | 主な制限内容 |
|----|--------------------|-------------|
| Linux | seccomp-bpf + Namespaces | システムコールのフィルタリング、PID/ネットワーク名前空間による隔離 |
| macOS | Seatbelt (sandbox_init) | プロファイルベースのリソースアクセス制御 |
| Windows | Restricted Token + LPAC | トークンの権限削減、AppContainer による隔離 |
| Android | SELinux + seccomp-bpf | 強制アクセス制御 + システムコール制限 |
| ChromeOS | Minijail + Namespaces | 最小権限のジェイルプロセス |

#### Linux におけるサンドボックスの詳細

Linux 上の Chromium では、seccomp-bpf（Secure Computing mode with Berkeley Packet Filter）を用いてシステムコールをフィルタリングする。レンダラープロセスが呼び出せるシステムコールは厳密にホワイトリスト化されており、ファイルの open()、ネットワークソケットの作成、プロセスの生成などは禁止される。

```c
// seccomp-bpf フィルタの概念的な疑似コード
// （Chromium の実際の実装を簡略化したもの）

struct sock_filter filter[] = {
    // アーキテクチャの検証
    BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
             offsetof(struct seccomp_data, arch)),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, AUDIT_ARCH_X86_64, 1, 0),
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL),

    // システムコール番号を取得
    BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
             offsetof(struct seccomp_data, nr)),

    // 許可されるシステムコール
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_read, 0, 1),
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_write, 0, 1),
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_mmap, 0, 1),
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_mprotect, 0, 1),
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_futex, 0, 1),
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),

    // 上記以外はすべて拒否（SIGSYS シグナルで通知）
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_TRAP),
};

struct sock_fprog prog = {
    .len = (unsigned short)(sizeof(filter) / sizeof(filter[0])),
    .filter = filter,
};

// サンドボックスの適用
prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0);
prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &prog);
```

### 1.4 サンドボックスが制限する操作と許可する操作

```
サンドボックスによる権限分離:

  ┌──────────────────────────────────────────────────────────────┐
  │                  レンダラープロセス（サンドボックス内）          │
  │                                                              │
  │  制限される操作:                                               │
  │  ✗ ファイルシステムへの直接アクセス（open, stat, unlink等）    │
  │  ✗ ネットワークソケットの直接作成（socket, connect 等）        │
  │  ✗ 新規プロセスの生成（fork, execve 等）                      │
  │  ✗ OSのAPIへの直接アクセス                                    │
  │  ✗ 他のプロセスメモリへのアクセス                              │
  │  ✗ ハードウェアデバイスの直接制御                              │
  │  ✗ カーネルモジュールのロード                                  │
  │                                                              │
  │  IPC経由で許可される操作:                                      │
  │  ✓ ブラウザプロセスへのリソースリクエスト（fetch等）            │
  │  ✓ ユーザーが許可した機能へのアクセス（Permissions API 経由）  │
  │  ✓ 共有メモリ領域の読み書き（GPU描画用）                       │
  │  ✓ オリジン単位で隔離されたストレージへのアクセス               │
  │                                                              │
  │  JavaScript API 経由で許可される操作:                          │
  │  ✓ fetch() でHTTPリクエスト（CORSの範囲内）                    │
  │  ✓ <input type="file"> でユーザーが選択したファイル読み込み    │
  │  ✓ Geolocation API（ユーザー許可後）                           │
  │  ✓ Camera/Microphone（ユーザー許可後）                         │
  │  ✓ localStorage / IndexedDB（オリジン単位で隔離）              │
  │  ✓ Web Workers / Service Workers の生成                        │
  └──────────────────────────────────────────────────────────────┘

  Permissions Policy（旧 Feature Policy）:
  → Webページが使用できるブラウザ機能をHTTPヘッダーで制限
  → iframe に対しても機能制限を適用可能

  例: Permissions-Policy: camera=(), microphone=(), geolocation=(self)
```

### 1.5 iframe のサンドボックス属性

HTML の `<iframe>` 要素には `sandbox` 属性を指定でき、埋め込みコンテンツに対して追加の制限を課すことができる。

```html
<!-- 最も制限の厳しい設定（全機能をブロック） -->
<iframe src="https://untrusted.example.com" sandbox></iframe>

<!-- 必要な機能のみ選択的に許可 -->
<iframe src="https://payment.example.com"
        sandbox="allow-scripts allow-forms allow-same-origin">
</iframe>

<!-- sandbox 属性で制御可能なフラグ一覧 -->
<!--
  allow-forms            : フォーム送信を許可
  allow-modals           : alert(), confirm() 等のモーダルを許可
  allow-orientation-lock : 画面の向きロックを許可
  allow-pointer-lock     : Pointer Lock API を許可
  allow-popups           : window.open() やtarget="_blank" を許可
  allow-popups-to-escape-sandbox : ポップアップにサンドボックスを継承しない
  allow-presentation     : Presentation API を許可
  allow-same-origin      : 同一オリジンとして扱う（※注意が必要）
  allow-scripts          : JavaScript の実行を許可
  allow-top-navigation   : 親フレームのナビゲーションを許可
  allow-downloads        : ファイルダウンロードを許可
-->
```

**注意**: `allow-scripts` と `allow-same-origin` を同時に指定すると、埋め込みコンテンツが自身の sandbox 属性を JavaScript で除去できてしまうため、信頼できないコンテンツに対してはこの組み合わせを避けること。

---

## 2. Same-Origin Policy（同一オリジンポリシー）

### 2.1 オリジンの定義

Same-Origin Policy（SOP）はブラウザセキュリティの最も基本的な仕組みであり、1995年に Netscape Navigator 2.0 で初めて導入された。SOPの核心は「オリジン」の概念にある。

**オリジン** = スキーム（プロトコル） + ホスト（ドメイン） + ポート番号

```
オリジンの判定例:

  基準URL: https://www.example.com:443/path/page.html

  ┌─────────────────────────────────────────┬───────────┬──────────────────┐
  │ 比較対象のURL                             │ 同一オリジン│ 理由              │
  ├─────────────────────────────────────────┼───────────┼──────────────────┤
  │ https://www.example.com:443/other.html  │ Yes       │ パスのみ異なる     │
  │ https://www.example.com/other.html      │ Yes       │ 443は省略可能      │
  │ http://www.example.com/page.html        │ No        │ スキームが異なる   │
  │ https://api.example.com/page.html       │ No        │ ホストが異なる     │
  │ https://www.example.com:8080/page.html  │ No        │ ポートが異なる     │
  │ https://example.com/page.html           │ No        │ サブドメインが異なる│
  └─────────────────────────────────────────┴───────────┴──────────────────┘
```

### 2.2 SOPが制御する対象

Same-Origin Policy は、異なるオリジン間でのリソースアクセスを以下のように制御する。

| 操作カテゴリ | 具体例 | クロスオリジンでの動作 |
|-------------|--------|----------------------|
| 読み取り（Read） | DOM アクセス、Cookie 読み取り、AJAX レスポンス | 原則禁止 |
| 書き込み（Write） | リンク、リダイレクト、フォーム送信 | 原則許可 |
| 埋め込み（Embed） | `<script>`, `<img>`, `<iframe>`, `<link>` | 原則許可 |

```javascript
// Same-Origin Policy の動作例

// --- 同一オリジン（許可される）---
// 現在のページ: https://app.example.com/dashboard

// DOM アクセス
const iframe = document.getElementById('settings-frame');
// iframe のソースが同一オリジンなら DOM にアクセス可能
const innerDoc = iframe.contentDocument;  // OK

// AJAX リクエスト
const response = await fetch('https://app.example.com/api/data');
const data = await response.json();  // OK: 同一オリジン

// --- 異なるオリジン（制限される）---

// DOM アクセスの制限
const externalFrame = document.getElementById('external-frame');
// iframe のソースが異なるオリジンの場合
try {
    const doc = externalFrame.contentDocument;  // SecurityError
} catch (e) {
    console.error('Cross-origin DOM access blocked:', e.message);
}

// AJAX リクエストの制限（CORSなし）
try {
    const resp = await fetch('https://other-site.com/api/data');
    // サーバーが適切な CORS ヘッダーを返さない場合
    const data = await resp.json();  // TypeError: Failed to fetch
} catch (e) {
    console.error('Cross-origin request blocked:', e.message);
}

// window.postMessage による安全なクロスオリジン通信
// 送信側（親ウィンドウ）
const targetOrigin = 'https://trusted-partner.com';
externalFrame.contentWindow.postMessage(
    { type: 'greeting', payload: 'Hello!' },
    targetOrigin  // 必ず具体的なオリジンを指定（'*' は避ける）
);

// 受信側（iframe 内のスクリプト）
window.addEventListener('message', (event) => {
    // オリジンの検証は必須
    if (event.origin !== 'https://app.example.com') {
        console.warn('Rejected message from untrusted origin:', event.origin);
        return;
    }
    console.log('Received:', event.data);
});
```

### 2.3 SOP の例外と緩和メカニズム

Same-Origin Policy には、正当なユースケースに対応するためのいくつかの例外と緩和メカニズムが存在する。

#### document.domain による緩和（非推奨）

```javascript
// https://app.example.com のページ
document.domain = 'example.com';

// https://api.example.com のページでも同様に設定
document.domain = 'example.com';

// これにより両ページが同一オリジンとして扱われる
// ※ 注意: この機能は非推奨であり、将来的に削除予定
// 代替手段: postMessage, CORS, Channel Messaging API
```

#### CORS（Cross-Origin Resource Sharing）

CORS は SOP を安全に緩和するための標準メカニズムであり、サーバーが明示的に許可したクロスオリジンリクエストのみを通過させる。

```
CORS のリクエストフロー（プリフライトあり）:

  ブラウザ                                   サーバー
    │                                          │
    │  ① OPTIONS /api/data HTTP/1.1            │
    │  Origin: https://app.example.com         │
    │  Access-Control-Request-Method: POST     │
    │  Access-Control-Request-Headers:         │
    │    Content-Type, Authorization           │
    │ ──────────────────────────────────────>   │
    │                                          │
    │  ② 200 OK                                │
    │  Access-Control-Allow-Origin:            │
    │    https://app.example.com               │
    │  Access-Control-Allow-Methods:           │
    │    GET, POST, PUT                        │
    │  Access-Control-Allow-Headers:           │
    │    Content-Type, Authorization           │
    │  Access-Control-Max-Age: 86400           │
    │ <──────────────────────────────────────   │
    │                                          │
    │  ③ POST /api/data HTTP/1.1               │
    │  Origin: https://app.example.com         │
    │  Content-Type: application/json          │
    │  Authorization: Bearer token123          │
    │  {"key": "value"}                        │
    │ ──────────────────────────────────────>   │
    │                                          │
    │  ④ 200 OK                                │
    │  Access-Control-Allow-Origin:            │
    │    https://app.example.com               │
    │  {"result": "success"}                   │
    │ <──────────────────────────────────────   │
    │                                          │
```

```javascript
// サーバー側の CORS 設定例（Node.js / Express）

const express = require('express');
const app = express();

// 方法1: 手動での CORS ヘッダー設定
app.use((req, res, next) => {
    // 許可するオリジンのホワイトリスト
    const allowedOrigins = [
        'https://app.example.com',
        'https://staging.example.com'
    ];

    const origin = req.headers.origin;
    if (allowedOrigins.includes(origin)) {
        res.setHeader('Access-Control-Allow-Origin', origin);
    }

    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE');
    res.setHeader('Access-Control-Allow-Headers',
        'Content-Type, Authorization, X-Requested-With');
    res.setHeader('Access-Control-Allow-Credentials', 'true');
    res.setHeader('Access-Control-Max-Age', '86400');

    // プリフライトリクエストへの応答
    if (req.method === 'OPTIONS') {
        return res.status(204).end();
    }

    next();
});

// 方法2: cors ミドルウェアの使用
const cors = require('cors');
app.use(cors({
    origin: ['https://app.example.com', 'https://staging.example.com'],
    methods: ['GET', 'POST', 'PUT', 'DELETE'],
    allowedHeaders: ['Content-Type', 'Authorization'],
    credentials: true,
    maxAge: 86400
}));
```

### 2.4 SOP とストレージの隔離

ブラウザのストレージ機構はオリジン単位で隔離されている。

| ストレージ種別 | 隔離単位 | 容量目安 | 備考 |
|---------------|---------|---------|------|
| Cookie | ドメイン + パス | 4KB/cookie, 約50個/ドメイン | SameSite属性で送信制御 |
| localStorage | オリジン | 5-10MB | 同期API、メインスレッドをブロック |
| sessionStorage | オリジン + タブ | 5-10MB | タブを閉じると消失 |
| IndexedDB | オリジン | 数百MB-数GB | 非同期API、大容量データ向け |
| Cache API | オリジン | ブラウザ依存 | Service Worker で使用 |
| Web SQL | オリジン | 5MB（初期） | 非推奨、新規使用禁止 |

---

## 3. CSP（Content Security Policy）

### 3.1 CSP の設計思想

CSP は、XSS（Cross-Site Scripting）攻撃の影響を緩和するために設計されたセキュリティレイヤーである。XSS 攻撃は入力バリデーションの不備によって発生するが、CSP は「たとえ XSS の脆弱性が存在しても、攻撃者が注入したスクリプトの実行を防ぐ」という二次的な防御線として機能する。

CSP の基本原則は、Webページがロードできるリソースの出所をホワイトリスト方式で制限することである。

### 3.2 CSP の設定方法

CSP は以下の2つの方法で設定できる。

```
方法 1: HTTP レスポンスヘッダー（推奨）

  Content-Security-Policy: default-src 'self'; script-src 'self' https://cdn.example.com

方法 2: HTML の <meta> タグ

  <meta http-equiv="Content-Security-Policy"
        content="default-src 'self'; script-src 'self' https://cdn.example.com">

  ※ meta タグでは frame-ancestors, report-uri, sandbox ディレクティブは使用不可
  ※ HTTPヘッダーのほうが早く処理されるため、HTTPヘッダー方式を推奨
```

### 3.3 CSP ディレクティブの完全リファレンス

```
主要なCSPディレクティブ一覧:

  ┌──────────────────┬──────────────────────────────────────────────┐
  │ ディレクティブ     │ 制御対象                                      │
  ├──────────────────┼──────────────────────────────────────────────┤
  │ default-src      │ 他のディレクティブのフォールバック              │
  │ script-src       │ JavaScript ファイルとインラインスクリプト       │
  │ script-src-elem  │ <script> 要素のみ（イベントハンドラ除外）      │
  │ script-src-attr  │ インラインイベントハンドラのみ（onclick等）     │
  │ style-src        │ CSS ファイルとインラインスタイル                │
  │ style-src-elem   │ <style> 要素と <link rel="stylesheet">        │
  │ style-src-attr   │ style 属性のみ                                │
  │ img-src          │ 画像（<img>, CSS background-image 等）        │
  │ connect-src      │ fetch, XHR, WebSocket, EventSource の接続先   │
  │ font-src         │ Web フォント                                  │
  │ frame-src        │ <iframe>, <frame> の読み込み元                 │
  │ child-src        │ Web Worker と iframe（frame-src優先）          │
  │ worker-src       │ Worker, SharedWorker, ServiceWorker            │
  │ media-src        │ <audio>, <video> メディア                     │
  │ object-src       │ <object>, <embed>, <applet>                   │
  │ manifest-src     │ Web App Manifest                              │
  │ base-uri         │ <base> 要素の href                            │
  │ form-action      │ <form> の action 属性（送信先）               │
  │ frame-ancestors  │ このページを埋め込める親フレーム               │
  │ navigate-to      │ ナビゲーション先の制限（実験的）               │
  │ report-uri       │ 違反レポートの送信先（非推奨）                 │
  │ report-to        │ 違反レポートの送信先グループ                   │
  │ require-trusted-types-for │ Trusted Types の強制適用            │
  │ trusted-types    │ 許可する Trusted Type ポリシー名               │
  │ upgrade-insecure-requests │ HTTP を HTTPS に自動アップグレード  │
  │ sandbox          │ iframe の sandbox と同等の制限を適用           │
  └──────────────────┴──────────────────────────────────────────────┘

  ソース値の指定:

  'self'             — 同一オリジンのみ許可
  'none'             — すべてブロック
  'unsafe-inline'    — インラインスクリプト/スタイル許可（非推奨）
  'unsafe-eval'      — eval(), Function(), setTimeout(string) 許可（非推奨）
  'unsafe-hashes'    — 特定のインラインイベントハンドラを許可
  'nonce-{base64}'   — 指定 nonce を持つ要素のみ許可
  'sha256-{hash}'    — 指定ハッシュに一致するインラインコードのみ許可
  'strict-dynamic'   — 信頼されたスクリプトが動的にロードするスクリプトも許可
  https:             — HTTPS スキームのリソースのみ許可
  data:              — data: URI を許可
  blob:              — blob: URI を許可
  mediastream:       — mediastream: URI を許可
  *.example.com      — ワイルドカードによるホスト指定
```

### 3.4 CSP のレベル別実践設定

#### レベル1: 基本的な XSS 防御

```
Content-Security-Policy:
    default-src 'self';
    script-src 'self';
    style-src 'self' 'unsafe-inline';
    img-src 'self' data: https:;
    font-src 'self';
    object-src 'none';
    base-uri 'self';
    frame-ancestors 'self';
    form-action 'self';
```

#### レベル2: nonce ベースの厳格な設定（推奨）

```html
<!-- サーバーサイドでリクエストごとにランダムな nonce を生成 -->
<!-- HTTP ヘッダー -->
<!-- Content-Security-Policy:
    default-src 'self';
    script-src 'nonce-dGhpcyBpcyBhIHNhbXBsZQ==' 'strict-dynamic';
    style-src 'nonce-dGhpcyBpcyBhIHNhbXBsZQ==';
    img-src 'self' data: https:;
    connect-src 'self' https://api.example.com;
    font-src 'self';
    object-src 'none';
    base-uri 'self';
    frame-ancestors 'none';
    form-action 'self';
    upgrade-insecure-requests;
-->

<!DOCTYPE html>
<html>
<head>
    <!-- nonce が一致するスクリプトのみ実行される -->
    <script nonce="dGhpcyBpcyBhIHNhbXBsZQ==">
        // このスクリプトは実行される
        console.log('Trusted script executed');
    </script>

    <!-- nonce のないスクリプトはブロックされる -->
    <script>
        // このスクリプトはブロックされる
        console.log('This will not execute');
    </script>

    <!-- 攻撃者が注入したスクリプトもブロックされる -->
    <!-- <script>alert('XSS')</script> → ブロック -->

    <style nonce="dGhpcyBpcyBhIHNhbXBsZQ==">
        body { font-family: sans-serif; }
    </style>
</head>
<body>
    <h1>CSP Nonce Example</h1>
    <!-- strict-dynamic により、信頼されたスクリプトが動的に
         ロードするスクリプトも自動的に許可される -->
</body>
</html>
```

#### レベル3: ハッシュベースの設定

```javascript
// Node.js でインラインスクリプトのハッシュを計算する例
const crypto = require('crypto');

const inlineScript = `console.log('Hello, World!');`;
const hash = crypto.createHash('sha256')
    .update(inlineScript)
    .digest('base64');

console.log(`'sha256-${hash}'`);
// 出力例: 'sha256-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'

// このハッシュを CSP ヘッダーに含める
// Content-Security-Policy: script-src 'sha256-xxxxxxxx...'
```

### 3.5 CSP 違反レポートの活用

CSP にはレポート機能があり、ポリシー違反が発生した際にサーバーへ自動的に通知を送信できる。これにより、本番環境でのセキュリティ問題を早期に発見できる。

```javascript
// Report-Only モードで段階的に導入する例

// ステップ1: Report-Only で影響を調査（ブロックはしない）
// Content-Security-Policy-Report-Only:
//     default-src 'self';
//     script-src 'self' 'nonce-abc123';
//     report-uri /csp-report;
//     report-to csp-endpoint;

// ステップ2: レポートエンドポイントの実装（Express）
const express = require('express');
const app = express();

app.post('/csp-report', express.json({ type: 'application/csp-report' }),
    (req, res) => {
    const report = req.body['csp-report'];

    console.log('CSP Violation:', {
        blockedUri:       report['blocked-uri'],
        violatedDirective: report['violated-directive'],
        documentUri:      report['document-uri'],
        sourceFile:       report['source-file'],
        lineNumber:       report['line-number'],
        columnNumber:     report['column-number'],
        originalPolicy:   report['original-policy']
    });

    // 本番環境ではログ集約サービスに送信
    // await logService.send('csp-violation', report);

    res.status(204).end();
});

// Reporting API v1 (新しい標準) のヘッダー設定例
// Report-To: {"group":"csp-endpoint",
//             "max_age":86400,
//             "endpoints":[{"url":"https://reports.example.com/csp"}]}
// Content-Security-Policy: ... report-to csp-endpoint;
```

### 3.6 CSP と主要フレームワークの統合

| フレームワーク | CSP との相性 | 主な課題 | 推奨対策 |
|--------------|-------------|---------|---------|
| React | 良好 | dangerouslySetInnerHTML 使用時 | Trusted Types と併用 |
| Next.js | 良好 | nonce の SSR 対応が必要 | next.config.js で CSP 設定 |
| Vue.js | 注意が必要 | テンプレートコンパイルに eval が必要な場合あり | ランタイムのみビルドを使用 |
| Angular | 注意が必要 | AOT コンパイルなしでは unsafe-eval が必要 | 必ず AOT コンパイルを使用 |
| Svelte | 良好 | コンパイル済みのため eval 不要 | 標準の nonce ベース CSP で対応可 |
| jQuery | 要注意 | .html() や .append() による DOM 操作 | jQuery を段階的に置換 |

---

## 4. サイトアイソレーション（Site Isolation）

### 4.1 サイトアイソレーションの背景

サイトアイソレーションは、Chromium が 2018年（Chrome 67）から本格導入したセキュリティアーキテクチャである。従来のブラウザでは、複数のサイトのコンテンツが同一のレンダラープロセス内で実行される場合があったが、サイトアイソレーションでは異なるサイトのコンテンツを必ず別のプロセスで実行する。

この仕組みが必要になった直接的な契機は、2018年1月に公表された **Spectre** 脆弱性である。Spectre により、同一プロセス内の任意のメモリ領域を読み取ることが理論的に可能となったため、異なるサイトのデータが同一プロセス内に共存する状況が深刻なセキュリティリスクとなった。

### 4.2 サイト（Site）とオリジン（Origin）の違い

```
サイトアイソレーションにおける「サイト」の定義:

  サイト = スキーム + eTLD+1 (effective Top-Level Domain + 1)

  eTLD+1 の例:
    URL: https://mail.google.com/inbox
    eTLD: com
    eTLD+1: google.com
    サイト: https://google.com

  ┌──────────────────────────────────┬──────────────────┬────────────┐
  │ URL                              │ サイト             │ オリジン    │
  ├──────────────────────────────────┼──────────────────┼────────────┤
  │ https://www.example.com/page     │ https://example.com│ https://www│
  │                                  │                    │ .example.com│
  │ https://app.example.com/dash     │ https://example.com│ https://app│
  │                                  │                    │ .example.com│
  │ https://www.example.co.uk/page   │ https://example    │ https://www│
  │                                  │ .co.uk             │ .example   │
  │                                  │                    │ .co.uk     │
  │ https://user.github.io/repo      │ https://user       │ https://   │
  │                                  │ .github.io         │ user.github│
  │                                  │                    │ .io        │
  └──────────────────────────────────┴──────────────────┴────────────┘

  重要な違い:
  ・Same-Origin Policy → オリジン単位（スキーム+ホスト+ポート）
  ・Site Isolation     → サイト単位（スキーム+eTLD+1）
  ・SameSite Cookie    → サイト単位（スキーム+eTLD+1）

  ※ github.io のような Public Suffix では、
    user1.github.io と user2.github.io は異なるサイトとして扱われる
```

### 4.3 サイトアイソレーションのアーキテクチャ

```
サイトアイソレーション有効時のプロセス配置:

  タブ1: https://app.example.com/dashboard
  ┌──────────────────────────────────────────────────┐
  │  レンダラープロセス A (サイト: example.com)         │
  │  ┌─────────────────────────────────────────┐      │
  │  │ メインフレーム: app.example.com          │      │
  │  └─────────────────────────────────────────┘      │
  └──────────────────────────────────────────────────┘

  タブ1内の iframe: https://ads.partner.com/banner
  ┌──────────────────────────────────────────────────┐
  │  レンダラープロセス B (サイト: partner.com)         │
  │  ┌─────────────────────────────────────────┐      │
  │  │ サブフレーム: ads.partner.com            │      │
  │  └─────────────────────────────────────────┘      │
  └──────────────────────────────────────────────────┘

  タブ2: https://mail.example.com/inbox
  ┌──────────────────────────────────────────────────┐
  │  レンダラープロセス A (サイト: example.com) ← 再利用│
  │  ┌─────────────────────────────────────────┐      │
  │  │ メインフレーム: mail.example.com         │      │
  │  └─────────────────────────────────────────┘      │
  └──────────────────────────────────────────────────┘

  タブ3: https://social.other-site.com
  ┌──────────────────────────────────────────────────┐
  │  レンダラープロセス C (サイト: other-site.com)      │
  │  ┌─────────────────────────────────────────┐      │
  │  │ メインフレーム: social.other-site.com    │      │
  │  └─────────────────────────────────────────┘      │
  └──────────────────────────────────────────────────┘

  ※ 同一サイトのフレームは同一プロセスで実行されるが、
    異なるサイトのフレームは必ず異なるプロセスで実行される
```

### 4.4 Spectre 脆弱性とブラウザの対策

Spectre は CPU の投機的実行（Speculative Execution）を悪用するサイドチャネル攻撃であり、攻撃者は同一プロセス内のメモリを高精度のタイマーを用いて間接的に読み取ることができる。

ブラウザにおける Spectre 対策は多層的に行われている。

| 対策 | 説明 | 導入時期 |
|------|------|---------|
| サイトアイソレーション | 異なるサイトを別プロセスで実行 | Chrome 67 (2018) |
| performance.now() の精度低下 | タイマーの解像度を下げてタイミング攻撃を困難化 | 2018年1月 |
| SharedArrayBuffer の無効化 | 高精度タイマーの構築手段を除去 | 2018年1月 |
| Cross-Origin Isolation | SharedArrayBuffer を安全に再有効化 | Chrome 91 (2021) |
| CORB (Cross-Origin Read Blocking) | クロスオリジンレスポンスのプロセス内読み込みを防止 | Chrome 67 (2018) |
| ORB (Opaque Response Blocking) | CORB の後継、より広範なリソースを保護 | 段階的導入中 |

### 4.5 Cross-Origin Isolation

Cross-Origin Isolation は、SharedArrayBuffer や高精度タイマーなどの機能を安全に使用するための仕組みである。以下の HTTP ヘッダーを設定することで有効化される。

```
# Cross-Origin Isolation を有効にするためのヘッダー

# 1. Cross-Origin-Opener-Policy (COOP)
# → 同一オリジン以外のウィンドウとの browsing context group を分離
Cross-Origin-Opener-Policy: same-origin

# 2. Cross-Origin-Embedder-Policy (COEP)
# → ページに埋め込まれるすべてのリソースが CORS または CORP で
#    明示的に許可されていることを要求
Cross-Origin-Embedder-Policy: require-corp

# これらを両方設定すると:
# ・self.crossOriginIsolated === true になる
# ・SharedArrayBuffer が使用可能になる
# ・performance.now() の精度が回復する（5マイクロ秒）
# ・performance.measureUserAgentSpecificMemory() が使用可能になる
```

```javascript
// Cross-Origin Isolation の状態確認
if (self.crossOriginIsolated) {
    console.log('Cross-Origin Isolated: SharedArrayBuffer 使用可能');

    // 高精度タイマーの使用
    const start = performance.now();
    // ... 処理 ...
    const elapsed = performance.now() - start;
    console.log(`Elapsed: ${elapsed} ms (高精度)`);

    // SharedArrayBuffer の使用（Web Worker との共有メモリ）
    const sharedBuffer = new SharedArrayBuffer(1024);
    const view = new Int32Array(sharedBuffer);

    const worker = new Worker('worker.js');
    worker.postMessage({ buffer: sharedBuffer });
} else {
    console.warn('Cross-Origin Isolated ではありません');
    console.warn('COOP と COEP ヘッダーを確認してください');
}
```

### 4.6 CORB と ORB

Cross-Origin Read Blocking（CORB）は、レンダラープロセスに到達する前にクロスオリジンのセンシティブなリソースをブロックする仕組みである。

```
CORB の動作フロー:

  悪意あるページ: https://evil.com
    │
    │  <img src="https://bank.com/api/account"> を試行
    │  （画像タグを使ってAPI応答を読み取ろうとする攻撃）
    │
    ▼
  ネットワークプロセス
    │  レスポンスの Content-Type を確認
    │  Content-Type: application/json → HTML/XML/JSON と判定
    │
    │  <img> タグのリクエストに JSON レスポンスは不適切
    │  → CORB によりレスポンスボディを空に置換
    │
    ▼
  レンダラープロセス（evil.com）
    │  空のレスポンスボディを受信
    │  → 機密データはプロセスのメモリ空間に到達しない
    │  → Spectre 攻撃でも読み取れない
    │
    結果: 画像の読み込み失敗（これは正常な動作）
```

---

## 5. Cookie セキュリティ

### 5.1 Cookie のセキュリティ属性の完全ガイド

Cookie はHTTPのステートレスな性質を補完するための仕組みだが、適切なセキュリティ属性を設定しないと攻撃の対象となりやすい。

```
Set-Cookie ヘッダーのセキュリティ属性:

  Set-Cookie: session_id=a1b2c3d4e5f6;
    Secure;                    ← HTTPS 接続でのみ送信
    HttpOnly;                  ← JavaScript (document.cookie) からアクセス不可
    SameSite=Lax;              ← クロスサイトリクエストでの送信制限
    Path=/;                    ← Cookie の有効パス
    Domain=.example.com;       ← Cookie の有効ドメイン
    Max-Age=86400;             ← 有効期限（秒単位、86400秒 = 24時間）
    Partitioned;               ← CHIPS: トップレベルサイトごとに分離

  各属性の重要度と推奨設定:

  ┌──────────────┬───────────┬──────────────────────────────────────┐
  │ 属性          │ 推奨設定   │ 未設定時のリスク                      │
  ├──────────────┼───────────┼──────────────────────────────────────┤
  │ Secure       │ 常に設定   │ HTTP通信でCookieが平文送信される       │
  │ HttpOnly     │ 常に設定   │ XSS でCookieが窃取される              │
  │ SameSite     │ Lax以上   │ CSRF 攻撃のリスク                     │
  │ Path         │ 最小範囲   │ 不要なパスにCookieが送信される         │
  │ Domain       │ 必要最小限 │ サブドメインでCookieが共有される       │
  │ Max-Age      │ 用途に応じ │ セッションCookieとして扱われる         │
  │ __Host- 接頭辞│ 推奨      │ Domain属性の上書き攻撃の可能性         │
  │ Partitioned  │ 3P用途で推奨│ トラッキングに悪用される可能性          │
  └──────────────┴───────────┴──────────────────────────────────────┘
```

### 5.2 SameSite 属性の詳細

```
SameSite の値と動作の比較:

  ┌───────────┬──────────────────────────────────────────────────────┐
  │ 値         │ 動作                                                 │
  ├───────────┼──────────────────────────────────────────────────────┤
  │ Strict    │ クロスサイトリクエストでは一切送信しない               │
  │           │ → 外部サイトからのリンク遷移でもCookieなし             │
  │           │ → 最も安全だが UX に影響する場合あり                   │
  │           │                                                      │
  │           │ 使用例: 銀行サイトの認証Cookie、管理画面               │
  ├───────────┼──────────────────────────────────────────────────────┤
  │ Lax       │ トップレベルナビゲーション（GET リンク遷移）では送信   │
  │ (デフォルト)│ POST、iframe、AJAX、画像ロードでは送信しない          │
  │           │ → CSRF の主要な攻撃ベクトルを防御しつつ               │
  │           │   外部リンクからの遷移には対応                        │
  │           │                                                      │
  │           │ 使用例: 一般的なWebアプリのセッションCookie            │
  ├───────────┼──────────────────────────────────────────────────────┤
  │ None      │ 常に送信される（Secure 属性の同時指定が必須）          │
  │           │ → サードパーティCookieとして動作                      │
  │           │ → ブラウザの制限強化により段階的に廃止傾向             │
  │           │                                                      │
  │           │ 使用例: 認証連携、埋め込みウィジェット                 │
  └───────────┴──────────────────────────────────────────────────────┘

  SameSite による送信制御の具体的シナリオ:

  ユーザーが https://blog.com を閲覧中に、
  https://shop.example.com へのリクエストが発生する場合:

  シナリオ                          Strict    Lax    None
  ─────────────────────────────────────────────────────
  <a href="shop.example.com">       送信しない  送信   送信
  <form method="GET" action="...">  送信しない  送信   送信
  <form method="POST" action="..."> 送信しない  送信しない 送信
  <img src="shop.example.com/...">  送信しない  送信しない 送信
  fetch("shop.example.com/...")     送信しない  送信しない 送信
  <iframe src="shop.example.com">   送信しない  送信しない 送信
```

### 5.3 Cookie プレフィックスによる保護

```javascript
// __Host- プレフィックス: 最も厳格な Cookie
// 要件: Secure 必須、Domain 属性なし、Path=/ 必須
// → Cookie のスコープが確実に現在のホストに限定される

// サーバー側の設定例（Express）
app.use((req, res, next) => {
    // セッション Cookie には __Host- プレフィックスを推奨
    res.cookie('__Host-session', sessionId, {
        secure: true,
        httpOnly: true,
        sameSite: 'lax',
        path: '/',
        maxAge: 24 * 60 * 60 * 1000  // 24時間
        // domain を指定してはいけない（__Host- の要件）
    });

    // __Secure- プレフィックス: Secure 属性のみ必須
    // Domain 属性の指定は許可される
    res.cookie('__Secure-preferences', prefsToken, {
        secure: true,
        httpOnly: true,
        sameSite: 'lax',
        domain: '.example.com',
        path: '/',
        maxAge: 30 * 24 * 60 * 60 * 1000  // 30日
    });

    next();
});
```

### 5.4 サードパーティ Cookie の廃止と代替技術

```
サードパーティ Cookie の廃止状況:

  ┌──────────┬──────────────────────────────────────────────────────┐
  │ ブラウザ  │ 状況                                                 │
  ├──────────┼──────────────────────────────────────────────────────┤
  │ Safari   │ ITP (Intelligent Tracking Prevention) で既にブロック  │
  │          │ 2020年以降、サードパーティ Cookie は完全ブロック       │
  ├──────────┼──────────────────────────────────────────────────────┤
  │ Firefox  │ ETP (Enhanced Tracking Protection) でブロック         │
  │          │ Total Cookie Protection でストレージも分離            │
  ├──────────┼──────────────────────────────────────────────────────┤
  │ Chrome   │ Privacy Sandbox に段階的移行                         │
  │          │ CHIPS (Cookies Having Independent Partitioned State) │
  │          │ Topics API、Attribution Reporting API 等で代替        │
  └──────────┴──────────────────────────────────────────────────────┘

  代替技術の比較:

  ┌─────────────────────┬────────────────────┬─────────────────────┐
  │ 用途                 │ 従来の手法          │ 代替技術             │
  ├─────────────────────┼────────────────────┼─────────────────────┤
  │ 広告ターゲティング   │ 3rd party Cookie   │ Topics API           │
  │ コンバージョン計測   │ 3rd party Cookie   │ Attribution Reporting│
  │ 認証連携 (SSO)      │ 3rd party Cookie   │ FedCM API            │
  │ 埋め込みウィジェット │ 3rd party Cookie   │ CHIPS (Partitioned)  │
  │ 不正検知            │ 3rd party Cookie   │ Private State Tokens │
  └─────────────────────┴────────────────────┴─────────────────────┘
```

---

## 6. その他のセキュリティ機構

### 6.1 Subresource Integrity（SRI）

SRI は、CDN 等の外部から読み込むリソースが改ざんされていないことを暗号学的に検証する仕組みである。

```html
<!-- SRI の使用例 -->
<script
    src="https://cdn.example.com/lib/react.production.min.js"
    integrity="sha384-oqVuAfXRKap7fdgcCY5uykM6+R9GqQ8K/uxy9rx7HNQlGYl1kPzQho1wx4JwY8w"
    crossorigin="anonymous">
</script>

<link
    rel="stylesheet"
    href="https://cdn.example.com/css/bootstrap.min.css"
    integrity="sha384-xOolHFLEh07PJGoPkLv1IbcEPTNtaed2xpHsD9ESMhqIYd0nLMwNLD69Npy4HI+"
    crossorigin="anonymous">

<!--
  SRI の動作:
  1. ブラウザがリソースをダウンロード
  2. ダウンロードしたリソースの SHA ハッシュを計算
  3. integrity 属性のハッシュ値と比較
  4. 一致 → リソースを使用
     不一致 → リソースをブロック（ネットワークエラー扱い）

  crossorigin="anonymous" が必要な理由:
  → SRI はレスポンスボディのハッシュを検証するため、
    CORS によりレスポンスにアクセスできる必要がある
-->
```

```bash
# SRI ハッシュの生成方法
# コマンドラインで sha384 ハッシュを生成
cat react.production.min.js | openssl dgst -sha384 -binary | openssl base64 -A

# 複数のハッシュアルゴリズムを指定可能（フォールバック）
# integrity="sha256-xxx sha384-yyy sha512-zzz"
# → ブラウザは最も強いアルゴリズムを選択
```

### 6.2 Trusted Types

Trusted Types は、DOM XSS を根本的に防止するための API である。innerHTML 等の危険な DOM API に対して、文字列の直接代入を禁止し、サニタイズ済みの「信頼された型」のみを受け入れるようにする。

```javascript
// Trusted Types の設定と使用例

// CSP ヘッダーで Trusted Types を強制
// Content-Security-Policy: require-trusted-types-for 'script';
//                          trusted-types myPolicy default;

// ポリシーの作成
const sanitizePolicy = trustedTypes.createPolicy('myPolicy', {
    createHTML: (input) => {
        // DOMPurify 等でサニタイズ
        return DOMPurify.sanitize(input, {
            ALLOWED_TAGS: ['b', 'i', 'em', 'strong', 'a', 'p', 'br'],
            ALLOWED_ATTR: ['href', 'title']
        });
    },
    createScriptURL: (input) => {
        // 許可された URL のみ返す
        const url = new URL(input, document.baseURI);
        if (url.origin === location.origin) {
            return url.href;
        }
        throw new TypeError(`Untrusted script URL: ${input}`);
    },
    createScript: (input) => {
        // スクリプトの直接生成は原則禁止
        throw new TypeError('Script creation is not allowed');
    }
});

// 使用例: innerHTML への代入
const userContent = '<p>Hello <b>World</b></p><script>alert("XSS")</script>';
const trustedHTML = sanitizePolicy.createHTML(userContent);
// → <p>Hello <b>World</b></p> （scriptタグは除去される）

document.getElementById('content').innerHTML = trustedHTML;  // OK

// Trusted Types なしでの直接代入はブロックされる
// document.getElementById('content').innerHTML = userContent;
// → TypeError: Failed to set 'innerHTML': This document requires
//   'TrustedHTML' assignment.

// デフォルトポリシー（フォールバック用）
trustedTypes.createPolicy('default', {
    createHTML: (input) => {
        console.warn('Uncontrolled innerHTML usage detected:', input);
        return DOMPurify.sanitize(input);
    }
});
```

### 6.3 Referrer Policy

```
Referrer-Policy の値と動作:

  ┌─────────────────────────────────┬──────────────────────────────────┐
  │ ポリシー                         │ 送信されるリファラー              │
  ├─────────────────────────────────┼──────────────────────────────────┤
  │ no-referrer                     │ リファラーを一切送信しない        │
  │ no-referrer-when-downgrade      │ HTTPS→HTTP では送信しない         │
  │ origin                          │ オリジンのみ送信                  │
  │                                 │ (https://example.com/)            │
  │ origin-when-cross-origin        │ 同一オリジン: フルURL             │
  │                                 │ クロスオリジン: オリジンのみ      │
  │ same-origin                     │ 同一オリジン: フルURL             │
  │                                 │ クロスオリジン: 送信しない        │
  │ strict-origin                   │ HTTPS→HTTPS: オリジンのみ        │
  │                                 │ HTTPS→HTTP: 送信しない           │
  │ strict-origin-when-cross-origin │ 同一オリジン: フルURL             │
  │ (デフォルト)                     │ クロスオリジン: オリジンのみ      │
  │                                 │ HTTPS→HTTP: 送信しない           │
  │ unsafe-url                      │ 常にフルURLを送信（非推奨）       │
  └─────────────────────────────────┴──────────────────────────────────┘

  推奨設定: strict-origin-when-cross-origin（多くのブラウザのデフォルト）
  機密情報を含むURL（トークン等）がある場合: no-referrer
```

### 6.4 セキュリティヘッダーの総合設定例

```nginx
# Nginx での推奨セキュリティヘッダー設定

server {
    listen 443 ssl http2;
    server_name example.com;

    # --- TLS 設定 ---
    ssl_certificate     /etc/ssl/certs/example.com.pem;
    ssl_certificate_key /etc/ssl/private/example.com.key;
    ssl_protocols       TLSv1.2 TLSv1.3;
    ssl_ciphers         HIGH:!aNULL:!MD5;

    # --- セキュリティヘッダー ---

    # HSTS: HTTPS の強制（max-age=2年、サブドメイン含む）
    add_header Strict-Transport-Security
        "max-age=63072000; includeSubDomains; preload" always;

    # CSP: リソース読み込みの制限
    # ※ nonce はアプリケーション側で動的に生成
    add_header Content-Security-Policy
        "default-src 'self'; script-src 'self' 'strict-dynamic'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; connect-src 'self' https://api.example.com; font-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'; form-action 'self'; upgrade-insecure-requests;"
        always;

    # X-Content-Type-Options: MIME スニッフィング防止
    add_header X-Content-Type-Options "nosniff" always;

    # X-Frame-Options: クリックジャッキング防止
    # （CSP frame-ancestors と併用推奨）
    add_header X-Frame-Options "DENY" always;

    # Referrer Policy
    add_header Referrer-Policy
        "strict-origin-when-cross-origin" always;

    # Permissions Policy: 不要な機能の無効化
    add_header Permissions-Policy
        "camera=(), microphone=(), geolocation=(self), payment=(self)"
        always;

    # Cross-Origin Isolation（必要な場合）
    # add_header Cross-Origin-Opener-Policy "same-origin" always;
    # add_header Cross-Origin-Embedder-Policy "require-corp" always;

    # Cross-Origin Resource Policy
    add_header Cross-Origin-Resource-Policy "same-origin" always;
}
```

---

## 7. アンチパターン

### 7.1 アンチパターン 1: CSP に `unsafe-inline` と `unsafe-eval` を安易に使用する

**問題のあるコード:**

```
Content-Security-Policy:
    default-src 'self';
    script-src 'self' 'unsafe-inline' 'unsafe-eval';
    style-src 'self' 'unsafe-inline';
```

**なぜ問題か:**

`unsafe-inline` を `script-src` に指定すると、CSP による XSS 防御がほぼ無効化される。攻撃者が HTML インジェクションに成功した場合、`<script>alert(document.cookie)</script>` のようなインラインスクリプトがそのまま実行されてしまう。同様に `unsafe-eval` を指定すると、`eval()` や `Function()` コンストラクタ、`setTimeout('string')` といった文字列からコードを生成する API が許可され、攻撃面が大幅に広がる。

CSP を導入する主目的は XSS の影響緩和であり、`unsafe-inline` と `unsafe-eval` の使用はその目的を損なう。特に `script-src` にこれらを指定することは、鍵のかからないドアに防犯カメラだけ設置するようなもので、根本的な防御にならない。

**正しいアプローチ:**

```
Content-Security-Policy:
    default-src 'self';
    script-src 'self' 'nonce-{server-generated-random}' 'strict-dynamic';
    style-src 'self' 'nonce-{server-generated-random}';
```

nonce ベースの CSP を使用し、サーバーサイドでリクエストごとにランダムな nonce を生成して、正規のスクリプト要素にのみ付与する。`strict-dynamic` を併用することで、信頼されたスクリプトが動的にロードするスクリプトも自動的に許可される。

### 7.2 アンチパターン 2: CORS で `Access-Control-Allow-Origin: *` と `credentials: true` を併用しようとする

**問題のあるコード:**

```javascript
// サーバー側
app.use((req, res, next) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Credentials', 'true');
    next();
});

// クライアント側
fetch('https://api.example.com/user/profile', {
    credentials: 'include'  // Cookie を含めたクロスオリジンリクエスト
});
```

**なぜ問題か:**

ブラウザは仕様上、`Access-Control-Allow-Origin: *` と `Access-Control-Allow-Credentials: true` の組み合わせを拒否する。`credentials: true` を使用する場合、`Access-Control-Allow-Origin` には具体的なオリジンを指定しなければならない。

しかし、この制約を回避しようとして「リクエストの Origin ヘッダーをそのまま `Access-Control-Allow-Origin` にエコーバックする」というパターンが散見される。これは事実上すべてのオリジンを許可するのと同じであり、CSRF 攻撃に対して脆弱になる。

**正しいアプローチ:**

```javascript
// サーバー側: 許可するオリジンをホワイトリストで管理
const allowedOrigins = new Set([
    'https://app.example.com',
    'https://staging.example.com',
    'https://admin.example.com'
]);

app.use((req, res, next) => {
    const origin = req.headers.origin;

    if (allowedOrigins.has(origin)) {
        res.setHeader('Access-Control-Allow-Origin', origin);
        res.setHeader('Access-Control-Allow-Credentials', 'true');
        res.setHeader('Vary', 'Origin');  // キャッシュの正確性のため必須
    }

    next();
});
```

### 7.3 アンチパターン 3: postMessage で origin を検証しない

**問題のあるコード:**

```javascript
// 受信側: origin の検証なし
window.addEventListener('message', (event) => {
    // 危険: 任意のオリジンからのメッセージを処理してしまう
    const data = event.data;
    document.getElementById('output').innerHTML = data.html;
});
```

**なぜ問題か:**

`postMessage` はクロスオリジン通信のための安全な API だが、受信側で `event.origin` を検証しないと、攻撃者のページから任意のメッセージを送信できてしまう。上記の例では、さらに受信したデータを `innerHTML` に直接代入しているため、DOM XSS の脆弱性も生じている。

**正しいアプローチ:**

```javascript
// 受信側: origin の検証あり
window.addEventListener('message', (event) => {
    // 送信元オリジンの検証は必須
    if (event.origin !== 'https://trusted-partner.com') {
        console.warn('Message from untrusted origin rejected:', event.origin);
        return;
    }

    // データの型と構造も検証
    if (typeof event.data !== 'object' || event.data.type !== 'update') {
        return;
    }

    // innerHTML ではなく textContent を使用（XSS 防止）
    document.getElementById('output').textContent = event.data.text;
});
```

---

## 8. エッジケース分析

### 8.1 エッジケース 1: `blob:` URL と `data:` URL のオリジン

`blob:` URL と `data:` URL は通常の HTTP URL とは異なるオリジン判定ルールを持つ。

```javascript
// blob: URL のオリジン
// → 作成元ドキュメントのオリジンを継承する

const htmlContent = '<html><body><script>alert(document.domain)</script></body></html>';
const blob = new Blob([htmlContent], { type: 'text/html' });
const blobUrl = URL.createObjectURL(blob);
// blobUrl = "blob:https://example.com/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
// → このblob URLのオリジンは https://example.com

// data: URL のオリジン
// → Opaque Origin（不透明なオリジン）として扱われる
// → どのオリジンとも同一オリジンにならない

const dataUrl = 'data:text/html,<script>alert(document.domain)</script>';
// → data: URL で開いたページの document.domain は "" (空文字列)
// → 同一オリジンポリシーの観点では、他のどのオリジンともマッチしない

// セキュリティ上の注意点:
// 1. CSP で data: を許可すると、data: URL からのリソース読み込みが可能になる
//    → script-src に data: を指定するのは危険
//       攻撃者が data:text/javascript,alert(1) を注入できる

// 2. blob: URL はオリジンを継承するため、
//    CSP の script-src 'self' で blob: からのスクリプト実行が許可される
//    ブラウザによっては追加の制限あり

// 3. iframe で data: URL を使用する場合
const iframe = document.createElement('iframe');
iframe.src = 'data:text/html,<h1>Hello</h1>';
// → iframe 内は Opaque Origin
// → 親ページからの DOM アクセスは SecurityError になる
```

### 8.2 エッジケース 2: Service Worker のスコープとセキュリティ境界

Service Worker は強力な機能を持つが、そのスコープとセキュリティ境界には注意が必要である。

```javascript
// Service Worker のスコープ制限

// Service Worker のスクリプトURLがそのスコープの上限を決定する
// /sw.js でSWを登録 → スコープは / 以下全体
// /app/sw.js でSWを登録 → スコープは /app/ 以下

// ケース1: スコープの上限を超えようとする（エラー）
navigator.serviceWorker.register('/app/sw.js', {
    scope: '/'  // エラー: /app/sw.js のスコープは /app/ まで
});

// ケース2: Service-Worker-Allowed ヘッダーで上限を拡張
// サーバーが SW スクリプトのレスポンスに以下を付与:
// Service-Worker-Allowed: /
// → これにより /app/sw.js のスコープを / まで拡張可能

// セキュリティ上の注意点:

// 1. Service Worker は HTTPS（または localhost）でのみ登録可能
// 2. Service Worker はオリジン単位で隔離される
// 3. importScripts() で読み込むスクリプトも同一オリジンが必要
//    （CORS が設定された外部スクリプトは可）

// 4. Service Worker がキャッシュしたレスポンスと CSP の関係
self.addEventListener('fetch', (event) => {
    event.respondWith(
        caches.match(event.request).then((cachedResponse) => {
            if (cachedResponse) {
                // キャッシュからのレスポンスにも CSP は適用される
                // ただし、CSP ヘッダーはキャッシュされたレスポンスの
                // ヘッダーが使用される（元のサーバー応答時のもの）
                return cachedResponse;
            }
            return fetch(event.request);
        })
    );
});

// 5. Navigation Preload と Service Worker
// → Navigation Preload を使用すると、SW の起動と
//    ネットワークリクエストが並行して実行される
// → レスポンスのセキュリティヘッダーは
//    ネットワークレスポンスのものが使用される
```

### 8.3 エッジケース 3: WebSocket と Same-Origin Policy

```javascript
// WebSocket は Same-Origin Policy の制約を受けない
// → 任意のオリジンへの WebSocket 接続が可能

// これは仕様上の設計判断であり、以下の理由による:
// 1. WebSocket のハンドシェイクは HTTP で行われ、
//    サーバー側で Origin ヘッダーを検証できる
// 2. WebSocket はブラウザの自動的な Cookie 送信をサポートするため、
//    サーバー側での認証チェックが可能

// サーバー側での Origin 検証（必須）
const WebSocket = require('ws');
const wss = new WebSocket.Server({ port: 8080 });

wss.on('connection', (ws, req) => {
    const origin = req.headers.origin;
    const allowedOrigins = ['https://app.example.com'];

    if (!allowedOrigins.includes(origin)) {
        ws.close(1008, 'Origin not allowed');
        return;
    }

    // 接続を受け入れる
    ws.on('message', (message) => {
        // メッセージ処理
    });
});

// CSP の connect-src は WebSocket にも適用される
// Content-Security-Policy: connect-src 'self' wss://ws.example.com
```

---

## 9. 演習

### 9.1 演習 1: 基礎 — CSP ヘッダーの設計

以下の要件を満たす CSP ヘッダーを設計せよ。

**要件:**
- 自社ドメイン `https://app.example.com` からのみスクリプトを読み込む
- CDN `https://cdn.jsdelivr.net` からスタイルシートとフォントを読み込む
- API サーバー `https://api.example.com` への fetch リクエストを許可する
- 画像は自社ドメインと HTTPS の任意のソースから読み込む
- iframe への埋め込みは一切禁止する
- インラインスクリプトは nonce ベースで制御する
- フォームの送信先は自社ドメインのみ

**模範解答:**

```
Content-Security-Policy:
    default-src 'none';
    script-src 'self' 'nonce-{random}' 'strict-dynamic';
    style-src 'self' https://cdn.jsdelivr.net 'nonce-{random}';
    img-src 'self' https:;
    font-src 'self' https://cdn.jsdelivr.net;
    connect-src 'self' https://api.example.com;
    frame-src 'none';
    frame-ancestors 'none';
    form-action 'self';
    base-uri 'self';
    upgrade-insecure-requests;
```

**解説:**
- `default-src 'none'` で全リソースをデフォルトでブロックし、必要なものだけ個別に許可するホワイトリスト方式を採用
- `script-src` には `'nonce-{random}'` を指定し、サーバーサイドでリクエストごとにランダムな nonce を生成
- `'strict-dynamic'` により、nonce 付きスクリプトが動的にロードするスクリプトも許可
- `frame-ancestors 'none'` でクリックジャッキングを防止（X-Frame-Options: DENY と同等）
- `upgrade-insecure-requests` で HTTP リクエストを自動的に HTTPS に昇格

### 9.2 演習 2: 中級 — CORS の設定とデバッグ

以下のエラーメッセージが発生した場合の原因と対策を述べよ。

**シナリオ:**
`https://app.example.com` のフロントエンドから `https://api.example.com/users` に POST リクエストを送信したところ、以下のエラーが発生した。

```
Access to fetch at 'https://api.example.com/users' from origin
'https://app.example.com' has been blocked by CORS policy:
Response to preflight request doesn't pass access control check:
No 'Access-Control-Allow-Origin' header is present on the requested resource.
```

**模範解答:**

原因: `https://api.example.com` のサーバーが、プリフライトリクエスト（OPTIONS メソッド）に対して適切な CORS ヘッダーを返していない。POST リクエストで `Content-Type: application/json` や Authorization ヘッダーを使用している場合、単純リクエスト（Simple Request）の条件を満たさないため、ブラウザは本リクエストの前にプリフライトリクエストを送信する。

対策:

```javascript
// サーバー側（Express）の修正
app.options('/users', (req, res) => {
    // プリフライトリクエストへの応答
    res.setHeader('Access-Control-Allow-Origin', 'https://app.example.com');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE');
    res.setHeader('Access-Control-Allow-Headers',
        'Content-Type, Authorization');
    res.setHeader('Access-Control-Max-Age', '86400');
    res.status(204).end();
});

app.post('/users', (req, res) => {
    res.setHeader('Access-Control-Allow-Origin', 'https://app.example.com');
    // ... ビジネスロジック
    res.json({ success: true });
});
```

デバッグのポイント:
1. ブラウザの DevTools の Network タブで OPTIONS リクエストの有無を確認
2. OPTIONS レスポンスのステータスコードが 2xx であることを確認
3. レスポンスヘッダーに必要な `Access-Control-Allow-*` が含まれていることを確認
4. `Vary: Origin` ヘッダーが設定されていることを確認（CDN/プロキシのキャッシュ対策）

### 9.3 演習 3: 上級 — セキュリティヘッダーの総合監査

以下の HTTP レスポンスヘッダーを監査し、セキュリティ上の問題点をすべて指摘し、改善案を提示せよ。

```
HTTP/1.1 200 OK
Content-Type: text/html
Set-Cookie: session=abc123; Path=/
X-Powered-By: Express 4.18.2
Server: nginx/1.24.0
```

**模範解答:**

| # | 問題点 | リスク | 改善案 |
|---|--------|--------|--------|
| 1 | CSP ヘッダーがない | XSS 攻撃の影響が最大化される | `Content-Security-Policy` を追加 |
| 2 | HSTS ヘッダーがない | ダウングレード攻撃（HTTP接続）のリスク | `Strict-Transport-Security` を追加 |
| 3 | Cookie に Secure 属性がない | HTTP 通信でセッション Cookie が平文送信される | `Secure` を追加 |
| 4 | Cookie に HttpOnly 属性がない | XSS でセッション Cookie が窃取される | `HttpOnly` を追加 |
| 5 | Cookie に SameSite 属性がない | CSRF 攻撃のリスク（ブラウザデフォルトは Lax だが明示推奨） | `SameSite=Lax` を追加 |
| 6 | X-Powered-By ヘッダーが露出 | フレームワークのバージョン情報が攻撃者に漏洩 | `X-Powered-By` を削除 |
| 7 | Server ヘッダーにバージョン情報 | サーバーソフトウェアの脆弱性を特定される | バージョン番号を非表示に |
| 8 | X-Content-Type-Options がない | MIME スニッフィング攻撃のリスク | `X-Content-Type-Options: nosniff` を追加 |
| 9 | X-Frame-Options がない | クリックジャッキング攻撃のリスク | `X-Frame-Options: DENY` を追加 |
| 10 | Referrer-Policy がない | 機密パス情報がリファラーで漏洩する可能性 | `Referrer-Policy: strict-origin-when-cross-origin` を追加 |
| 11 | Permissions-Policy がない | 不要なブラウザ機能が悪用される可能性 | `Permissions-Policy` で不要な機能を無効化 |
| 12 | Cookie に __Host- プレフィックスがない | Cookie のスコープが広すぎる可能性 | `__Host-session` に変更 |

改善後のレスポンスヘッダー:

```
HTTP/1.1 200 OK
Content-Type: text/html; charset=utf-8
Set-Cookie: __Host-session=abc123; Secure; HttpOnly; SameSite=Lax; Path=/; Max-Age=86400
Strict-Transport-Security: max-age=63072000; includeSubDomains; preload
Content-Security-Policy: default-src 'self'; script-src 'self' 'nonce-xxx' 'strict-dynamic'; object-src 'none'; base-uri 'self'; frame-ancestors 'none';
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
Referrer-Policy: strict-origin-when-cross-origin
Permissions-Policy: camera=(), microphone=(), geolocation=(self)
```

---

## 10. FAQ

### Q1: CSP を導入したら既存のサイトが壊れてしまいます。段階的に導入するにはどうすればよいですか？

CSP の段階的な導入には `Content-Security-Policy-Report-Only` ヘッダーを活用する。このヘッダーを使用すると、ポリシー違反はレポートされるがリソースのブロックは行われない。

推奨される導入手順:

1. **調査フェーズ**: `Content-Security-Policy-Report-Only` を緩めのポリシーで設定し、`report-uri` でレポートを収集する。これにより、サイトが読み込んでいるすべてのリソースの出所を把握できる。

2. **分析フェーズ**: 収集したレポートを分析し、正規のリソースと不要なリソースを区別する。インラインスクリプトやインラインスタイルの使用箇所を特定し、nonce やハッシュへの移行計画を立てる。

3. **段階的適用**: まず影響の少ないディレクティブ（`object-src 'none'`, `base-uri 'self'`）から本番の CSP ヘッダーに移行し、徐々にスコープを広げていく。

4. **本番適用**: すべてのディレクティブを `Content-Security-Policy` ヘッダーに移行し、`Report-Only` は次のポリシー変更のテスト用に残しておく。

### Q2: Same-Origin Policy があるのに、なぜ CSRF 攻撃が成立するのですか？

Same-Origin Policy は「レスポンスの読み取り」を制限するが、「リクエストの送信」自体は制限しない。フォーム送信（`<form method="POST">`）やイメージタグ（`<img src="...">`）によるリクエストは、クロスオリジンであっても送信される。このとき、ブラウザはターゲットサイトの Cookie を自動的に付与する。

攻撃者のページ:
```html
<!-- 攻撃者が evil.com に設置したページ -->
<form id="csrf-form"
      action="https://bank.example.com/transfer"
      method="POST">
    <input type="hidden" name="to" value="attacker-account">
    <input type="hidden" name="amount" value="1000000">
</form>
<script>document.getElementById('csrf-form').submit();</script>
```

この場合:
- ブラウザは `bank.example.com` への POST リクエストを送信する
- ユーザーが `bank.example.com` にログイン中であれば、Cookie が自動的に付与される
- サーバーは正規のリクエストと区別できない

**対策:**
- `SameSite=Lax` または `SameSite=Strict` の Cookie 属性
- CSRF トークン（サーバーサイドで生成したランダムなトークンをフォームに埋め込む）
- Origin ヘッダーの検証
- カスタムヘッダーの要求（`X-Requested-With` 等。プリフライトが発生するため CSRF が困難になる）

### Q3: Content-Security-Policy の `strict-dynamic` はどのように動作しますか？

`strict-dynamic` は CSP Level 3 で導入されたソース式であり、nonce またはハッシュで信頼されたスクリプトが動的に生成・読み込みするスクリプトにも信頼を伝播させる仕組みである。

```javascript
// CSP ヘッダー:
// Content-Security-Policy: script-src 'nonce-abc123' 'strict-dynamic'

// 以下の nonce 付きスクリプトは実行される
// <script nonce="abc123">
//     // このスクリプト内で動的にロードするスクリプトも許可される
//     const script = document.createElement('script');
//     script.src = 'https://any-cdn.com/library.js';
//     document.head.appendChild(script);
//     // → 'strict-dynamic' により、このスクリプトは実行される
//     //   （ホワイトリストに any-cdn.com がなくても）
// </script>
```

`strict-dynamic` が有効な場合の動作:
- nonce/ハッシュで直接信頼されたスクリプトから `createElement('script')` で追加されたスクリプトは自動的に許可される
- `document.write()` で挿入されたスクリプトはブロックされる（パーサー挿入型は危険なため）
- `https:` や `http:` 等の URL ベースのソース式は無視される（`strict-dynamic` が優先）
- `'self'` や具体的なホスト名も無視される

これにより、既存のスクリプトローダーやモジュールバンドラーとの互換性を維持しつつ、攻撃者が直接注入したインラインスクリプトはブロックされる。

### Q4: なぜ `X-Frame-Options` と CSP の `frame-ancestors` を両方設定する必要がありますか？

`X-Frame-Options` は古いヘッダーであり、`DENY` と `SAMEORIGIN` の2つの値のみサポートする。CSP の `frame-ancestors` はより柔軟であり、特定のオリジンを指定できる。両方を設定する理由は、古いブラウザが CSP の `frame-ancestors` をサポートしていない場合のフォールバックとしてである。

ただし、両方が設定されている場合、CSP `frame-ancestors` が優先される（CSP 仕様による）。そのため、CSP をサポートするモダンブラウザでは `frame-ancestors` の値が使用され、CSP をサポートしないレガシーブラウザでは `X-Frame-Options` が使用される。

### Q5: ブラウザのセキュリティモデルにおいて、拡張機能（Extension）はどのような位置づけですか？

ブラウザ拡張機能は通常のWebページよりも高い権限を持ち、セキュリティモデルの特殊な位置にある。

- 拡張機能は `manifest.json` で宣言した権限に基づいて動作する
- `content_scripts` はWebページのDOMにアクセスできるが、独立した JavaScript 実行環境（isolated world）で動作する
- `background` スクリプト（Service Worker）はブラウザAPIへの特権アクセスを持つ
- CSP はWebページに対して適用されるが、拡張機能自体には拡張機能用の CSP が適用される
- 拡張機能は `webRequest` API でネットワークリクエストを傍受・変更できる（Manifest V3 では `declarativeNetRequest` に移行）

拡張機能のインストールはユーザーの明示的な操作が必要であり、ストアの審査プロセスを経るため、一定の信頼性が担保されている。しかし、悪意ある拡張機能はブラウザのセキュリティモデルを迂回できるため、インストールする拡張機能の選別は重要である。

### Q6: CSP の設定ベストプラクティスを教えてください

本番環境で推奨される CSP 設定のベストプラクティスは以下のとおり:

**1. nonce ベースの CSP を採用する**
```http
Content-Security-Policy:
  script-src 'nonce-{ランダム値}' 'strict-dynamic';
  object-src 'none';
  base-uri 'none';
```

- リクエストごとに異なる nonce を生成し、信頼するスクリプトタグに付与する
- `'strict-dynamic'` により、nonce 付きスクリプトから動的にロードされるスクリプトも許可される
- `'unsafe-inline'` や `'unsafe-eval'` は避ける（攻撃者がインラインスクリプトを注入できる）

**2. すべての重要なディレクティブを明示的に設定する**
```http
Content-Security-Policy:
  default-src 'self';
  script-src 'nonce-{random}' 'strict-dynamic';
  style-src 'self' 'nonce-{random}';
  img-src 'self' https: data:;
  font-src 'self';
  connect-src 'self';
  frame-src 'none';
  frame-ancestors 'none';
  form-action 'self';
  base-uri 'none';
  object-src 'none';
  upgrade-insecure-requests;
```

**3. Report-Only モードでテストする**
本番適用前に `Content-Security-Policy-Report-Only` で違反をモニタリングし、誤検知を防ぐ。

**4. レポート収集エンドポイントを設定する**
```http
Content-Security-Policy: ...; report-uri /csp-violation-report;
```
CSP 違反を収集・分析することで、攻撃の試みや設定ミスを検知できる。

**5. 段階的に厳格化する**
最初は `default-src 'self'` から始め、徐々に `'unsafe-inline'` を排除し、nonce/ハッシュベースに移行する。

### Q7: ブラウザのサンドボックス化の仕組みを詳しく教えてください

ブラウザのサンドボックスは、OSレベルの権限制限メカニズムを利用して、レンダラープロセスが実行できる操作を厳しく制限する仕組みである。

**Windows でのサンドボックス実装:**
- **Job Objects**: プロセスグループに対してリソース制限を適用
- **Integrity Levels**: プロセスに「Low Integrity」ラベルを付与し、より高い Integrity Level のリソースへのアクセスを禁止
- **Restricted Tokens**: プロセスのアクセストークンから多くの権限を削除
- **AppContainer**: Windows 8 以降で導入された、UWP アプリと同様のサンドボックス環境

**macOS でのサンドボックス実装:**
- **Seatbelt (sandbox-exec)**: Apple 独自のサンドボックスフレームワーク
- プロファイルベースでアクセス可能なリソースを定義（ファイルシステム、ネットワーク、IPC など）
- レンダラープロセスは極めて制限されたプロファイルで起動される

**Linux でのサンドボックス実装:**
- **namespaces**: プロセスから見えるリソース（PID、ネットワーク、マウントポイント等）を分離
- **seccomp-bpf**: システムコールをフィルタリングし、許可されたシステムコールのみ実行可能にする
- **cgroups**: リソース使用量（CPU、メモリ等）を制限

**サンドボックスの制限内容:**
- ファイルシステムへの直接アクセス禁止（ブラウザプロセス経由でのみアクセス可能）
- ネットワークソケットの直接作成禁止
- デバイスドライバーへのアクセス禁止
- 他のプロセスへのアクセス禁止
- ウィンドウシステムへの直接アクセス制限

これにより、たとえレンダラープロセスが攻撃者に乗っ取られても、ユーザーのファイルを読み取ったり、マルウェアをインストールしたりすることはできない。攻撃者がさらにシステムを侵害するには、サンドボックスをエスケープする脆弱性を発見する必要がある（サンドボックスエスケープは高度な攻撃であり、通常は報奨金プログラムで高額の報酬が支払われる）。

---

## 11. ブラウザセキュリティの進化と将来展望

### 11.1 Privacy Sandbox

Google が推進する Privacy Sandbox は、サードパーティ Cookie に依存しないWeb エコシステムの構築を目指すイニシアチブである。以下の主要 API で構成される。

| API 名 | 用途 | サードパーティ Cookie の代替 |
|--------|------|---------------------------|
| Topics API | 興味関心ベースの広告 | Cookie ベースのユーザープロファイリング |
| Protected Audience (FLEDGE) | リターゲティング広告 | サードパーティ Cookie によるリターゲティング |
| Attribution Reporting | コンバージョン計測 | Cookie ベースのアトリビューション |
| Private State Tokens | 不正防止（Bot検知） | サードパーティ Cookie による信頼性判定 |
| FedCM | 認証連携（SSO） | サードパーティ Cookie による SSO |
| CHIPS | パーティション化 Cookie | 無制限のサードパーティ Cookie |
| Fenced Frames | 広告表示の分離 | iframe + サードパーティ Cookie |
| Shared Storage | 制限付きクロスサイトストレージ | サードパーティ Cookie による状態共有 |

### 11.2 Speculation Rules API とセキュリティ

Speculation Rules API は、ページのプリレンダリングやプリフェッチを宣言的に制御する仕組みである。セキュリティ面では以下の点に注意が必要である。

```html
<!-- Speculation Rules の記述例 -->
<script type="speculationrules">
{
    "prerender": [
        {
            "where": {
                "href_matches": "/products/*"
            },
            "eagerness": "moderate"
        }
    ],
    "prefetch": [
        {
            "urls": ["/api/featured-products"],
            "requires": ["anonymous-client-ip-when-cross-origin"]
        }
    ]
}
</script>

<!--
  セキュリティ上の考慮事項:
  1. プリレンダリングされたページは、ユーザーが実際にナビゲーションする前に
     副作用（API呼び出し、アナリティクス等）を発生させる可能性がある
  2. クロスオリジンのプリフェッチでは、ユーザーのIPアドレスが
     プリフェッチ先に漏洩する可能性がある
     → "requires": ["anonymous-client-ip-when-cross-origin"] で対策
  3. CSP はプリレンダリングされたページにも適用される
-->
```

---

## FAQ

### Q1: CSPを導入する際に、既存のインラインスクリプトが動作しなくなるのを防ぐにはどうすればよいですか?
CSPを段階的に導入するには、まず `Content-Security-Policy-Report-Only` ヘッダーを使って違反レポートのみを収集し、影響範囲を把握します。その後、インラインスクリプトには `nonce` 属性（サーバー側で毎回ランダム生成する値）を付与し、CSPヘッダーに `'nonce-<値>'` を指定することで、正当なインラインスクリプトのみ実行を許可できます。`'unsafe-inline'` の使用は最終手段とし、可能な限り `nonce` + `strict-dynamic` の組み合わせを推奨します。

### Q2: Same-Origin PolicyとCORSの関係はどのように整理すればよいですか?
Same-Origin Policy（SOP）はブラウザのデフォルトのセキュリティポリシーで、異なるオリジン間のリソースアクセスを制限します。CORS（Cross-Origin Resource Sharing）はSOPの例外を安全に設けるための仕組みです。サーバーが `Access-Control-Allow-Origin` ヘッダーで許可するオリジンを明示することで、ブラウザはクロスオリジンリクエストのレスポンスをJavaScriptに公開します。SOPが「デフォルトで拒否」、CORSが「明示的に許可」という関係にあります。

### Q3: SameSite Cookie属性のLax、Strict、Noneの使い分けはどうすべきですか?
`SameSite=Lax`（デフォルト）は、トップレベルナビゲーション（リンクのクリック）ではCookieが送信されますが、iframe内やAJAXリクエストではクロスサイトCookieが送信されません。これが最も汎用的な選択肢です。`SameSite=Strict` はクロスサイトリクエストでは一切Cookieを送信しないため、外部サイトからのリンク経由でもログイン状態が維持されません。`SameSite=None; Secure` はクロスサイトでもCookieを送信しますが、HTTPS必須で、サードパーティCookie廃止の動きにより今後制約が強まります。認証Cookieには `Lax` を、埋め込みウィジェット用には `None` を使用するのが一般的です。

---

## まとめ

### セキュリティ機構の対応表

| 概念 | 防御対象 | 設定場所 | ポイント |
|------|---------|---------|---------|
| サンドボックス | プロセス権限昇格 | OS/ブラウザ内部 | レンダラーの権限制限、OS隔離 |
| Same-Origin Policy | クロスオリジンデータ窃取 | ブラウザ内部（自動） | スキーム+ホスト+ポートで判定 |
| CSP | XSS の影響緩和 | HTTP ヘッダー | nonce + strict-dynamic を推奨 |
| サイトアイソレーション | Spectre 等のサイドチャネル | ブラウザ内部（自動） | 異なるサイトを別プロセスで実行 |
| Cookie セキュリティ | セッション窃取、CSRF | Set-Cookie ヘッダー | Secure + HttpOnly + SameSite=Lax |
| SRI | CDN リソースの改ざん | HTML の integrity 属性 | sha384 以上のハッシュを推奨 |
| CORS | 安全なクロスオリジン通信 | HTTP レスポンスヘッダー | ホワイトリスト + Vary: Origin |
| Trusted Types | DOM XSS | CSP + JavaScript API | innerHTML 等への文字列代入を禁止 |
| HSTS | ダウングレード攻撃 | HTTP レスポンスヘッダー | preload リストへの登録を推奨 |
| Permissions Policy | 不要な機能の悪用 | HTTP ヘッダー / iframe 属性 | 使わない機能は明示的に無効化 |

### セキュリティチェックリスト

本番環境にデプロイする前に、以下の項目を確認することを推奨する。

- [ ] HTTPS が有効であり、HTTP からのリダイレクトが設定されている
- [ ] HSTS ヘッダーが設定されている（`max-age` は十分に長い値）
- [ ] CSP ヘッダーが設定され、`unsafe-inline` / `unsafe-eval` を使用していない
- [ ] Cookie に Secure, HttpOnly, SameSite 属性が設定されている
- [ ] X-Content-Type-Options: nosniff が設定されている
- [ ] X-Frame-Options または CSP frame-ancestors が設定されている
- [ ] CDN のリソースに SRI（integrity 属性）が付与されている
- [ ] CORS の設定がホワイトリスト方式になっている
- [ ] Server / X-Powered-By ヘッダーのバージョン情報が非表示になっている
- [ ] Referrer-Policy が適切に設定されている
- [ ] Permissions-Policy で不要な機能が無効化されている

---

## 次に読むべきガイド

- [レンダリングパイプライン](../01-rendering/00-rendering-pipeline.md)
- ブラウザストレージ（Cookie, localStorage, IndexedDB の詳細）
- Fetch API と CORS の実践

---

## 参考文献

1. Chromium Project. "Site Isolation Design Document." The Chromium Projects, 2018. https://www.chromium.org/Home/chromium-security/site-isolation/
2. MDN Web Docs. "Content Security Policy (CSP)." Mozilla, 2024. https://developer.mozilla.org/en-US/docs/Web/HTTP/CSP
3. W3C. "Content Security Policy Level 3." W3C Working Draft, 2023. https://www.w3.org/TR/CSP3/
4. Reis, C., Moshchuk, A., and Oskov, N. "Site Isolation: Process Separation for Web Sites within the Browser." USENIX Security Symposium, 2019.
5. Kocher, P., Horn, J., Fogh, A. et al. "Spectre Attacks: Exploiting Speculative Execution." IEEE S&P, 2019.
6. Google. "Privacy Sandbox." Web.dev, 2024. https://web.dev/privacy-sandbox/
7. OWASP. "OWASP Secure Headers Project." OWASP Foundation, 2024. https://owasp.org/www-project-secure-headers/
8. Barth, A. "The Web Origin Concept." RFC 6454, IETF, 2011.
9. West, M. "Incrementally Better Cookies." RFC 6265bis, IETF, 2024.
10. W3C. "Trusted Types." W3C Working Draft, 2023. https://www.w3.org/TR/trusted-types/
```
