# ブラウザセキュリティモデル

> ブラウザはユーザーとWeb上のコンテンツの間に立つセキュリティの要。サンドボックス、同一オリジンポリシー、CSP、サイト分離、Cookie のセキュリティ属性を理解する。

## この章で学ぶこと

- [ ] ブラウザのサンドボックスモデルを理解する
- [ ] CSP（Content Security Policy）の設定方法を把握する
- [ ] Cookieのセキュリティ属性を学ぶ

---

## 1. サンドボックス

```
ブラウザのサンドボックス:
  → レンダラープロセスの権限を制限
  → Webコンテンツがユーザーのシステムに直接アクセスできない

  制限される操作:
  ✗ ファイルシステムへの直接アクセス
  ✗ ネットワークの直接操作
  ✗ OSのAPIへのアクセス
  ✗ 他のプロセスへのアクセス
  ✗ ハードウェアの直接制御

  許可される操作（API経由）:
  ✓ fetch() でHTTPリクエスト（CORSの範囲内）
  ✓ <input type="file"> でユーザーが選択したファイル
  ✓ Geolocation API（ユーザー許可後）
  ✓ Camera/Microphone（ユーザー許可後）
  ✓ localStorage / IndexedDB（オリジン単位で隔離）

  Permissions Policy:
  → Webページが使用できるブラウザ機能を制限
  Permissions-Policy: camera=(), microphone=(), geolocation=(self)
```

---

## 2. CSP（Content Security Policy）

```
CSP = Webページが読み込めるリソースを制限

  目的: XSS攻撃の被害を軽減

  設定方法:
  ① HTTPヘッダー（推奨）:
  Content-Security-Policy: default-src 'self'; script-src 'self' https://cdn.example.com

  ② メタタグ:
  <meta http-equiv="Content-Security-Policy" content="default-src 'self'">

主要なディレクティブ:
  ┌────────────────┬──────────────────────────────────┐
  │ ディレクティブ  │ 制御対象                          │
  ├────────────────┼──────────────────────────────────┤
  │ default-src    │ フォールバック（全リソース）       │
  │ script-src     │ JavaScript                       │
  │ style-src      │ CSS                              │
  │ img-src        │ 画像                             │
  │ connect-src    │ fetch/XHR/WebSocket の接続先      │
  │ font-src       │ フォント                         │
  │ frame-src      │ iframe                           │
  │ media-src      │ 動画/音声                        │
  │ object-src     │ <object>, <embed>                │
  │ base-uri       │ <base> タグのURI                 │
  │ form-action    │ フォームの送信先                  │
  │ frame-ancestors│ このページをiframeに入れられる親  │
  └────────────────┴──────────────────────────────────┘

  値の指定:
  'self'           — 同一オリジンのみ
  'none'           — 全てブロック
  'unsafe-inline'  — インラインスクリプト/スタイル許可（非推奨）
  'unsafe-eval'    — eval() 許可（非推奨）
  'nonce-xxx'      — 指定nonceを持つスクリプトのみ許可
  'strict-dynamic' — 信頼されたスクリプトが読み込むスクリプトも許可
  https:           — HTTPSのリソースのみ
  data:            — data: URIを許可

推奨設定（厳格）:
  Content-Security-Policy:
    default-src 'self';
    script-src 'self' 'nonce-{random}';
    style-src 'self' 'nonce-{random}';
    img-src 'self' data: https:;
    connect-src 'self' https://api.example.com;
    font-src 'self';
    object-src 'none';
    base-uri 'self';
    frame-ancestors 'none';
```

---

## 3. Cookie セキュリティ

```
Cookie のセキュリティ属性:

  Set-Cookie: session=abc123;
    Secure;            ← HTTPS のみ送信
    HttpOnly;          ← JavaScriptからアクセス不可（XSS対策）
    SameSite=Lax;      ← クロスサイトリクエストでの送信制限
    Path=/;            ← Cookie の有効パス
    Domain=.example.com;← Cookie の有効ドメイン
    Max-Age=86400;     ← 有効期限（秒）

SameSite の値:
  ┌───────────┬──────────────────────────────────────────┐
  │ 値        │ 動作                                      │
  ├───────────┼──────────────────────────────────────────┤
  │ Strict    │ クロスサイトリクエストでは一切送信しない   │
  │           │ → 外部リンクからの遷移でもCookieなし       │
  ├───────────┼──────────────────────────────────────────┤
  │ Lax       │ トップレベルナビゲーション（リンク遷移）は │
  │ (デフォルト)│ 送信。POSTやiframeでは送信しない         │
  ├───────────┼──────────────────────────────────────────┤
  │ None      │ 常に送信（Secure属性が必須）               │
  │           │ → サードパーティCookie                    │
  └───────────┴──────────────────────────────────────────┘

サードパーティCookieの廃止:
  → Chrome: Privacy Sandbox に移行（段階的）
  → Safari: ITP で既にブロック
  → Firefox: ETP でブロック
  → 影響: 広告トラッキング、SSO、埋め込みウィジェット
```

---

## 4. その他のセキュリティ機構

```
① Subresource Integrity（SRI）:
  → CDNのリソースが改ざんされていないことを検証
  <script src="https://cdn.example.com/lib.js"
    integrity="sha384-oqVuAfXRKap7fdgcCY5uykM6+R9GqQ8K..."
    crossorigin="anonymous"></script>

② Referrer Policy:
  → リファラー情報の送信範囲を制御
  Referrer-Policy: strict-origin-when-cross-origin
  → 同一オリジン: フルURL送信
  → クロスオリジン: オリジンのみ送信
  → HTTPS→HTTP: 送信しない

③ Feature Policy / Permissions Policy:
  → ページおよびiframeで使用可能な機能を制限
  Permissions-Policy: camera=(), microphone=(), payment=(self)

④ Cross-Origin Isolation:
  → SharedArrayBuffer 等の高精度APIを安全に使用
  Cross-Origin-Opener-Policy: same-origin
  Cross-Origin-Embedder-Policy: require-corp

⑤ Trusted Types:
  → DOM XSSを防ぐためのAPI
  → innerHTML等への文字列直接代入を禁止
  Content-Security-Policy: require-trusted-types-for 'script'
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| サンドボックス | レンダラーの権限制限、OS隔離 |
| CSP | リソース読み込み元の制限（XSS軽減） |
| Cookie | Secure + HttpOnly + SameSite=Lax |
| SRI | CDNリソースの改ざん検知 |
| Site Isolation | 異なるサイトを別プロセスで実行 |

---

## 次に読むべきガイド
→ [[../01-rendering/00-rendering-pipeline.md]] — レンダリングパイプライン

---

## 参考文献
1. MDN Web Docs. "Content Security Policy (CSP)." Mozilla, 2024.
2. W3C. "Content Security Policy Level 3." 2023.
