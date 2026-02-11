# OAuth 2.0 フロー

> OAuth 2.0 は「認可の委譲」のための標準プロトコル。Authorization Code + PKCE、Client Credentials、Device Code、Implicit（非推奨）の各フローの仕組み、適用場面、セキュリティ上の注意点を網羅的に解説する。

## この章で学ぶこと

- [ ] OAuth 2.0 の役割とアクターを理解する
- [ ] 各フローの仕組みと適用場面を把握する
- [ ] PKCE による SPA/モバイル向けセキュリティ強化を実装できる

---

## 1. OAuth 2.0 の基本概念

```
OAuth 2.0 の役割:
  「第三者アプリケーションにユーザーのリソースへの
   限定的なアクセスを許可する仕組み」

  例: 「GitHub のリポジトリ一覧を MyApp に許可する」
  → ユーザーは GitHub のパスワードを MyApp に渡さない
  → MyApp は必要な権限（スコープ）のみ取得

4つのアクター:

  ┌─────────────────────────────────────────┐
  │                                         │
  │  Resource Owner（リソースオーナー）        │
  │  → ユーザー本人                           │
  │                                         │
  │  Client（クライアント）                    │
  │  → アクセスを要求するアプリケーション        │
  │  → MyApp                                │
  │                                         │
  │  Authorization Server（認可サーバー）      │
  │  → アクセス許可を発行するサーバー            │
  │  → GitHub の OAuth サーバー               │
  │                                         │
  │  Resource Server（リソースサーバー）        │
  │  → 保護されたリソースを持つサーバー          │
  │  → GitHub API                           │
  │                                         │
  └─────────────────────────────────────────┘

フローの選択:

  アプリ種別              │ 推奨フロー
  ──────────────────────┼──────────────────────
  Web アプリ（サーバーあり）│ Authorization Code
  SPA（サーバーなし）      │ Authorization Code + PKCE
  モバイルアプリ           │ Authorization Code + PKCE
  サーバー間通信           │ Client Credentials
  IoT / テレビ            │ Device Code
  SPA（レガシー）         │ Implicit（非推奨）
```

---

## 2. Authorization Code フロー

```
Authorization Code フロー（最も安全、最も一般的）:

  ユーザー    クライアント   認可サーバー    リソースサーバー
    │           │            │               │
    │ ログイン   │            │               │
    │──────────>│            │               │
    │           │ ① 認可リクエスト             │
    │           │───────────>│               │
    │           │            │               │
    │ ② ログイン画面          │               │
    │<──────────────────────│               │
    │           │            │               │
    │ ③ 認証 + 権限承認      │               │
    │──────────────────────>│               │
    │           │            │               │
    │ ④ redirect_uri +      │               │
    │   authorization_code  │               │
    │──────────>│            │               │
    │           │            │               │
    │           │ ⑤ code → token 交換        │
    │           │───────────>│               │
    │           │            │               │
    │           │ ⑥ access_token +           │
    │           │   refresh_token             │
    │           │<───────────│               │
    │           │            │               │
    │           │ ⑦ API リクエスト             │
    │           │ + Bearer access_token       │
    │           │───────────────────────────>│
    │           │            │               │
    │           │ ⑧ リソース                  │
    │           │<───────────────────────────│
    │ レスポンス  │            │               │
    │<──────────│            │               │
```

```typescript
// Authorization Code フロー実装

// ① 認可リクエスト（クライアント → 認可サーバー）
function getAuthorizationUrl(state: string): string {
  const params = new URLSearchParams({
    response_type: 'code',
    client_id: process.env.GITHUB_CLIENT_ID!,
    redirect_uri: 'https://myapp.com/callback',
    scope: 'read:user repo',
    state, // CSRF 防御用ランダム値
  });

  return `https://github.com/login/oauth/authorize?${params}`;
}

// ④⑤ コールバック処理（認可コード → トークン交換）
async function handleCallback(code: string, state: string) {
  // state の検証（CSRF 防御）
  if (state !== storedState) {
    throw new Error('Invalid state parameter');
  }

  // ⑤ トークン交換（サーバー側で実行 → client_secret を安全に使用）
  const tokenResponse = await fetch('https://github.com/login/oauth/access_token', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
    body: JSON.stringify({
      client_id: process.env.GITHUB_CLIENT_ID,
      client_secret: process.env.GITHUB_CLIENT_SECRET, // サーバーのみ
      code,
      redirect_uri: 'https://myapp.com/callback',
    }),
  });

  const { access_token, refresh_token, scope, token_type } = await tokenResponse.json();

  // ⑦ リソース取得
  const userResponse = await fetch('https://api.github.com/user', {
    headers: { Authorization: `Bearer ${access_token}` },
  });

  return userResponse.json();
}
```

---

## 3. PKCE（Proof Key for Code Exchange）

```
PKCE の必要性:

  SPA / モバイルアプリの問題:
  → client_secret を安全に保持できない
  → 認可コードの横取り攻撃（Authorization Code Interception）
  → 悪意あるアプリが redirect_uri を傍受して code を取得

  PKCE の仕組み:
  → code_verifier: クライアントが生成するランダム文字列
  → code_challenge: code_verifier のハッシュ（SHA-256）
  → 認可リクエスト時に challenge を送信
  → トークン交換時に verifier を送信
  → サーバーが verifier をハッシュして challenge と比較

  PKCE フロー:
  ① code_verifier = ランダム文字列（43〜128文字）
  ② code_challenge = BASE64URL(SHA256(code_verifier))
  ③ 認可リクエスト: code_challenge + code_challenge_method=S256
  ④ コールバック: authorization_code を受信
  ⑤ トークン交換: code + code_verifier を送信
  ⑥ サーバー: SHA256(code_verifier) == code_challenge を検証
```

```typescript
// PKCE 実装

// code_verifier と code_challenge の生成
function generatePKCE(): { verifier: string; challenge: string } {
  // code_verifier: 43〜128文字の暗号ランダム文字列
  const verifier = base64URLEncode(crypto.getRandomValues(new Uint8Array(32)));

  // code_challenge: SHA-256 ハッシュの Base64URL エンコード
  const encoder = new TextEncoder();
  const data = encoder.encode(verifier);
  const hashBuffer = await crypto.subtle.digest('SHA-256', data);
  const challenge = base64URLEncode(new Uint8Array(hashBuffer));

  return { verifier, challenge };
}

function base64URLEncode(buffer: Uint8Array): string {
  return btoa(String.fromCharCode(...buffer))
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=+$/, '');
}

// ① 認可リクエスト（PKCE 付き）
async function startAuthFlow() {
  const { verifier, challenge } = await generatePKCE();
  const state = crypto.randomUUID();

  // verifier と state をセッションに保存
  sessionStorage.setItem('pkce_verifier', verifier);
  sessionStorage.setItem('oauth_state', state);

  const params = new URLSearchParams({
    response_type: 'code',
    client_id: 'my-spa-client-id',
    redirect_uri: 'https://myapp.com/callback',
    scope: 'openid profile email',
    state,
    code_challenge: challenge,
    code_challenge_method: 'S256',
  });

  window.location.href = `https://auth.example.com/authorize?${params}`;
}

// ⑤ コールバック（PKCE + トークン交換）
async function handlePKCECallback(code: string, state: string) {
  // state 検証
  const storedState = sessionStorage.getItem('oauth_state');
  if (state !== storedState) throw new Error('Invalid state');

  const verifier = sessionStorage.getItem('pkce_verifier');

  // トークン交換（client_secret 不要）
  const response = await fetch('https://auth.example.com/token', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({
      grant_type: 'authorization_code',
      code,
      redirect_uri: 'https://myapp.com/callback',
      client_id: 'my-spa-client-id',
      code_verifier: verifier!,  // client_secret の代わり
    }),
  });

  const tokens = await response.json();
  // { access_token, refresh_token, id_token, token_type, expires_in }

  // クリーンアップ
  sessionStorage.removeItem('pkce_verifier');
  sessionStorage.removeItem('oauth_state');

  return tokens;
}
```

---

## 4. Client Credentials フロー

```
Client Credentials フロー:

  用途: サーバー間通信（ユーザーが介在しない）
  例: バッチ処理、マイクロサービス間、バックエンド API

  クライアント          認可サーバー         リソースサーバー
    │                   │                   │
    │ client_id +       │                   │
    │ client_secret     │                   │
    │──────────────────>│                   │
    │                   │                   │
    │ access_token      │                   │
    │<──────────────────│                   │
    │                   │                   │
    │ Bearer token      │                   │
    │──────────────────────────────────────>│
    │                   │                   │
    │ リソース           │                   │
    │<──────────────────────────────────────│

  特徴:
  → ユーザーの同意画面なし
  → client_id + client_secret で認証
  → refresh_token は発行されない
  → トークンの有効期限で管理
```

```typescript
// Client Credentials フロー
async function getServiceToken(): Promise<string> {
  const response = await fetch('https://auth.example.com/token', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
      // Basic 認証で client_id:client_secret を送信
      'Authorization': `Basic ${btoa(`${CLIENT_ID}:${CLIENT_SECRET}`)}`,
    },
    body: new URLSearchParams({
      grant_type: 'client_credentials',
      scope: 'read:data write:data',
    }),
  });

  const { access_token } = await response.json();
  return access_token;
}
```

---

## 5. Device Code フロー

```
Device Code フロー:

  用途: キーボード入力が困難なデバイス（テレビ、IoT、CLI）
  例: 「テレビで https://example.com/activate にアクセスして
       コード ABC-123 を入力してください」

  デバイス        認可サーバー      ユーザーのスマホ/PC
    │               │                 │
    │ device auth    │                 │
    │──────────────>│                 │
    │               │                 │
    │ device_code + │                 │
    │ user_code +   │                 │
    │ verification  │                 │
    │ _uri          │                 │
    │<──────────────│                 │
    │               │                 │
    │ 画面に表示:    │                 │
    │ "コード:       │                 │
    │  ABC-123"     │                 │
    │               │ ユーザーが        │
    │               │ verification_uri│
    │               │ にアクセス        │
    │               │<────────────────│
    │               │ コード入力+承認   │
    │               │<────────────────│
    │               │                 │
    │ ポーリング      │                 │
    │ (device_code)  │                 │
    │──────────────>│                 │
    │               │                 │
    │ access_token  │                 │
    │<──────────────│                 │
```

```typescript
// Device Code フロー（CLI ツール等）
async function deviceCodeFlow() {
  // デバイス認可リクエスト
  const deviceRes = await fetch('https://auth.example.com/device/code', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({
      client_id: 'my-cli-app',
      scope: 'openid profile',
    }),
  });

  const {
    device_code,
    user_code,
    verification_uri,
    verification_uri_complete,
    expires_in,
    interval,
  } = await deviceRes.json();

  // ユーザーに表示
  console.log(`ブラウザで ${verification_uri} にアクセスし、`);
  console.log(`コード ${user_code} を入力してください。`);

  // ポーリングでトークン取得を試行
  const deadline = Date.now() + expires_in * 1000;

  while (Date.now() < deadline) {
    await sleep(interval * 1000);

    const tokenRes = await fetch('https://auth.example.com/token', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({
        grant_type: 'urn:ietf:params:oauth:grant-type:device_code',
        device_code,
        client_id: 'my-cli-app',
      }),
    });

    const data = await tokenRes.json();

    if (data.error === 'authorization_pending') continue;
    if (data.error === 'slow_down') {
      interval += 5; // ポーリング間隔を延長
      continue;
    }
    if (data.error) throw new Error(data.error_description);

    return data; // { access_token, refresh_token, ... }
  }

  throw new Error('Device code expired');
}
```

---

## 6. スコープ設計

```
スコープの設計パターン:

  resource:action 形式:
    → read:user, write:user, delete:user
    → read:repo, write:repo, admin:repo

  GitHub のスコープ例:
    → repo: リポジトリ全般
    → read:user: ユーザープロフィール読取
    → user:email: メールアドレス読取
    → admin:org: 組織管理

  設計原則:
    → 最小権限: 必要最小限のスコープのみ要求
    → 粒度: 細かすぎず粗すぎず
    → 命名: resource:action の一貫したパターン
    → ドキュメント: 各スコープの意味を明確に
```

---

## まとめ

| フロー | 用途 | セキュリティ |
|--------|------|------------|
| Authorization Code | Web アプリ（サーバーあり） | 最高 |
| Auth Code + PKCE | SPA、モバイル | 高い |
| Client Credentials | サーバー間 | 高い |
| Device Code | IoT、CLI | 中程度 |
| Implicit | 非推奨 | 低い |

---

## 次に読むべきガイド
→ [[02-openid-connect.md]] — OpenID Connect

---

## 参考文献
1. RFC 6749. "The OAuth 2.0 Authorization Framework." IETF, 2012.
2. RFC 7636. "Proof Key for Code Exchange by OAuth Public Clients." IETF, 2015.
3. RFC 8628. "OAuth 2.0 Device Authorization Grant." IETF, 2019.
4. OAuth.net. "OAuth 2.0." oauth.net, 2024.
