# セッション vs トークン

> 認証状態の管理には「セッション方式」と「トークン方式」の2つの主要なアプローチがある。ステートフルとステートレスの本質的な違い、それぞれのメリット・デメリット、そしてプロジェクトに応じた正しい選定基準を解説する。

## この章で学ぶこと

- [ ] セッション方式とトークン方式の仕組みと違いを理解する
- [ ] 各方式のセキュリティ上のトレードオフを把握する
- [ ] プロジェクト要件に基づく適切な方式を選定できるようになる

---

## 1. 2つの方式の全体像

```
セッション方式（ステートフル）:

  ユーザー             サーバー              セッションストア
    │                   │                      │
    │ ログイン           │                      │
    │──────────────────>│                      │
    │                   │ セッション作成         │
    │                   │─────────────────────>│
    │                   │     session_id        │
    │                   │<─────────────────────│
    │ Set-Cookie:       │                      │
    │ session_id=abc123 │                      │
    │<──────────────────│                      │
    │                   │                      │
    │ Cookie:           │                      │
    │ session_id=abc123 │                      │
    │──────────────────>│                      │
    │                   │ セッションデータ取得   │
    │                   │─────────────────────>│
    │                   │ { userId, role, ... } │
    │                   │<─────────────────────│
    │ レスポンス         │                      │
    │<──────────────────│                      │

  特徴:
  → 認証状態をサーバー側で管理
  → Cookie でセッション ID のみ送信
  → サーバーが「誰がログインしているか」を把握


トークン方式（ステートレス）:

  ユーザー             サーバー
    │                   │
    │ ログイン           │
    │──────────────────>│
    │                   │ JWT 生成（署名付き）
    │ { accessToken }   │
    │<──────────────────│
    │                   │
    │ Authorization:    │
    │ Bearer eyJhbG...  │
    │──────────────────>│
    │                   │ JWT 検証（署名確認のみ）
    │                   │ → サーバーに状態なし
    │ レスポンス         │
    │<──────────────────│

  特徴:
  → 認証状態をトークンに含む（自己完結型）
  → サーバーは状態を持たない
  → トークンの署名を検証するだけ
```

---

## 2. 詳細比較

```
比較表:

  項目            │ セッション          │ トークン（JWT）
  ──────────────┼───────────────────┼──────────────────
  状態管理       │ サーバー側           │ クライアント側
  ストレージ     │ Redis / DB          │ 不要（検証のみ）
  スケーラビリティ│ セッションストア     │ ステートレス
                │ の共有が必要          │ （スケール容易）
  失効           │ サーバー側で即時     │ 有効期限まで
                │ 削除可能             │ 失効不可（※）
  データサイズ   │ Cookie: 小           │ JWT: 大（~1KB）
  CSRF           │ 対策が必要           │ 不要（Bearer）
  XSS            │ HttpOnly で保護     │ localStorage は
                │                     │ XSS に脆弱
  モバイル対応   │ Cookie 管理が面倒    │ ヘッダーで簡単
  マイクロ       │ セッション共有       │ 各サービスで
  サービス       │ が困難               │ 独立検証可能

  ※ JWT の失効対策:
    → 短い有効期限（15分）+ Refresh Token
    → ブラックリスト（サーバー側に状態が必要）
    → Token Version（ユーザーごとのバージョン番号）
```

---

## 3. セキュリティの比較

```
セッション方式のセキュリティ:

  利点:
  ✓ サーバー側で即時無効化可能
  ✓ HttpOnly Cookie でXSS耐性
  ✓ セッションデータはサーバーに安全に保管
  ✓ セッション固定攻撃の防御が確立

  リスク:
  ✗ CSRF攻撃に脆弱（対策必須）
  ✗ セッションハイジャック（ID漏洩時）
  ✗ セッションストアのSPOF（単一障害点）

  対策:
  → SameSite=Lax/Strict Cookie
  → CSRF トークン
  → セッション ID のローテーション
  → Redis Sentinel / Cluster


トークン方式（JWT）のセキュリティ:

  利点:
  ✓ CSRF攻撃の心配なし（Authorizationヘッダー）
  ✓ サーバーに状態不要
  ✓ マイクロサービス間の認証が容易

  リスク:
  ✗ 即時失効が困難
  ✗ トークンサイズが大きい（ヘッダー肥大）
  ✗ localStorage保存 → XSS で窃取可能
  ✗ トークンの改ざん検知のみ（暗号化ではない）
  ✗ 秘密鍵が漏洩すると全トークンが偽造可能

  対策:
  → 短い有効期限（15分以下）
  → Refresh Token Rotation
  → HttpOnly Cookie に JWT を保存（ハイブリッド）
  → RS256/ES256（非対称鍵）で署名
```

---

## 4. ハイブリッドアプローチ

```
推奨: JWT を HttpOnly Cookie に保存するハイブリッド:

  トークンの利点（ステートレス）+ Cookie の利点（XSS耐性）

  ┌──────────────────────────────────────────────┐
  │                                              │
  │  ログイン時:                                   │
  │    サーバー → JWT 生成                          │
  │    サーバー → Set-Cookie: token=eyJ..;          │
  │               HttpOnly; Secure; SameSite=Lax   │
  │                                              │
  │  リクエスト時:                                  │
  │    ブラウザ → Cookie: token=eyJ..              │
  │    サーバー → JWT 検証（署名確認）               │
  │                                              │
  │  利点:                                         │
  │  ✓ JavaScript からトークンにアクセス不可（XSS耐性）│
  │  ✓ ステートレス検証（サーバー側に状態不要）        │
  │  ✓ CSRF は SameSite で防御                     │
  │                                              │
  └──────────────────────────────────────────────┘
```

```typescript
// ハイブリッドアプローチの実装
import { SignJWT, jwtVerify } from 'jose';
import { cookies } from 'next/headers';

const secret = new TextEncoder().encode(process.env.JWT_SECRET);

// ログイン: JWT を HttpOnly Cookie に設定
async function login(userId: string, role: string) {
  const token = await new SignJWT({ sub: userId, role })
    .setProtectedHeader({ alg: 'HS256' })
    .setIssuedAt()
    .setExpirationTime('15m')  // 短い有効期限
    .sign(secret);

  const cookieStore = await cookies();
  cookieStore.set('token', token, {
    httpOnly: true,     // JavaScript からアクセス不可
    secure: true,       // HTTPS のみ
    sameSite: 'lax',    // CSRF 防御
    path: '/',
    maxAge: 15 * 60,    // 15分
  });

  // Refresh Token は別の Cookie（より長い有効期限）
  const refreshToken = await createRefreshToken(userId);
  cookieStore.set('refresh_token', refreshToken, {
    httpOnly: true,
    secure: true,
    sameSite: 'strict',  // Refresh は Strict
    path: '/api/auth/refresh',  // リフレッシュ API のみ
    maxAge: 7 * 24 * 60 * 60,  // 7日
  });
}

// リクエスト検証
async function verifyAuth(): Promise<{ userId: string; role: string } | null> {
  const cookieStore = await cookies();
  const token = cookieStore.get('token')?.value;
  if (!token) return null;

  try {
    const { payload } = await jwtVerify(token, secret);
    return { userId: payload.sub as string, role: payload.role as string };
  } catch {
    return null;
  }
}

// トークンリフレッシュ
async function refreshAccessToken() {
  const cookieStore = await cookies();
  const refreshToken = cookieStore.get('refresh_token')?.value;
  if (!refreshToken) throw new Error('No refresh token');

  const { userId } = await verifyRefreshToken(refreshToken);

  // 新しい Access Token を発行
  await login(userId, (await getUser(userId)).role);

  // Refresh Token Rotation: 新しい Refresh Token も発行
  await rotateRefreshToken(refreshToken, userId);
}
```

---

## 5. 選定ガイドライン

```
プロジェクトタイプ別の推奨:

  ┌───────────────────────────────────────────────┐
  │                                               │
  │  Next.js / フルスタック Web アプリ:              │
  │  → ハイブリッド（JWT in HttpOnly Cookie）        │
  │  → 理由: SSR + SPA 両対応、CSRF/XSS 対策済み    │
  │                                               │
  │  SPA + 別バックエンド API:                      │
  │  → JWT in HttpOnly Cookie（BFF経由）            │
  │  → または短命 Access Token + Refresh Token       │
  │                                               │
  │  モバイルアプリ + API:                           │
  │  → JWT（Secure Storage に保存）                 │
  │  → Access Token(15分) + Refresh Token(30日)    │
  │                                               │
  │  マイクロサービス間通信:                          │
  │  → JWT（サービス間は短命トークン）                │
  │  → mTLS（相互TLS認証）                          │
  │                                               │
  │  伝統的 Web アプリ（MPA）:                      │
  │  → セッション + Cookie                          │
  │  → 最もシンプルで安全                            │
  │                                               │
  │  B2B エンタープライズ:                           │
  │  → セッション（即時無効化が重要）                  │
  │  → SAML / OIDC for SSO                        │
  │                                               │
  └───────────────────────────────────────────────┘

判断フローチャート:

  モバイルアプリ？ ─yes→ JWT + Secure Storage
       │no
  マイクロサービス？ ─yes→ JWT（サービス間）
       │no
  即時失効が必要？ ─yes→ セッション
       │no
  SPA？ ─yes→ JWT in HttpOnly Cookie
       │no
  → セッション or ハイブリッド
```

---

## 6. トークン保存場所の比較

```
ブラウザでのトークン保存場所:

  保存場所        │ XSS耐性 │ CSRF耐性 │ 推奨度
  ──────────────┼────────┼────────┼───────
  HttpOnly Cookie│ ✓ 安全  │ △ 対策要 │ ◎ 推奨
  localStorage   │ ✗ 脆弱  │ ✓ 安全   │ ✗ 非推奨
  sessionStorage │ ✗ 脆弱  │ ✓ 安全   │ ✗ 非推奨
  メモリ（変数）  │ ✓ 安全  │ ✓ 安全   │ △ リロードで消失

  HttpOnly Cookie が推奨される理由:
  → XSS でトークンを読み取れない
  → SameSite 属性で CSRF も防御可能
  → ブラウザが自動送信（実装が簡潔）
  → Secure 属性で HTTPS 強制

  localStorage が非推奨の理由:
  → XSS 脆弱性1つでトークン窃取
  → document.cookie とは異なり HttpOnly がない
  → CSP でも完全には防げない

  メモリ保存（Auth0 のアプローチ）:
  → トークンを JavaScript 変数に保持
  → XSS でもアクセス困難（スコープ外）
  → リロード時に再認証が必要（Refresh Token で対応）
```

---

## まとめ

| 方式 | 最適な用途 | 注意点 |
|------|----------|--------|
| セッション | MPA、即時失効が必要 | CSRF対策必須、ストア管理 |
| JWT（Bearer） | モバイル、API間 | 即時失効困難、サイズ大 |
| JWT + Cookie | Next.js、SPA | 推奨。XSS/CSRF両対策 |
| Refresh Token | 長期セッション | ローテーション必須 |

---

## 次に読むべきガイド
→ [[../01-session-auth/00-cookie-and-session.md]] — Cookie とセッション管理

---

## 参考文献
1. OWASP. "Session Management Cheat Sheet." cheatsheetseries.owasp.org, 2024.
2. Auth0. "Token Storage." auth0.com/docs, 2024.
3. RFC 6750. "The OAuth 2.0 Authorization Framework: Bearer Token Usage." IETF, 2012.
