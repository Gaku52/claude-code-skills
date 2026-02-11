# 認証方式

> Web APIの認証方式を体系的に理解する。Basic認証、Bearer Token、OAuth 2.0、JWTの仕組みと使い分けを学び、安全な認証システムを設計する。

## この章で学ぶこと

- [ ] 主要な認証方式の仕組みと違いを理解する
- [ ] OAuth 2.0のフローを把握する
- [ ] JWTの構造とセキュリティ上の注意点を学ぶ

---

## 1. 認証と認可

```
認証（Authentication）:
  → 「あなたは誰？」
  → ユーザーの本人確認
  → ログイン処理

認可（Authorization）:
  → 「あなたは何ができる？」
  → アクセス権限の確認
  → ロールベースアクセス制御（RBAC）

  例:
  認証: ログインして「Taroさん」であることを確認
  認可: 「Taroさん」は管理者なので全データにアクセス可能
```

---

## 2. Basic認証

```
Basic認証:
  → ユーザー名:パスワードをBase64エンコード
  → リクエストごとに送信

  Authorization: Basic dGFybzpwYXNzd29yZA==
                       ↑ "taro:password" のBase64

  利点:
  ✓ 実装が極めてシンプル
  ✓ サーバー側の状態管理不要

  欠点:
  ✗ パスワードが毎回送信される（Base64は暗号化ではない）
  ✗ HTTPS必須（平文で流れる）
  ✗ ログアウト機能がない

  用途:
  → 内部ツール、CI/CDのAPI認証
  → 本番APIには非推奨
```

---

## 3. Bearer Token / API Key

```
Bearer Token:
  Authorization: Bearer eyJhbGciOiJIUzI1NiIs...

  → サーバーが発行したトークンをヘッダーに含める
  → トークンの形式は自由（JWT、ランダム文字列等）

API Key:
  X-API-Key: sk_live_abcdef123456
  または
  ?api_key=sk_live_abcdef123456

  → サービス間連携でよく使用
  → ユーザーではなくアプリケーションの認証

セッションベース vs トークンベース:
  ┌──────────────┬──────────────────┬──────────────────┐
  │              │ セッション        │ トークン          │
  ├──────────────┼──────────────────┼──────────────────┤
  │ 状態管理     │ サーバー側        │ クライアント側    │
  │ スケーラビリティ│ 低い（共有必要）│ 高い（ステートレス）│
  │ 無効化       │ 容易             │ 困難              │
  │ ストレージ   │ サーバーメモリ/DB│ クライアント       │
  │ CSRF        │ 脆弱             │ 安全              │
  └──────────────┴──────────────────┴──────────────────┘
```

---

## 4. JWT（JSON Web Token）

```
JWTの構造:
  eyJhbGciOiJIUzI1NiIs.eyJzdWIiOiIxMjM0NTY3.SflKxwRJSMeKKF2QT4
  ↑ ヘッダー               ↑ ペイロード          ↑ 署名

  ヘッダー（Base64URL）:
  {
    "alg": "HS256",     // 署名アルゴリズム
    "typ": "JWT"
  }

  ペイロード（Base64URL）:
  {
    "sub": "user_123",        // Subject（ユーザーID）
    "name": "Taro",
    "role": "admin",
    "iat": 1704067200,        // Issued At（発行時刻）
    "exp": 1704070800,        // Expiration（有効期限）
    "iss": "api.example.com"  // Issuer（発行者）
  }

  署名:
  HMACSHA256(
    base64UrlEncode(header) + "." + base64UrlEncode(payload),
    secret
  )

  重要: ペイロードは暗号化されていない（Base64デコードで読める）
  → 機密情報（パスワード等）を含めてはいけない
  → 署名は改ざん検知のため（暗号化ではない）

トークンの種類:
  Access Token:
    → 短い有効期限（15分〜1時間）
    → APIアクセスに使用
    → ステートレス検証

  Refresh Token:
    → 長い有効期限（7日〜30日）
    → Access Token の再発行に使用
    → サーバー側で管理（無効化可能）

フロー:
  1. ログイン → Access Token + Refresh Token を取得
  2. API呼び出し → Access Token をヘッダーに付与
  3. Access Token 期限切れ → Refresh Token で再取得
  4. Refresh Token 期限切れ → 再ログイン

JWTのセキュリティ注意:
  ✗ alg: "none" を許可しない（署名なし攻撃）
  ✗ ペイロードに機密情報を入れない
  ✗ 有効期限を長くしすぎない
  ✓ HTTPS で送信
  ✓ HttpOnly Cookie に保存（XSS対策）
  ✓ RS256（公開鍵）を推奨（HS256はシークレット共有が必要）
```

---

## 5. OAuth 2.0

```
OAuth 2.0 = 認可のフレームワーク（認証ではない）
  → 第三者アプリにリソースへのアクセス権を委譲

登場人物:
  Resource Owner:  ユーザー
  Client:          アプリ（アクセスを要求する側）
  Authorization Server: 認可サーバー（Google, GitHub等）
  Resource Server: リソースサーバー（API）

Authorization Code Flow（推奨）:

  ユーザー     アプリ       認可サーバー    リソースサーバー
    │           │              │                │
    │──ログイン→│              │                │
    │           │──認可リクエスト→│              │
    │←───── ログイン画面 ────│                │
    │── 同意 ──→│              │                │
    │           │←── 認可コード──│              │
    │           │── コード + シークレット →│    │
    │           │←── Access Token ────│        │
    │           │── API呼び出し + Token ──────→│
    │           │←── リソースデータ ───────── │

  Authorization Code + PKCE（SPA/モバイル向け）:
  → クライアントシークレットが不要
  → code_verifier / code_challenge で保護
  → 現在のSPAでの推奨方式

主要なグラントタイプ:
  ┌─────────────────────┬──────────────────────────────┐
  │ グラントタイプ       │ 用途                          │
  ├─────────────────────┼──────────────────────────────┤
  │ Authorization Code  │ サーバーサイドアプリ（推奨）  │
  │ Auth Code + PKCE    │ SPA/モバイル（推奨）          │
  │ Client Credentials  │ マシン間通信（サーバー間）    │
  │ Device Code         │ TV/IoT等の入力制限デバイス    │
  └─────────────────────┴──────────────────────────────┘

  廃止されたグラント:
  ✗ Implicit: セキュリティ上の問題で非推奨
  ✗ Resource Owner Password: 非推奨
```

---

## 6. OpenID Connect（OIDC）

```
OIDC = OAuth 2.0 + 認証レイヤー
  → OAuth 2.0は認可のみ、OIDCは認証も提供

  OAuth 2.0: 「このアプリにGoogleドライブへのアクセスを許可」
  OIDC:      「このアプリにGoogleアカウントでログイン」

  Access Token: リソースへのアクセス権
  ID Token:     ユーザーの認証情報（JWT形式）

  ID Token の中身:
  {
    "iss": "https://accounts.google.com",
    "sub": "110169484474386276334",
    "aud": "my-app-client-id",
    "email": "user@gmail.com",
    "name": "Taro Yamada",
    "picture": "https://...",
    "iat": 1704067200,
    "exp": 1704070800
  }

主要なOIDCプロバイダー:
  Google, Microsoft, Apple, Auth0, Okta, Keycloak
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| Basic認証 | シンプルだが本番非推奨 |
| Bearer Token | ステートレスなAPI認証 |
| JWT | ヘッダー.ペイロード.署名、暗号化ではない |
| OAuth 2.0 | 認可フレームワーク、Auth Code + PKCE推奨 |
| OIDC | OAuth 2.0 + 認証（ID Token） |

---

## 次に読むべきガイド
→ [[02-common-attacks.md]] — ネットワーク攻撃

---

## 参考文献
1. RFC 7519. "JSON Web Token (JWT)." IETF, 2015.
2. RFC 6749. "The OAuth 2.0 Authorization Framework." IETF, 2012.
3. OpenID Connect Core 1.0. OpenID Foundation, 2014.
