# 認証・認可（Authentication & Authorization）

> 認証と認可はWebアプリケーションセキュリティの根幹。パスワード管理、セッション、JWT、OAuth 2.0、OpenID Connect、RBAC/ABAC、多要素認証からNextAuth.js実装まで、安全なアクセス制御の全てを体系的に解説する。

## このSkillの対象者

- Web アプリケーションに認証機能を実装するエンジニア
- セキュリティを意識した設計・実装を学びたい開発者
- OAuth 2.0 / OIDC の仕組みを深く理解したい方
- RBAC/ABAC による権限管理を設計する方

## 前提知識

- HTTP の基礎（ヘッダー、Cookie、ステータスコード）
- JavaScript / TypeScript の基礎
- Web アプリケーションの基本構造（フロントエンド / バックエンド）

## 学習ガイド

### 00-fundamentals — 認証・認可の基礎

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/00-fundamentals/00-authentication-vs-authorization.md]] | 認証と認可の違い、脅威モデル、セキュリティ原則 |
| 01 | [[docs/00-fundamentals/01-password-security.md]] | パスワードハッシュ（bcrypt/Argon2）、ポリシー、漏洩検知 |
| 02 | [[docs/00-fundamentals/02-multi-factor-authentication.md]] | TOTP、WebAuthn/Passkeys、SMS、リカバリーコード |
| 03 | [[docs/00-fundamentals/03-session-vs-token.md]] | セッション方式 vs トークン方式の比較と選定基準 |

### 01-session-auth — セッションベース認証

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/01-session-auth/00-cookie-and-session.md]] | Cookie属性、セッション管理、HttpOnly/Secure/SameSite |
| 01 | [[docs/01-session-auth/01-session-store.md]] | メモリ/Redis/DB セッションストア、スケーリング戦略 |
| 02 | [[docs/01-session-auth/02-csrf-protection.md]] | CSRF攻撃と防御（Synchronizer Token、Double Submit Cookie、SameSite） |

### 02-token-auth — トークンベース認証

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/02-token-auth/00-jwt-deep-dive.md]] | JWT構造、署名アルゴリズム、クレーム設計、検証フロー |
| 01 | [[docs/02-token-auth/01-oauth2-flows.md]] | OAuth 2.0 全フロー（Authorization Code、PKCE、Client Credentials、Device Code） |
| 02 | [[docs/02-token-auth/02-openid-connect.md]] | OIDC プロトコル、ID Token、UserInfo、Discovery |
| 03 | [[docs/02-token-auth/03-token-management.md]] | Access/Refresh Token、ローテーション、失効、ストレージ戦略 |

### 03-authorization — 認可設計

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/03-authorization/00-rbac.md]] | ロールベースアクセス制御、権限モデル、階層ロール |
| 01 | [[docs/03-authorization/01-abac-and-policies.md]] | 属性ベースアクセス制御、ポリシーエンジン、CASL |
| 02 | [[docs/03-authorization/02-api-authorization.md]] | スコープ設計、API キー管理、リソースベース認可 |
| 03 | [[docs/03-authorization/03-frontend-authorization.md]] | フロントエンド権限制御、ルートガード、UIの条件表示 |

### 04-implementation — 実装パターン

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/04-implementation/00-nextauth-setup.md]] | NextAuth.js (Auth.js) セットアップと基本設定 |
| 01 | [[docs/04-implementation/01-social-login.md]] | ソーシャルログイン（Google、GitHub、Apple）実装 |
| 02 | [[docs/04-implementation/02-email-password-auth.md]] | メール・パスワード認証（登録、ログイン、パスワードリセット） |
| 03 | [[docs/04-implementation/03-sso-and-enterprise.md]] | SSO（SAML、OIDC）、エンタープライズ認証、ディレクトリ連携 |

## クイックリファレンス

```
認証方式の選定:
  個人開発・小規模 → NextAuth.js + ソーシャルログイン
  B2C サービス → OAuth 2.0 + PKCE + メール認証
  B2B SaaS → OIDC + SAML SSO + RBAC
  API サービス → API Key + OAuth 2.0 Client Credentials
  モバイルアプリ → OAuth 2.0 + PKCE + Refresh Token Rotation

セキュリティチェックリスト:
  ✓ パスワードは bcrypt/Argon2 でハッシュ
  ✓ JWT は RS256/ES256 で署名
  ✓ Cookie は HttpOnly + Secure + SameSite=Lax
  ✓ CSRF トークンを実装
  ✓ Refresh Token はローテーション + 失効検知
  ✓ レート制限をログインに適用
  ✓ MFA を重要操作に要求
```

## 参考文献

1. OWASP. "Authentication Cheat Sheet." cheatsheetseries.owasp.org, 2024.
2. RFC 6749. "The OAuth 2.0 Authorization Framework." IETF, 2012.
3. RFC 7519. "JSON Web Token (JWT)." IETF, 2015.
4. OpenID Foundation. "OpenID Connect Core 1.0." openid.net, 2014.
5. Auth.js. "Documentation." authjs.dev, 2024.
