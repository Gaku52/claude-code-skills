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

### 01-session-auth — セッションベース認証

| # | ファイル | 内容 |
|---|---------|------|

### 02-token-auth — トークンベース認証

| # | ファイル | 内容 |
|---|---------|------|

### 03-authorization — 認可設計

| # | ファイル | 内容 |
|---|---------|------|

### 04-implementation — 実装パターン

| # | ファイル | 内容 |
|---|---------|------|

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
