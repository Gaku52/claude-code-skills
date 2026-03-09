# セキュリティ基礎

> セキュリティはソフトウェア開発の基盤。OWASP Top 10、暗号技術、ネットワークセキュリティ、アプリケーションセキュリティ、クラウドセキュリティ、セキュリティ運用まで、エンジニアに必要なセキュリティ知識を体系的に解説する。

## このSkillの対象者

- セキュリティの基礎を体系的に学びたいエンジニア
- セキュアなアプリケーション開発を目指す方
- セキュリティ監査・インシデント対応を担当する方

## 前提知識

- Web アプリケーションの基本構造
- ネットワークの基礎知識（TCP/IP、HTTP）
- Linux の基本操作

## 学習ガイド

### 00-basics — セキュリティの基礎

| # | ファイル | 内容 |
|---|---------|------|

### 01-web-security — Web セキュリティ

| # | ファイル | 内容 |
|---|---------|------|

### 02-cryptography — 暗号技術

| # | ファイル | 内容 |
|---|---------|------|

### 03-network-security — ネットワークセキュリティ

| # | ファイル | 内容 |
|---|---------|------|

### 04-application-security — アプリケーションセキュリティ

| # | ファイル | 内容 |
|---|---------|------|

### 05-cloud-security — クラウドセキュリティ

| # | ファイル | 内容 |
|---|---------|------|

### 06-operations — セキュリティ運用

| # | ファイル | 内容 |
|---|---------|------|

## クイックリファレンス

```
セキュリティチェックリスト:

  Web アプリケーション:
    ✓ 入力検証（サーバーサイド必須）
    ✓ パラメータ化クエリ（SQL Injection 防止）
    ✓ CSP ヘッダー設定（XSS 防止）
    ✓ CSRF トークン or SameSite=Lax
    ✓ HttpOnly + Secure Cookie
    ✓ HTTPS 強制（HSTS）

  認証・認可:
    ✓ bcrypt/Argon2 パスワードハッシュ
    ✓ MFA（TOTP or WebAuthn）
    ✓ JWT 署名検証（ES256 推奨）
    ✓ 最小権限の原則

  インフラ:
    ✓ 依存関係の脆弱性スキャン
    ✓ コンテナイメージスキャン
    ✓ シークレット管理（.env をコミットしない）
    ✓ ログ・監査証跡の保持

  OWASP Top 10 (2021):
    A01: Broken Access Control
    A02: Cryptographic Failures
    A03: Injection
    A04: Insecure Design
    A05: Security Misconfiguration
    A06: Vulnerable Components
    A07: Auth Failures
    A08: Software/Data Integrity
    A09: Logging Failures
    A10: SSRF
```

## 参考文献

1. OWASP. "Top 10 Web Application Security Risks." owasp.org, 2021.
2. NIST. "Cybersecurity Framework." nist.gov, 2024.
3. Mozilla. "Web Security Guidelines." infosec.mozilla.org, 2024.
