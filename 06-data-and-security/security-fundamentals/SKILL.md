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
| 00 | [[docs/00-basics/00-security-principles.md]] | CIA triad、脅威モデリング、リスク評価、Defense in Depth |
| 01 | [[docs/00-basics/01-owasp-top10.md]] | OWASP Top 10（2021）全項目の詳細解説と対策 |
| 02 | [[docs/00-basics/02-secure-development-lifecycle.md]] | SSDLC、脅威モデリング（STRIDE）、セキュリティレビュー |
| 03 | [[docs/00-basics/03-compliance-and-standards.md]] | ISO 27001、SOC 2、GDPR、PCI DSS、NIST フレームワーク |

### 01-web-security — Web セキュリティ

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/01-web-security/00-xss-prevention.md]] | XSS の種類（Reflected/Stored/DOM）、CSP、サニタイゼーション |
| 01 | [[docs/01-web-security/01-injection-attacks.md]] | SQL Injection、NoSQL Injection、Command Injection、ORM |
| 02 | [[docs/01-web-security/02-csrf-and-cors.md]] | CSRF 対策、CORS 設定、SameSite Cookie、Origin 検証 |
| 03 | [[docs/01-web-security/03-security-headers.md]] | CSP、HSTS、X-Frame-Options、Permissions-Policy、helmet.js |

### 02-cryptography — 暗号技術

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/02-cryptography/00-encryption-basics.md]] | 対称鍵暗号（AES）、非対称鍵暗号（RSA/ECDSA）、ハッシュ |
| 01 | [[docs/02-cryptography/01-tls-and-certificates.md]] | TLS 1.3、証明書チェーン、Let's Encrypt、mTLS |
| 02 | [[docs/02-cryptography/02-password-and-key-management.md]] | パスワードハッシュ、KDF、鍵管理、HSM、Vault |
| 03 | [[docs/02-cryptography/03-digital-signatures.md]] | デジタル署名、JWT 署名、コード署名、PKI |

### 03-network-security — ネットワークセキュリティ

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/03-network-security/00-firewall-and-ids.md]] | ファイアウォール、IDS/IPS、WAF、DDoS 対策 |
| 01 | [[docs/03-network-security/01-vpn-and-zero-trust.md]] | VPN、ゼロトラスト、BeyondCorp、SASE |
| 02 | [[docs/03-network-security/02-dns-security.md]] | DNSSEC、DoH/DoT、DNS 攻撃と対策 |

### 04-application-security — アプリケーションセキュリティ

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/04-application-security/00-input-validation.md]] | 入力検証、サニタイゼーション、型安全、Zod/Joi |
| 01 | [[docs/04-application-security/01-dependency-security.md]] | npm audit、Dependabot、Snyk、SCA、サプライチェーン攻撃 |
| 02 | [[docs/04-application-security/02-api-security.md]] | API 認証、レート制限、入力検証、BOLA/BFLA |
| 03 | [[docs/04-application-security/03-container-security.md]] | イメージスキャン、最小権限、rootless、ランタイム保護 |

### 05-cloud-security — クラウドセキュリティ

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/05-cloud-security/00-aws-security.md]] | IAM ベストプラクティス、VPC セキュリティ、GuardDuty |
| 01 | [[docs/05-cloud-security/01-secrets-management.md]] | シークレット管理（Vault、AWS SM、.env）、ローテーション |
| 02 | [[docs/05-cloud-security/02-infrastructure-security.md]] | IaC セキュリティ、CSPM、設定監査（Config/SecurityHub） |

### 06-operations — セキュリティ運用

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/06-operations/00-logging-and-monitoring.md]] | セキュリティログ、SIEM、アラート設計、監査ログ |
| 01 | [[docs/06-operations/01-incident-response.md]] | インシデント対応フロー、フォレンジック、ポストモーテム |
| 02 | [[docs/06-operations/02-penetration-testing.md]] | ペネトレーションテスト、バグバウンティ、CTF、ツール |

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
