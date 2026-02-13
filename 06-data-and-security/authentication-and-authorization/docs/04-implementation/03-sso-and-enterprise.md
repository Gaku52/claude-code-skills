# SSO とエンタープライズ認証

> B2B SaaS では SSO（Single Sign-On）対応がエンタープライズ契約の必須要件。SAML 2.0、OIDC ベースの SSO、ディレクトリ連携（SCIM）、テナント別認証設定まで、エンタープライズ認証の全体像を解説する。エンタープライズ顧客の 80% 以上が SSO を要求しており、SSO 非対応は事実上のエンタープライズ市場からの脱落を意味する。

## この章で学ぶこと

- [ ] SSO の概念と SAML / OIDC の違いを理解する
- [ ] SAML 2.0 の認証フローを把握する
- [ ] OIDC ベースの SSO を実装する
- [ ] テナント別 SSO 設定の設計と実装を学ぶ
- [ ] SCIM によるディレクトリ連携を実装する
- [ ] エンタープライズ向けの認証セキュリティ要件を把握する
- [ ] Just-in-Time プロビジョニングを実装する

## 前提知識

- HTTP プロトコルの基礎
- OAuth 2.0 / OpenID Connect の基本概念
- TypeScript / Node.js の基本
- マルチテナントアーキテクチャの基礎

## 関連ガイド

- [[../02-token-auth/02-openid-connect.md]] — OpenID Connect の詳細
- [[../01-session-auth/00-cookie-and-session.md]] — セッション管理
- [[../03-authorization/01-abac-and-policies.md]] — 認可とポリシー
- [[02-email-password-auth.md]] — メール・パスワード認証

---

## 1. SSO の基本概念

```
SSO（Single Sign-On）とは:

  1回のログインで複数のアプリケーションにアクセス可能にする仕組み

  ┌─────────────────────────────────────────┐
  │                                         │
  │  ユーザー: Okta にログイン（1回だけ）       │
  │                                         │
  │  → Slack にアクセス → 自動ログイン          │
  │  → Jira にアクセス → 自動ログイン           │
  │  → 自社アプリにアクセス → 自動ログイン       │
  │                                         │
  │  Identity Provider（IdP）:               │
  │    → Okta, Azure AD, Google Workspace    │
  │    → ユーザーの認証を一元管理               │
  │                                         │
  │  Service Provider（SP）:                  │
  │    → 自社アプリ（SSO を受け入れる側）        │
  │    → IdP の認証結果を信頼                  │
  │                                         │
  └─────────────────────────────────────────┘

SSO のメリット:
  企業側:
    → ユーザー管理の一元化（1箇所で全アプリの権限制御）
    → 退職時の即時アクセス無効化（IdP で無効化 → 全アプリ遮断）
    → コンプライアンス対応（監査ログの一元化）
    → IT コスト削減（パスワードリセット問い合わせの減少）

  ユーザー側:
    → パスワード記憶の削減（1つのパスワードで全アプリ）
    → ログイン手順の簡素化
    → パスワード疲れの解消

  セキュリティ:
    → MFA の一元適用（IdP で MFA → 全アプリに波及）
    → パスワード漏洩リスクの低減
    → フィッシング対策の強化
    → 一貫したセキュリティポリシーの適用
```

### 1.1 SSO プロトコルの比較

```
SSO プロトコル:

  ┌────────────┬──────────────────┬──────────────────┬──────────────────┐
  │ 項目       │ SAML 2.0          │ OIDC              │ LDAP             │
  ├────────────┼──────────────────┼──────────────────┼──────────────────┤
  │ データ形式 │ XML               │ JSON              │ バイナリ(ASN.1)   │
  │ トークン   │ Assertion         │ ID Token(JWT)     │ セッション        │
  │ 署名方式   │ XML Signature     │ JWS(JWT)          │ SASL/TLS         │
  │ 対象       │ エンタープライズ    │ コンシューマー+エンタープライズ│ 社内ネットワーク │
  │ 策定年     │ 2005年            │ 2014年            │ 1993年           │
  │ 複雑さ     │ 高い              │ 中程度            │ 高い             │
  │ ブラウザ   │ リダイレクト/POST  │ リダイレクト       │ 不要(TCP直接)     │
  │ モバイル   │ △(XMLパースが重い) │ ◎(JSON,軽量)      │ ×(社内のみ)       │
  │ IdP例      │ Okta,Azure AD,    │ Okta,Azure AD,    │ Active Directory │
  │           │ OneLogin          │ Auth0,Google      │ OpenLDAP         │
  │ 採用率     │ エンタープライズの  │ 増加中            │ レガシー          │
  │           │ デファクト標準      │                  │ （縮小傾向）       │
  └────────────┴──────────────────┴──────────────────┴──────────────────┘

  選定ガイドライン:
    SAML 2.0 を選ぶ場合:
      → 大企業の既存 IdP が SAML のみ対応
      → Okta、OneLogin 等のレガシー連携
      → セキュリティ要件が XML Signature を要求

    OIDC を選ぶ場合:
      → モダンな IdP（Azure AD、Google Workspace）
      → モバイルアプリ対応が必要
      → 開発コストを最小化したい
      → JSON ベースで扱いやすい

    両方サポートする場合（推奨）:
      → エンタープライズ顧客の要件に柔軟に対応
      → SAML と OIDC の両方を統一的に扱う抽象レイヤーを設計
```

---

## 2. SAML 2.0

### 2.1 SAML 2.0 の認証フロー

```
SAML 2.0 の認証フロー（SP-Initiated）:

  ユーザー     SP（自社アプリ）    IdP（Okta等）
    │            │                │
    │ アクセス    │                │
    │───────────>│                │
    │            │ 未認証を検知     │
    │            │                │
    │ ① SAMLRequest             │
    │ （リダイレクト）              │
    │<───────────│                │
    │────────────────────────────>│
    │            │                │
    │            │  ② ログイン画面  │
    │            │                │
    │<───────────────────────────│
    │ 認証情報入力│                │
    │────────────────────────────>│
    │            │                │
    │ ③ SAMLResponse            │
    │ （署名付きアサーション）       │
    │<───────────────────────────│
    │────────────>│               │
    │            │ ④ 署名検証      │
    │            │ セッション作成   │
    │ ⑤ ログイン  │               │
    │   完了      │               │
    │<───────────│               │

IdP-Initiated SSO:
  ユーザーが IdP のダッシュボードからアプリを選択して直接アクセス

  ユーザー     IdP（Okta等）      SP（自社アプリ）
    │            │                │
    │ Okta にログイン              │
    │───────────>│                │
    │            │                │
    │ アプリ選択  │                │
    │───────────>│                │
    │            │ SAMLResponse   │
    │            │ 生成            │
    │<───────────│                │
    │ POST /saml/callback         │
    │────────────────────────────>│
    │            │                │ 署名検証
    │            │                │ セッション作成
    │ ログイン完了│                │
    │<───────────────────────────│

SAML の構成要素:
  → Assertion: ユーザー情報を含む XML（署名付き）
     → Authentication Statement: いつ、どのように認証されたか
     → Attribute Statement: ユーザー属性（email, name, groups）
     → Authorization Decision Statement: 認可判定（あまり使われない）
  → Metadata: SP/IdP の設定情報（エンドポイント、証明書）
  → Binding: 通信方式（HTTP-Redirect, HTTP-POST, SOAP）
  → Profile: ユースケースの定義（Web Browser SSO Profile 等）
```

### 2.2 SAML 2.0 の実装

```typescript
// SAML 2.0 実装（samlify ライブラリ）
// npm install samlify @authenio/samlify-node-xmllint

import * as samlify from 'samlify';
import * as validator from '@authenio/samlify-node-xmllint';
import fs from 'fs';

// XML 署名の検証を有効化（本番では必須）
samlify.setSchemaValidator(validator);

// SP（自社アプリ）の設定
const sp = samlify.ServiceProvider({
  entityID: 'https://myapp.com/saml/metadata',
  assertionConsumerService: [{
    Binding: 'urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST',
    Location: 'https://myapp.com/api/auth/saml/callback',
  }],
  singleLogoutService: [{
    Binding: 'urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect',
    Location: 'https://myapp.com/api/auth/saml/logout',
  }],
  nameIDFormat: ['urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress'],
  signingCert: fs.readFileSync('./certs/sp-cert.pem', 'utf8'),
  privateKey: fs.readFileSync('./certs/sp-key.pem', 'utf8'),
  // 署名アルゴリズム（SHA-256 推奨、SHA-1 は非推奨）
  requestSignatureAlgorithm: 'http://www.w3.org/2001/04/xmldsig-more#rsa-sha256',
  // Assertion の暗号化を要求（オプション、高セキュリティ環境）
  wantAssertionsSigned: true,
});

// IdP の設定（Okta の Metadata から自動構成）
async function createIdPFromMetadata(metadataUrl: string) {
  const response = await fetch(metadataUrl);
  const metadata = await response.text();

  return samlify.IdentityProvider({
    metadata,
    // Assertion 内の署名を検証
    wantMessageSigned: true,
  });
}

// IdP の設定（手動構成）
function createIdPManually(config: {
  entityId: string;
  ssoUrl: string;
  certificate: string;
}) {
  return samlify.IdentityProvider({
    entityID: config.entityId,
    singleSignOnService: [{
      Binding: 'urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect',
      Location: config.ssoUrl,
    }],
    signingCert: config.certificate,
    wantMessageSigned: true,
  });
}

// SP Metadata エンドポイント（IdP に提供する）
app.get('/api/auth/saml/metadata', (req, res) => {
  res.type('application/xml');
  res.send(sp.getMetadata());
});

// SAML ログイン開始（SP-Initiated）
app.get('/api/auth/saml/login', async (req, res) => {
  const { orgId } = req.query;

  // テナントの SSO 設定を取得
  const org = await prisma.organization.findUnique({
    where: { id: orgId as string },
  });

  if (!org?.ssoEnabled || org.ssoProvider !== 'saml') {
    return res.status(400).json({ error: 'SAML SSO not configured' });
  }

  // テナント固有の IdP を構成
  const idp = org.ssoMetadataUrl
    ? await createIdPFromMetadata(org.ssoMetadataUrl)
    : createIdPManually({
        entityId: org.ssoEntityId!,
        ssoUrl: org.ssoSignOnUrl!,
        certificate: org.ssoCertificate!,
      });

  // RelayState にリダイレクト先を保存
  const relayState = JSON.stringify({
    orgId: org.id,
    redirectTo: req.query.redirectTo || '/dashboard',
  });

  const { context } = sp.createLoginRequest(idp, 'redirect');

  // RelayState をクエリパラメータに追加
  const redirectUrl = new URL(context);
  redirectUrl.searchParams.set('RelayState', relayState);

  res.redirect(redirectUrl.toString());
});

// SAML コールバック（Assertion 受信・検証）
app.post('/api/auth/saml/callback', async (req, res) => {
  try {
    // RelayState からテナント情報を取得
    const relayState = JSON.parse(req.body.RelayState || '{}');
    const { orgId, redirectTo } = relayState;

    const org = await prisma.organization.findUnique({
      where: { id: orgId },
    });

    if (!org) {
      return res.redirect('/login?error=org_not_found');
    }

    // テナント固有の IdP を構成
    const idp = org.ssoMetadataUrl
      ? await createIdPFromMetadata(org.ssoMetadataUrl)
      : createIdPManually({
          entityId: org.ssoEntityId!,
          ssoUrl: org.ssoSignOnUrl!,
          certificate: org.ssoCertificate!,
        });

    // SAML Response の検証
    const { extract } = await sp.parseLoginResponse(idp, 'post', {
      body: req.body,
    });

    // ユーザー情報の取得
    const email = extract.nameID;
    const attributes = extract.attributes || {};
    const sessionIndex = extract.sessionIndex?.sessionIndex;

    // email のドメインが組織のドメインと一致するか検証
    const emailDomain = email.split('@')[1];
    if (org.domain && emailDomain !== org.domain) {
      await logSecurityEvent({
        type: 'saml_domain_mismatch',
        orgId: org.id,
        email,
        expectedDomain: org.domain,
        severity: 'high',
      });
      return res.redirect('/login?error=domain_mismatch');
    }

    // JIT（Just-in-Time）プロビジョニング
    const user = await findOrCreateSAMLUser({
      email,
      name: formatUserName(attributes),
      orgId: org.id,
      samlNameId: extract.nameID,
      samlSessionIndex: sessionIndex,
      groups: attributes.groups || attributes.memberOf || [],
      attributes,
    });

    // セッション作成
    const { sessionId } = await sessionManager.create(
      { userId: user.id, role: user.role },
      req
    );

    setSessionCookie(res, sessionId);

    // 監査ログ
    await logAuthEvent({
      type: 'saml_login',
      userId: user.id,
      orgId: org.id,
      idpEntityId: org.ssoEntityId,
      ip: getClientIP(req),
    });

    res.redirect(redirectTo || '/dashboard');
  } catch (error) {
    console.error('SAML validation failed:', error);

    await logSecurityEvent({
      type: 'saml_validation_failed',
      error: error instanceof Error ? error.message : 'Unknown error',
      severity: 'high',
    });

    res.redirect('/login?error=saml_failed');
  }
});

// ユーザー名のフォーマット（IdP ごとに属性名が異なる）
function formatUserName(attributes: Record<string, unknown>): string {
  // Okta
  if (attributes.firstName && attributes.lastName) {
    return `${attributes.firstName} ${attributes.lastName}`;
  }
  // Azure AD
  if (attributes['http://schemas.xmlsoap.org/ws/2005/05/identity/claims/givenname']) {
    const given = attributes['http://schemas.xmlsoap.org/ws/2005/05/identity/claims/givenname'];
    const surname = attributes['http://schemas.xmlsoap.org/ws/2005/05/identity/claims/surname'];
    return `${given} ${surname}`;
  }
  // Google Workspace
  if (attributes.displayName) {
    return attributes.displayName as string;
  }
  // フォールバック
  return attributes.name as string || 'Unknown User';
}
```

### 2.3 SAML 証明書管理

```
SAML 証明書のライフサイクル:

  ┌─────────────────────────────────────────────────────────┐
  │              証明書管理の重要性                            │
  │                                                         │
  │  SAML では XML 署名に X.509 証明書を使用                  │
  │  証明書の期限切れ = SSO の停止 = 全ユーザーがログイン不能    │
  │                                                         │
  │  証明書の種類:                                            │
  │  1. IdP の署名証明書: IdP が SAMLResponse に署名          │
  │     → SP は IdP の公開鍵で署名を検証                      │
  │     → IdP の管理画面からダウンロード                       │
  │                                                         │
  │  2. SP の署名証明書: SP が SAMLRequest に署名             │
  │     → IdP は SP の公開鍵でリクエストを検証                 │
  │     → 自社で生成して IdP に登録                           │
  │                                                         │
  │  3. SP の暗号化証明書: IdP が Assertion を暗号化          │
  │     → SP が秘密鍵で復号                                  │
  │     → 高セキュリティ環境でのみ使用                         │
  │                                                         │
  │  証明書ローテーション:                                     │
  │  1. 新しい証明書を生成                                    │
  │  2. Metadata に新旧両方の証明書を掲載                      │
  │  3. IdP に新しい Metadata を登録                          │
  │  4. 旧証明書を削除                                        │
  │  ⚠️ ダウンタイムなしでのローテーションが必要                │
  └─────────────────────────────────────────────────────────┘
```

```typescript
// SP 証明書の生成と管理
import { execSync } from 'child_process';
import crypto from 'crypto';

class SAMLCertificateManager {
  // 自己署名証明書の生成（SP 用）
  static generateSPCertificate(options: {
    commonName: string;
    organization: string;
    validityDays: number;
    outputDir: string;
  }): { certPath: string; keyPath: string } {
    const { commonName, organization, validityDays, outputDir } = options;

    const keyPath = `${outputDir}/sp-key.pem`;
    const certPath = `${outputDir}/sp-cert.pem`;

    // RSA 2048ビット秘密鍵の生成
    execSync(`openssl genrsa -out ${keyPath} 2048`);

    // 自己署名証明書の生成
    execSync(
      `openssl req -new -x509 -key ${keyPath} -out ${certPath} ` +
      `-days ${validityDays} ` +
      `-subj "/CN=${commonName}/O=${organization}"`
    );

    return { certPath, keyPath };
  }

  // 証明書の有効期限チェック
  static async checkCertificateExpiry(certPem: string): Promise<{
    expiresAt: Date;
    daysUntilExpiry: number;
    isExpired: boolean;
    isExpiringSoon: boolean; // 30日以内
  }> {
    const cert = new crypto.X509Certificate(certPem);
    const expiresAt = new Date(cert.validTo);
    const now = new Date();
    const daysUntilExpiry = Math.floor(
      (expiresAt.getTime() - now.getTime()) / (1000 * 60 * 60 * 24)
    );

    return {
      expiresAt,
      daysUntilExpiry,
      isExpired: daysUntilExpiry < 0,
      isExpiringSoon: daysUntilExpiry < 30,
    };
  }

  // 定期的な証明書チェック（cron で実行）
  static async checkAllOrganizationCertificates(): Promise<void> {
    const orgs = await prisma.organization.findMany({
      where: { ssoEnabled: true, ssoProvider: 'saml' },
    });

    for (const org of orgs) {
      if (!org.ssoCertificate) continue;

      const status = await SAMLCertificateManager.checkCertificateExpiry(
        org.ssoCertificate
      );

      if (status.isExpired) {
        // 期限切れ → 管理者に緊急通知
        await notifyOrgAdmins(org.id, {
          type: 'certificate_expired',
          message: 'SSO 証明書が期限切れです。SSO が機能しません。',
          severity: 'critical',
        });
      } else if (status.isExpiringSoon) {
        // 期限間近 → 管理者に警告
        await notifyOrgAdmins(org.id, {
          type: 'certificate_expiring_soon',
          message: `SSO 証明書が ${status.daysUntilExpiry} 日後に期限切れになります。`,
          severity: 'warning',
        });
      }
    }
  }
}
```

---

## 3. OIDC ベースの SSO

```
OIDC SSO のフロー:

  OIDC は SAML より軽量でモダンな SSO プロトコル
  JSON ベース、JWT 使用、モバイル対応

  ユーザー     SP（自社アプリ）    IdP（Azure AD等）
    │            │                │
    │ アクセス    │                │
    │───────────>│                │
    │            │ 未認証を検知     │
    │            │                │
    │ ① Authorization Request    │
    │ (PKCE + state + nonce)     │
    │<───────────│                │
    │────────────────────────────>│
    │            │                │
    │            │  ② ログイン画面  │
    │<───────────────────────────│
    │ 認証情報入力│                │
    │────────────────────────────>│
    │            │                │
    │ ③ Authorization Code       │
    │<───────────────────────────│
    │────────────>│               │
    │            │                │
    │            │ ④ Token Request│
    │            │ (code + PKCE)  │
    │            │ ──────────────>│
    │            │                │
    │            │ ⑤ ID Token +   │
    │            │ Access Token   │
    │            │ <──────────────│
    │            │                │
    │            │ ⑥ ID Token 検証 │
    │            │ セッション作成   │
    │ ⑦ ログイン  │               │
    │   完了      │               │
    │<───────────│               │

  SAML との違い:
    → ID Token は JWT（JSON）なので軽量
    → PKCE でセキュリティ強化
    → UserInfo エンドポイントで追加情報取得
    → リフレッシュトークンでセッション延長可能
```

```typescript
// OIDC ベースの SSO 実装
import { Issuer, Client, generators, TokenSet } from 'openid-client';

class OIDCSSOManager {
  private clients: Map<string, Client> = new Map();

  // テナント固有の OIDC クライアント取得
  async getClient(org: Organization): Promise<Client> {
    const cacheKey = org.id;

    // キャッシュチェック
    if (this.clients.has(cacheKey)) {
      return this.clients.get(cacheKey)!;
    }

    // OIDC Discovery でプロバイダー情報を自動取得
    const issuer = await Issuer.discover(org.ssoIssuer!);

    const client = new issuer.Client({
      client_id: org.ssoClientId!,
      client_secret: decrypt(org.ssoClientSecret!),
      redirect_uris: [`https://myapp.com/api/auth/oidc/${org.id}/callback`],
      response_types: ['code'],
      token_endpoint_auth_method: 'client_secret_post',
    });

    this.clients.set(cacheKey, client);
    return client;
  }

  // SSO ログイン開始
  async initiateLogin(
    orgId: string,
    redirectTo: string
  ): Promise<{ url: string; state: string; nonce: string; codeVerifier: string }> {
    const org = await prisma.organization.findUnique({
      where: { id: orgId },
    });

    if (!org?.ssoEnabled || org.ssoProvider !== 'oidc') {
      throw new Error('OIDC SSO not configured');
    }

    const client = await this.getClient(org);

    // PKCE
    const codeVerifier = generators.codeVerifier();
    const codeChallenge = generators.codeChallenge(codeVerifier);

    // state（CSRF 防御）
    const state = generators.state();

    // nonce（リプレイ攻撃防御）
    const nonce = generators.nonce();

    const url = client.authorizationUrl({
      scope: 'openid email profile groups',
      state,
      nonce,
      code_challenge: codeChallenge,
      code_challenge_method: 'S256',
      // ログインヒント（ドメイン指定で IdP のログイン画面を省略）
      login_hint: `@${org.domain}`,
    });

    // state, nonce, codeVerifier を Redis に保存（5分間有効）
    await redis.setex(
      `oidc_state:${state}`,
      300,
      JSON.stringify({ orgId, nonce, codeVerifier, redirectTo })
    );

    return { url, state, nonce, codeVerifier };
  }

  // コールバック処理
  async handleCallback(
    orgId: string,
    params: Record<string, string>,
    req: Request
  ): Promise<{ user: User; sessionId: string }> {
    const org = await prisma.organization.findUnique({
      where: { id: orgId },
    });

    if (!org) throw new Error('Organization not found');

    const client = await this.getClient(org);

    // state の検証
    const stateData = await redis.get(`oidc_state:${params.state}`);
    if (!stateData) throw new Error('Invalid or expired state');

    const { nonce, codeVerifier, redirectTo } = JSON.parse(stateData);

    // state を使用済みにする（リプレイ攻撃防止）
    await redis.del(`oidc_state:${params.state}`);

    // Authorization Code を Token に交換
    const tokenSet: TokenSet = await client.callback(
      `https://myapp.com/api/auth/oidc/${orgId}/callback`,
      params,
      {
        state: params.state,
        nonce,
        code_verifier: codeVerifier,
      }
    );

    // ID Token の検証は openid-client が自動で行う
    // → 署名検証、issuer 検証、audience 検証、nonce 検証、有効期限検証
    const claims = tokenSet.claims();

    // email 検証
    if (!claims.email_verified) {
      throw new Error('Email not verified at IdP');
    }

    // ドメイン検証
    const emailDomain = (claims.email as string).split('@')[1];
    if (org.domain && emailDomain !== org.domain) {
      throw new Error('Email domain mismatch');
    }

    // UserInfo エンドポイントで追加情報取得（groups 等）
    let userInfo = claims;
    try {
      const additionalInfo = await client.userinfo(tokenSet.access_token!);
      userInfo = { ...claims, ...additionalInfo };
    } catch {
      // UserInfo が利用できない場合は claims のみ使用
    }

    // JIT プロビジョニング
    const user = await findOrCreateOIDCUser({
      email: claims.email as string,
      name: claims.name as string,
      sub: claims.sub,
      orgId: org.id,
      groups: (userInfo as any).groups || [],
      picture: claims.picture as string,
    });

    // セッション作成
    const { sessionId } = await sessionManager.create(
      { userId: user.id, role: user.role },
      req
    );

    // 監査ログ
    await logAuthEvent({
      type: 'oidc_sso_login',
      userId: user.id,
      orgId: org.id,
      issuer: org.ssoIssuer,
      ip: getClientIP(req),
    });

    return { user, sessionId };
  }
}
```

---

## 4. テナント別 SSO 設定

```
マルチテナント SSO:

  テナント A: Okta で SAML SSO
  テナント B: Azure AD で OIDC SSO
  テナント C: Google Workspace で OIDC SSO
  テナント D: SSO なし（メール・パスワード）

  ログインフロー:

  ┌─────────────────────────────────────────────────────────┐
  │                                                         │
  │  ① ユーザーがメールアドレスを入力                          │
  │     alice@company-a.com                                  │
  │                                                         │
  │  ② ドメインから組織を特定                                 │
  │     company-a.com → テナント A                           │
  │                                                         │
  │  ③ テナントの SSO 設定を確認                              │
  │     テナント A: ssoEnabled=true, ssoProvider="saml"      │
  │                                                         │
  │  ④ SSO が設定されている場合                               │
  │     → IdP にリダイレクト（Okta のログイン画面へ）           │
  │                                                         │
  │  ⑤ SSO が未設定の場合                                    │
  │     → パスワード入力画面を表示                             │
  │                                                         │
  │  ⑥ SSO 強制（enforceSSO=true）の場合                     │
  │     → パスワードログインを拒否                             │
  │     → 「この組織は SSO でのログインが必要です」             │
  │                                                         │
  └─────────────────────────────────────────────────────────┘
```

### 4.1 データモデル

```typescript
// テナント別 SSO 設定のデータモデル
// schema.prisma

// model Organization {
//   id               String   @id @default(cuid())
//   name             String
//   slug             String   @unique          // URL 用のスラッグ
//   domain           String?  @unique          // "company-a.com"
//   domains          String[] @default([])     // 複数ドメイン対応
//
//   // SSO 設定
//   ssoEnabled       Boolean  @default(false)
//   ssoProvider      String?                   // "saml" | "oidc"
//   enforceSSO       Boolean  @default(false)  // SSO を強制
//
//   // SAML 設定
//   ssoEntityId      String?                   // IdP Entity ID
//   ssoSignOnUrl     String?                   // IdP SSO URL
//   ssoLogoutUrl     String?                   // IdP SLO URL
//   ssoCertificate   String?  @db.Text         // IdP 署名証明書（PEM）
//   ssoMetadataUrl   String?                   // IdP Metadata URL（自動更新用）
//   ssoMetadataXml   String?  @db.Text         // IdP Metadata XML
//
//   // OIDC 設定
//   ssoClientId      String?
//   ssoClientSecret  String?                   // 暗号化保存
//   ssoIssuer        String?                   // OIDC issuer URL
//
//   // セキュリティ設定
//   mfaRequired      Boolean  @default(false)  // MFA 強制
//   sessionMaxAge    Int?                      // セッション最大時間（秒）
//   ipAllowlist      String[] @default([])     // IP ホワイトリスト
//
//   // SCIM 設定
//   scimEnabled      Boolean  @default(false)
//   scimToken        String?                   // SCIM API トークン（ハッシュ化）
//   scimTokenSalt    String?
//
//   // 関連
//   members          OrganizationMember[]
//   ssoConnections   SSOConnection[]
//   auditLogs        AuditLog[]
//
//   createdAt        DateTime @default(now())
//   updatedAt        DateTime @updatedAt
// }
//
// model SSOConnection {
//   id               String   @id @default(cuid())
//   orgId            String
//   org              Organization @relation(fields: [orgId], references: [id])
//   provider         String   // "saml" | "oidc"
//   name             String   // "Okta Production", "Azure AD" 等
//   isActive         Boolean  @default(true)
//   isPrimary        Boolean  @default(false)  // メインの SSO 接続
//   config           Json     // プロバイダー固有の設定
//   lastTestedAt     DateTime?
//   lastUsedAt       DateTime?
//   createdAt        DateTime @default(now())
//   updatedAt        DateTime @updatedAt
//
//   @@unique([orgId, isPrimary])  // 組織ごとにプライマリは1つ
// }
```

### 4.2 ログインフロー実装

```typescript
// ドメインから組織を特定
async function getOrgByEmailDomain(email: string): Promise<Organization | null> {
  const domain = email.split('@')[1].toLowerCase();

  // 完全一致で検索
  let org = await prisma.organization.findUnique({
    where: { domain },
  });

  if (org) return org;

  // 複数ドメイン対応（domains 配列で検索）
  org = await prisma.organization.findFirst({
    where: { domains: { has: domain } },
  });

  return org;
}

// ログイン開始（メールアドレスから SSO 判定）
app.post('/api/auth/login/check', async (req, res) => {
  const { email } = req.body;

  if (!email || !isValidEmail(email)) {
    return res.status(400).json({ error: 'Valid email required' });
  }

  const org = await getOrgByEmailDomain(email);

  if (org?.ssoEnabled) {
    // SSO にリダイレクト
    return res.json({
      method: 'sso',
      provider: org.ssoProvider,
      orgId: org.id,
      orgName: org.name,
      redirectUrl: `/api/auth/sso/${org.id}/login`,
    });
  }

  if (org?.enforceSSO) {
    // SSO 強制だが SSO が設定されていない → エラー
    return res.status(403).json({
      error: 'SSO is required for this organization but not configured',
      contactAdmin: true,
    });
  }

  // パスワードログイン
  return res.json({
    method: 'password',
    // SSO が利用可能な場合はヒントを表示
    ssoAvailable: org?.ssoEnabled || false,
  });
});

// SSO ログインの統一エントリーポイント
app.get('/api/auth/sso/:orgId/login', async (req, res) => {
  const { orgId } = req.params;
  const { redirectTo } = req.query;

  const org = await prisma.organization.findUnique({
    where: { id: orgId },
  });

  if (!org?.ssoEnabled) {
    return res.redirect('/login?error=sso_not_configured');
  }

  switch (org.ssoProvider) {
    case 'saml': {
      // SAML ログインフローへ
      const idp = await getSAMLIdP(org);
      const { context } = sp.createLoginRequest(idp, 'redirect');
      const url = new URL(context);
      url.searchParams.set('RelayState', JSON.stringify({
        orgId: org.id,
        redirectTo: redirectTo || '/dashboard',
      }));
      return res.redirect(url.toString());
    }

    case 'oidc': {
      // OIDC ログインフローへ
      const oidcManager = new OIDCSSOManager();
      const { url } = await oidcManager.initiateLogin(
        orgId,
        (redirectTo as string) || '/dashboard'
      );
      return res.redirect(url);
    }

    default:
      return res.redirect('/login?error=unknown_sso_provider');
  }
});
```

### 4.3 SSO 管理画面（組織管理者用）

```typescript
// SSO 設定 API（組織管理者用）

// SSO 設定の取得
app.get('/api/admin/sso/config', requireOrgAdmin, async (req, res) => {
  const org = await prisma.organization.findUnique({
    where: { id: req.orgId },
    select: {
      ssoEnabled: true,
      ssoProvider: true,
      enforceSSO: true,
      ssoEntityId: true,
      ssoSignOnUrl: true,
      ssoMetadataUrl: true,
      ssoClientId: true,
      ssoIssuer: true,
      // 秘密情報は返さない
      // ssoCertificate, ssoClientSecret は除外
    },
  });

  // SP Metadata URL（IdP に設定するための情報）
  const spInfo = {
    entityId: 'https://myapp.com/saml/metadata',
    acsUrl: `https://myapp.com/api/auth/saml/${req.orgId}/callback`,
    metadataUrl: 'https://myapp.com/api/auth/saml/metadata',
    sloUrl: `https://myapp.com/api/auth/saml/${req.orgId}/logout`,
    // OIDC の場合
    redirectUri: `https://myapp.com/api/auth/oidc/${req.orgId}/callback`,
  };

  res.json({ config: org, spInfo });
});

// SAML SSO の設定
app.put('/api/admin/sso/saml', requireOrgAdmin, async (req, res) => {
  const { metadataUrl, metadataXml, entityId, signOnUrl, certificate } = req.body;

  // Metadata URL からの自動取得
  let resolvedConfig: any = {};

  if (metadataUrl) {
    try {
      const metadataResponse = await fetch(metadataUrl);
      const metadata = await metadataResponse.text();
      resolvedConfig = parseSAMLMetadata(metadata);
      resolvedConfig.ssoMetadataUrl = metadataUrl;
      resolvedConfig.ssoMetadataXml = metadata;
    } catch (error) {
      return res.status(400).json({
        error: 'Failed to fetch SAML metadata',
        details: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  } else if (metadataXml) {
    resolvedConfig = parseSAMLMetadata(metadataXml);
    resolvedConfig.ssoMetadataXml = metadataXml;
  } else {
    // 手動設定
    if (!entityId || !signOnUrl || !certificate) {
      return res.status(400).json({
        error: 'entityId, signOnUrl, certificate are required for manual configuration',
      });
    }

    // 証明書の形式検証
    if (!isValidX509Certificate(certificate)) {
      return res.status(400).json({ error: 'Invalid X.509 certificate' });
    }

    resolvedConfig = {
      ssoEntityId: entityId,
      ssoSignOnUrl: signOnUrl,
      ssoCertificate: certificate,
    };
  }

  // 設定を保存
  await prisma.organization.update({
    where: { id: req.orgId },
    data: {
      ssoEnabled: true,
      ssoProvider: 'saml',
      ...resolvedConfig,
    },
  });

  // 監査ログ
  await logAdminEvent({
    type: 'sso_saml_configured',
    orgId: req.orgId,
    adminId: req.userId,
  });

  res.json({ success: true, message: 'SAML SSO configured' });
});

// SSO 接続テスト
app.post('/api/admin/sso/test', requireOrgAdmin, async (req, res) => {
  const org = await prisma.organization.findUnique({
    where: { id: req.orgId },
  });

  if (!org?.ssoEnabled) {
    return res.status(400).json({ error: 'SSO not configured' });
  }

  try {
    switch (org.ssoProvider) {
      case 'saml': {
        // SAML Metadata の取得テスト
        if (org.ssoMetadataUrl) {
          const metadataRes = await fetch(org.ssoMetadataUrl, { signal: AbortSignal.timeout(10000) });
          if (!metadataRes.ok) throw new Error(`Metadata fetch failed: ${metadataRes.status}`);
        }

        // 証明書の有効期限チェック
        if (org.ssoCertificate) {
          const certStatus = await SAMLCertificateManager.checkCertificateExpiry(org.ssoCertificate);
          if (certStatus.isExpired) {
            return res.json({
              success: false,
              error: 'IdP certificate has expired',
              details: certStatus,
            });
          }
        }

        return res.json({ success: true, provider: 'saml' });
      }

      case 'oidc': {
        // OIDC Discovery の取得テスト
        const discoveryUrl = `${org.ssoIssuer}/.well-known/openid-configuration`;
        const discoveryRes = await fetch(discoveryUrl, { signal: AbortSignal.timeout(10000) });
        if (!discoveryRes.ok) throw new Error(`Discovery fetch failed: ${discoveryRes.status}`);

        const discovery = await discoveryRes.json();

        return res.json({
          success: true,
          provider: 'oidc',
          issuer: discovery.issuer,
          endpoints: {
            authorization: discovery.authorization_endpoint,
            token: discovery.token_endpoint,
            userinfo: discovery.userinfo_endpoint,
          },
        });
      }
    }
  } catch (error) {
    return res.json({
      success: false,
      error: error instanceof Error ? error.message : 'Connection test failed',
    });
  }
});

// SSO の有効化/無効化
app.put('/api/admin/sso/toggle', requireOrgAdmin, async (req, res) => {
  const { enabled, enforceSSO } = req.body;

  // SSO 無効化時の確認
  if (!enabled) {
    const activeUsers = await prisma.organizationMember.count({
      where: { orgId: req.orgId },
    });

    // SSO でのみログインしているユーザーにパスワードリセットを送信
    if (activeUsers > 0) {
      const ssoOnlyUsers = await prisma.user.findMany({
        where: {
          memberships: { some: { orgId: req.orgId } },
          passwordHash: null, // パスワード未設定（SSO でのみログイン）
        },
      });

      if (ssoOnlyUsers.length > 0) {
        return res.status(400).json({
          error: 'Cannot disable SSO',
          reason: `${ssoOnlyUsers.length} users have no password set`,
          suggestion: 'Send password reset emails before disabling SSO',
          affectedUsers: ssoOnlyUsers.map((u) => u.email),
        });
      }
    }
  }

  await prisma.organization.update({
    where: { id: req.orgId },
    data: {
      ssoEnabled: enabled,
      enforceSSO: enforceSSO ?? false,
    },
  });

  await logAdminEvent({
    type: enabled ? 'sso_enabled' : 'sso_disabled',
    orgId: req.orgId,
    adminId: req.userId,
    enforceSSO,
  });

  res.json({ success: true });
});
```

---

## 5. JIT（Just-in-Time）プロビジョニング

```
JIT プロビジョニング:

  SSO ログイン時にユーザーアカウントを自動作成する仕組み

  ┌─────────────────────────────────────────────────────────┐
  │                                                         │
  │  従来の方法:                                             │
  │  1. 管理者がユーザーを手動作成                             │
  │  2. ユーザーに招待メールを送信                             │
  │  3. ユーザーがアカウントを有効化                           │
  │  → 管理者の手間 + ユーザーの待ち時間                       │
  │                                                         │
  │  JIT プロビジョニング:                                    │
  │  1. ユーザーが IdP でログイン                              │
  │  2. SSO コールバック時にユーザーが存在しなければ自動作成     │
  │  3. IdP の属性（groups 等）からロールを自動割当             │
  │  → 管理者の手間ゼロ + 即時アクセス                        │
  │                                                         │
  │  注意点:                                                 │
  │  → 想定外のユーザーが作成される可能性（ドメイン検証必須）    │
  │  → ロール/権限の自動マッピングルールの設計が重要            │
  │  → 不要なアカウントのクリーンアップが必要                   │
  │                                                         │
  └─────────────────────────────────────────────────────────┘
```

```typescript
// JIT プロビジョニングの実装

interface SSOUserData {
  email: string;
  name: string;
  orgId: string;
  samlNameId?: string;
  samlSessionIndex?: string;
  sub?: string;           // OIDC subject
  groups?: string[];      // IdP グループ
  picture?: string;
  attributes?: Record<string, unknown>;
}

async function findOrCreateSSOUser(data: SSOUserData): Promise<User> {
  // 1. 既存ユーザーを検索
  let user = await prisma.user.findFirst({
    where: {
      email: data.email.toLowerCase(),
      memberships: { some: { orgId: data.orgId } },
    },
    include: { memberships: true },
  });

  if (user) {
    // 2a. 既存ユーザーの情報を更新
    user = await prisma.user.update({
      where: { id: user.id },
      data: {
        name: data.name,
        picture: data.picture,
        samlNameId: data.samlNameId,
        oidcSub: data.sub,
        lastLoginAt: new Date(),
        // グループベースのロール更新
        memberships: {
          update: {
            where: {
              userId_orgId: { userId: user.id, orgId: data.orgId },
            },
            data: {
              role: mapGroupsToRole(data.groups || [], data.orgId),
            },
          },
        },
      },
      include: { memberships: true },
    });

    return user;
  }

  // 2b. 新規ユーザーの自動作成（JIT プロビジョニング）
  const org = await prisma.organization.findUnique({
    where: { id: data.orgId },
  });

  if (!org) throw new Error('Organization not found');

  // JIT プロビジョニングが許可されているか確認
  if (!org.jitProvisioningEnabled) {
    throw new Error('Just-in-Time provisioning is not enabled for this organization');
  }

  // ドメイン検証
  const emailDomain = data.email.split('@')[1].toLowerCase();
  const allowedDomains = [org.domain, ...org.domains].filter(Boolean);
  if (!allowedDomains.includes(emailDomain)) {
    throw new Error(`Email domain ${emailDomain} is not allowed for this organization`);
  }

  // グループからロールをマッピング
  const role = mapGroupsToRole(data.groups || [], data.orgId);

  // ユーザー作成
  user = await prisma.user.create({
    data: {
      email: data.email.toLowerCase(),
      name: data.name,
      picture: data.picture,
      samlNameId: data.samlNameId,
      oidcSub: data.sub,
      emailVerified: true,  // IdP で認証済みなので verified
      lastLoginAt: new Date(),
      memberships: {
        create: {
          orgId: data.orgId,
          role,
          joinedVia: 'sso_jit',
        },
      },
    },
    include: { memberships: true },
  });

  // 監査ログ
  await logAuthEvent({
    type: 'jit_user_created',
    userId: user.id,
    orgId: data.orgId,
    email: data.email,
    role,
    groups: data.groups,
  });

  // 組織管理者に通知
  await notifyOrgAdmins(data.orgId, {
    type: 'new_user_via_jit',
    message: `${data.email} が SSO 経由で自動作成されました`,
    userId: user.id,
  });

  return user;
}

// グループからロールへのマッピング
function mapGroupsToRole(groups: string[], orgId: string): string {
  // 組織固有のマッピングルールを取得
  // 例: Okta のグループ "MyApp-Admins" → admin ロール
  const mappingRules = [
    { pattern: /admin/i, role: 'admin' },
    { pattern: /manager/i, role: 'manager' },
    { pattern: /editor/i, role: 'editor' },
    { pattern: /viewer/i, role: 'viewer' },
  ];

  for (const rule of mappingRules) {
    if (groups.some((g) => rule.pattern.test(g))) {
      return rule.role;
    }
  }

  return 'member'; // デフォルトロール
}
```

---

## 6. SCIM（ディレクトリ連携）

```
SCIM（System for Cross-domain Identity Management）:

  IdP のユーザーディレクトリと自社アプリを同期:
  → ユーザーの自動作成（プロビジョニング）
  → ユーザーの自動無効化（デプロビジョニング）
  → グループメンバーシップの同期
  → 属性変更の同期（名前変更、部署変更等）

  JIT プロビジョニングとの違い:

  ┌──────────────────┬──────────────────────┬──────────────────────┐
  │ 項目             │ JIT プロビジョニング   │ SCIM                 │
  ├──────────────────┼──────────────────────┼──────────────────────┤
  │ タイミング       │ ログイン時            │ IdP での変更時         │
  │ 方向             │ IdP → SP（ログイン時） │ IdP → SP（随時）      │
  │ ユーザー作成     │ ✓                    │ ✓                    │
  │ ユーザー無効化   │ ✗（ログインしなければ│ ✓（即時無効化）        │
  │                 │   検知できない）       │                      │
  │ 属性更新        │ △（ログイン時のみ）    │ ✓（即時反映）          │
  │ グループ同期    │ △（ログイン時のみ）    │ ✓（即時反映）          │
  │ 実装コスト      │ 低い                  │ 高い                  │
  │ 推奨           │ スタートアップ          │ エンタープライズ       │
  └──────────────────┴──────────────────────┴──────────────────────┘

  SCIM API エンドポイント（RFC 7644）:

    GET    /scim/v2/Users           — ユーザー一覧
    POST   /scim/v2/Users           — ユーザー作成
    GET    /scim/v2/Users/:id       — ユーザー取得
    PUT    /scim/v2/Users/:id       — ユーザー更新（全属性）
    PATCH  /scim/v2/Users/:id       — ユーザー部分更新
    DELETE /scim/v2/Users/:id       — ユーザー削除

    GET    /scim/v2/Groups          — グループ一覧
    POST   /scim/v2/Groups          — グループ作成
    GET    /scim/v2/Groups/:id      — グループ取得
    PUT    /scim/v2/Groups/:id      — グループ更新
    PATCH  /scim/v2/Groups/:id      — グループ部分更新
    DELETE /scim/v2/Groups/:id      — グループ削除

    GET    /scim/v2/ServiceProviderConfig — SP 機能情報
    GET    /scim/v2/Schemas         — スキーマ定義
    GET    /scim/v2/ResourceTypes   — リソースタイプ
```

```typescript
// SCIM エンドポイントの実装

import express from 'express';

const scimRouter = express.Router();

// SCIM 認証ミドルウェア
async function scimAuth(
  req: express.Request,
  res: express.Response,
  next: express.NextFunction
) {
  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return res.status(401).json({
      schemas: ['urn:ietf:params:scim:api:messages:2.0:Error'],
      detail: 'Authentication required',
      status: '401',
    });
  }

  const token = authHeader.replace('Bearer ', '');

  // トークンから組織を特定
  const org = await findOrgByScimToken(token);
  if (!org) {
    return res.status(401).json({
      schemas: ['urn:ietf:params:scim:api:messages:2.0:Error'],
      detail: 'Invalid SCIM token',
      status: '401',
    });
  }

  (req as any).orgId = org.id;
  next();
}

scimRouter.use(scimAuth);

// SCIM ユーザーへの変換
function toScimUser(user: User & { memberships: OrganizationMember[] }): any {
  return {
    schemas: ['urn:ietf:params:scim:schemas:core:2.0:User'],
    id: user.id,
    externalId: user.externalId,
    userName: user.email,
    name: {
      givenName: user.name?.split(' ')[0] || '',
      familyName: user.name?.split(' ').slice(1).join(' ') || '',
      formatted: user.name || '',
    },
    emails: [{
      value: user.email,
      type: 'work',
      primary: true,
    }],
    active: user.active,
    groups: user.memberships.map((m) => ({
      value: m.teamId,
      display: m.teamName,
    })),
    meta: {
      resourceType: 'User',
      created: user.createdAt.toISOString(),
      lastModified: user.updatedAt.toISOString(),
      location: `https://myapp.com/scim/v2/Users/${user.id}`,
    },
  };
}

// ユーザー一覧
scimRouter.get('/Users', async (req, res) => {
  const orgId = (req as any).orgId;
  const startIndex = parseInt(req.query.startIndex as string) || 1;
  const count = Math.min(parseInt(req.query.count as string) || 100, 200);

  // フィルター解析（例: filter=userName eq "alice@example.com"）
  const filter = req.query.filter as string;
  let where: any = {
    memberships: { some: { orgId } },
  };

  if (filter) {
    const match = filter.match(/userName eq "(.+)"/);
    if (match) {
      where.email = match[1];
    }
  }

  const [users, totalResults] = await Promise.all([
    prisma.user.findMany({
      where,
      include: { memberships: { where: { orgId } } },
      skip: startIndex - 1,
      take: count,
    }),
    prisma.user.count({ where }),
  ]);

  res.json({
    schemas: ['urn:ietf:params:scim:api:messages:2.0:ListResponse'],
    totalResults,
    startIndex,
    itemsPerPage: count,
    Resources: users.map(toScimUser),
  });
});

// ユーザー作成
scimRouter.post('/Users', async (req, res) => {
  const orgId = (req as any).orgId;
  const scimUser = req.body;

  // 既存ユーザーチェック
  const existingUser = await prisma.user.findFirst({
    where: {
      email: scimUser.userName || scimUser.emails?.[0]?.value,
      memberships: { some: { orgId } },
    },
  });

  if (existingUser) {
    return res.status(409).json({
      schemas: ['urn:ietf:params:scim:api:messages:2.0:Error'],
      detail: 'User already exists',
      status: '409',
      scimType: 'uniqueness',
    });
  }

  const user = await prisma.user.create({
    data: {
      externalId: scimUser.externalId,
      email: (scimUser.userName || scimUser.emails?.[0]?.value).toLowerCase(),
      name: scimUser.name?.formatted ||
        `${scimUser.name?.givenName || ''} ${scimUser.name?.familyName || ''}`.trim(),
      active: scimUser.active ?? true,
      emailVerified: true,
      memberships: {
        create: {
          orgId,
          role: 'member',
          joinedVia: 'scim',
        },
      },
    },
    include: { memberships: { where: { orgId } } },
  });

  await logAuditEvent({
    type: 'scim_user_created',
    orgId,
    userId: user.id,
    email: user.email,
  });

  res.status(201).json(toScimUser(user));
});

// ユーザー取得
scimRouter.get('/Users/:id', async (req, res) => {
  const orgId = (req as any).orgId;

  const user = await prisma.user.findFirst({
    where: {
      id: req.params.id,
      memberships: { some: { orgId } },
    },
    include: { memberships: { where: { orgId } } },
  });

  if (!user) {
    return res.status(404).json({
      schemas: ['urn:ietf:params:scim:api:messages:2.0:Error'],
      detail: 'User not found',
      status: '404',
    });
  }

  res.json(toScimUser(user));
});

// ユーザー部分更新（PATCH）
scimRouter.patch('/Users/:id', async (req, res) => {
  const orgId = (req as any).orgId;
  const operations = req.body.Operations || [];

  const user = await prisma.user.findFirst({
    where: {
      id: req.params.id,
      memberships: { some: { orgId } },
    },
  });

  if (!user) {
    return res.status(404).json({
      schemas: ['urn:ietf:params:scim:api:messages:2.0:Error'],
      detail: 'User not found',
      status: '404',
    });
  }

  const updateData: any = {};

  for (const op of operations) {
    switch (op.op.toLowerCase()) {
      case 'replace': {
        if (op.path === 'active') {
          updateData.active = op.value;

          if (op.value === false) {
            // ユーザー無効化（デプロビジョニング）
            updateData.deactivatedAt = new Date();

            // 全セッションを無効化
            await sessionManager.destroyAllForUser(user.id);

            // Remember Me トークンを全て無効化
            await rememberMeManager.revokeAllForUser(user.id);

            await logAuditEvent({
              type: 'scim_user_deactivated',
              orgId,
              userId: user.id,
              email: user.email,
            });
          } else {
            // ユーザー再有効化
            updateData.deactivatedAt = null;

            await logAuditEvent({
              type: 'scim_user_reactivated',
              orgId,
              userId: user.id,
              email: user.email,
            });
          }
        }

        if (op.path === 'userName') {
          updateData.email = op.value.toLowerCase();
        }

        if (op.path === 'name.givenName' || op.path === 'name.familyName') {
          // 名前の更新は完全な名前を再構築
          const currentName = user.name?.split(' ') || ['', ''];
          if (op.path === 'name.givenName') currentName[0] = op.value;
          if (op.path === 'name.familyName') currentName[1] = op.value;
          updateData.name = currentName.join(' ').trim();
        }
        break;
      }

      case 'add': {
        // グループへの追加等
        break;
      }

      case 'remove': {
        // グループからの削除等
        break;
      }
    }
  }

  const updatedUser = await prisma.user.update({
    where: { id: user.id },
    data: updateData,
    include: { memberships: { where: { orgId } } },
  });

  res.json(toScimUser(updatedUser));
});

// ユーザー削除（完全削除 or 無効化）
scimRouter.delete('/Users/:id', async (req, res) => {
  const orgId = (req as any).orgId;

  const user = await prisma.user.findFirst({
    where: {
      id: req.params.id,
      memberships: { some: { orgId } },
    },
  });

  if (!user) {
    return res.status(404).json({
      schemas: ['urn:ietf:params:scim:api:messages:2.0:Error'],
      detail: 'User not found',
      status: '404',
    });
  }

  // ソフトデリート（推奨: 完全削除ではなく無効化）
  await prisma.user.update({
    where: { id: user.id },
    data: {
      active: false,
      deactivatedAt: new Date(),
    },
  });

  // 全セッションを無効化
  await sessionManager.destroyAllForUser(user.id);

  await logAuditEvent({
    type: 'scim_user_deleted',
    orgId,
    userId: user.id,
    email: user.email,
  });

  res.status(204).send();
});

// ServiceProviderConfig（SCIM サーバーの機能情報）
scimRouter.get('/ServiceProviderConfig', (req, res) => {
  res.json({
    schemas: ['urn:ietf:params:scim:schemas:core:2.0:ServiceProviderConfig'],
    documentationUri: 'https://myapp.com/docs/scim',
    patch: { supported: true },
    bulk: { supported: false, maxOperations: 0, maxPayloadSize: 0 },
    filter: { supported: true, maxResults: 200 },
    changePassword: { supported: false },
    sort: { supported: false },
    etag: { supported: false },
    authenticationSchemes: [{
      type: 'oauthbearertoken',
      name: 'OAuth Bearer Token',
      description: 'Authentication scheme using the OAuth Bearer Token Standard',
    }],
  });
});

// ルーターのマウント
app.use('/scim/v2', scimRouter);
```

---

## 7. エンタープライズ認証のセキュリティ要件

```
B2B SaaS の認証チェックリスト:

  SSO:
  ✓ SAML 2.0 対応（SP-Initiated + IdP-Initiated）
  ✓ OIDC 対応（Authorization Code Flow + PKCE）
  ✓ テナント別 SSO 設定
  ✓ SSO 強制オプション（パスワード無効化）
  ✓ IdP Metadata の自動取得（URL指定）
  ✓ 証明書ローテーション（ダウンタイムなし）
  ✓ SSO 接続テスト機能

  ディレクトリ:
  ✓ SCIM プロビジョニング
  ✓ 退職者の自動無効化（デプロビジョニング）
  ✓ グループ/ロール同期
  ✓ JIT プロビジョニング

  セキュリティ:
  ✓ MFA の組織レベル強制
  ✓ セッション有効期限の組織設定
  ✓ IP ホワイトリスト
  ✓ 監査ログ（ログイン/ログアウト/権限変更）
  ✓ API キー管理
  ✓ パスワードポリシーのカスタマイズ

  コンプライアンス:
  ✓ SOC 2 Type II
  ✓ GDPR 対応（データ削除要求）
  ✓ HIPAA 対応（医療向け）
  ✓ データ保管場所の選択（リージョン指定）
  ✓ データ暗号化（保存時/転送時）
  ✓ 保持期間ポリシー
```

### 7.1 組織レベルの MFA 強制

```typescript
// 組織レベルの MFA 強制
app.use('/api', async (req, res, next) => {
  const session = await getSession(req);
  if (!session) return next();

  // 組織のセキュリティ設定を取得
  const membership = await prisma.organizationMember.findFirst({
    where: { userId: session.userId },
    include: { org: true },
  });

  if (!membership) return next();

  const org = membership.org;

  // 1. MFA 強制チェック
  if (org.mfaRequired) {
    const user = await prisma.user.findUnique({
      where: { id: session.userId },
    });

    if (!user?.mfaEnabled) {
      return res.status(403).json({
        error: 'MFA required',
        code: 'MFA_SETUP_REQUIRED',
        message: 'Your organization requires MFA. Please set up MFA to continue.',
        setupUrl: '/settings/security/mfa',
      });
    }
  }

  // 2. IP ホワイトリストチェック
  if (org.ipAllowlist.length > 0) {
    const clientIP = getClientIP(req);
    const isAllowed = org.ipAllowlist.some((allowed: string) => {
      if (allowed.includes('/')) {
        // CIDR 表記
        return isIPInCIDR(clientIP, allowed);
      }
      return clientIP === allowed;
    });

    if (!isAllowed) {
      await logSecurityEvent({
        type: 'ip_blocked',
        userId: session.userId,
        orgId: org.id,
        ip: clientIP,
        allowedIPs: org.ipAllowlist,
      });

      return res.status(403).json({
        error: 'Access denied',
        code: 'IP_NOT_ALLOWED',
        message: 'Your IP address is not in the organization allowlist.',
      });
    }
  }

  // 3. セッション最大時間チェック
  if (org.sessionMaxAge) {
    const sessionAge = Date.now() - session.createdAt;
    if (sessionAge > org.sessionMaxAge * 1000) {
      await sessionManager.destroy(req.cookies['__Host-session_id']);
      return res.status(401).json({
        error: 'Session expired',
        code: 'SESSION_EXPIRED_BY_ORG_POLICY',
        message: 'Your session has expired per organization policy.',
      });
    }
  }

  next();
});
```

### 7.2 監査ログ

```typescript
// エンタープライズ向け監査ログ

interface AuditLogEntry {
  id: string;
  timestamp: Date;
  orgId: string;
  actorId: string;         // 操作者（ユーザー or システム）
  actorType: 'user' | 'admin' | 'system' | 'scim' | 'api';
  action: string;          // 'login', 'logout', 'create_user', etc.
  resource: string;        // 'session', 'user', 'team', etc.
  resourceId?: string;
  details: Record<string, unknown>;
  ipAddress: string;
  userAgent: string;
  success: boolean;
  errorReason?: string;
}

class AuditLogger {
  // 監査ログの記録
  async log(entry: Omit<AuditLogEntry, 'id' | 'timestamp'>): Promise<void> {
    await prisma.auditLog.create({
      data: {
        ...entry,
        details: entry.details as any,
        timestamp: new Date(),
      },
    });

    // リアルタイム通知（セキュリティイベント）
    if (this.isSecurityEvent(entry.action)) {
      await this.notifySecurityTeam(entry);
    }
  }

  // 監査ログの検索（管理画面用）
  async search(orgId: string, filters: {
    startDate?: Date;
    endDate?: Date;
    actorId?: string;
    action?: string;
    resource?: string;
    success?: boolean;
    limit?: number;
    cursor?: string;
  }): Promise<{ entries: AuditLogEntry[]; nextCursor?: string }> {
    const where: any = { orgId };

    if (filters.startDate || filters.endDate) {
      where.timestamp = {};
      if (filters.startDate) where.timestamp.gte = filters.startDate;
      if (filters.endDate) where.timestamp.lte = filters.endDate;
    }
    if (filters.actorId) where.actorId = filters.actorId;
    if (filters.action) where.action = filters.action;
    if (filters.resource) where.resource = filters.resource;
    if (filters.success !== undefined) where.success = filters.success;

    const limit = Math.min(filters.limit || 50, 200);

    if (filters.cursor) {
      where.id = { lt: filters.cursor };
    }

    const entries = await prisma.auditLog.findMany({
      where,
      orderBy: { timestamp: 'desc' },
      take: limit + 1,
    });

    const hasMore = entries.length > limit;
    if (hasMore) entries.pop();

    return {
      entries: entries as AuditLogEntry[],
      nextCursor: hasMore ? entries[entries.length - 1].id : undefined,
    };
  }

  // 監査ログのエクスポート（CSV）
  async exportCSV(orgId: string, startDate: Date, endDate: Date): Promise<string> {
    const entries = await prisma.auditLog.findMany({
      where: {
        orgId,
        timestamp: { gte: startDate, lte: endDate },
      },
      orderBy: { timestamp: 'asc' },
    });

    const header = 'Timestamp,Actor,Action,Resource,IP Address,Success,Details\n';
    const rows = entries.map((e) =>
      `${e.timestamp.toISOString()},${e.actorId},${e.action},${e.resource},${e.ipAddress},${e.success},"${JSON.stringify(e.details).replace(/"/g, '""')}"`
    ).join('\n');

    return header + rows;
  }

  private isSecurityEvent(action: string): boolean {
    return [
      'login_failed',
      'mfa_failed',
      'ip_blocked',
      'session_hijack_detected',
      'scim_user_deactivated',
      'sso_config_changed',
      'admin_role_granted',
    ].includes(action);
  }

  private async notifySecurityTeam(
    entry: Omit<AuditLogEntry, 'id' | 'timestamp'>
  ): Promise<void> {
    // Slack / PagerDuty / メール等への通知
    // 組織のセキュリティ通知設定に基づく
  }
}

// 監査ログ API
app.get('/api/admin/audit-logs', requireOrgAdmin, async (req, res) => {
  const auditLogger = new AuditLogger();
  const result = await auditLogger.search(req.orgId, {
    startDate: req.query.startDate ? new Date(req.query.startDate as string) : undefined,
    endDate: req.query.endDate ? new Date(req.query.endDate as string) : undefined,
    actorId: req.query.actorId as string,
    action: req.query.action as string,
    limit: parseInt(req.query.limit as string) || 50,
    cursor: req.query.cursor as string,
  });

  res.json(result);
});
```

---

## 8. 認証 SaaS の活用

```
認証 SaaS の比較:

  ┌────────────┬──────┬──────┬────────────────┬──────────────────────┐
  │ サービス    │ SSO  │ SCIM │ 価格           │ 特徴                  │
  ├────────────┼──────┼──────┼────────────────┼──────────────────────┤
  │ Auth0      │ ✓    │ ✓    │ 高い($23k+/年) │ 最も機能豊富           │
  │ WorkOS     │ ✓    │ ✓    │ 中($49/接続/月) │ エンタープライズ SSO 特化│
  │ Clerk      │ ✓    │ △    │ 中($0.02/MAU)  │ React UI コンポーネント │
  │ Kinde      │ ✓    │ △    │ 安い($0/～)     │ 新興、DX 良好          │
  │ Keycloak   │ ✓    │ ✓    │ 無料(自前運用)  │ セルフホスト           │
  │ Stytch     │ ✓    │ ✓    │ 中             │ B2B 認証に強い         │
  │ FusionAuth │ ✓    │ ✓    │ 中(自前運用可)  │ 柔軟なカスタマイズ     │
  └────────────┴──────┴──────┴────────────────┴──────────────────────┘

  選定ガイド:

  ┌─────────────────────────────────────────────────────────┐
  │                                                         │
  │  スタートアップ（～シリーズA）:                             │
  │    → Auth.js（無料）でメール/パスワード + ソーシャルログイン │
  │    → SSO が必要になったら WorkOS を追加                    │
  │    → 理由: 初期コスト最小、必要に応じてスケール             │
  │                                                         │
  │  成長期（シリーズB～）:                                    │
  │    → Clerk or Auth0 でフルスタック認証                     │
  │    → SCIM 対応が必要なら WorkOS or Auth0                  │
  │    → 理由: 開発リソースを製品に集中                        │
  │                                                         │
  │  エンタープライズ:                                        │
  │    → Auth0（フルスタック）or WorkOS（SSO/SCIM特化）        │
  │    → セルフホスト要件 → Keycloak or FusionAuth            │
  │    → 理由: コンプライアンス、SLA、サポート                  │
  │                                                         │
  │  セルフホスト要件:                                        │
  │    → Keycloak（Java）or Authentik（Python）               │
  │    → 理由: データ主権、規制要件                            │
  │                                                         │
  └─────────────────────────────────────────────────────────┘
```

```typescript
// WorkOS を使った SSO 実装（最小限のコード）
// npm install @workos-inc/node

import WorkOS from '@workos-inc/node';

const workos = new WorkOS(process.env.WORKOS_API_KEY!);

// SSO ログイン開始
app.get('/api/auth/sso/login', async (req, res) => {
  const { email } = req.query;

  // WorkOS が自動的にドメインから組織を特定
  const authorizationUrl = workos.sso.getAuthorizationURL({
    clientId: process.env.WORKOS_CLIENT_ID!,
    // ドメインベース or 接続 ID ベース
    domain: email ? (email as string).split('@')[1] : undefined,
    connection: req.query.connectionId as string,
    redirectUri: 'https://myapp.com/api/auth/sso/callback',
    state: JSON.stringify({
      redirectTo: req.query.redirectTo || '/dashboard',
    }),
  });

  res.redirect(authorizationUrl);
});

// SSO コールバック
app.get('/api/auth/sso/callback', async (req, res) => {
  const { code, state } = req.query;
  const { redirectTo } = JSON.parse(state as string);

  try {
    // WorkOS がプロファイル取得と検証を一括処理
    const { profile } = await workos.sso.getProfileAndToken({
      clientId: process.env.WORKOS_CLIENT_ID!,
      code: code as string,
    });

    // profile: { id, email, first_name, last_name, organization_id, ... }

    const user = await findOrCreateUser({
      email: profile.email,
      name: `${profile.first_name} ${profile.last_name}`,
      externalId: profile.id,
      orgExternalId: profile.organization_id,
    });

    const { sessionId } = await sessionManager.create(
      { userId: user.id, role: user.role },
      req as any
    );

    setSessionCookie(res, sessionId);
    res.redirect(redirectTo || '/dashboard');
  } catch (error) {
    console.error('WorkOS SSO error:', error);
    res.redirect('/login?error=sso_failed');
  }
});

// WorkOS SCIM ディレクトリ同期（Webhook）
app.post('/api/webhooks/workos', async (req, res) => {
  const payload = workos.webhooks.constructEvent({
    payload: req.body,
    sigHeader: req.headers['workos-signature'] as string,
    secret: process.env.WORKOS_WEBHOOK_SECRET!,
  });

  switch (payload.event) {
    case 'dsync.user.created': {
      const { data } = payload;
      await prisma.user.create({
        data: {
          email: data.emails[0].value,
          name: `${data.first_name} ${data.last_name}`,
          externalId: data.id,
          active: data.state === 'active',
          emailVerified: true,
        },
      });
      break;
    }

    case 'dsync.user.deleted':
    case 'dsync.user.deactivated': {
      const { data } = payload;
      await prisma.user.update({
        where: { externalId: data.id },
        data: { active: false, deactivatedAt: new Date() },
      });
      await sessionManager.destroyAllForUser(data.id);
      break;
    }

    case 'dsync.group.user_added':
    case 'dsync.group.user_removed': {
      // グループメンバーシップの更新
      break;
    }
  }

  res.json({ received: true });
});
```

---

## 9. アンチパターン

```
エンタープライズ認証のアンチパターン:

  ❌ アンチパターン1: SSO を後付けで実装
     → マルチテナントを考慮せずに認証を実装
     → SSO 追加時に大規模リファクタリングが必要
     ○ 正しい方法: 初期設計からテナント別認証を考慮

  ❌ アンチパターン2: SAML 署名の検証を省略
     → 開発時に署名検証を無効化 → 本番にそのまま投入
     → IdP のなりすましが可能になる
     ○ 正しい方法: 必ず XML 署名を検証、テストでも有効化

  ❌ アンチパターン3: SCIM トークンを平文保存
     → SCIM の Bearer トークンを DB に平文で保存
     → DB 漏洩時にディレクトリ連携が悪用される
     ○ 正しい方法: トークンはハッシュ化して保存

  ❌ アンチパターン4: SSO ユーザーにパスワードリセットを許可
     → SSO 強制の組織でパスワードリセットが可能
     → SSO バイパスのセキュリティホール
     ○ 正しい方法: SSO 強制時はパスワード関連機能を無効化

  ❌ アンチパターン5: 退職者のセッションを放置
     → SCIM でユーザーを無効化したがセッションは有効のまま
     → 退職者が引き続きアクセス可能
     ○ 正しい方法: 無効化時に全セッションを即座に破棄
```

---

## 10. エッジケース

```
SSO のエッジケース:

  ① IdP 障害時のフォールバック:
     → IdP がダウン → SSO ログイン不可
     → 対策: 緊急アクセス用の「ブレークグラス」アカウント
     → 組織管理者にのみバックアップパスワードを許可
     → IdP 障害の自動検知と通知

  ② 証明書ローテーション中の検証エラー:
     → IdP が新しい証明書に切り替え → 旧証明書での検証失敗
     → 対策: Metadata URL を定期取得して証明書を自動更新
     → 複数の証明書を同時に信頼する猶予期間

  ③ ドメインの所有権変更:
     → 企業買収で example.com のドメイン所有者が変わった
     → 旧組織のユーザーが新組織の SSO で認証される可能性
     → 対策: ドメイン所有権の定期検証（DNS TXT レコード）

  ④ 複数組織に所属するユーザー:
     → alice@consultant.com が複数の顧客組織に所属
     → どの組織の SSO でログインすべきか？
     → 対策: 組織選択画面を表示、または組織ごとに異なる URL

  ⑤ SSO 設定ミスによるロックアウト:
     → 組織管理者が誤った SSO 設定を保存
     → 全ユーザーがログイン不能
     → 対策: SSO 設定変更前にテスト接続を必須化
     → バックドアとしてのスーパー管理者アカウント
     → SSO 設定の即時ロールバック機能

  ⑥ SAML Response のリプレイ攻撃:
     → 攻撃者が過去の SAMLResponse を再利用
     → 対策: InResponseTo の検証、Assertion の有効期限チェック
     → 一度使用した AssertionID の記録（リプレイ防止）
```

```typescript
// ブレークグラス（緊急アクセス）の実装
app.post('/api/auth/break-glass', async (req, res) => {
  const { email, password, breakGlassCode } = req.body;

  // ブレークグラスコードの検証（事前に組織管理者に配布）
  const org = await getOrgByEmailDomain(email);
  if (!org) {
    return res.status(400).json({ error: 'Organization not found' });
  }

  const isValidCode = await verifyBreakGlassCode(org.id, breakGlassCode);
  if (!isValidCode) {
    await logSecurityEvent({
      type: 'break_glass_failed',
      email,
      orgId: org.id,
      ip: getClientIP(req),
      severity: 'critical',
    });
    return res.status(403).json({ error: 'Invalid break glass code' });
  }

  // 通常の認証（SSO バイパス）
  const user = await authenticateWithPassword(email, password);
  if (!user) {
    return res.status(401).json({ error: 'Invalid credentials' });
  }

  // セッション作成（短い有効期限）
  const { sessionId } = await sessionManager.create(
    { userId: user.id, role: user.role },
    req
  );

  // 全管理者に緊急通知
  await notifyAllAdmins({
    type: 'break_glass_used',
    email,
    orgId: org.id,
    ip: getClientIP(req),
    severity: 'critical',
    message: `Emergency access used by ${email}. SSO may be down.`,
  });

  await logAuditEvent({
    type: 'break_glass_login',
    orgId: org.id,
    userId: user.id,
    ip: getClientIP(req),
  });

  setSessionCookie(res, sessionId);
  res.json({ success: true, warning: 'Emergency access logged and monitored' });
});
```

---

## 11. 演習問題

### 演習1（基礎）: テナント別 SSO 判定の実装

```
課題:
  以下を実装せよ:
  1. メールアドレスのドメインから組織を特定する関数
  2. 組織の SSO 設定に基づいて認証方法を判定する API
  3. SSO 強制時にパスワードログインを拒否する処理

検証ポイント:
  - ドメイン検索が大文字/小文字を区別しないか
  - SSO 未設定の組織ではパスワードログインが許可されるか
  - SSO 強制時にパスワードログインが適切に拒否されるか
```

### 演習2（応用）: SAML SSO の実装

```
課題:
  samlify ライブラリを使って以下を実装せよ:
  1. SP Metadata の生成と配信エンドポイント
  2. SP-Initiated SSO ログインフロー
  3. SAML Assertion の検証とユーザー作成（JIT プロビジョニング）
  4. テナント別の IdP 設定管理

検証ポイント:
  - XML 署名の検証が有効になっているか
  - NameID とドメインの一致確認が行われているか
  - JIT プロビジョニング時にグループからロールがマッピングされるか
  - 監査ログが記録されているか
```

### 演習3（発展）: SCIM プロビジョニングの実装

```
課題:
  RFC 7644 準拠の SCIM エンドポイントを実装せよ:
  1. Bearer トークン認証
  2. CRUD エンドポイント（Users）
  3. PATCH 操作によるユーザー無効化（デプロビジョニング）
  4. 無効化時の全セッション破棄
  5. フィルター対応（userName eq "..."）

検証ポイント:
  - SCIM トークンがハッシュ化されて保存されているか
  - デプロビジョニング時に全セッションが破棄されるか
  - SCIM レスポンスが RFC 7644 のスキーマに準拠しているか
  - エラーレスポンスが SCIM 標準形式か
```

---

## 12. FAQ / トラブルシューティング

```
Q: SAML ログインで「Signature validation failed」エラーが出る
A: 以下をチェック:
   1. IdP の証明書が正しく設定されているか（PEM 形式、ヘッダー/フッター含む）
   2. 証明書が期限切れでないか
   3. IdP が Response と Assertion のどちらに署名しているか確認
   4. 署名アルゴリズムが一致しているか（SHA-1 vs SHA-256）
   5. IdP の Metadata を再取得して更新

Q: OIDC SSO で「Invalid redirect_uri」エラーが出る
A: 以下をチェック:
   1. IdP に登録した redirect_uri と完全一致しているか（末尾のスラッシュ含む）
   2. https:// が正しく使用されているか
   3. ポート番号が一致しているか（開発環境で localhost:3000 等）

Q: SCIM でユーザーが二重作成される
A: 以下をチェック:
   1. externalId で既存ユーザーを検索しているか
   2. email の大文字/小文字を正規化しているか（.toLowerCase()）
   3. IdP 側で「User uniqueness」の設定が正しいか
   4. 同時リクエストの排他制御（Unique 制約、トランザクション）

Q: SSO 設定後に管理者がログインできなくなった
A: 対策:
   1. ブレークグラスアカウントで緊急アクセス
   2. DB で直接 ssoEnabled を false に変更
   3. 事前にスーパー管理者のパスワードログインを維持
   4. SSO 設定変更前にテスト接続を必須化

Q: IdP-Initiated SSO でユーザーが「Invalid RelayState」エラーになる
A: IdP-Initiated SSO では SP が RelayState を設定しないため:
   1. RelayState が空/null の場合のデフォルト処理を追加
   2. IdP-Initiated SSO のコールバック URL を別途用意
   3. IdP 側で RelayState にデフォルト値を設定

Q: SCIM トークンのローテーションはどう行うべきか
A: 以下の手順:
   1. 新しいトークンを生成して DB に保存
   2. IdP の SCIM 設定を新しいトークンに更新
   3. 旧トークンを猶予期間（例: 24時間）有効に保持
   4. 猶予期間後に旧トークンを無効化
   5. ダウンタイムなしのローテーションが可能
```

---

## まとめ

| 項目 | ポイント |
|------|---------|
| SSO プロトコル | SAML 2.0（エンタープライズ標準）+ OIDC（モダン） |
| テナント別 | ドメインから組織特定 → SSO 判定 → IdP リダイレクト |
| JIT プロビジョニング | SSO ログイン時にユーザー自動作成 + グループロールマッピング |
| SCIM | 自動プロビジョニング/デプロビジョニング、即時セッション無効化 |
| セキュリティ | MFA 強制、IP ホワイトリスト、監査ログ、証明書管理 |
| 緊急対応 | ブレークグラスアカウント、IdP 障害時のフォールバック |
| 認証 SaaS | WorkOS（SSO/SCIM 特化）、Auth0（フルスタック）、Keycloak（セルフホスト） |

---

## 次に読むべきガイド

- [[../02-token-auth/02-openid-connect.md]] — OpenID Connect の詳細実装
- [[../01-session-auth/00-cookie-and-session.md]] — セッション管理のベストプラクティス
- [[../03-authorization/01-abac-and-policies.md]] — テナント別の認可ポリシー
- [[02-email-password-auth.md]] — SSO フォールバック時のパスワード認証

---

## 参考文献

1. OASIS. "SAML 2.0 Technical Overview." docs.oasis-open.org, 2008.
2. RFC 7644. "System for Cross-domain Identity Management: Protocol." IETF, 2015.
3. RFC 7643. "System for Cross-domain Identity Management: Core Schema." IETF, 2015.
4. OpenID Foundation. "OpenID Connect Core 1.0." openid.net, 2014.
5. WorkOS. "Enterprise SSO." workos.com/docs, 2024.
6. Auth0. "Enterprise Connections." auth0.com/docs, 2024.
7. Okta. "SAML 2.0 Overview." developer.okta.com, 2024.
8. Microsoft. "Azure AD SAML Protocol." learn.microsoft.com, 2024.
9. OWASP. "SAML Security Cheat Sheet." cheatsheetseries.owasp.org, 2024.
10. NIST SP 800-63C. "Digital Identity Guidelines: Federation and Assertions." NIST, 2017.
