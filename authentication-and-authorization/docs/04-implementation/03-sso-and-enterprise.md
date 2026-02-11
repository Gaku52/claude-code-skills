# SSO とエンタープライズ認証

> B2B SaaS では SSO（Single Sign-On）対応がエンタープライズ契約の必須要件。SAML 2.0、OIDC ベースの SSO、ディレクトリ連携（SCIM）、テナント別認証設定まで、エンタープライズ認証の全体像を解説する。

## この章で学ぶこと

- [ ] SSO の概念と SAML / OIDC の違いを理解する
- [ ] SAML 2.0 の認証フローを把握する
- [ ] エンタープライズ向けの認証設計を学ぶ

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
  企業側: ユーザー管理の一元化、退職時の即時アクセス無効化
  ユーザー側: パスワード記憶の削減、利便性向上
  セキュリティ: MFA の一元適用、パスワード漏洩リスク低減

SSO プロトコル:
  SAML 2.0:  エンタープライズの標準（XML ベース）
  OIDC:      モダンな標準（JSON ベース）
  LDAP:      ディレクトリサービス（社内ネットワーク）
```

---

## 2. SAML 2.0

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

  SAML の構成要素:
    → Assertion: ユーザー情報を含む XML（署名付き）
    → Metadata: SP/IdP の設定情報（エンドポイント、証明書）
    → Binding: 通信方式（HTTP-Redirect, HTTP-POST）
```

```typescript
// SAML 2.0 実装（samlify ライブラリ）
// npm install samlify

import * as samlify from 'samlify';

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
});

// IdP の設定（Okta の Metadata から）
const idp = samlify.IdentityProvider({
  metadata: fs.readFileSync('./idp-metadata.xml', 'utf8'),
  // または個別指定
  // entityID: 'https://company.okta.com/...',
  // singleSignOnService: [{ ... }],
  // signingCert: '...',
});

// SP Metadata エンドポイント（IdP に提供する）
app.get('/api/auth/saml/metadata', (req, res) => {
  res.type('application/xml');
  res.send(sp.getMetadata());
});

// SAML ログイン開始
app.get('/api/auth/saml/login', (req, res) => {
  const { context } = sp.createLoginRequest(idp, 'redirect');
  res.redirect(context);
});

// SAML コールバック（Assertion 受信・検証）
app.post('/api/auth/saml/callback', async (req, res) => {
  try {
    const { extract } = await sp.parseLoginResponse(idp, 'post', { body: req.body });

    // ユーザー情報の取得
    const email = extract.nameID;
    const attributes = extract.attributes;
    // attributes: { firstName, lastName, groups, ... }

    // ユーザーの作成 or ログイン
    const user = await findOrCreateSAMLUser({
      email,
      name: `${attributes.firstName} ${attributes.lastName}`,
      samlNameId: extract.nameID,
      samlSessionIndex: extract.sessionIndex?.sessionIndex,
    });

    // セッション作成
    const session = await createSession(user.id);
    setSessionCookie(res, session);

    res.redirect('/dashboard');
  } catch (error) {
    console.error('SAML validation failed:', error);
    res.redirect('/login?error=saml_failed');
  }
});
```

---

## 3. テナント別 SSO 設定

```
マルチテナント SSO:

  テナント A: Okta で SSO
  テナント B: Azure AD で SSO
  テナント C: SSO なし（メール・パスワード）

  ログインフロー:
  ① ユーザーがメールアドレスを入力
  ② ドメインから組織を特定（alice@company-a.com → テナント A）
  ③ テナントの SSO 設定を確認
  ④ SSO が設定されている → IdP にリダイレクト
  ⑤ SSO が未設定 → パスワードログイン
```

```typescript
// テナント別 SSO 設定のデータモデル
// schema.prisma
// model Organization {
//   id              String  @id @default(cuid())
//   name            String
//   domain          String  @unique  // "company-a.com"
//   ssoEnabled      Boolean @default(false)
//   ssoProvider     String? // "saml" | "oidc"
//   ssoMetadataUrl  String? // IdP Metadata URL
//   ssoMetadataXml  String? // IdP Metadata XML（直接アップロード）
//   ssoClientId     String? // OIDC の場合
//   ssoClientSecret String? // OIDC の場合（暗号化保存）
//   ssoIssuer       String? // OIDC の issuer
//   enforceSSO      Boolean @default(false) // SSO を強制
// }

// ドメインから組織を特定
async function getOrgByEmailDomain(email: string) {
  const domain = email.split('@')[1];
  return prisma.organization.findUnique({
    where: { domain },
  });
}

// ログイン開始（メールアドレスから SSO 判定）
app.post('/api/auth/login/check', async (req, res) => {
  const { email } = req.body;
  const org = await getOrgByEmailDomain(email);

  if (org?.ssoEnabled) {
    // SSO にリダイレクト
    return res.json({
      method: 'sso',
      provider: org.ssoProvider,
      redirectUrl: `/api/auth/sso/${org.id}/login`,
    });
  }

  // パスワードログイン
  return res.json({ method: 'password' });
});

// SSO 管理画面（組織管理者用）
async function configureSSOForOrg(orgId: string, config: {
  provider: 'saml' | 'oidc';
  metadataUrl?: string;
  metadataXml?: string;
  clientId?: string;
  clientSecret?: string;
  issuer?: string;
}) {
  const session = await auth();
  if (!session) throw new Error('Unauthorized');

  // 組織管理者権限チェック
  const membership = await prisma.organizationMember.findUnique({
    where: { userId_orgId: { userId: session.user.id, orgId } },
  });

  if (membership?.role !== 'admin') {
    throw new Error('Only organization admins can configure SSO');
  }

  // SSO 設定を保存
  await prisma.organization.update({
    where: { id: orgId },
    data: {
      ssoEnabled: true,
      ssoProvider: config.provider,
      ssoMetadataUrl: config.metadataUrl,
      ssoMetadataXml: config.metadataXml,
      ssoClientId: config.clientId,
      ssoClientSecret: config.clientSecret ? encrypt(config.clientSecret) : undefined,
      ssoIssuer: config.issuer,
    },
  });
}
```

---

## 4. SCIM（ディレクトリ連携）

```
SCIM（System for Cross-domain Identity Management）:

  IdP のユーザーディレクトリと自社アプリを同期:
  → ユーザーの自動作成（プロビジョニング）
  → ユーザーの自動無効化（デプロビジョニング）
  → グループメンバーシップの同期
  → 属性変更の同期

  SCIM API エンドポイント:
    GET    /scim/v2/Users           — ユーザー一覧
    POST   /scim/v2/Users           — ユーザー作成
    GET    /scim/v2/Users/:id       — ユーザー取得
    PUT    /scim/v2/Users/:id       — ユーザー更新
    PATCH  /scim/v2/Users/:id       — ユーザー部分更新
    DELETE /scim/v2/Users/:id       — ユーザー削除
    GET    /scim/v2/Groups          — グループ一覧
    POST   /scim/v2/Groups          — グループ作成
```

```typescript
// SCIM エンドポイントの基本実装

// Bearer トークン認証
function scimAuth(req: Request, res: Response, next: Function) {
  const token = req.headers.authorization?.replace('Bearer ', '');
  if (!token || token !== process.env.SCIM_API_TOKEN) {
    return res.status(401).json({
      schemas: ['urn:ietf:params:scim:api:messages:2.0:Error'],
      detail: 'Authentication required',
      status: '401',
    });
  }
  next();
}

// ユーザー作成
app.post('/scim/v2/Users', scimAuth, async (req, res) => {
  const scimUser = req.body;

  const user = await prisma.user.create({
    data: {
      externalId: scimUser.externalId,
      email: scimUser.emails?.[0]?.value,
      name: `${scimUser.name?.givenName} ${scimUser.name?.familyName}`,
      active: scimUser.active ?? true,
      orgId: getOrgFromScimToken(req),
    },
  });

  res.status(201).json(toScimUser(user));
});

// ユーザー無効化（退職時の自動デプロビジョニング）
app.patch('/scim/v2/Users/:id', scimAuth, async (req, res) => {
  const operations = req.body.Operations;

  for (const op of operations) {
    if (op.op === 'replace' && op.path === 'active' && op.value === false) {
      // ユーザーを無効化
      await prisma.user.update({
        where: { externalId: req.params.id },
        data: {
          active: false,
          deactivatedAt: new Date(),
        },
      });

      // 全セッションを無効化
      await prisma.session.deleteMany({
        where: { user: { externalId: req.params.id } },
      });
    }
  }

  const user = await prisma.user.findUnique({
    where: { externalId: req.params.id },
  });

  res.json(toScimUser(user!));
});
```

---

## 5. エンタープライズ認証のチェックリスト

```
B2B SaaS の認証チェックリスト:

  SSO:
  ✓ SAML 2.0 対応
  ✓ OIDC 対応
  ✓ テナント別 SSO 設定
  ✓ SSO 強制オプション（パスワード無効化）
  ✓ IdP Metadata の自動取得（URL指定）

  ディレクトリ:
  ✓ SCIM プロビジョニング
  ✓ 退職者の自動無効化
  ✓ グループ/ロール同期

  セキュリティ:
  ✓ MFA の組織レベル強制
  ✓ セッション有効期限の組織設定
  ✓ IP ホワイトリスト
  ✓ 監査ログ（ログイン履歴）
  ✓ API キー管理

  コンプライアンス:
  ✓ SOC 2 Type II
  ✓ GDPR 対応
  ✓ データ保管場所の選択
  ✓ データ削除対応
```

---

## 6. 認証 SaaS の活用

```
認証 SaaS の比較:

  サービス     │ SSO  │ SCIM │ 価格           │ 特徴
  ────────────┼─────┼─────┼───────────────┼──────────
  Auth0       │ ✓   │ ✓   │ 高い           │ 最も機能豊富
  Clerk       │ ✓   │ △   │ 中             │ React UI コンポーネント
  WorkOS      │ ✓   │ ✓   │ 中             │ エンタープライズ特化
  Kinde       │ ✓   │ △   │ 安い           │ 新興、DX 良好
  Keycloak    │ ✓   │ ✓   │ 無料(自前運用)  │ セルフホスト

  選定ガイド:
    スタートアップ → Auth.js（無料）+ WorkOS（SSO のみ）
    中規模 → Clerk（UI付き）or Auth0
    エンタープライズ → Auth0 or WorkOS
    セルフホスト → Keycloak
```

---

## まとめ

| 項目 | ポイント |
|------|---------|
| SSO | SAML 2.0（エンタープライズ標準）+ OIDC |
| テナント別 | ドメインから組織特定→SSO 判定 |
| SCIM | 自動プロビジョニング/デプロビジョニング |
| セキュリティ | MFA 強制、監査ログ、IP 制限 |
| 認証 SaaS | WorkOS（SSO特化）、Auth0（フルスタック） |

---

## 参考文献
1. OASIS. "SAML 2.0 Technical Overview." docs.oasis-open.org, 2008.
2. RFC 7644. "System for Cross-domain Identity Management: Protocol." IETF, 2015.
3. WorkOS. "Enterprise SSO." workos.com/docs, 2024.
4. Auth0. "Enterprise Connections." auth0.com/docs, 2024.
