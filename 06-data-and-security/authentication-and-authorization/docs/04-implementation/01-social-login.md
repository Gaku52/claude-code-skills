# ソーシャルログイン

> Google、GitHub、Apple のソーシャルログインは現代の Web アプリに不可欠。各プロバイダーの OAuth 2.0 / OpenID Connect フロー、設定手順、アカウントリンク、プロフィール同期、プロバイダー固有の注意点、セキュリティ上の落とし穴を網羅的に解説する。

## この章で学ぶこと

- [ ] OAuth 2.0 / OpenID Connect の仕組みとソーシャルログインにおける役割を理解する
- [ ] 主要プロバイダー（Google、GitHub、Apple）の設定と固有の制約を把握する
- [ ] アカウントリンクとプロバイダー切り替えを安全に実装する
- [ ] プロバイダー固有のエッジケースとトラブルシューティングを学ぶ
- [ ] ソーシャルログインの UX 最適化とセキュリティ強化を実践する

### 前提知識

- OAuth 2.0 の基本概念（Authorization Code Grant）
- Next.js App Router と Auth.js v5 の基礎（→ [[00-nextauth-setup.md]]）
- Prisma による DB 操作の基礎

---

## 1. ソーシャルログインの全体像

### 1.1 なぜソーシャルログインが必要なのか

```
ソーシャルログインの利点:

  ユーザー視点:
  ┌─────────────────────────────────────────────────────┐
  │                                                     │
  │  ① パスワード不要 → 記憶・管理の負担なし             │
  │  ② ワンクリックログイン → 離脱率の低下               │
  │  ③ 信頼性 → Google / Apple という既知のブランド       │
  │  ④ プロフィール自動入力 → サインアップの簡略化        │
  │                                                     │
  └─────────────────────────────────────────────────────┘

  開発者視点:
  ┌─────────────────────────────────────────────────────┐
  │                                                     │
  │  ① パスワードハッシュ管理が不要                       │
  │  ② メール検証をプロバイダーに委任可能                 │
  │  ③ 2FA / MFA をプロバイダー側で実施                   │
  │  ④ ブルートフォース対策が不要                         │
  │  ⑤ パスワードリセットフローが不要                     │
  │                                                     │
  └─────────────────────────────────────────────────────┘

  統計（業界データ）:
    → ソーシャルログイン導入でサインアップ率が 20-50% 向上
    → Google が最も利用率が高い（約 60%）
    → Apple は iOS ユーザーで急速に普及中
    → GitHub は開発者向けサービスでほぼ必須
```

### 1.2 OAuth 2.0 Authorization Code Flow（PKCE 付き）

```
ソーシャルログインの認証フロー:

  ┌──────────┐     ┌──────────┐     ┌──────────────┐     ┌──────────┐
  │ ブラウザ  │     │ App Server│     │ Auth Provider │     │ Database │
  │          │     │ (Auth.js) │     │ (Google等)    │     │          │
  └────┬─────┘     └────┬──────┘     └──────┬────────┘     └────┬─────┘
       │                │                    │                   │
       │ ① ログインボタン│                    │                   │
       │ クリック        │                    │                   │
       ├───────────────→│                    │                   │
       │                │                    │                   │
       │                │ ② state + PKCE     │                   │
       │                │ code_verifier 生成  │                   │
       │                │                    │                   │
       │ ③ リダイレクト  │                    │                   │
       │←───────────────│                    │                   │
       │ (認可エンドポイント)                  │                   │
       │                │                    │                   │
       │ ④ ユーザーに同意画面表示              │                   │
       ├────────────────────────────────────→│                   │
       │                │                    │                   │
       │ ⑤ 同意 → Authorization Code         │                   │
       │←────────────────────────────────────│                   │
       │                │                    │                   │
       │ ⑥ Code をコールバック URL で送信      │                   │
       ├───────────────→│                    │                   │
       │                │                    │                   │
       │                │ ⑦ Code + code_verifier                 │
       │                │ → Token Exchange    │                   │
       │                ├───────────────────→│                   │
       │                │                    │                   │
       │                │ ⑧ Access Token +   │                   │
       │                │ ID Token + Refresh  │                   │
       │                │←───────────────────│                   │
       │                │                    │                   │
       │                │ ⑨ UserInfo取得/     │                   │
       │                │ ID Token 検証       │                   │
       │                │                    │                   │
       │                │ ⑩ ユーザー作成/更新                     │
       │                ├──────────────────────────────────────→│
       │                │                                        │
       │                │ ⑪ セッション作成                        │
       │                │←──────────────────────────────────────│
       │                │                    │                   │
       │ ⑫ セッション Cookie                  │                   │
       │←───────────────│                    │                   │
       │                │                    │                   │
  ※ PKCE (Proof Key for Code Exchange):
    → Authorization Code 横取り攻撃を防止
    → code_verifier: クライアントで生成したランダム文字列
    → code_challenge: code_verifier の SHA256 ハッシュ
    → ⑦ で code_verifier を送信し、サーバーが検証
```

### 1.3 OpenID Connect と OAuth 2.0 の違い

```
OAuth 2.0 vs OpenID Connect:

  ┌─────────────────┬──────────────────┬──────────────────────┐
  │ 項目             │ OAuth 2.0        │ OpenID Connect       │
  ├─────────────────┼──────────────────┼──────────────────────┤
  │ 目的             │ 認可（API アクセス）│ 認証（ユーザー特定） │
  │ 返却トークン      │ Access Token     │ ID Token + Access T. │
  │ ユーザー情報      │ API で取得       │ ID Token に含有      │
  │ 標準スコープ      │ なし             │ openid, profile等    │
  │ Discovery        │ なし             │ .well-known/openid-c │
  │ 利用プロバイダー   │ GitHub          │ Google, Apple        │
  └─────────────────┴──────────────────┴──────────────────────┘

  ID Token (JWT) の構造:
    {
      "iss": "https://accounts.google.com",   // 発行者
      "sub": "1234567890",                     // ユーザー識別子
      "aud": "your-client-id.apps...",         // クライアント ID
      "exp": 1700000000,                       // 有効期限
      "iat": 1699996400,                       // 発行時刻
      "email": "user@gmail.com",               // メール
      "email_verified": true,                  // メール検証済み
      "name": "Taro Yamada",                   // 名前
      "picture": "https://..."                 // アバター URL
    }

  重要: GitHub は純粋な OAuth 2.0 のみ（OpenID Connect 非対応）
  → ID Token がないため、/user API でユーザー情報を取得
```

---

## 2. Google ログイン

### 2.1 Google Cloud Console 設定

```
Google Cloud Console での詳細設定手順:

  ① プロジェクト作成:
     → console.cloud.google.com
     → 新しいプロジェクトを作成（または既存プロジェクトを選択）

  ② OAuth 同意画面の設定:
     → APIs & Services > OAuth consent screen
     → User Type:
        ・Internal: Google Workspace ユーザーのみ（組織内用）
        ・External: 全 Google ユーザー（一般公開用）
     → アプリ名、サポートメール、デベロッパー連絡先
     → スコープの追加:
        ・openid（必須）
        ・email（メール取得）
        ・profile（名前・アバター取得）
     → テストユーザー追加（External の場合、公開審査前に必要）

  ③ 認証情報の作成:
     → APIs & Services > Credentials
     → Create Credentials > OAuth client ID
     → アプリケーションの種類: Web application
     → 承認済みの JavaScript 生成元:
        開発: http://localhost:3000
        本番: https://myapp.com
     → 承認済みのリダイレクト URI:
        開発: http://localhost:3000/api/auth/callback/google
        本番: https://myapp.com/api/auth/callback/google

  ④ Client ID と Client Secret を取得
     → .env.local に設定

  ⑤ 本番公開:
     → OAuth 同意画面で「PUBLISH APP」
     → Google による審査（数日〜数週間）
     → プライバシーポリシーとサービス利用規約の URL が必要

  注意事項:
  ┌────────────────────────────────────────────────────┐
  │ ・テストモードでは最大 100 人のテストユーザーまで     │
  │ ・公開後は Google のブランドガイドラインに準拠必須     │
  │ ・Sensitive scope（カレンダー等）は追加審査が必要      │
  │ ・Client Secret は絶対にフロントエンドに露出させない   │
  └────────────────────────────────────────────────────┘
```

### 2.2 Google プロバイダー設定（Auth.js v5）

```typescript
// auth.ts - Google プロバイダーの完全設定
import Google from 'next-auth/providers/google';
import type { NextAuthConfig } from 'next-auth';

export const googleProvider = Google({
  clientId: process.env.GOOGLE_CLIENT_ID!,
  clientSecret: process.env.GOOGLE_CLIENT_SECRET!,

  // 認可パラメータのカスタマイズ
  authorization: {
    params: {
      scope: 'openid email profile',
      prompt: 'consent',         // 毎回同意画面を表示（Refresh Token 取得に必要）
      access_type: 'offline',    // Refresh Token を取得
      response_type: 'code',     // Authorization Code Flow
      // hd: 'mycompany.com',    // 特定ドメインに限定（Google Workspace）
    },
  },

  // ID Token から取得するプロフィール情報のマッピング
  profile(profile) {
    return {
      id: profile.sub,                        // Google の一意識別子
      name: profile.name,                     // フルネーム
      email: profile.email,                   // メールアドレス
      image: profile.picture,                 // アバター URL
      emailVerified: profile.email_verified,  // メール検証済みフラグ
      // カスタムフィールド
      role: 'viewer',                         // デフォルトロール
      locale: profile.locale,                 // ロケール（ja, en 等）
    };
  },
});
```

### 2.3 Google 固有の注意点

```
Google ログインの重要な注意事項:

  ① prompt パラメータの挙動:
     ┌──────────────┬──────────────────────────────────────┐
     │ 値            │ 挙動                                 │
     ├──────────────┼──────────────────────────────────────┤
     │ none          │ 自動ログイン（同意済みの場合）         │
     │ consent       │ 毎回同意画面を表示                    │
     │ select_account│ アカウント選択画面を表示               │
     │ login         │ 再認証を要求                          │
     └──────────────┴──────────────────────────────────────┘

     推奨: prompt: 'consent' + access_type: 'offline'
     → Refresh Token を確実に取得するため

  ② Refresh Token の取得条件:
     → access_type: 'offline' が必要
     → 初回認可時のみ発行される（デフォルト）
     → prompt: 'consent' で毎回発行を強制
     → Refresh Token がない場合、Access Token 期限切れでログアウト

  ③ Google One Tap:
     → ページ上にポップアップで表示されるログイン
     → OAuth フローではなく、credential response を直接受信
     → Auth.js では直接サポートされていないため、
       カスタム実装が必要

  ④ Google Workspace（旧 G Suite）制限:
     → hd パラメータで組織ドメインを制限可能
     → 管理者がアプリへのアクセスを制御可能
     → Workspace ユーザーは管理者の許可が必要な場合あり

  ⑤ Google のアバター URL の有効期限:
     → Google のプロフィール画像 URL は変更される可能性がある
     → 定期的に更新するか、ローカルにキャッシュ
```

```typescript
// Google Workspace ドメイン制限の実装
callbacks: {
  async signIn({ account, profile }) {
    if (account?.provider === 'google') {
      // 特定ドメインのみ許可
      const allowedDomains = ['mycompany.com', 'partner.com'];
      const email = profile?.email;

      if (!email) return false;

      const domain = email.split('@')[1];
      if (!allowedDomains.includes(domain)) {
        return '/login?error=DomainNotAllowed';
      }

      // email_verified チェック
      if (!profile?.email_verified) {
        return '/login?error=EmailNotVerified';
      }
    }
    return true;
  },
}
```

```typescript
// Google Refresh Token Rotation の実装
import { google } from 'googleapis';

async function refreshGoogleAccessToken(refreshToken: string) {
  const oauth2Client = new google.auth.OAuth2(
    process.env.GOOGLE_CLIENT_ID,
    process.env.GOOGLE_CLIENT_SECRET,
  );

  oauth2Client.setCredentials({
    refresh_token: refreshToken,
  });

  try {
    const { credentials } = await oauth2Client.refreshAccessToken();

    return {
      access_token: credentials.access_token!,
      expires_at: Math.floor(credentials.expiry_date! / 1000),
      refresh_token: credentials.refresh_token ?? refreshToken,
      // Google は Refresh Token を毎回返さない場合がある
      // → 既存の Refresh Token を保持
    };
  } catch (error) {
    console.error('Failed to refresh Google access token:', error);
    throw error;
  }
}

// Auth.js の jwt コールバックで使用
callbacks: {
  async jwt({ token, account }) {
    // 初回ログイン時: トークンを保存
    if (account) {
      return {
        ...token,
        access_token: account.access_token,
        expires_at: account.expires_at,
        refresh_token: account.refresh_token,
        provider: account.provider,
      };
    }

    // Access Token が有効な場合はそのまま返す
    if (token.expires_at && Date.now() < (token.expires_at as number) * 1000) {
      return token;
    }

    // Access Token が期限切れの場合はリフレッシュ
    if (token.provider === 'google' && token.refresh_token) {
      try {
        const refreshed = await refreshGoogleAccessToken(
          token.refresh_token as string
        );
        return {
          ...token,
          access_token: refreshed.access_token,
          expires_at: refreshed.expires_at,
          refresh_token: refreshed.refresh_token,
        };
      } catch {
        // リフレッシュ失敗 → 再ログインを促す
        return { ...token, error: 'RefreshAccessTokenError' };
      }
    }

    return token;
  },
}
```

---

## 3. GitHub ログイン

### 3.1 GitHub OAuth App vs GitHub App

```
GitHub の 2 種類の OAuth 方式:

  ┌─────────────────┬──────────────────┬──────────────────────┐
  │ 項目             │ OAuth App        │ GitHub App           │
  ├─────────────────┼──────────────────┼──────────────────────┤
  │ 作成場所         │ Developer settings│ Developer settings  │
  │                 │ > OAuth Apps     │ > GitHub Apps        │
  │ スコープ制御      │ ユーザーが選択   │ インストール時に設定   │
  │ 組織アクセス      │ 別途承認が必要   │ インストール単位       │
  │ Webhook          │ なし             │ あり                 │
  │ インストール      │ OAuth 認可のみ   │ アカウント/組織に install│
  │ Refresh Token    │ なし             │ あり（8時間有効）     │
  │ Rate Limit       │ 5,000 req/h      │ 5,000 req/h (user)  │
  │                 │                  │ + installation limit │
  │ 推奨用途         │ シンプルなログイン │ リポジトリ操作あり    │
  └─────────────────┴──────────────────┴──────────────────────┘

  ソーシャルログインだけなら OAuth App で十分。
  リポジトリアクセスが必要な場合は GitHub App を推奨。
```

### 3.2 GitHub OAuth App 設定

```
GitHub OAuth App の設定手順:

  ① Settings > Developer settings > OAuth Apps
  ② New OAuth App を作成
  ③ 設定項目:
     ・Application name: アプリ名（ユーザーに表示される）
     ・Homepage URL: https://myapp.com
     ・Application description: アプリの説明
     ・Authorization callback URL:
       開発: http://localhost:3000/api/auth/callback/github
       本番: https://myapp.com/api/auth/callback/github
  ④ Client ID と Client Secret を取得

  重要: GitHub OAuth App はコールバック URL を 1 つしか設定できない
  → 開発/ステージング/本番で別々の OAuth App が必要
  → GitHub App なら複数のコールバック URL を設定可能

  スコープ一覧（よく使うもの）:
  ┌──────────────────┬────────────────────────────────────┐
  │ スコープ          │ 説明                               │
  ├──────────────────┼────────────────────────────────────┤
  │ (なし)            │ 公開情報のみ（login, avatar 等）    │
  │ read:user        │ ユーザープロフィール読み取り          │
  │ user:email       │ メールアドレス取得                   │
  │ repo             │ プライベートリポジトリへのフルアクセス │
  │ read:org         │ 組織メンバーシップ読み取り            │
  │ gist             │ Gist 作成                           │
  │ admin:org        │ 組織の管理                          │
  └──────────────────┴────────────────────────────────────┘

  ログインだけなら: read:user user:email で十分
```

### 3.3 GitHub プロバイダー設定（Auth.js v5）

```typescript
// auth.ts - GitHub プロバイダーの完全設定
import GitHub from 'next-auth/providers/github';

export const githubProvider = GitHub({
  clientId: process.env.GITHUB_CLIENT_ID!,
  clientSecret: process.env.GITHUB_CLIENT_SECRET!,

  // スコープ設定
  authorization: {
    params: {
      scope: 'read:user user:email',
    },
  },

  // GitHub は OpenID Connect ではないため、/user API レスポンスを使用
  profile(profile) {
    return {
      id: String(profile.id),              // GitHub の数値 ID を文字列に
      name: profile.name || profile.login,  // name が null なら login を使用
      email: profile.email,                 // null の場合がある（後述）
      image: profile.avatar_url,            // アバター URL
      role: 'viewer',                       // デフォルトロール
    };
  },
});
```

### 3.4 GitHub のメール問題と解決策

```
GitHub のメール取得における問題:

  GitHub ユーザーのメールが null になるケース:
  ┌────────────────────────────────────────────────────┐
  │                                                    │
  │  ① メールを非公開に設定している                      │
  │     → Settings > Emails > "Keep my email private"  │
  │     → profile.email が null になる                  │
  │                                                    │
  │  ② メールを公開設定にしていない                      │
  │     → Settings > Profile > "Public email" が空     │
  │                                                    │
  │  ③ スコープに user:email がない                     │
  │     → /user API でもメールが返らない                │
  │                                                    │
  └────────────────────────────────────────────────────┘

  解決策: /user/emails API を使って取得
  → user:email スコープが必要
  → primary かつ verified なメールを使用
```

```typescript
// GitHub メール取得の完全実装
callbacks: {
  async signIn({ user, account, profile }) {
    // GitHub プロバイダーでメールが null の場合の対処
    if (account?.provider === 'github' && !user.email) {
      try {
        // GitHub API でメール一覧を取得
        const emailRes = await fetch('https://api.github.com/user/emails', {
          headers: {
            Authorization: `Bearer ${account.access_token}`,
            Accept: 'application/vnd.github+json',
            'X-GitHub-Api-Version': '2022-11-28',
          },
        });

        if (!emailRes.ok) {
          console.error('Failed to fetch GitHub emails:', emailRes.status);
          return '/login?error=EmailFetchFailed';
        }

        const emails: Array<{
          email: string;
          primary: boolean;
          verified: boolean;
          visibility: string | null;
        }> = await emailRes.json();

        // 優先順位: primary + verified > verified > primary
        const primary = emails.find((e) => e.primary && e.verified);
        const verified = emails.find((e) => e.verified);

        const selectedEmail = primary?.email ?? verified?.email;

        if (selectedEmail) {
          user.email = selectedEmail;
        } else {
          // 検証済みメールがない場合はログインを拒否
          return '/login?error=NoVerifiedEmail';
        }
      } catch (error) {
        console.error('Error fetching GitHub emails:', error);
        return '/login?error=EmailFetchFailed';
      }
    }
    return true;
  },
}
```

### 3.5 GitHub 組織メンバーシップによるアクセス制御

```typescript
// GitHub 組織メンバーシップを検証してアクセスを制限
async function checkGitHubOrgMembership(
  accessToken: string,
  allowedOrgs: string[]
): Promise<boolean> {
  const orgsRes = await fetch('https://api.github.com/user/orgs', {
    headers: {
      Authorization: `Bearer ${accessToken}`,
      Accept: 'application/vnd.github+json',
    },
  });

  if (!orgsRes.ok) return false;

  const orgs: Array<{ login: string }> = await orgsRes.json();
  return orgs.some((org) => allowedOrgs.includes(org.login));
}

// Auth.js コールバックで使用
callbacks: {
  async signIn({ account }) {
    if (account?.provider === 'github') {
      const isMember = await checkGitHubOrgMembership(
        account.access_token!,
        ['my-company', 'my-team']
      );

      if (!isMember) {
        return '/login?error=OrgMembershipRequired';
      }
    }
    return true;
  },
}
```

```typescript
// GitHub の特定チームメンバーシップ確認
async function checkGitHubTeamMembership(
  accessToken: string,
  org: string,
  teamSlug: string
): Promise<boolean> {
  // 自分のチームメンバーシップを確認
  const res = await fetch(
    `https://api.github.com/orgs/${org}/teams/${teamSlug}/memberships/${username}`,
    {
      headers: {
        Authorization: `Bearer ${accessToken}`,
        Accept: 'application/vnd.github+json',
      },
    }
  );

  if (!res.ok) return false;

  const data = await res.json();
  return data.state === 'active';
}

// チームに基づくロール割り当て
async function assignRoleByTeam(
  accessToken: string,
  org: string
): Promise<string> {
  // admin チームのメンバーか確認
  if (await checkGitHubTeamMembership(accessToken, org, 'admin')) {
    return 'admin';
  }

  // editors チームのメンバーか確認
  if (await checkGitHubTeamMembership(accessToken, org, 'editors')) {
    return 'editor';
  }

  // それ以外は viewer
  return 'viewer';
}
```

---

## 4. Apple ログイン

### 4.1 Apple Developer 設定

```
Apple Developer での詳細設定手順:

  前提条件:
  ┌────────────────────────────────────────────────────┐
  │ ・Apple Developer Program（年額 $99）への加入が必須  │
  │ ・Web 向けは Services ID が必要                     │
  │ ・iOS アプリと Web で設定が異なる                    │
  └────────────────────────────────────────────────────┘

  ① App ID の登録:
     → Certificates, Identifiers & Profiles
     → Identifiers > App IDs
     → Sign in with Apple の Capability を有効化
     → Bundle ID: com.mycompany.myapp

  ② Services ID の作成（Web 向け）:
     → Identifiers > Services IDs
     → Identifier: com.mycompany.myapp.web
     → Sign in with Apple を有効化
     → Configure:
       ・Primary App ID: ① で作成した App ID
       ・Website URLs:
         ・Domains and Subdomains: myapp.com
         ・Return URLs: https://myapp.com/api/auth/callback/apple

  ③ Key の作成:
     → Keys > Create a key
     → Key Name: MyApp Auth Key
     → Sign in with Apple を有効化
     → Configure で Primary App ID を選択
     → Register → ダウンロード（AuthKey_XXXXXXXXXX.p8）
     → Key ID をメモ

  ④ 必要な環境変数:
     APPLE_CLIENT_ID=com.mycompany.myapp.web  (Services ID)
     APPLE_TEAM_ID=XXXXXXXXXX                 (Team ID)
     APPLE_KEY_ID=XXXXXXXXXX                  (Key ID)
     APPLE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\n..."

  重要な制約:
  ┌────────────────────────────────────────────────────┐
  │ ・p8 ファイルは一度しかダウンロードできない          │
  │ ・Key は最大 2 つまで作成可能                       │
  │ ・Return URL は最大 10 個まで設定可能               │
  │ ・localhost は Return URL に使用できない             │
  │   → 開発時は ngrok 等のトンネルが必要               │
  │ ・clientSecret は JWT 形式で動的生成が必要           │
  └────────────────────────────────────────────────────┘
```

### 4.2 Apple プロバイダー設定（Auth.js v5）

```typescript
// auth.ts - Apple プロバイダーの完全設定
import Apple from 'next-auth/providers/apple';

export const appleProvider = Apple({
  clientId: process.env.APPLE_CLIENT_ID!,
  // clientSecret は動的に生成する必要がある
  clientSecret: generateAppleClientSecret(),
});

// Apple clientSecret の生成
// p8 秘密鍵から ES256 JWT を生成
import * as jose from 'jose';

async function generateAppleClientSecret(): Promise<string> {
  // 環境変数から秘密鍵を取得（改行を復元）
  const privateKeyPem = process.env.APPLE_PRIVATE_KEY!.replace(/\\n/g, '\n');

  // PKCS#8 形式の秘密鍵をインポート
  const privateKey = await jose.importPKCS8(privateKeyPem, 'ES256');

  // JWT を生成
  const jwt = await new jose.SignJWT({})
    .setProtectedHeader({
      alg: 'ES256',
      kid: process.env.APPLE_KEY_ID!,  // Key ID
    })
    .setIssuedAt()
    .setExpirationTime('180d')  // 最大 6 ヶ月
    .setAudience('https://appleid.apple.com')
    .setIssuer(process.env.APPLE_TEAM_ID!)   // Team ID
    .setSubject(process.env.APPLE_CLIENT_ID!) // Services ID
    .sign(privateKey);

  return jwt;
}
```

### 4.3 Apple 固有の注意点とエッジケース

```
Apple ログインの特殊な挙動:

  ① name と email は初回認可時のみ返却される:
  ┌────────────────────────────────────────────────────┐
  │                                                    │
  │  初回ログイン:                                      │
  │  {                                                 │
  │    "sub": "001234.abcdef...",                      │
  │    "email": "user@icloud.com",                     │
  │    "email_verified": true,                         │
  │    "name": { "firstName": "太郎", "lastName": "山田" }│
  │  }                                                 │
  │                                                    │
  │  2回目以降:                                         │
  │  {                                                 │
  │    "sub": "001234.abcdef..."                       │
  │    // email も name も含まれない！                   │
  │  }                                                 │
  │                                                    │
  │  対策:                                              │
  │  → 初回ログイン時に必ず DB に保存する               │
  │  → 保存に失敗した場合、ユーザーは Apple ID 設定で    │
  │    「Stop using Apple ID」→ 再度認可が必要          │
  │                                                    │
  └────────────────────────────────────────────────────┘

  ② Private Email Relay（メール非公開機能）:
  ┌────────────────────────────────────────────────────┐
  │                                                    │
  │  ユーザーが「メールを非公開」を選択した場合:          │
  │  → xxxxx@privaterelay.appleid.com が返される        │
  │  → Apple のリレーサービスを経由してメール転送        │
  │                                                    │
  │  リレーメールの設定:                                 │
  │  → Certificates > More > Configure                 │
  │  → Email Sources に送信元ドメインを登録             │
  │  → SPF / DKIM の設定が必要                         │
  │                                                    │
  │  注意:                                              │
  │  → 同じユーザーが別のメールアドレスを返す可能性      │
  │  → メールアドレスでのアカウント照合が困難            │
  │  → sub（ユーザー ID）で照合するのが確実             │
  │                                                    │
  └────────────────────────────────────────────────────┘

  ③ clientSecret の有効期限:
     → JWT の有効期限は最大 6 ヶ月
     → 6 ヶ月ごとに再生成が必要
     → アプリ起動時に動的生成するのが推奨

  ④ Apple のレビューガイドライン:
     → iOS アプリでソーシャルログインを提供する場合、
       Sign in with Apple の提供が必須
     → Web のみのサービスでは任意
```

```typescript
// Apple ログインの初回データ保存実装
callbacks: {
  async signIn({ user, account, profile }) {
    if (account?.provider === 'apple') {
      // Apple の profile は初回のみ name を含む
      const appleProfile = profile as {
        sub: string;
        email?: string;
        email_verified?: boolean;
        is_private_email?: string;
      };

      // user オブジェクトには Auth.js が name を設定済み
      // （POST ボディの user パラメータから取得）

      const existingUser = await prisma.user.findFirst({
        where: {
          accounts: {
            some: {
              provider: 'apple',
              providerAccountId: appleProfile.sub,
            },
          },
        },
      });

      if (!existingUser && user.email) {
        // 新規ユーザー: Private Relay メールのフラグを保存
        await prisma.user.create({
          data: {
            email: user.email,
            name: user.name,
            image: user.image,
            isPrivateEmail: appleProfile.is_private_email === 'true',
          },
        });
      }
    }
    return true;
  },
}
```

```typescript
// Apple Private Relay メールへの送信設定
// Apple にリレーメール送信元として登録が必要

// メール送信時の分岐
async function sendEmail(to: string, subject: string, body: string) {
  const isAppleRelay = to.endsWith('@privaterelay.appleid.com');

  if (isAppleRelay) {
    // Apple Relay 経由の場合、送信元ドメインが登録済みである必要がある
    // From: noreply@myapp.com（Apple に登録したドメイン）
    await sendViaRegisteredDomain(to, subject, body);
  } else {
    await sendNormally(to, subject, body);
  }
}
```

---

## 5. その他のプロバイダー

### 5.1 Microsoft (Azure AD / Entra ID)

```typescript
// Microsoft プロバイダー設定
import MicrosoftEntraID from 'next-auth/providers/microsoft-entra-id';

MicrosoftEntraID({
  clientId: process.env.AZURE_AD_CLIENT_ID!,
  clientSecret: process.env.AZURE_AD_CLIENT_SECRET!,
  // テナント ID（特定組織のみ許可する場合）
  tenantId: process.env.AZURE_AD_TENANT_ID,
  // 'common': 全 Microsoft アカウント
  // 'organizations': 組織アカウントのみ
  // 'consumers': 個人アカウントのみ
  // 特定テナントID: その組織のみ

  authorization: {
    params: {
      scope: 'openid email profile User.Read',
    },
  },

  profile(profile) {
    return {
      id: profile.sub,
      name: profile.name,
      email: profile.email,
      image: null, // Microsoft は picture を直接返さない
      role: 'viewer',
    };
  },
});
```

### 5.2 Discord

```typescript
// Discord プロバイダー設定
import Discord from 'next-auth/providers/discord';

Discord({
  clientId: process.env.DISCORD_CLIENT_ID!,
  clientSecret: process.env.DISCORD_CLIENT_SECRET!,

  authorization: {
    params: {
      scope: 'identify email guilds', // guilds でサーバー情報も取得可能
    },
  },

  profile(profile) {
    // Discord のアバター URL 構築
    const avatarUrl = profile.avatar
      ? `https://cdn.discordapp.com/avatars/${profile.id}/${profile.avatar}.${
          profile.avatar.startsWith('a_') ? 'gif' : 'png'
        }`
      : `https://cdn.discordapp.com/embed/avatars/${
          Number(profile.discriminator) % 5
        }.png`;

    return {
      id: profile.id,
      name: profile.username,
      email: profile.email,
      image: avatarUrl,
      role: 'viewer',
    };
  },
});
```

### 5.3 プロバイダー比較表

```
主要プロバイダーの比較:

  ┌─────────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
  │ 項目         │ Google   │ GitHub   │ Apple    │ Microsoft│ Discord  │
  ├─────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
  │ プロトコル    │ OIDC     │ OAuth2.0 │ OIDC     │ OIDC     │ OAuth2.0 │
  │ ID Token    │ ✓        │ ✗        │ ✓        │ ✓        │ ✗        │
  │ email保証    │ ✓ 常時   │ △ null可 │ △ 初回のみ│ ✓ 常時   │ ✓ 常時   │
  │ email検証    │ ✓        │ ✓(別API) │ ✓        │ ✓        │ △ 未検証可│
  │ Refresh T.  │ ✓        │ ✗(OAuth) │ ✓        │ ✓        │ ✓        │
  │ name取得     │ ✓ 常時   │ △ null可 │ △ 初回のみ│ ✓ 常時   │ ✓ 常時   │
  │ アバター     │ ✓        │ ✓        │ ✗        │ △(別API) │ ✓        │
  │ 費用         │ 無料     │ 無料     │ $99/年   │ 無料     │ 無料     │
  │ 審査         │ 必要     │ 不要     │ 不要     │ 不要     │ 不要     │
  │ localhost   │ ✓        │ ✓        │ ✗        │ ✓        │ ✓        │
  │ 推奨対象     │ 全般     │ 開発者   │ iOS      │ 企業     │ ゲーマー │
  └─────────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
```

---

## 6. アカウントリンク

### 6.1 アカウントリンクの問題と戦略

```
アカウントリンクの問題:

  シナリオ:
  ┌────────────────────────────────────────────────────┐
  │                                                    │
  │  ① Alice が Google (alice@gmail.com) でサインアップ │
  │  ② 後日 Alice が GitHub (alice@gmail.com) でログイン│
  │                                                    │
  │  問題: 同じメールだが、どう扱うべきか？              │
  │                                                    │
  │  選択肢:                                            │
  │  (a) 自動リンク → 便利だがセキュリティリスク         │
  │  (b) 別アカウント作成 → 安全だが UX が悪い          │
  │  (c) エラー表示 → 安全だが UX が悪い                │
  │  (d) 条件付き自動リンク → バランスの取れた方法       │
  │                                                    │
  └────────────────────────────────────────────────────┘

  攻撃シナリオ（自動リンクが危険な理由）:
  ┌────────────────────────────────────────────────────┐
  │                                                    │
  │  ① 攻撃者が victim@example.com を知っている         │
  │  ② 攻撃者が GitHub で victim@example.com を登録     │
  │    （GitHub はメール検証なしでも登録可能）            │
  │  ③ 攻撃者が GitHub で対象サービスにログイン          │
  │  ④ 自動リンクにより被害者のアカウントに紐付け        │
  │  ⑤ 攻撃者が被害者のデータにアクセス可能に！         │
  │                                                    │
  │  対策: email_verified が true の場合のみ自動リンク   │
  │                                                    │
  └────────────────────────────────────────────────────┘

  Auth.js のデフォルト動作:
  → allowDangerousEmailAccountLinking: false（デフォルト）
  → 同じメールの既存アカウントにはリンクしない
  → OAuthAccountNotLinked エラーが発生
```

### 6.2 安全なアカウントリンクの実装

```typescript
// 条件付き自動アカウントリンクの完全実装
// auth.ts

import { PrismaAdapter } from '@auth/prisma-adapter';
import type { Account, Profile, User } from 'next-auth';

export const authConfig = {
  adapter: PrismaAdapter(prisma),

  callbacks: {
    async signIn({
      user,
      account,
      profile,
    }: {
      user: User;
      account: Account | null;
      profile?: Profile;
    }) {
      if (!account || !user.email) return true;

      // 同じメールの既存ユーザーをチェック
      const existingUser = await prisma.user.findUnique({
        where: { email: user.email },
        include: {
          accounts: {
            select: {
              provider: true,
              providerAccountId: true,
            },
          },
        },
      });

      // 新規ユーザーの場合はそのまま
      if (!existingUser) return true;

      // 既にこのプロバイダーでリンク済みの場合はそのまま
      const isLinked = existingUser.accounts.some(
        (a) => a.provider === account.provider
      );
      if (isLinked) return true;

      // メールの検証状態を確認
      const isEmailVerified = checkEmailVerification(
        account.provider,
        profile
      );

      if (!isEmailVerified) {
        // 検証されていない場合: エラーページにリダイレクト
        return `/login?error=OAuthAccountNotLinked&provider=${account.provider}`;
      }

      // 検証済みの場合: 既存ユーザーにアカウントをリンク
      try {
        await prisma.account.create({
          data: {
            userId: existingUser.id,
            type: account.type,
            provider: account.provider,
            providerAccountId: account.providerAccountId,
            access_token: account.access_token,
            refresh_token: account.refresh_token,
            expires_at: account.expires_at,
            token_type: account.token_type,
            scope: account.scope,
            id_token: account.id_token,
          },
        });

        // リンクイベントを記録
        await prisma.auditLog.create({
          data: {
            userId: existingUser.id,
            action: 'account_linked',
            metadata: {
              provider: account.provider,
              providerAccountId: account.providerAccountId,
              linkedAt: new Date().toISOString(),
            },
          },
        });

        // Auth.js のユーザー ID を既存ユーザーに合わせる
        user.id = existingUser.id;
      } catch (error) {
        console.error('Failed to link account:', error);
        return '/login?error=LinkFailed';
      }

      return true;
    },
  },
};

// プロバイダーごとのメール検証チェック
function checkEmailVerification(
  provider: string,
  profile?: Profile
): boolean {
  switch (provider) {
    case 'google':
      // Google: email_verified フィールドで判定
      return (profile as any)?.email_verified === true;

    case 'github':
      // GitHub: /user/emails API で verified のみ返すため
      // user:email スコープがあれば検証済みとみなせる
      return true;

    case 'apple':
      // Apple: email_verified は常に true（Apple が保証）
      return (profile as any)?.email_verified === true;

    case 'microsoft-entra-id':
      // Microsoft: email_verified フィールドで判定
      return (profile as any)?.email_verified === true;

    default:
      // 不明なプロバイダーは安全側に倒す
      return false;
  }
}
```

### 6.3 手動アカウントリンク（設定画面）

```typescript
// app/settings/accounts/page.tsx - アカウント管理画面（Server Component）
import { auth } from '@/auth';
import { redirect } from 'next/navigation';
import { LinkedAccountsList } from './linked-accounts-list';

export default async function AccountSettingsPage() {
  const session = await auth();
  if (!session) redirect('/login');

  // ユーザーのリンク済みアカウントを取得
  const accounts = await prisma.account.findMany({
    where: { userId: session.user.id },
    select: {
      id: true,
      provider: true,
      providerAccountId: true,
      createdAt: true,
    },
    orderBy: { createdAt: 'asc' },
  });

  // パスワードの有無を確認
  const user = await prisma.user.findUnique({
    where: { id: session.user.id },
    select: { password: true },
  });

  const hasPassword = !!user?.password;
  const canUnlink = accounts.length > 1 || hasPassword;

  return (
    <div>
      <h1>アカウント連携</h1>
      <LinkedAccountsList
        accounts={accounts}
        canUnlink={canUnlink}
      />
    </div>
  );
}
```

```typescript
// app/settings/accounts/actions.ts - Server Actions
'use server';

import { auth, signIn } from '@/auth';
import { revalidatePath } from 'next/cache';

// アカウントリンク開始
export async function linkAccount(provider: string) {
  const session = await auth();
  if (!session) throw new Error('Unauthorized');

  // signIn を呼び出してOAuthフローを開始
  await signIn(provider, {
    redirectTo: '/settings/accounts',
  });
}

// アカウントリンク解除
export async function unlinkAccount(
  _prevState: any,
  formData: FormData
): Promise<{ error?: string; success?: boolean }> {
  const session = await auth();
  if (!session) return { error: 'Unauthorized' };

  const provider = formData.get('provider') as string;
  if (!provider) return { error: 'Provider is required' };

  // リンク済みアカウント数を確認
  const accountCount = await prisma.account.count({
    where: { userId: session.user.id },
  });

  // パスワードの有無を確認
  const user = await prisma.user.findUnique({
    where: { id: session.user.id },
    select: { password: true },
  });

  // 最後のログイン方法は削除不可
  if (accountCount <= 1 && !user?.password) {
    return {
      error: 'ログイン方法が他にないため、この連携は解除できません。',
    };
  }

  try {
    await prisma.account.deleteMany({
      where: {
        userId: session.user.id,
        provider,
      },
    });

    // 監査ログ
    await prisma.auditLog.create({
      data: {
        userId: session.user.id,
        action: 'account_unlinked',
        metadata: { provider },
      },
    });

    revalidatePath('/settings/accounts');
    return { success: true };
  } catch {
    return { error: 'アカウント連携の解除に失敗しました。' };
  }
}
```

```typescript
// app/settings/accounts/linked-accounts-list.tsx - クライアントコンポーネント
'use client';

import { useActionState } from 'react';
import { linkAccount, unlinkAccount } from './actions';

const providerInfo: Record<string, { name: string; icon: string; color: string }> = {
  google:   { name: 'Google',    icon: '/icons/google.svg',  color: '#4285F4' },
  github:   { name: 'GitHub',    icon: '/icons/github.svg',  color: '#333333' },
  apple:    { name: 'Apple',     icon: '/icons/apple.svg',   color: '#000000' },
  discord:  { name: 'Discord',   icon: '/icons/discord.svg', color: '#5865F2' },
};

const allProviders = ['google', 'github', 'apple'];

export function LinkedAccountsList({
  accounts,
  canUnlink,
}: {
  accounts: Array<{ id: string; provider: string; createdAt: Date }>;
  canUnlink: boolean;
}) {
  const [state, dispatch] = useActionState(unlinkAccount, {});

  const linkedProviders = new Set(accounts.map((a) => a.provider));
  const unlinkedProviders = allProviders.filter((p) => !linkedProviders.has(p));

  return (
    <div className="space-y-4">
      {/* リンク済みアカウント */}
      <h2>連携済みアカウント</h2>
      {accounts.map((account) => {
        const info = providerInfo[account.provider];
        return (
          <div key={account.id} className="flex items-center justify-between p-4 border rounded">
            <div className="flex items-center gap-3">
              <img src={info?.icon} alt="" className="w-6 h-6" />
              <span>{info?.name ?? account.provider}</span>
              <span className="text-sm text-gray-500">
                連携日: {new Date(account.createdAt).toLocaleDateString('ja-JP')}
              </span>
            </div>
            {canUnlink && (
              <form action={dispatch}>
                <input type="hidden" name="provider" value={account.provider} />
                <button type="submit" className="text-red-500 hover:text-red-700">
                  連携解除
                </button>
              </form>
            )}
          </div>
        );
      })}

      {/* 未リンクのプロバイダー */}
      {unlinkedProviders.length > 0 && (
        <>
          <h2>追加の連携</h2>
          {unlinkedProviders.map((provider) => {
            const info = providerInfo[provider];
            return (
              <div key={provider} className="flex items-center justify-between p-4 border rounded">
                <div className="flex items-center gap-3">
                  <img src={info?.icon} alt="" className="w-6 h-6" />
                  <span>{info?.name ?? provider}</span>
                </div>
                <form action={() => linkAccount(provider)}>
                  <button type="submit" className="text-blue-500 hover:text-blue-700">
                    連携する
                  </button>
                </form>
              </div>
            );
          })}
        </>
      )}

      {state.error && (
        <p className="text-red-500">{state.error}</p>
      )}
    </div>
  );
}
```

---

## 7. プロフィール同期

### 7.1 ログイン時のプロフィール同期

```typescript
// プロフィール同期の完全実装
// 各ログイン時にプロバイダーからの最新情報でプロフィールを更新

callbacks: {
  async signIn({ user, account, profile }) {
    if (!account || !user.id) return true;

    // プロフィール同期データの構築
    const syncData: Record<string, any> = {
      lastLoginAt: new Date(),
      lastLoginProvider: account.provider,
    };

    // プロバイダーごとのプロフィール情報取得
    switch (account.provider) {
      case 'google': {
        const googleProfile = profile as {
          name?: string;
          picture?: string;
          locale?: string;
        };
        if (googleProfile.name) syncData.name = googleProfile.name;
        if (googleProfile.picture) syncData.image = googleProfile.picture;
        if (googleProfile.locale) syncData.locale = googleProfile.locale;
        break;
      }

      case 'github': {
        const githubProfile = profile as {
          name?: string;
          login: string;
          avatar_url?: string;
          bio?: string;
          company?: string;
          location?: string;
          blog?: string;
        };
        if (githubProfile.name) syncData.name = githubProfile.name;
        if (githubProfile.avatar_url) syncData.image = githubProfile.avatar_url;
        // 追加フィールド（スキーマに存在する場合）
        if (githubProfile.bio) syncData.bio = githubProfile.bio;
        if (githubProfile.company) syncData.company = githubProfile.company;
        break;
      }

      case 'apple': {
        // Apple は 2 回目以降 name を返さないため、既存データを上書きしない
        // sub のみで識別
        break;
      }
    }

    // メールは変更しない（セキュリティ上の理由）
    // → メール変更は別のフローで行う

    try {
      await prisma.user.update({
        where: { id: user.id },
        data: syncData,
      });
    } catch (error) {
      // 同期失敗はログインを妨げない
      console.error('Profile sync failed:', error);
    }

    // Access Token / Refresh Token の更新
    await prisma.account.update({
      where: {
        provider_providerAccountId: {
          provider: account.provider,
          providerAccountId: account.providerAccountId,
        },
      },
      data: {
        access_token: account.access_token,
        refresh_token: account.refresh_token ?? undefined,
        expires_at: account.expires_at,
        token_type: account.token_type,
        scope: account.scope,
        id_token: account.id_token,
      },
    });

    return true;
  },
}
```

### 7.2 アバター画像のローカルキャッシュ

```typescript
// プロバイダーのアバター URL は変更される可能性がある
// ローカルにキャッシュして安定した URL を提供

import { S3Client, PutObjectCommand } from '@aws-sdk/client-s3';
import sharp from 'sharp';

const s3 = new S3Client({ region: process.env.AWS_REGION });

async function cacheAvatar(
  userId: string,
  avatarUrl: string
): Promise<string> {
  try {
    // 画像をダウンロード
    const response = await fetch(avatarUrl);
    if (!response.ok) throw new Error('Failed to fetch avatar');

    const buffer = Buffer.from(await response.arrayBuffer());

    // リサイズと最適化
    const optimized = await sharp(buffer)
      .resize(256, 256, { fit: 'cover' })
      .webp({ quality: 80 })
      .toBuffer();

    // S3 にアップロード
    const key = `avatars/${userId}.webp`;
    await s3.send(new PutObjectCommand({
      Bucket: process.env.AVATAR_BUCKET!,
      Key: key,
      Body: optimized,
      ContentType: 'image/webp',
      CacheControl: 'public, max-age=86400', // 1 日キャッシュ
    }));

    return `${process.env.CDN_URL}/${key}`;
  } catch (error) {
    console.error('Avatar cache failed:', error);
    return avatarUrl; // フォールバック: 元の URL をそのまま返す
  }
}
```

---

## 8. ソーシャルログインの UX

### 8.1 ログインページの実装

```typescript
// app/login/page.tsx - ソーシャルログインページ
import { signIn } from '@/auth';
import { redirect } from 'next/navigation';
import { auth } from '@/auth';

export default async function LoginPage({
  searchParams,
}: {
  searchParams: { callbackUrl?: string; error?: string };
}) {
  // 既にログイン済みならリダイレクト
  const session = await auth();
  if (session) redirect(searchParams.callbackUrl || '/dashboard');

  const callbackUrl = searchParams.callbackUrl || '/dashboard';

  // エラーメッセージのマッピング
  const errorMessages: Record<string, string> = {
    OAuthAccountNotLinked:
      'このメールアドレスは別のログイン方法で登録されています。元の方法でログインしてください。',
    OAuthSignin: 'ログインの開始に失敗しました。もう一度お試しください。',
    OAuthCallback: '認証に失敗しました。もう一度お試しください。',
    Callback: '認証処理中にエラーが発生しました。',
    AccessDenied: 'アクセスが拒否されました。',
    DomainNotAllowed: 'このドメインのアカウントではログインできません。',
    OrgMembershipRequired: '組織のメンバーシップが必要です。',
    EmailFetchFailed: 'メールアドレスの取得に失敗しました。',
    NoVerifiedEmail: '検証済みのメールアドレスが見つかりません。',
    Default: 'ログインに失敗しました。もう一度お試しください。',
  };

  const error = searchParams.error;
  const errorMessage = error
    ? errorMessages[error] || errorMessages.Default
    : null;

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <div className="max-w-md w-full space-y-8 p-8 bg-white rounded-xl shadow-md">
        <div className="text-center">
          <h1 className="text-2xl font-bold">ログイン</h1>
          <p className="mt-2 text-gray-600">
            アカウントにログインしてください
          </p>
        </div>

        {errorMessage && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4">
            <p className="text-red-700 text-sm">{errorMessage}</p>
          </div>
        )}

        <div className="space-y-3">
          {/* Google */}
          <form
            action={async () => {
              'use server';
              await signIn('google', { redirectTo: callbackUrl });
            }}
          >
            <button
              type="submit"
              className="w-full flex items-center justify-center gap-3 px-4 py-3
                         border border-gray-300 rounded-lg hover:bg-gray-50
                         transition-colors"
            >
              <GoogleIcon className="w-5 h-5" />
              <span>Google で続ける</span>
            </button>
          </form>

          {/* Apple */}
          <form
            action={async () => {
              'use server';
              await signIn('apple', { redirectTo: callbackUrl });
            }}
          >
            <button
              type="submit"
              className="w-full flex items-center justify-center gap-3 px-4 py-3
                         bg-black text-white rounded-lg hover:bg-gray-900
                         transition-colors"
            >
              <AppleIcon className="w-5 h-5" />
              <span>Apple で続ける</span>
            </button>
          </form>

          {/* GitHub */}
          <form
            action={async () => {
              'use server';
              await signIn('github', { redirectTo: callbackUrl });
            }}
          >
            <button
              type="submit"
              className="w-full flex items-center justify-center gap-3 px-4 py-3
                         bg-gray-800 text-white rounded-lg hover:bg-gray-700
                         transition-colors"
            >
              <GitHubIcon className="w-5 h-5" />
              <span>GitHub で続ける</span>
            </button>
          </form>
        </div>

        {/* 区切り線 */}
        <div className="relative">
          <div className="absolute inset-0 flex items-center">
            <div className="w-full border-t border-gray-300" />
          </div>
          <div className="relative flex justify-center text-sm">
            <span className="px-2 bg-white text-gray-500">または</span>
          </div>
        </div>

        {/* メール・パスワードフォーム（フォールバック） */}
        <EmailPasswordForm callbackUrl={callbackUrl} />

        <p className="text-center text-xs text-gray-500">
          ログインすることで、
          <a href="/terms" className="underline">利用規約</a>
          と
          <a href="/privacy" className="underline">プライバシーポリシー</a>
          に同意したものとみなします。
        </p>
      </div>
    </div>
  );
}
```

### 8.2 UX のベストプラクティス

```
ソーシャルログインの UX ベストプラクティス:

  ① ボタンの順序と配置:
  ┌────────────────────────────────────────────────────┐
  │                                                    │
  │  推奨順序（利用率順）:                               │
  │    1. Google（最も普及、約 60%）                     │
  │    2. Apple（iOS ユーザーに人気）                    │
  │    3. GitHub（開発者向けサービスの場合は上位に）      │
  │    4. メール・パスワード（フォールバック）            │
  │                                                    │
  │  ターゲットに応じた調整:                             │
  │  ・開発者向け → GitHub を最上位に                    │
  │  ・B2B SaaS  → Google + Microsoft を上位に         │
  │  ・ゲーム    → Discord を追加                       │
  │  ・日本市場  → LINE ログインを検討                   │
  │                                                    │
  └────────────────────────────────────────────────────┘

  ② ボタンのデザイン:
  ┌────────────────────────────────────────────────────┐
  │                                                    │
  │  ✓ 推奨:                                           │
  │  → 各プロバイダーのブランドガイドラインに準拠        │
  │  → 「Continue with Google」形式のラベル             │
  │  → 公式ロゴの使用                                  │
  │  → 十分な大きさのタッチターゲット（44px 以上）       │
  │                                                    │
  │  ✗ 避けるべき:                                     │
  │  → 「Sign in with」ではなく「Continue with」を使用  │
  │    （サインアップとログインの区別をなくす）           │
  │  → 自前のアイコン使用（ブランドガイドライン違反）    │
  │  → 5 個以上のプロバイダーを並べない（選択疲れ）      │
  │  → プロバイダー名だけのテキストリンク               │
  │                                                    │
  └────────────────────────────────────────────────────┘

  ③ エラーハンドリング:
  ┌────────────────────────────────────────────────────┐
  │                                                    │
  │  ユーザーに分かりやすいメッセージ:                    │
  │                                                    │
  │  ✓ 「このメールアドレスは Google で登録されています。│
  │     Google でログインしてください。」                │
  │                                                    │
  │  ✗ 「OAuthAccountNotLinked」                       │
  │                                                    │
  │  ✓ 「ログインがキャンセルされました。               │
  │     もう一度お試しください。」                       │
  │                                                    │
  │  ✗ 「access_denied: user denied access」           │
  │                                                    │
  └────────────────────────────────────────────────────┘

  ④ callbackUrl の処理:
     → ログイン前にアクセスしようとしたページに戻す
     → オープンリダイレクト攻撃を防ぐためバリデーション必須
     → 外部 URL へのリダイレクトを禁止
```

```typescript
// callbackUrl のバリデーション
function validateCallbackUrl(url: string | undefined): string {
  const defaultUrl = '/dashboard';

  if (!url) return defaultUrl;

  try {
    const parsed = new URL(url, process.env.NEXTAUTH_URL);
    const appHost = new URL(process.env.NEXTAUTH_URL!).host;

    // 同一ホストのみ許可（オープンリダイレクト防止）
    if (parsed.host !== appHost) {
      return defaultUrl;
    }

    // 特定のパスを禁止
    const blockedPaths = ['/api/', '/auth/'];
    if (blockedPaths.some((p) => parsed.pathname.startsWith(p))) {
      return defaultUrl;
    }

    return parsed.pathname + parsed.search;
  } catch {
    return defaultUrl;
  }
}
```

---

## 9. セキュリティ強化

### 9.1 state パラメータと CSRF 防御

```
OAuth の state パラメータ:

  目的: CSRF 攻撃の防止

  攻撃シナリオ（state なし）:
  ┌────────────────────────────────────────────────────┐
  │                                                    │
  │  ① 攻撃者が自分の Google アカウントで OAuth 開始    │
  │  ② 攻撃者がリダイレクト URL（code 付き）を取得      │
  │  ③ 攻撃者がこの URL を被害者に踏ませる             │
  │  ④ 被害者のブラウザが code をサーバーに送信         │
  │  ⑤ サーバーが攻撃者のアカウントでセッション作成     │
  │  ⑥ 被害者が攻撃者のアカウントで操作（情報漏洩）     │
  │                                                    │
  └────────────────────────────────────────────────────┘

  Auth.js は state を自動で処理:
  → ランダムな state 値を生成
  → Cookie に保存（HttpOnly, SameSite=Lax）
  → コールバック時に一致を検証
  → 不一致の場合は認証を拒否

  PKCE も同様に Auth.js が自動処理:
  → code_verifier を生成・保存
  → code_challenge を認可リクエストに含める
  → トークン交換時に code_verifier を送信
```

### 9.2 Token 保管のセキュリティ

```typescript
// Access Token / Refresh Token の安全な保管

// Prisma Schema - Token の暗号化
// model Account {
//   ...
//   access_token_encrypted   String?
//   refresh_token_encrypted  String?
//   ...
// }

import { createCipheriv, createDecipheriv, randomBytes } from 'crypto';

const ENCRYPTION_KEY = Buffer.from(process.env.TOKEN_ENCRYPTION_KEY!, 'hex');
// 32 bytes = 256-bit key for AES-256

function encryptToken(token: string): string {
  const iv = randomBytes(16);
  const cipher = createCipheriv('aes-256-gcm', ENCRYPTION_KEY, iv);

  let encrypted = cipher.update(token, 'utf8', 'hex');
  encrypted += cipher.final('hex');

  const authTag = cipher.getAuthTag().toString('hex');

  // iv:authTag:encrypted の形式で保存
  return `${iv.toString('hex')}:${authTag}:${encrypted}`;
}

function decryptToken(encryptedToken: string): string {
  const [ivHex, authTagHex, encrypted] = encryptedToken.split(':');

  const iv = Buffer.from(ivHex, 'hex');
  const authTag = Buffer.from(authTagHex, 'hex');
  const decipher = createDecipheriv('aes-256-gcm', ENCRYPTION_KEY, iv);
  decipher.setAuthTag(authTag);

  let decrypted = decipher.update(encrypted, 'hex', 'utf8');
  decrypted += decipher.final('utf8');

  return decrypted;
}

// Auth.js のアダプターをラップして暗号化
import { PrismaAdapter } from '@auth/prisma-adapter';

function encryptedAdapter(prisma: PrismaClient) {
  const adapter = PrismaAdapter(prisma);

  return {
    ...adapter,
    async linkAccount(account: any) {
      // トークンを暗号化してから保存
      const encryptedAccount = {
        ...account,
        access_token: account.access_token
          ? encryptToken(account.access_token)
          : null,
        refresh_token: account.refresh_token
          ? encryptToken(account.refresh_token)
          : null,
      };
      return adapter.linkAccount!(encryptedAccount);
    },
  };
}
```

### 9.3 レート制限とブルートフォース防止

```typescript
// ソーシャルログインのレート制限
// 短時間での大量のログイン試行を防止

import { Ratelimit } from '@upstash/ratelimit';
import { Redis } from '@upstash/redis';

const loginRatelimit = new Ratelimit({
  redis: Redis.fromEnv(),
  limiter: Ratelimit.slidingWindow(5, '1 m'), // 1分に5回まで
  analytics: true,
  prefix: 'ratelimit:login',
});

// middleware.ts でレート制限を適用
export async function middleware(request: NextRequest) {
  // /api/auth/signin へのリクエストをレート制限
  if (request.nextUrl.pathname.startsWith('/api/auth/signin')) {
    const ip = request.ip ?? request.headers.get('x-forwarded-for') ?? 'unknown';
    const { success, remaining, reset } = await loginRatelimit.limit(ip);

    if (!success) {
      return new NextResponse('Too Many Requests', {
        status: 429,
        headers: {
          'Retry-After': String(Math.ceil((reset - Date.now()) / 1000)),
          'X-RateLimit-Remaining': String(remaining),
        },
      });
    }
  }

  return NextResponse.next();
}
```

---

## 10. エッジケースとトラブルシューティング

### 10.1 よくある問題と解決策

```
ソーシャルログインのトラブルシューティング:

  ┌─────────────────────┬──────────────────────────────────────┐
  │ 問題                 │ 解決策                               │
  ├─────────────────────┼──────────────────────────────────────┤
  │ OAuthAccountNotLinked│ 同じメールが別プロバイダーで登録済み   │
  │                     │ → アカウントリンク機能を実装          │
  ├─────────────────────┼──────────────────────────────────────┤
  │ redirect_uri_mismatch│ コールバック URL が不一致              │
  │                     │ → プロバイダー設定を確認              │
  │                     │ → 末尾スラッシュの有無に注意          │
  ├─────────────────────┼──────────────────────────────────────┤
  │ CSRF エラー          │ state の不一致                       │
  │                     │ → Cookie 設定を確認                  │
  │                     │ → SameSite=Lax を確認                │
  ├─────────────────────┼──────────────────────────────────────┤
  │ Token 期限切れ       │ Refresh Token が取得できていない      │
  │                     │ → Google: prompt=consent +           │
  │                     │   access_type=offline を設定          │
  ├─────────────────────┼──────────────────────────────────────┤
  │ Apple name が null  │ 初回以降は name が返されない          │
  │                     │ → 初回ログイン時に必ず保存           │
  │                     │ → Apple ID 設定から再認可            │
  ├─────────────────────┼──────────────────────────────────────┤
  │ GitHub email null   │ メール非公開設定                      │
  │                     │ → user:email スコープ +              │
  │                     │   /user/emails API で取得            │
  ├─────────────────────┼──────────────────────────────────────┤
  │ NEXT_REDIRECT error │ Server Action 内で signIn 呼出し      │
  │                     │ → Next.js の仕様。try-catch で       │
  │                     │   NEXT_REDIRECT を再 throw          │
  ├─────────────────────┼──────────────────────────────────────┤
  │ 開発環境で動かない   │ localhost の設定漏れ                  │
  │                     │ → Google/GitHub: localhost を追加    │
  │                     │ → Apple: ngrok 等のトンネルが必要    │
  └─────────────────────┴──────────────────────────────────────┘
```

### 10.2 デバッグ方法

```typescript
// Auth.js のデバッグモードを有効化
// auth.ts
export const { handlers, auth, signIn, signOut } = NextAuth({
  debug: process.env.NODE_ENV === 'development',
  // これにより、コンソールに詳細なログが出力される

  logger: {
    error(code, ...message) {
      console.error(`[Auth Error] ${code}:`, ...message);
    },
    warn(code, ...message) {
      console.warn(`[Auth Warn] ${code}:`, ...message);
    },
    debug(code, ...message) {
      if (process.env.NODE_ENV === 'development') {
        console.log(`[Auth Debug] ${code}:`, ...message);
      }
    },
  },

  events: {
    async signIn({ user, account, profile, isNewUser }) {
      console.log('[Auth Event] signIn:', {
        userId: user.id,
        provider: account?.provider,
        isNewUser,
      });
    },
    async signOut(message) {
      console.log('[Auth Event] signOut:', message);
    },
    async linkAccount({ user, account }) {
      console.log('[Auth Event] linkAccount:', {
        userId: user.id,
        provider: account.provider,
      });
    },
    async createUser({ user }) {
      console.log('[Auth Event] createUser:', { userId: user.id });
    },
  },
});
```

---

## 11. アンチパターン

### 11.1 フロントエンドに Client Secret を露出

```typescript
// ✗ 危険: Client Secret をフロントエンドに露出
// フロントエンドのコード
const response = await fetch('https://oauth2.googleapis.com/token', {
  method: 'POST',
  body: JSON.stringify({
    client_id: 'xxx',
    client_secret: 'EXPOSED_SECRET', // 絶対にダメ！
    code: authorizationCode,
  }),
});

// ✓ 正しい: サーバーサイドでトークン交換
// Auth.js がサーバーサイドで自動的に処理
// Client Secret は環境変数から読み取り、サーバーでのみ使用
```

### 11.2 email_verified を確認せずにアカウントリンク

```typescript
// ✗ 危険: メール検証なしの自動リンク
callbacks: {
  async signIn({ user, account }) {
    const existing = await prisma.user.findUnique({
      where: { email: user.email! },
    });
    if (existing) {
      // メール検証なしでリンク → アカウント乗っ取り可能！
      await prisma.account.create({
        data: { userId: existing.id, ...account },
      });
    }
    return true;
  },
}

// ✓ 正しい: email_verified を確認してからリンク
// （前述の checkEmailVerification 関数を使用）
```

### 11.3 Apple の初回データを保存し忘れる

```typescript
// ✗ 問題: Apple のプロフィール情報を初回に保存していない
callbacks: {
  async signIn({ user, account }) {
    // Apple の name が null でも気にせず進む
    // → 2 回目以降は name が取得できなくなる！
    return true;
  },
}

// ✓ 正しい: 初回ログイン時に name / email を確実に保存
// （前述の Apple 実装を参照）
```

---

## 12. 演習問題

### 演習 1: 基本 — Google + GitHub ログインの実装（難易度: 基本）

```
課題:
  Next.js App Router + Auth.js v5 で、Google と GitHub の
  ソーシャルログインを実装してください。

要件:
  ① Google と GitHub プロバイダーを設定
  ② ログインページにプロバイダーボタンを配置
  ③ ログイン後にダッシュボードにリダイレクト
  ④ ユーザー情報をヘッダーに表示（名前 + アバター）
  ⑤ ログアウト機能

ヒント:
  → auth.ts でプロバイダーを設定
  → app/api/auth/[...nextauth]/route.ts でハンドラーを公開
  → SessionProvider でクライアントコンポーネントを wrap

確認ポイント:
  □ Google でログインできるか
  □ GitHub でログインできるか
  □ ログアウトが動作するか
  □ セッションが正しく保持されるか
```

### 演習 2: 応用 — アカウントリンク機能の実装（難易度: 応用）

```
課題:
  演習 1 の上に、条件付き自動アカウントリンク機能を
  実装してください。

要件:
  ① email_verified が true の場合のみ自動リンク
  ② 設定画面でリンク済みアカウント一覧を表示
  ③ アカウントのリンク解除機能
  ④ 最後のログイン方法は解除不可
  ⑤ リンク / 解除のイベントを監査ログに記録

ヒント:
  → signIn コールバックでリンク処理
  → /settings/accounts ページを作成
  → Server Actions で link/unlink を実装

確認ポイント:
  □ Google で登録 → 同じメールの GitHub で自動リンク
  □ アカウント一覧が正しく表示される
  □ 2 つ以上のアカウントがある場合のみ解除可能
  □ 監査ログが記録される
```

### 演習 3: 発展 — マルチテナント対応ソーシャルログイン（難易度: 発展）

```
課題:
  複数の組織（テナント）を持つ SaaS アプリケーションで、
  組織ごとに許可するプロバイダーを制御するソーシャルログインを
  実装してください。

要件:
  ① 組織ごとに許可プロバイダーを設定（管理画面）
  ② Google Workspace ドメイン制限
  ③ GitHub 組織メンバーシップによるアクセス制御
  ④ 新規ユーザーの自動組織割り当て（メールドメインベース）
  ⑤ 組織ごとの SSO 設定（Google Workspace / Azure AD）
  ⑥ ユーザーが複数組織に所属可能

ヒント:
  → Organization モデルに allowedProviders を追加
  → signIn コールバックでドメイン / 組織チェック
  → jwt コールバックで組織情報をトークンに含める

確認ポイント:
  □ 組織 A は Google のみ、組織 B は GitHub のみ
  □ ドメイン制限が正しく動作
  □ 新規ユーザーが正しい組織に割り当てられる
  □ 組織切り替えが動作する
```

---

## 13. FAQ

### Q1: ソーシャルログインのみでパスワードなしの運用は安全ですか？

```
A: 安全です。むしろパスワードレスの方がセキュリティは高くなります。

理由:
  → パスワードの漏洩リスクがない
  → ブルートフォース攻撃の対象にならない
  → プロバイダー側の 2FA / MFA を活用できる
  → パスワードリセットフローが不要

注意点:
  → プロバイダーのアカウントが乗っ取られた場合のリスク
  → プロバイダーのサービス障害時にログインできなくなる
  → 複数プロバイダーの提供を推奨（フォールバック用）
```

### Q2: 同じユーザーが Google と GitHub で異なるメールを使っている場合は？

```
A: 自動リンクは不可能です。手動リンク機能を提供します。

推奨フロー:
  ① ユーザーが Google (alice@gmail.com) でログイン
  ② 設定画面で「GitHub を連携」をクリック
  ③ GitHub OAuth フローを開始
  ④ コールバックで既存ユーザーにアカウントを追加
  ⑤ 以後、どちらでもログイン可能

実装のポイント:
  → ログイン済みの状態でリンクフローを開始
  → signIn コールバックでセッションの userId を確認
  → 別のメールでも同一ユーザーに紐付け可能
```

### Q3: Apple の Private Relay メールでメール送信ができません

```
A: Apple の Private Email Relay の設定が必要です。

手順:
  ① Apple Developer > Certificates > More > Configure
  ② 「Email Sources」に送信元ドメインを登録
  ③ SPF レコードの設定:
     v=spf1 include:_spf.appleid.apple.com ~all
  ④ DKIM の設定（メールサービスプロバイダーの手順に従う）
  ⑤ 登録したドメインからのみ送信可能

注意:
  → Apple が中継するため、バウンスの検知が難しい
  → ユーザーが Apple ID 設定から転送を停止する可能性
  → 重要な通知にはアプリ内通知も併用
```

### Q4: Auth.js v4 から v5 への移行でソーシャルログインの変更点は？

```
A: 主な変更点:

  ① import パスの変更:
     v4: import GoogleProvider from 'next-auth/providers/google'
     v5: import Google from 'next-auth/providers/google'
     → Provider は default export に変更

  ② 設定ファイルの構造:
     v4: pages/api/auth/[...nextauth].ts
     v5: auth.ts（ルートレベル）+ app/api/auth/[...nextauth]/route.ts

  ③ Adapter の変更:
     v4: @next-auth/prisma-adapter
     v5: @auth/prisma-adapter

  ④ コールバックの引数:
     v4: profile パラメータの型が any
     v5: profile の型がプロバイダーごとに定義

  ⑤ signIn / signOut の呼び出し:
     v4: import { signIn } from 'next-auth/react'
     v5: サーバー: import { signIn } from '@/auth'
         クライアント: import { signIn } from 'next-auth/react'
```

---

## まとめ

| 項目 | ポイント |
|------|---------|
| プロトコル | Google/Apple は OIDC、GitHub は OAuth 2.0 |
| Google | prompt=consent + access_type=offline で Refresh Token 取得 |
| GitHub | email が null の場合あり。/user/emails API で取得 |
| Apple | name/email は初回のみ。clientSecret は JWT 動的生成 |
| アカウントリンク | email_verified の場合のみ自動リンク。手動リンク UI も提供 |
| セキュリティ | state + PKCE は Auth.js が自動処理。Token は暗号化保管 |
| UX | 「Continue with」形式、ブランドガイドライン遵守、エラーは具体的に |

---

## 次に読むべきガイド

- [[02-email-password-auth.md]] — メール・パスワード認証
- [[00-nextauth-setup.md]] — NextAuth.js セットアップ
- [[../01-session-auth/01-session-store.md]] — セッションストア
- [[../03-authorization/03-frontend-authorization.md]] — フロントエンド認可

---

## 参考文献

1. Auth.js. "Providers." authjs.dev, 2024.
2. Google. "Sign in with Google for Web." developers.google.com/identity, 2024.
3. Apple. "Sign in with Apple." developer.apple.com/sign-in-with-apple, 2024.
4. GitHub. "Authorizing OAuth Apps." docs.github.com, 2024.
5. IETF. "RFC 6749 — The OAuth 2.0 Authorization Framework." tools.ietf.org, 2012.
6. IETF. "RFC 7636 — Proof Key for Code Exchange (PKCE)." tools.ietf.org, 2015.
7. OpenID Foundation. "OpenID Connect Core 1.0." openid.net/specs, 2014.
8. OWASP. "OAuth 2.0 Security." cheatsheetseries.owasp.org, 2024.
