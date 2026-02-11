# 認証と認可の基礎

> 認証（Authentication）は「あなたは誰か？」、認可（Authorization）は「あなたに何が許されているか？」。この2つの根本的な違いを理解し、脅威モデル、セキュリティ原則、認証フローの全体像を把握することが、安全なシステム構築の第一歩となる。

## この章で学ぶこと

- [ ] 認証と認可の違いを正確に理解する
- [ ] 主要な脅威モデルとセキュリティ原則を把握する
- [ ] 認証・認可の全体アーキテクチャを設計できるようになる

---

## 1. 認証と認可の違い

```
認証（Authentication / AuthN）:
  質問: 「あなたは誰ですか？」
  目的: ユーザーの身元を確認する
  例:   パスワード入力、指紋認証、Google ログイン

  ┌─────────────────────────────────────────┐
  │  ユーザー: 「私は alice@example.com です」  │
  │  システム: 「パスワードで証明してください」     │
  │  ユーザー: 「********」                    │
  │  システム: 「確認しました。alice さんですね」   │
  └─────────────────────────────────────────┘

認可（Authorization / AuthZ）:
  質問: 「あなたに何が許されていますか？」
  目的: ユーザーのアクセス権限を判定する
  例:   管理画面へのアクセス、ファイルの編集権限

  ┌─────────────────────────────────────────┐
  │  alice: 「管理画面にアクセスしたい」          │
  │  システム: 「alice の権限を確認中...」         │
  │  システム: 「admin ロール確認。アクセス許可」    │
  └─────────────────────────────────────────┘

重要な関係:
  認証 → 認可 の順序（必ず認証が先）
  認証なしに認可はできない
  認証されても認可されるとは限らない
```

```
比較表:

  項目       │ 認証（AuthN）      │ 認可（AuthZ）
  ───────────┼───────────────────┼──────────────────
  質問       │ Who are you?      │ What can you do?
  タイミング  │ 最初に実行          │ 認証後に実行
  入力       │ クレデンシャル       │ ユーザー属性・ロール
  出力       │ ユーザー ID         │ 許可 / 拒否
  失敗時     │ 401 Unauthorized   │ 403 Forbidden
  技術例     │ パスワード、JWT      │ RBAC、ABAC
  変更頻度   │ ユーザー主導        │ 管理者主導
```

---

## 2. 認証の要素（Authentication Factors）

```
3つの認証要素:

  ┌──────────────────────────────────────────────┐
  │                                              │
  │  ① Something You Know（知識要素）              │
  │     → パスワード、PIN、秘密の質問               │
  │     → 最も一般的だが最も脆弱                    │
  │                                              │
  │  ② Something You Have（所有要素）              │
  │     → スマートフォン、ハードウェアキー、ICカード   │
  │     → TOTP、SMS コード、FIDO2 セキュリティキー   │
  │                                              │
  │  ③ Something You Are（生体要素）               │
  │     → 指紋、顔認証、虹彩、声紋                  │
  │     → 変更不可（漏洩時のリスクが高い）            │
  │                                              │
  └──────────────────────────────────────────────┘

多要素認証（MFA）:
  2つ以上の異なる要素を組み合わせる
  例: パスワード（知識）+ TOTP（所有）= 2FA

  ✗ パスワード + 秘密の質問 = 単一要素（両方「知識」）
  ✓ パスワード + TOTP = 多要素（「知識」+「所有」）
  ✓ パスワード + 指紋 = 多要素（「知識」+「生体」）

強度の順:
  パスワードのみ < パスワード + SMS < パスワード + TOTP < パスワード + FIDO2
```

---

## 3. 脅威モデル

```
主要な認証への脅威:

  ┌─────────────────────┬─────────────────────────────────┐
  │ 脅威                 │ 説明                            │
  ├─────────────────────┼─────────────────────────────────┤
  │ ブルートフォース      │ パスワードの総当たり攻撃           │
  │ クレデンシャル        │ 漏洩したパスワードリストで         │
  │   スタッフィング      │ 他サービスにログイン試行           │
  │ フィッシング         │ 偽サイトでクレデンシャル詐取        │
  │ セッションハイジャック │ セッションIDの窃取                │
  │ CSRF               │ 認証済みユーザーの操作を偽造        │
  │ XSS → トークン窃取  │ スクリプト注入でトークン読取        │
  │ 中間者攻撃（MITM）   │ 通信の傍受・改ざん                │
  │ リプレイ攻撃         │ 正規のリクエストを再送             │
  │ 権限昇格            │ 一般ユーザーが管理者権限を取得      │
  └─────────────────────┴─────────────────────────────────┘

攻撃フロー例（クレデンシャルスタッフィング）:

  ① 攻撃者がサービスAの漏洩データを入手
  ② alice@example.com / password123 を取得
  ③ サービスB, C, D... に同じ認証情報でログイン試行
  ④ パスワード使い回しのユーザーがアカウント侵害される

  対策:
  → パスワード漏洩チェック（Have I Been Pwned API）
  → レート制限（ログイン試行回数制限）
  → MFA の強制
  → アカウントロックアウト
```

---

## 4. セキュリティ原則

```
認証・認可のセキュリティ原則:

  ① 最小権限の原則（Principle of Least Privilege）:
     → 必要最小限の権限のみ付与
     → デフォルトは「拒否」
     → 管理者権限は必要な人にのみ

  ② 多層防御（Defense in Depth）:
     → 単一の防御に頼らない
     → ネットワーク + アプリ + DB の各層で防御
     → 1つ突破されても次の層で止める

  ③ フェイルセキュア（Fail Secure）:
     → エラー時は安全側に倒す
     → 認証エラー → アクセス拒否（許可ではなく）
     → 例外発生 → ログアウト状態に

  ④ セキュリティバイデフォルト:
     → 安全な設定をデフォルトに
     → Cookie: HttpOnly=true, Secure=true, SameSite=Lax
     → HTTPS 強制

  ⑤ ゼロトラスト:
     → 「信頼しない、常に検証する」
     → 内部ネットワークも信頼しない
     → リクエストごとに認証・認可
```

---

## 5. 認証・認可の全体アーキテクチャ

```
一般的な認証フロー:

  ユーザー         フロントエンド       バックエンド        DB / IdP
    │                 │                  │                │
    │ ログイン画面     │                  │                │
    │────────────────>│                  │                │
    │                 │ POST /auth/login │                │
    │                 │────────────────>│                │
    │                 │                  │ パスワード検証   │
    │                 │                  │───────────────>│
    │                 │                  │ ユーザー情報     │
    │                 │                  │<───────────────│
    │                 │                  │                │
    │                 │  セッション or    │                │
    │                 │  トークン発行     │                │
    │                 │<────────────────│                │
    │ ログイン成功     │                  │                │
    │<────────────────│                  │                │
    │                 │                  │                │
    │ 保護リソース要求  │                  │                │
    │────────────────>│                  │                │
    │                 │ + Cookie/Bearer  │                │
    │                 │────────────────>│                │
    │                 │                  │ 認証チェック     │
    │                 │                  │ 認可チェック     │
    │                 │                  │───────────────>│
    │                 │  200 OK + データ  │                │
    │                 │<────────────────│                │
    │ データ表示       │                  │                │
    │<────────────────│                  │                │
```

```typescript
// 認証・認可ミドルウェアの基本構造
import { NextRequest, NextResponse } from 'next/server';

// 認証ミドルウェア
async function authenticate(req: NextRequest): Promise<User | null> {
  // セッション方式
  const sessionId = req.cookies.get('session_id')?.value;
  if (sessionId) {
    return await sessionStore.get(sessionId);
  }

  // トークン方式
  const token = req.headers.get('Authorization')?.replace('Bearer ', '');
  if (token) {
    return await verifyJWT(token);
  }

  return null;
}

// 認可ミドルウェア
function authorize(user: User, resource: string, action: string): boolean {
  // RBAC: ユーザーのロールに基づく権限チェック
  const permissions = getRolePermissions(user.role);
  return permissions.includes(`${resource}:${action}`);
}

// 組み合わせ
export async function middleware(req: NextRequest) {
  // 1. 認証
  const user = await authenticate(req);
  if (!user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  // 2. 認可
  const { resource, action } = parseRequest(req);
  if (!authorize(user, resource, action)) {
    return NextResponse.json({ error: 'Forbidden' }, { status: 403 });
  }

  // 3. リクエスト続行
  return NextResponse.next();
}
```

---

## 6. 認証方式の全体像

```
認証方式の分類:

  ┌─────────────────────────────────────────────────┐
  │                                                 │
  │  サーバーサイド（ステートフル）                     │
  │  ├── セッション + Cookie                          │
  │  │   → サーバーにセッション状態を保持               │
  │  │   → Cookie でセッション ID を送信               │
  │  │   → 伝統的な Web アプリに最適                   │
  │  │                                               │
  │  クライアントサイド（ステートレス）                   │
  │  ├── JWT（JSON Web Token）                        │
  │  │   → トークンに情報を含む（自己完結型）            │
  │  │   → API 認証に最適                             │
  │  │                                               │
  │  委譲型（第三者認証）                               │
  │  ├── OAuth 2.0                                   │
  │  │   → 認可の委譲（第三者アプリへのアクセス許可）     │
  │  ├── OpenID Connect                              │
  │  │   → OAuth 2.0 上の認証レイヤー                  │
  │  ├── SAML                                        │
  │  │   → エンタープライズ SSO                        │
  │  │                                               │
  │  パスワードレス                                    │
  │  ├── Magic Link                                  │
  │  │   → メールでワンタイムリンクを送信                │
  │  ├── WebAuthn / Passkeys                         │
  │  │   → 公開鍵暗号による認証                         │
  │  │   → フィッシング耐性が最強                       │
  │  └── OTP                                         │
  │      → ワンタイムパスワード（SMS / メール）           │
  │                                                 │
  └─────────────────────────────────────────────────┘
```

---

## 7. HTTPステータスコードと認証・認可

```
認証・認可関連のHTTPステータスコード:

  401 Unauthorized（認証エラー）:
    → 認証が必要、または認証に失敗
    → WWW-Authenticate ヘッダーを返すべき
    → 例: トークン未送信、トークン期限切れ

  403 Forbidden（認可エラー）:
    → 認証済みだが権限がない
    → 再認証しても結果は変わらない
    → 例: 一般ユーザーが管理画面にアクセス

  407 Proxy Authentication Required:
    → プロキシ認証が必要
    → Proxy-Authenticate ヘッダー

よくある間違い:
  ✗ 未ログインで 403 を返す → 401 が正しい
  ✗ 権限不足で 401 を返す → 403 が正しい
  ✗ リソースの存在を隠す場合 → 404 を返す（情報漏洩防止）
```

```typescript
// 正しいレスポンスの使い分け
function handleAuthError(req: Request): Response {
  const user = getAuthenticatedUser(req);

  // 認証されていない → 401
  if (!user) {
    return new Response(JSON.stringify({ error: 'Authentication required' }), {
      status: 401,
      headers: { 'WWW-Authenticate': 'Bearer realm="api"' },
    });
  }

  // 認証済みだが権限がない → 403
  if (!hasPermission(user, req.resource)) {
    return new Response(JSON.stringify({ error: 'Insufficient permissions' }), {
      status: 403,
    });
  }

  // リソースが存在しない（ただし存在を隠したい場合）
  // → 403 ではなく 404 を返して情報漏洩を防ぐ
  if (!resourceExists(req.resource) && shouldHideExistence(req.resource)) {
    return new Response(JSON.stringify({ error: 'Not found' }), {
      status: 404,
    });
  }
}
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| 認証 | 「誰か」を確認する。失敗時は 401 |
| 認可 | 「何ができるか」を判定する。失敗時は 403 |
| 認証要素 | 知識・所有・生体の3要素。MFAは異なる要素の組合せ |
| 脅威 | ブルートフォース、フィッシング、セッションハイジャック等 |
| 原則 | 最小権限、多層防御、フェイルセキュア、ゼロトラスト |
| 方式 | セッション、JWT、OAuth 2.0、Passkeys 等 |

---

## 次に読むべきガイド
→ [[01-password-security.md]] — パスワードセキュリティ

---

## 参考文献
1. OWASP. "Authentication Cheat Sheet." cheatsheetseries.owasp.org, 2024.
2. NIST. "SP 800-63B: Digital Identity Guidelines." nist.gov, 2020.
3. RFC 7235. "Hypertext Transfer Protocol (HTTP/1.1): Authentication." IETF, 2014.
