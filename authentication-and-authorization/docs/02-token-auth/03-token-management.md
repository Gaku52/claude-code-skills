# トークン管理

> Access Token と Refresh Token の適切な管理は認証セキュリティの要。トークンのライフサイクル、Refresh Token Rotation、失効戦略、安全なストレージ、トークンの監視まで、実践的なトークン管理を解説する。

## この章で学ぶこと

- [ ] Access Token と Refresh Token の役割と運用を理解する
- [ ] Refresh Token Rotation の仕組みと必要性を把握する
- [ ] トークンの失効・ストレージ・監視を設計できるようになる

---

## 1. トークンのライフサイクル

```
Access Token と Refresh Token:

  Access Token:
  → 短命（15分〜1時間）
  → API アクセスに使用
  → 失効時は Refresh Token で更新
  → ステートレス検証（署名確認のみ）

  Refresh Token:
  → 長命（7日〜30日）
  → 新しい Access Token の取得に使用
  → サーバー側で管理（失効可能）
  → 1回使用でローテーション推奨

  ライフサイクル:

  t=0m:   ログイン → AT(15m) + RT(7d) 発行
  t=14m:  API リクエスト → AT 有効 → 成功
  t=16m:  API リクエスト → AT 期限切れ → 401
  t=16m:  RT で AT 更新 → 新AT(15m) + 新RT(7d) 発行
  t=31m:  API リクエスト → 新AT 有効 → 成功
  ...
  t=7d:   RT 期限切れ → 再ログイン要求

  なぜ2つのトークンが必要か:
  → AT のみ: 長命にするとリスク（盗まれた場合の被害が大きい）
  → AT のみ: 短命にすると頻繁にログインが必要（UX悪化）
  → 2つの組合せ: AT は短命（セキュリティ）、RT で更新（UX）
```

---

## 2. Refresh Token Rotation

```
Refresh Token Rotation の仕組み:

  通常のリフレッシュ:
    RT-1 → 新AT + RT-1（同じRTを再利用）
    → RTが漏洩すると攻撃者が永久にATを取得可能

  Rotation:
    RT-1 → 新AT + RT-2（新しいRTを発行、RT-1は無効化）
    RT-2 → 新AT + RT-3（新しいRTを発行、RT-2は無効化）
    → 各RTは1回限り使用

  攻撃検知:
    正規ユーザー: RT-2 を使用
    攻撃者:      RT-1 を使用（既に無効化済み）
    → 無効化されたRTの使用を検知
    → そのユーザーの全トークンを無効化（リーク検知）

  ┌────────────────────────────────────────────┐
  │  正常フロー:                                │
  │    RT-1 使用 → RT-2 発行（RT-1 無効化）       │
  │    RT-2 使用 → RT-3 発行（RT-2 無効化）       │
  │                                            │
  │  攻撃検知:                                   │
  │    攻撃者が RT-1 を使用 → 既に無効化           │
  │    → 全トークン無効化（家族全体を失効）          │
  │    → ユーザーに再ログインを要求                 │
  └────────────────────────────────────────────┘
```

```typescript
// Refresh Token Rotation の実装
interface RefreshTokenRecord {
  id: string;
  token: string;         // ハッシュ化済み
  userId: string;
  familyId: string;      // トークンファミリー（関連するRTのグループ）
  expiresAt: Date;
  usedAt: Date | null;   // 使用済みフラグ
  replacedBy: string | null;
}

async function refreshTokens(refreshToken: string) {
  const hashedToken = hashToken(refreshToken);

  // 現在のRTを検索
  const currentRT = await db.refreshToken.findUnique({
    where: { token: hashedToken },
  });

  if (!currentRT) {
    throw new AuthError('Invalid refresh token');
  }

  // 期限切れチェック
  if (currentRT.expiresAt < new Date()) {
    throw new AuthError('Refresh token expired');
  }

  // 再利用検知（既に使用済みのRTが使われた → リーク）
  if (currentRT.usedAt) {
    // トークンファミリー全体を無効化
    await db.refreshToken.deleteMany({
      where: { familyId: currentRT.familyId },
    });
    // セキュリティアラート
    await notifyTokenReuse(currentRT.userId);
    throw new AuthError('Refresh token reuse detected');
  }

  // 新しいトークンペアを生成
  const newAccessToken = await issueAccessToken(currentRT.userId);
  const newRefreshToken = crypto.randomBytes(32).toString('hex');
  const hashedNewRT = hashToken(newRefreshToken);

  await db.$transaction([
    // 現在のRTを使用済みに
    db.refreshToken.update({
      where: { id: currentRT.id },
      data: {
        usedAt: new Date(),
        replacedBy: hashedNewRT,
      },
    }),
    // 新しいRTを作成
    db.refreshToken.create({
      data: {
        token: hashedNewRT,
        userId: currentRT.userId,
        familyId: currentRT.familyId,  // 同じファミリー
        expiresAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000),
      },
    }),
  ]);

  return {
    accessToken: newAccessToken,
    refreshToken: newRefreshToken,
  };
}

// ログイン時（新しいトークンファミリーの開始）
async function createTokenPair(userId: string) {
  const familyId = crypto.randomUUID();
  const accessToken = await issueAccessToken(userId);
  const refreshToken = crypto.randomBytes(32).toString('hex');

  await db.refreshToken.create({
    data: {
      token: hashToken(refreshToken),
      userId,
      familyId,
      expiresAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000),
    },
  });

  return { accessToken, refreshToken };
}
```

---

## 3. クライアント側のトークン更新

```typescript
// Axios インターセプターによる自動リフレッシュ
import axios from 'axios';

const api = axios.create({ baseURL: '/api' });

let isRefreshing = false;
let failedQueue: Array<{ resolve: Function; reject: Function }> = [];

function processQueue(error: any, token: string | null) {
  failedQueue.forEach(({ resolve, reject }) => {
    if (error) reject(error);
    else resolve(token);
  });
  failedQueue = [];
}

api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;

    if (error.response?.status !== 401 || originalRequest._retry) {
      return Promise.reject(error);
    }

    // 既にリフレッシュ中なら待機キューに入れる
    if (isRefreshing) {
      return new Promise((resolve, reject) => {
        failedQueue.push({ resolve, reject });
      }).then(() => api(originalRequest));
    }

    originalRequest._retry = true;
    isRefreshing = true;

    try {
      // トークンリフレッシュ（Cookie ベースなら自動送信）
      await fetch('/api/auth/refresh', { method: 'POST' });
      processQueue(null, null);
      return api(originalRequest);
    } catch (refreshError) {
      processQueue(refreshError, null);
      // リフレッシュ失敗 → ログインページへ
      window.location.href = '/login';
      return Promise.reject(refreshError);
    } finally {
      isRefreshing = false;
    }
  }
);
```

---

## 4. トークン失効戦略

```
トークン失効が必要な場面:

  → ユーザーがログアウト
  → パスワード変更
  → アカウント無効化
  → セキュリティ侵害の検知
  → ユーザーがデバイスを紛失
  → 管理者によるセッション強制終了

失効方法の比較:

  方法            │ 即時性 │ スケーラビリティ│ 複雑度
  ──────────────┼──────┼──────────────┼──────
  短命AT のみ     │ △     │ ◎             │ 低
  ブラックリスト  │ ◎     │ △             │ 中
  Token Version │ ○     │ ○             │ 中
  RT 削除        │ △     │ ○             │ 低

  短命AT のみ:
    → AT の有効期限（15分）まで待つ
    → 最もシンプルだが即時性がない

  ブラックリスト:
    → 失効したトークンの JTI を Redis に保存
    → 各リクエストでブラックリストをチェック
    → 即時失効可能だがステートフル

  Token Version:
    → ユーザーごとにバージョン番号を管理
    → AT に version を含め、検証時に比較
    → パスワード変更時にバージョンを上げる
```

```typescript
// Token Version による失効
async function issueAccessToken(userId: string): Promise<string> {
  const user = await db.user.findUnique({ where: { id: userId } });

  return new SignJWT({
    sub: userId,
    role: user!.role,
    token_version: user!.tokenVersion,  // バージョンを含める
  })
    .setProtectedHeader({ alg: 'ES256' })
    .setExpirationTime('15m')
    .sign(privateKey);
}

// 検証時にバージョンをチェック
async function verifyAccessToken(token: string) {
  const payload = await jwtVerify(token, publicKey);

  const user = await db.user.findUnique({
    where: { id: payload.sub as string },
    select: { tokenVersion: true },
  });

  // バージョン不一致 → 失効済み
  if (user?.tokenVersion !== payload.token_version) {
    throw new AuthError('Token has been revoked');
  }

  return payload;
}

// パスワード変更時: 全トークンを無効化
async function changePassword(userId: string, newPassword: string) {
  await db.user.update({
    where: { id: userId },
    data: {
      password: await hashPassword(newPassword),
      tokenVersion: { increment: 1 },  // バージョンを上げる
    },
  });

  // Refresh Token も全削除
  await db.refreshToken.deleteMany({ where: { userId } });
}
```

---

## 5. トークン有効期限の設計

```
推奨有効期限:

  トークン種別      │ 有効期限     │ 理由
  ────────────────┼────────────┼──────────────────
  Access Token    │ 15分        │ 標準的。漏洩リスクと UX のバランス
  Refresh Token   │ 7日         │ 週1回の再ログインは許容
  ID Token        │ 1時間       │ ユーザー情報の鮮度
  Remember Me     │ 30日        │ ユーザー選択の長期セッション
  Password Reset  │ 1時間       │ 短命にして悪用リスク低減
  Email Verify    │ 24時間      │ メール確認の猶予
  API Key         │ 無期限      │ ローテーションで管理

  環境別の調整:

  一般 Web アプリ:
    AT: 15分, RT: 7日

  金融/医療:
    AT: 5分, RT: 1時間
    重要操作時に再認証

  ソーシャルメディア:
    AT: 1時間, RT: 30日
    UX 重視

  モバイルアプリ:
    AT: 15分, RT: 90日
    バイオメトリクス再認証で延長
```

---

## 6. セキュリティベストプラクティス

```
トークン管理のベストプラクティス:

  ✓ 生成:
    → 暗号的に安全なランダム値（crypto.randomBytes）
    → 十分なエントロピー（256ビット以上）
    → JWT の JTI を一意に

  ✓ 保存:
    → サーバー: ハッシュ化して保存（平文は保存しない）
    → ブラウザ: HttpOnly Cookie（localStorage は使わない）
    → モバイル: Secure Enclave / Keychain / KeyStore

  ✓ 送信:
    → HTTPS のみ（TLS 必須）
    → Cookie: Secure + HttpOnly + SameSite
    → Authorization ヘッダー: Bearer スキーム

  ✓ 検証:
    → アルゴリズムを明示的に指定
    → issuer, audience を検証
    → 有効期限を検証
    → ブラックリスト/バージョンを確認

  ✓ 失効:
    → ログアウト時に RT を削除
    → パスワード変更時に全トークン無効化
    → 異常検知時にファミリー全体を無効化

  ✗ やってはいけないこと:
    → トークンを URL クエリパラメータに含める
    → トークンをログに出力
    → 平文で DB に保存
    → localStorage に保存
    → フロントエンドでトークンをデコードして認可判定
```

---

## まとめ

| 項目 | 推奨 |
|------|------|
| Access Token 期限 | 15分 |
| Refresh Token 期限 | 7日 |
| RT Rotation | 1回使用で新RT発行 |
| 再利用検知 | ファミリー全体を無効化 |
| 保存場所 | HttpOnly Cookie |
| 失効 | Token Version + RT 削除 |

---

## 次に読むべきガイド
→ [[../03-authorization/00-rbac.md]] — RBAC

---

## 参考文献
1. Auth0. "Refresh Token Rotation." auth0.com/docs, 2024.
2. RFC 6749 §1.5. "Refresh Token." IETF, 2012.
3. OWASP. "JSON Web Token Cheat Sheet." cheatsheetseries.owasp.org, 2024.
