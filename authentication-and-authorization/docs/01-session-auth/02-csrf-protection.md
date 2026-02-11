# CSRF 防御

> CSRF（Cross-Site Request Forgery）は認証済みユーザーの操作を偽造する攻撃。Synchronizer Token パターン、Double Submit Cookie、SameSite Cookie、Origin ヘッダー検証まで、CSRF 攻撃の仕組みと多層防御を解説する。

## この章で学ぶこと

- [ ] CSRF 攻撃の仕組みとリスクを理解する
- [ ] 主要な CSRF 防御パターンを実装できるようになる
- [ ] SameSite Cookie による防御の効果と限界を把握する

---

## 1. CSRF 攻撃の仕組み

```
CSRF 攻撃フロー:

  ① ユーザーが bank.com にログイン中（Cookie 有効）
  ② 攻撃者が evil.com に以下のHTMLを仕込む:
     <form action="https://bank.com/transfer" method="POST">
       <input type="hidden" name="to" value="attacker" />
       <input type="hidden" name="amount" value="1000000" />
     </form>
     <script>document.forms[0].submit();</script>
  ③ ユーザーが evil.com を訪問
  ④ フォームが自動送信
  ⑤ ブラウザが bank.com の Cookie を自動付与
  ⑥ bank.com はユーザーからの正規リクエストと判断
  ⑦ 送金が実行される

  なぜ成功するか:
  → ブラウザはクロスサイトリクエストでも Cookie を送信する
  → サーバーは Cookie のみでユーザーを認証している
  → リクエストが正規のユーザーからか判別できない

  攻撃例:
  → 送金、購入、パスワード変更
  → メールアドレス変更 → アカウント乗っ取り
  → 管理操作（ユーザー削除、権限変更）
```

---

## 2. 防御パターン

```
CSRF 防御の4つのパターン:

  ① Synchronizer Token Pattern（同期トークン）:
     → サーバーがランダムトークンを生成
     → フォームに hidden field として埋め込み
     → サーバーでトークンを検証
     → 最も確実な防御

  ② Double Submit Cookie:
     → トークンを Cookie と リクエスト両方に設定
     → 両者が一致するか検証
     → サーバーに状態不要

  ③ SameSite Cookie:
     → Cookie の SameSite 属性で制御
     → ブラウザレベルの防御
     → 追加実装不要

  ④ Origin / Referer ヘッダー検証:
     → リクエスト元のオリジンを検証
     → 補助的な防御
     → ヘッダーが省略される場合がある

  推奨: ③ SameSite=Lax + ① or ② の組合せ
```

```typescript
// ① Synchronizer Token Pattern
import crypto from 'crypto';

// トークン生成（セッションに紐付け）
function generateCSRFToken(sessionId: string): string {
  const token = crypto.randomBytes(32).toString('hex');
  // セッションにトークンを保存
  sessionStore.setCSRFToken(sessionId, token);
  return token;
}

// トークン検証ミドルウェア
async function csrfProtection(req: Request, res: Response, next: Function) {
  // GET, HEAD, OPTIONS はスキップ（安全なメソッド）
  if (['GET', 'HEAD', 'OPTIONS'].includes(req.method)) {
    return next();
  }

  const sessionId = req.cookies.session_id;
  const token = req.headers['x-csrf-token'] || req.body._csrf;
  const storedToken = await sessionStore.getCSRFToken(sessionId);

  if (!token || !storedToken || token !== storedToken) {
    return res.status(403).json({ error: 'Invalid CSRF token' });
  }

  // トークンを再生成（ワンタイム使用）
  const newToken = generateCSRFToken(sessionId);
  res.setHeader('X-CSRF-Token', newToken);

  next();
}

// フォームに埋め込み（サーバーサイドレンダリング）
// <input type="hidden" name="_csrf" value="${csrfToken}" />

// SPA の場合: メタタグ or API で取得
// <meta name="csrf-token" content="${csrfToken}" />
```

```typescript
// ② Double Submit Cookie
function doubleSubmitCSRF(req: Request, res: Response, next: Function) {
  if (['GET', 'HEAD', 'OPTIONS'].includes(req.method)) {
    // GET 時にトークンを Cookie に設定
    if (!req.cookies['csrf-token']) {
      const token = crypto.randomBytes(32).toString('hex');
      res.cookie('csrf-token', token, {
        httpOnly: false,   // JavaScript で読める必要あり
        secure: true,
        sameSite: 'strict',
        path: '/',
      });
    }
    return next();
  }

  // POST 時: Cookie とヘッダーのトークンを比較
  const cookieToken = req.cookies['csrf-token'];
  const headerToken = req.headers['x-csrf-token'];

  if (!cookieToken || !headerToken || cookieToken !== headerToken) {
    return res.status(403).json({ error: 'CSRF validation failed' });
  }

  next();
}

// クライアント側: リクエスト時にヘッダーに設定
// const csrfToken = document.cookie.match(/csrf-token=([^;]+)/)?.[1];
// fetch('/api/data', {
//   method: 'POST',
//   headers: { 'X-CSRF-Token': csrfToken },
//   body: JSON.stringify(data),
// });
```

---

## 3. SameSite Cookie による防御

```
SameSite 属性の効果:

  SameSite=Strict:
    → クロスサイトリクエストで Cookie を一切送信しない
    → CSRF を完全に防御
    → ただし: 外部リンクからのアクセスで未ログイン状態になる
    → 例: Google検索から bank.com をクリック → ログイン画面

  SameSite=Lax（推奨デフォルト）:
    → トップレベルの GET ナビゲーションのみ Cookie 送信
    → POST, iframe, img, fetch 等のクロスサイトリクエストはブロック
    → CSRF の主要な攻撃ベクターを防御
    → UX への影響が少ない

  SameSite=None:
    → すべてのクロスサイトリクエストで Cookie 送信
    → Secure 属性が必須
    → サードパーティ Cookie が必要な場合のみ

  SameSite=Lax でブロックされるもの:
    ✓ <form method="POST"> の自動送信
    ✓ <img src="https://bank.com/transfer?to=attacker">
    ✓ fetch('https://bank.com/api', { method: 'POST' })
    ✓ <iframe src="https://bank.com/transfer">

  SameSite=Lax で許可されるもの:
    ✓ <a href="https://bank.com">リンクをクリック</a>（GET）
    → GETリクエストが安全であることが前提
    → 状態変更はPOSTで行う設計が重要
```

```
SameSite の限界:

  ① GET リクエストでの状態変更:
     → GET /delete-account のようなAPIは SameSite=Lax でも攻撃可能
     → 対策: 状態変更は必ず POST/PUT/DELETE を使用

  ② サブドメイン間:
     → SameSite は eTLD+1 で判定
     → app.example.com と evil.example.com は同一サイト
     → サブドメインの信頼性に依存

  ③ 古いブラウザ:
     → SameSite をサポートしないブラウザが存在
     → 追加の防御策との併用が推奨
```

---

## 4. Next.js での CSRF 対策

```typescript
// Next.js App Router での CSRF 対策

// Server Actions は自動的に CSRF 保護される
// Next.js が内部的に Origin ヘッダーを検証

// API Routes の場合は手動で対策が必要
// middleware.ts
import { NextRequest, NextResponse } from 'next/server';

export function middleware(request: NextRequest) {
  // API ルートの POST/PUT/DELETE を保護
  if (
    request.nextUrl.pathname.startsWith('/api/') &&
    !['GET', 'HEAD', 'OPTIONS'].includes(request.method)
  ) {
    const origin = request.headers.get('origin');
    const host = request.headers.get('host');

    // Origin ヘッダーの検証
    if (origin) {
      const originUrl = new URL(origin);
      if (originUrl.host !== host) {
        return NextResponse.json(
          { error: 'CSRF validation failed' },
          { status: 403 }
        );
      }
    }
  }

  return NextResponse.next();
}
```

---

## 5. CSRF 対策が不要なケース

```
CSRF 対策が不要な場合:

  ① Bearer トークン認証:
     → Authorization ヘッダーは自動送信されない
     → JavaScript で明示的に設定する必要がある
     → 攻撃者はトークンを設定できない

  ② SameSite=Strict の Cookie:
     → クロスサイトリクエストで Cookie が送信されない
     → ただし UX に影響

  ③ API Key 認証:
     → カスタムヘッダーで送信
     → 自動送信されない

  CSRF 対策が必要な場合:
     → Cookie ベースのセッション認証
     → SameSite=None の Cookie
     → SameSite=Lax で GET に状態変更がある場合
```

---

## まとめ

| 防御方法 | 効果 | 状態 | 推奨 |
|---------|------|------|------|
| SameSite=Lax | 高 | 不要 | 最低限 |
| Synchronizer Token | 最高 | 必要 | 確実な防御 |
| Double Submit Cookie | 高 | 不要 | SPA向け |
| Origin 検証 | 中 | 不要 | 補助的 |

---

## 次に読むべきガイド
→ [[../02-token-auth/00-jwt-deep-dive.md]] — JWT 詳解

---

## 参考文献
1. OWASP. "Cross-Site Request Forgery Prevention Cheat Sheet." cheatsheetseries.owasp.org, 2024.
2. MDN. "SameSite cookies." developer.mozilla.org, 2024.
3. Next.js. "Server Actions and Mutations." nextjs.org/docs, 2024.
