# API 認可

> API のアクセス制御はサービスのセキュリティの要。スコープ設計、API キー管理、リソースベース認可、レート制限、マルチテナント API まで、安全な API 認可の設計と実装を解説する。

## この章で学ぶこと

- [ ] OAuth スコープの設計と実装を理解する
- [ ] API キーの安全な管理パターンを把握する
- [ ] リソースベースの認可をミドルウェアで実装できるようになる

---

## 1. スコープ設計

```
OAuth スコープの設計:

  命名パターン: resource:action

  例（GitHub 風）:
    read:user         — ユーザー情報読取
    user:email        — メールアドレス読取
    repo              — リポジトリ全般
    repo:status       — コミットステータス
    admin:org         — 組織管理

  例（Google 風）:
    https://www.googleapis.com/auth/userinfo.email
    https://www.googleapis.com/auth/drive.readonly
    https://www.googleapis.com/auth/calendar.events

  推奨パターン:
    → resource:action 形式が最も明確
    → 粒度は API の用途に合わせる
    → read は readonly の意味（副作用なし）

  スコープの粒度:

  粗すぎる:
    ✗ "all" — 全権限（危険）
    ✗ "admin" — 曖昧

  適切:
    ✓ "articles:read" — 記事の読取
    ✓ "articles:write" — 記事の作成・更新
    ✓ "users:read" — ユーザー情報の読取

  細かすぎる:
    △ "articles:read:title" — 記事タイトルのみ（過度）
    △ "users:read:email:verified" — 過度に細分化
```

```typescript
// スコープベースの API 認可

// スコープ定義
const SCOPES = {
  'articles:read': 'Read articles',
  'articles:write': 'Create and update articles',
  'articles:delete': 'Delete articles',
  'users:read': 'Read user profiles',
  'users:write': 'Update user profiles',
  'admin': 'Full administrative access',
} as const;

type Scope = keyof typeof SCOPES;

// スコープ検証ミドルウェア
function requireScope(...requiredScopes: Scope[]) {
  return (req: Request, res: Response, next: Function) => {
    const tokenScopes = req.auth?.scopes as Scope[] || [];

    // admin スコープは全権限
    if (tokenScopes.includes('admin')) return next();

    const hasAll = requiredScopes.every((s) => tokenScopes.includes(s));
    if (!hasAll) {
      return res.status(403).json({
        error: 'insufficient_scope',
        required_scopes: requiredScopes,
        granted_scopes: tokenScopes,
      });
    }

    next();
  };
}

// 使用例
app.get('/api/articles', requireScope('articles:read'), listArticles);
app.post('/api/articles', requireScope('articles:write'), createArticle);
app.delete('/api/articles/:id', requireScope('articles:delete'), deleteArticle);
```

---

## 2. API キー管理

```
API キーの設計:

  構造:
    prefix_randomstring
    例: sk_live_EXAMPLE_KEY_DO_NOT_USE_1234567890

  プレフィックス:
    sk_live_  — 本番用 Secret Key
    sk_test_  — テスト用 Secret Key
    pk_live_  — 本番用 Public Key
    pk_test_  — テスト用 Public Key

  セキュリティ:
  → API キーはハッシュ化して保存
  → 作成時のみ平文を表示（以降は取得不可）
  → プレフィックスは平文で保存（検索用）
```

```typescript
// API キーの生成と管理
import crypto from 'crypto';

// API キー生成
function generateApiKey(type: 'secret' | 'public', env: 'live' | 'test'): {
  key: string;
  prefix: string;
  hash: string;
} {
  const prefixMap = {
    'secret-live': 'sk_live_',
    'secret-test': 'sk_test_',
    'public-live': 'pk_live_',
    'public-test': 'pk_test_',
  };

  const prefix = prefixMap[`${type}-${env}`];
  const randomPart = crypto.randomBytes(24).toString('base64url');
  const key = `${prefix}${randomPart}`;
  const hash = crypto.createHash('sha256').update(key).digest('hex');

  return { key, prefix, hash };
}

// API キー保存
async function createApiKey(userId: string, name: string, scopes: string[]) {
  const { key, prefix, hash } = generateApiKey('secret', 'live');

  await db.apiKey.create({
    data: {
      userId,
      name,
      prefix,          // 検索・表示用（平文）
      keyHash: hash,   // 検証用（ハッシュ）
      scopes,
      lastUsedAt: null,
      expiresAt: null,  // 無期限（ローテーションで管理）
    },
  });

  // この時点でのみ完全なキーを返す
  return { key, prefix };
}

// API キー検証
async function validateApiKey(key: string): Promise<ApiKeyData | null> {
  const hash = crypto.createHash('sha256').update(key).digest('hex');

  const apiKey = await db.apiKey.findUnique({
    where: { keyHash: hash },
    include: { user: true },
  });

  if (!apiKey) return null;
  if (apiKey.revokedAt) return null;
  if (apiKey.expiresAt && apiKey.expiresAt < new Date()) return null;

  // 最終使用日時を更新（非同期、リクエストをブロックしない）
  db.apiKey.update({
    where: { id: apiKey.id },
    data: { lastUsedAt: new Date() },
  }).catch(() => {});

  return {
    userId: apiKey.userId,
    scopes: apiKey.scopes,
    keyId: apiKey.id,
  };
}

// API キー認証ミドルウェア
async function apiKeyAuth(req: Request, res: Response, next: Function) {
  const key = req.headers['x-api-key'] as string
    || req.headers.authorization?.replace('Bearer ', '');

  if (!key) {
    return res.status(401).json({ error: 'API key required' });
  }

  const apiKeyData = await validateApiKey(key);
  if (!apiKeyData) {
    return res.status(401).json({ error: 'Invalid API key' });
  }

  req.auth = apiKeyData;
  next();
}
```

---

## 3. リソースベース認可

```typescript
// リソースの所有者チェック
async function authorizeResourceAccess(
  userId: string,
  resourceType: string,
  resourceId: string,
  action: string
): Promise<boolean> {
  switch (resourceType) {
    case 'article': {
      const article = await db.article.findUnique({ where: { id: resourceId } });
      if (!article) return false;

      // 公開記事は誰でも読める
      if (action === 'read' && article.status === 'published') return true;

      // 自分の記事は操作可能
      if (article.authorId === userId) return true;

      // 同じ組織のadminは操作可能
      const user = await db.user.findUnique({ where: { id: userId } });
      if (user?.role === 'admin' && user?.orgId === article.orgId) return true;

      return false;
    }

    case 'comment': {
      const comment = await db.comment.findUnique({ where: { id: resourceId } });
      if (!comment) return false;

      if (action === 'read') return true;
      return comment.authorId === userId;
    }

    default:
      return false;
  }
}

// ミドルウェアとして使用
function authorizeResource(resourceType: string, action: string) {
  return async (req: Request, res: Response, next: Function) => {
    const resourceId = req.params.id;
    const userId = req.auth.userId;

    const allowed = await authorizeResourceAccess(userId, resourceType, resourceId, action);

    if (!allowed) {
      return res.status(403).json({ error: 'Access denied' });
    }

    next();
  };
}

// ルート定義
app.get('/api/articles/:id', apiKeyAuth, authorizeResource('article', 'read'), getArticle);
app.put('/api/articles/:id', apiKeyAuth, authorizeResource('article', 'update'), updateArticle);
```

---

## 4. マルチテナント API 認可

```typescript
// テナント分離ミドルウェア
function tenantIsolation() {
  return async (req: Request, res: Response, next: Function) => {
    const userId = req.auth.userId;
    const user = await db.user.findUnique({ where: { id: userId } });

    if (!user?.orgId) {
      return res.status(403).json({ error: 'No organization assigned' });
    }

    // リクエストにテナントコンテキストを設定
    req.tenantId = user.orgId;

    // Prisma のクエリに自動でテナントフィルターを追加
    req.prisma = prisma.$extends({
      query: {
        $allModels: {
          async findMany({ args, query }) {
            args.where = { ...args.where, orgId: req.tenantId };
            return query(args);
          },
          async findFirst({ args, query }) {
            args.where = { ...args.where, orgId: req.tenantId };
            return query(args);
          },
          async create({ args, query }) {
            args.data = { ...args.data, orgId: req.tenantId };
            return query(args);
          },
        },
      },
    });

    next();
  };
}
```

---

## 5. 認可レスポンスのベストプラクティス

```
API 認可エラーの設計:

  401 Unauthorized — 認証エラー:
  {
    "error": "unauthorized",
    "message": "Authentication required"
  }

  403 Forbidden — 認可エラー:
  {
    "error": "forbidden",
    "message": "Insufficient permissions",
    "required_permissions": ["articles:delete"]
  }

  403（スコープ不足）:
  {
    "error": "insufficient_scope",
    "required_scopes": ["articles:write"],
    "granted_scopes": ["articles:read"]
  }

  404 vs 403 の判断:
  → リソースの存在を隠す必要がある場合 → 404
  → 権限不足を明示する場合 → 403
  → 一般的には 403 が適切（デバッグしやすい）
  → セキュリティが重要な場合は 404（情報漏洩防止）
```

---

## まとめ

| 項目 | ポイント |
|------|---------|
| スコープ | resource:action 形式、最小権限 |
| API キー | ハッシュ保存、プレフィックス付き |
| リソース認可 | 所有者チェック + ロールチェック |
| マルチテナント | テナント ID でデータ分離 |
| エラー | 401 vs 403 の正しい使い分け |

---

## 次に読むべきガイド
→ [[03-frontend-authorization.md]] — フロントエンド認可

---

## 参考文献
1. OWASP. "Authorization Cheat Sheet." cheatsheetseries.owasp.org, 2024.
2. Stripe. "API Keys." stripe.com/docs, 2024.
3. RFC 6750 §3.1. "Insufficient Scope." IETF, 2012.
