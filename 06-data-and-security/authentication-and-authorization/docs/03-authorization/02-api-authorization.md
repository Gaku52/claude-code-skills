# API 認可

> API のアクセス制御はサービスのセキュリティの要。スコープ設計、API キー管理、リソースベース認可、レート制限、マルチテナント API まで、安全な API 認可の設計と実装を解説する。

## 前提知識

- [[../../02-token-auth/00-jwt-deep-dive.md]] — JWT の基礎
- [[../../02-token-auth/01-oauth-openid.md]] — OAuth 2.0 / OpenID Connect
- HTTP ステータスコードの基本（401, 403, 404）
- Express / Next.js のミドルウェアパターン

## この章で学ぶこと

- [ ] OAuth スコープの設計と実装を理解する
- [ ] API キーの安全な管理パターンを把握する
- [ ] リソースベースの認可をミドルウェアで実装できるようになる
- [ ] マルチテナント API のデータ分離を設計できる
- [ ] レート制限と API 認可の組み合わせを実装できる
- [ ] 認可エラーの適切なレスポンス設計を把握する

---

## 1. スコープ設計

### 1.1 スコープの基本概念

```
スコープとは:

  OAuth 2.0 におけるスコープは、アクセストークンに付与される
  「権限の範囲」を定義する文字列

  ┌──────────────────────────────────────────────────────┐
  │                   アクセス制御の階層                    │
  │                                                       │
  │  ┌─────────────────────────────────────────────────┐  │
  │  │ Authentication（認証）                           │  │
  │  │ → 「誰であるか」を確認                            │  │
  │  │ → JWT / Session / API Key                       │  │
  │  └────────────────────────┬────────────────────────┘  │
  │                           │                           │
  │  ┌────────────────────────┴────────────────────────┐  │
  │  │ Authorization（認可）                            │  │
  │  │                                                 │  │
  │  │  ┌───────────┐  ┌───────────┐  ┌────────────┐  │  │
  │  │  │ Scope     │  │ Role      │  │ Resource   │  │  │
  │  │  │ ベース    │  │ ベース    │  │ ベース     │  │  │
  │  │  │           │  │           │  │            │  │  │
  │  │  │ API 単位  │  │ ユーザー  │  │ リソース    │  │  │
  │  │  │ の権限    │  │ 単位の    │  │ 所有者     │  │  │
  │  │  │           │  │ 権限      │  │ チェック   │  │  │
  │  │  └───────────┘  └───────────┘  └────────────┘  │  │
  │  └─────────────────────────────────────────────────┘  │
  └──────────────────────────────────────────────────────┘
```

### 1.2 スコープの命名パターン

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

  例（Stripe 風）:
    charges:read      — 課金情報の読取
    charges:write     — 課金の作成・更新
    customers:read    — 顧客情報の読取

  推奨パターン:
    → resource:action 形式が最も明確
    → 粒度は API の用途に合わせる
    → read は readonly の意味（副作用なし）
```

### 1.3 スコープの粒度設計

```
スコープの粒度:

  ┌──────────────┬──────────────────────┬────────────────┬──────────┐
  │ 粒度          │ 例                   │ 利点            │ 欠点     │
  ├──────────────┼──────────────────────┼────────────────┼──────────┤
  │ 粗すぎる      │ "all"               │ シンプル        │ 危険     │
  │              │ "admin"             │ 管理しやすい     │ 過剰権限 │
  ├──────────────┼──────────────────────┼────────────────┼──────────┤
  │ 適切          │ "articles:read"     │ 最小権限原則    │ 設計が    │
  │              │ "articles:write"    │ 明確な範囲      │ 必要     │
  │              │ "users:read"        │                │          │
  ├──────────────┼──────────────────────┼────────────────┼──────────┤
  │ 細かすぎる    │ "articles:read:title"│ 精密な制御      │ 複雑     │
  │              │ "users:read:email"  │                │ 管理困難 │
  └──────────────┴──────────────────────┴────────────────┴──────────┘

  適切な粒度の判断基準:
  → ユーザーが理解できるか（OAuth 同意画面で表示）
  → API のエンドポイント設計と一致するか
  → 最小権限原則を満たしているか
  → 将来の拡張性があるか

  スコープの階層関係:
  ┌──────────────────────────────────────┐
  │ admin（全権限）                       │
  │  ├── articles:write（記事の書込み）   │
  │  │    └── articles:read（記事の読取） │
  │  ├── users:write（ユーザーの書込み）   │
  │  │    └── users:read（ユーザーの読取） │
  │  └── org:settings（組織設定）         │
  └──────────────────────────────────────┘
  → write は read を暗黙的に含む設計が一般的
```

### 1.4 スコープベースの認可実装

```typescript
// スコープベースの API 認可

// スコープ定義
const SCOPES = {
  'articles:read': 'Read articles',
  'articles:write': 'Create and update articles',
  'articles:delete': 'Delete articles',
  'users:read': 'Read user profiles',
  'users:write': 'Update user profiles',
  'org:settings': 'Manage organization settings',
  'org:billing': 'Manage billing',
  'admin': 'Full administrative access',
} as const;

type Scope = keyof typeof SCOPES;

// スコープの包含関係を定義
const SCOPE_HIERARCHY: Record<string, Scope[]> = {
  'admin': Object.keys(SCOPES) as Scope[],
  'articles:write': ['articles:read'],
  'articles:delete': ['articles:read', 'articles:write'],
  'users:write': ['users:read'],
  'org:settings': [],
  'org:billing': [],
};

// スコープの展開（階層を考慮）
function expandScopes(scopes: Scope[]): Set<Scope> {
  const expanded = new Set<Scope>(scopes);

  for (const scope of scopes) {
    const implied = SCOPE_HIERARCHY[scope];
    if (implied) {
      implied.forEach(s => expanded.add(s));
    }
  }

  return expanded;
}

// スコープ検証ミドルウェア
function requireScope(...requiredScopes: Scope[]) {
  return (req: Request, res: Response, next: Function) => {
    const tokenScopes = req.auth?.scopes as Scope[] || [];
    const expandedScopes = expandScopes(tokenScopes);

    const hasAll = requiredScopes.every(s => expandedScopes.has(s));
    if (!hasAll) {
      return res.status(403).json({
        error: 'insufficient_scope',
        message: 'You do not have the required permissions',
        required_scopes: requiredScopes,
        granted_scopes: tokenScopes,
        // WWW-Authenticate ヘッダー（RFC 6750 §3.1）
      });
    }

    next();
  };
}

// 使用例
app.get('/api/articles', requireScope('articles:read'), listArticles);
app.post('/api/articles', requireScope('articles:write'), createArticle);
app.delete('/api/articles/:id', requireScope('articles:delete'), deleteArticle);
app.get('/api/users', requireScope('users:read'), listUsers);
app.put('/api/users/:id', requireScope('users:write'), updateUser);
app.get('/api/org/settings', requireScope('org:settings'), getOrgSettings);
```

### 1.5 OAuth 同意画面のスコープ表示

```typescript
// スコープの説明を人間可読にする
const SCOPE_DESCRIPTIONS: Record<Scope, { title: string; description: string; risk: 'low' | 'medium' | 'high' }> = {
  'articles:read': {
    title: '記事の閲覧',
    description: 'あなたのアカウントの記事を読み取ります',
    risk: 'low',
  },
  'articles:write': {
    title: '記事の作成・編集',
    description: 'あなたのアカウントで記事を作成・編集します',
    risk: 'medium',
  },
  'articles:delete': {
    title: '記事の削除',
    description: 'あなたのアカウントの記事を削除します',
    risk: 'high',
  },
  'users:read': {
    title: 'ユーザー情報の閲覧',
    description: 'あなたのプロフィール情報を読み取ります',
    risk: 'low',
  },
  'users:write': {
    title: 'ユーザー情報の更新',
    description: 'あなたのプロフィール情報を更新します',
    risk: 'medium',
  },
  'org:settings': {
    title: '組織設定の管理',
    description: '組織の設定を変更します',
    risk: 'high',
  },
  'org:billing': {
    title: '課金情報の管理',
    description: '支払い情報やプランを変更します',
    risk: 'high',
  },
  'admin': {
    title: '管理者アクセス',
    description: '全ての操作を実行できます',
    risk: 'high',
  },
};

// 同意画面のAPI
app.get('/api/oauth/consent', async (req, res) => {
  const requestedScopes = (req.query.scope as string).split(' ') as Scope[];

  const scopeDetails = requestedScopes.map(scope => ({
    scope,
    ...SCOPE_DESCRIPTIONS[scope],
  }));

  res.json({
    client: {
      name: 'Third Party App',
      logo: 'https://example.com/logo.png',
    },
    scopes: scopeDetails,
    hasHighRisk: scopeDetails.some(s => s.risk === 'high'),
  });
});
```

---

## 2. API キー管理

### 2.1 API キーの設計原則

```
API キーの設計:

  構造:
    prefix_randomstring
    例: sk_live_EXAMPLE_DO_NOT_USE_1234567890

  プレフィックス:
    sk_live_  — 本番用 Secret Key
    sk_test_  — テスト用 Secret Key
    pk_live_  — 本番用 Public Key（クライアント向け）
    pk_test_  — テスト用 Public Key

  ┌──────────────────────────────────────────────────────────┐
  │                  API キーのライフサイクル                   │
  │                                                          │
  │  生成 → 表示(1回のみ) → 使用 → ローテーション → 失効     │
  │                                                          │
  │  ┌────────┐  ┌────────────┐  ┌────────┐  ┌───────────┐  │
  │  │ 生成   │→│  ハッシュ化 │→│ DB保存 │→│ 検証時に  │  │
  │  │ random │  │  SHA-256   │  │ hash   │  │ 比較     │  │
  │  │ bytes  │  │            │  │ prefix │  │          │  │
  │  └────────┘  └────────────┘  └────────┘  └───────────┘  │
  │                                                          │
  │  セキュリティ要件:                                         │
  │  → API キーはハッシュ化して保存（平文保存しない）           │
  │  → 作成時のみ平文を表示（以降は取得不可）                  │
  │  → プレフィックスは平文で保存（検索・表示用）              │
  │  → 定期的なローテーションを推奨                            │
  │  → 失効・取り消しの仕組みを用意                           │
  └──────────────────────────────────────────────────────────┘
```

### 2.2 API キーの生成と管理

```typescript
// API キーの生成と管理
import crypto from 'crypto';

// API キー生成
function generateApiKey(type: 'secret' | 'public', env: 'live' | 'test'): {
  key: string;
  prefix: string;
  hash: string;
  lastFour: string;
} {
  const prefixMap = {
    'secret-live': 'sk_live_',
    'secret-test': 'sk_test_',
    'public-live': 'pk_live_',
    'public-test': 'pk_test_',
  };

  const prefix = prefixMap[`${type}-${env}`];
  const randomPart = crypto.randomBytes(24).toString('base64url');
  // 24 bytes = 32 文字の base64url = 192 ビットのエントロピー
  const key = `${prefix}${randomPart}`;
  const hash = crypto.createHash('sha256').update(key).digest('hex');
  const lastFour = randomPart.slice(-4);

  return { key, prefix, hash, lastFour };
}

// API キー保存
async function createApiKey(
  userId: string,
  name: string,
  scopes: string[],
  options: {
    expiresInDays?: number;
    rateLimit?: number;
    ipWhitelist?: string[];
  } = {}
) {
  const { key, prefix, hash, lastFour } = generateApiKey('secret', 'live');

  const expiresAt = options.expiresInDays
    ? new Date(Date.now() + options.expiresInDays * 24 * 60 * 60 * 1000)
    : null;

  await db.apiKey.create({
    data: {
      userId,
      name,
      prefix,            // 検索・表示用（平文）
      keyHash: hash,     // 検証用（ハッシュ）
      lastFour,          // 表示用（末尾4文字）
      scopes,
      rateLimit: options.rateLimit ?? 1000, // リクエスト/時間
      ipWhitelist: options.ipWhitelist ?? [],
      lastUsedAt: null,
      expiresAt,
      revokedAt: null,
    },
  });

  // この時点でのみ完全なキーを返す
  return {
    key,
    prefix,
    lastFour,
    message: 'This key will only be shown once. Please save it securely.',
  };
}

// API キーの一覧取得（ハッシュは返さない）
async function listApiKeys(userId: string) {
  const keys = await db.apiKey.findMany({
    where: { userId },
    select: {
      id: true,
      name: true,
      prefix: true,
      lastFour: true,
      scopes: true,
      lastUsedAt: true,
      expiresAt: true,
      revokedAt: true,
      createdAt: true,
    },
    orderBy: { createdAt: 'desc' },
  });

  return keys.map(k => ({
    ...k,
    // 表示用: sk_live_****abcd
    maskedKey: `${k.prefix}****${k.lastFour}`,
    isExpired: k.expiresAt ? k.expiresAt < new Date() : false,
    isRevoked: !!k.revokedAt,
  }));
}
```

### 2.3 API キーの検証

```typescript
// API キー検証
async function validateApiKey(key: string): Promise<ApiKeyData | null> {
  // プレフィックスの確認
  const validPrefixes = ['sk_live_', 'sk_test_', 'pk_live_', 'pk_test_'];
  const hasValidPrefix = validPrefixes.some(p => key.startsWith(p));
  if (!hasValidPrefix) return null;

  // ハッシュ化して検索
  const hash = crypto.createHash('sha256').update(key).digest('hex');

  const apiKey = await db.apiKey.findUnique({
    where: { keyHash: hash },
    include: { user: true },
  });

  // 存在チェック
  if (!apiKey) return null;

  // 取り消しチェック
  if (apiKey.revokedAt) {
    console.warn(`Revoked API key used: ${apiKey.id}`);
    return null;
  }

  // 有効期限チェック
  if (apiKey.expiresAt && apiKey.expiresAt < new Date()) {
    console.warn(`Expired API key used: ${apiKey.id}`);
    return null;
  }

  // 最終使用日時を更新（非同期、リクエストをブロックしない）
  db.apiKey.update({
    where: { id: apiKey.id },
    data: { lastUsedAt: new Date() },
  }).catch(() => {});

  return {
    userId: apiKey.userId,
    scopes: apiKey.scopes,
    keyId: apiKey.id,
    rateLimit: apiKey.rateLimit,
    ipWhitelist: apiKey.ipWhitelist,
    isTestKey: key.includes('_test_'),
  };
}

// API キーのローテーション
async function rotateApiKey(userId: string, oldKeyId: string) {
  // 古いキーの情報を取得
  const oldKey = await db.apiKey.findFirst({
    where: { id: oldKeyId, userId },
  });

  if (!oldKey) throw new Error('API key not found');

  // 新しいキーを生成
  const newKeyData = await createApiKey(userId, oldKey.name, oldKey.scopes, {
    rateLimit: oldKey.rateLimit,
    ipWhitelist: oldKey.ipWhitelist,
  });

  // 古いキーにグレースピリオドを設定（即座に無効化しない）
  const gracePeriod = new Date(Date.now() + 24 * 60 * 60 * 1000); // 24時間
  await db.apiKey.update({
    where: { id: oldKeyId },
    data: { expiresAt: gracePeriod },
  });

  return {
    newKey: newKeyData,
    oldKeyExpiresAt: gracePeriod,
    message: 'Old key will remain valid for 24 hours. Please update your integration.',
  };
}

// API キーの取り消し
async function revokeApiKey(userId: string, keyId: string) {
  await db.apiKey.update({
    where: { id: keyId, userId },
    data: { revokedAt: new Date() },
  });
}
```

### 2.4 API キー認証ミドルウェア

```typescript
// API キー認証ミドルウェア
async function apiKeyAuth(req: Request, res: Response, next: Function) {
  // API キーの取得元（優先順位）
  const key =
    req.headers['x-api-key'] as string ||                    // カスタムヘッダー
    req.headers.authorization?.replace('Bearer ', '') ||     // Authorization ヘッダー
    req.query.api_key as string;                             // クエリパラメータ（非推奨）

  if (!key) {
    return res.status(401).json({
      error: 'unauthorized',
      message: 'API key is required. Provide it via X-API-Key header or Authorization: Bearer header.',
    });
  }

  // クエリパラメータ経由の場合は警告
  if (req.query.api_key) {
    res.setHeader('X-Warning', 'Passing API key via query parameter is deprecated. Use X-API-Key header instead.');
  }

  const apiKeyData = await validateApiKey(key);
  if (!apiKeyData) {
    return res.status(401).json({
      error: 'invalid_api_key',
      message: 'The provided API key is invalid, expired, or revoked.',
    });
  }

  // IP ホワイトリストのチェック
  if (apiKeyData.ipWhitelist.length > 0) {
    const clientIp = req.ip || req.headers['x-forwarded-for'];
    if (!apiKeyData.ipWhitelist.includes(clientIp as string)) {
      return res.status(403).json({
        error: 'ip_not_allowed',
        message: 'Request from this IP address is not allowed.',
      });
    }
  }

  // テストキーの分離
  if (apiKeyData.isTestKey && process.env.NODE_ENV === 'production') {
    // テストキーは本番環境ではテストデータのみアクセス可能
    req.isTestMode = true;
  }

  req.auth = apiKeyData;
  next();
}
```

---

## 3. リソースベース認可

### 3.1 リソースベース認可の設計

```
リソースベース認可:

  スコープやロールだけでは不十分な場合:
  → 「記事の編集権限がある」だけでなく
  → 「この記事の編集権限がある」かを判定

  判定ロジック:
  ┌──────────────────────────────────────────────────┐
  │ リソースアクセスの判定フロー:                      │
  │                                                   │
  │ ① リソースの存在確認                              │
  │    → 見つからない場合: 404                         │
  │                                                   │
  │ ② 公開リソースかチェック                           │
  │    → 公開 + read アクション: 許可                  │
  │                                                   │
  │ ③ 所有者チェック                                   │
  │    → リソースの authorId === userId: 許可           │
  │                                                   │
  │ ④ 組織内の権限チェック                             │
  │    → 同じ組織 + 適切なロール: 許可                  │
  │                                                   │
  │ ⑤ 共有設定のチェック                               │
  │    → 明示的に共有されている: 許可                   │
  │                                                   │
  │ ⑥ 上記すべて不可: 403（または 404）                │
  └──────────────────────────────────────────────────┘
```

### 3.2 リソースベース認可の実装

```typescript
// リソースの所有者チェック
interface ResourcePolicy {
  resourceType: string;
  actions: string[];
  check: (userId: string, resourceId: string, action: string) => Promise<boolean>;
}

// ポリシー定義
const articlePolicy: ResourcePolicy = {
  resourceType: 'article',
  actions: ['read', 'update', 'delete', 'publish'],

  async check(userId: string, resourceId: string, action: string): Promise<boolean> {
    const article = await db.article.findUnique({
      where: { id: resourceId },
      include: { author: true },
    });

    if (!article) return false;

    // 公開記事は誰でも読める
    if (action === 'read' && article.status === 'published') return true;

    // 自分の記事は操作可能
    if (article.authorId === userId) return true;

    // 同じ組織のadminは操作可能
    const user = await db.user.findUnique({ where: { id: userId } });
    if (user?.role === 'admin' && user?.orgId === article.orgId) return true;

    // editor はドラフト以外を publish できる
    if (action === 'publish' && user?.role === 'editor' && user?.orgId === article.orgId) {
      return true;
    }

    return false;
  },
};

const commentPolicy: ResourcePolicy = {
  resourceType: 'comment',
  actions: ['read', 'update', 'delete'],

  async check(userId: string, resourceId: string, action: string): Promise<boolean> {
    const comment = await db.comment.findUnique({
      where: { id: resourceId },
      include: { article: true },
    });

    if (!comment) return false;

    // コメントは誰でも読める（親記事が公開の場合）
    if (action === 'read' && comment.article.status === 'published') return true;

    // 自分のコメントは更新・削除可能
    if (comment.authorId === userId) return true;

    // 記事の著者はコメントを削除可能
    if (action === 'delete' && comment.article.authorId === userId) return true;

    // admin は全コメントを操作可能
    const user = await db.user.findUnique({ where: { id: userId } });
    if (user?.role === 'admin') return true;

    return false;
  },
};

// ポリシーレジストリ
const policyRegistry = new Map<string, ResourcePolicy>([
  ['article', articlePolicy],
  ['comment', commentPolicy],
]);

// 汎用的なリソース認可関数
async function authorizeResourceAccess(
  userId: string,
  resourceType: string,
  resourceId: string,
  action: string
): Promise<boolean> {
  const policy = policyRegistry.get(resourceType);
  if (!policy) {
    console.warn(`No policy defined for resource type: ${resourceType}`);
    return false;
  }

  if (!policy.actions.includes(action)) {
    console.warn(`Unknown action '${action}' for resource type: ${resourceType}`);
    return false;
  }

  return policy.check(userId, resourceId, action);
}

// ミドルウェアとして使用
function authorizeResource(resourceType: string, action: string) {
  return async (req: Request, res: Response, next: Function) => {
    const resourceId = req.params.id;
    const userId = req.auth.userId;

    const allowed = await authorizeResourceAccess(userId, resourceType, resourceId, action);

    if (!allowed) {
      // セキュリティ上の理由で 404 を返す場合
      // → リソースの存在を隠したい場合
      const hideExistence = resourceType === 'article';
      const statusCode = hideExistence ? 404 : 403;

      return res.status(statusCode).json({
        error: statusCode === 404 ? 'not_found' : 'forbidden',
        message: statusCode === 404
          ? 'Resource not found'
          : 'You do not have permission to perform this action',
      });
    }

    next();
  };
}

// ルート定義
app.get('/api/articles/:id', apiKeyAuth, authorizeResource('article', 'read'), getArticle);
app.put('/api/articles/:id', apiKeyAuth, requireScope('articles:write'), authorizeResource('article', 'update'), updateArticle);
app.delete('/api/articles/:id', apiKeyAuth, requireScope('articles:delete'), authorizeResource('article', 'delete'), deleteArticle);
app.post('/api/articles/:id/publish', apiKeyAuth, requireScope('articles:write'), authorizeResource('article', 'publish'), publishArticle);
```

### 3.3 フィールドレベルの認可

```typescript
// フィールドレベルの認可
// → リソースへのアクセスは許可するが、一部のフィールドを隠す

interface FieldFilter {
  [field: string]: boolean | ((user: AuthData) => boolean);
}

const articleFieldFilters: Record<string, FieldFilter> = {
  // 公開記事: 全フィールド
  viewer: {
    id: true,
    title: true,
    content: true,
    author: true,
    createdAt: true,
    // 内部フィールドは非表示
    internalNotes: false,
    moderationStatus: false,
    revenue: false,
  },
  // 著者: 内部メモも表示
  author: {
    id: true,
    title: true,
    content: true,
    author: true,
    createdAt: true,
    internalNotes: true,
    moderationStatus: true,
    revenue: false,
  },
  // 管理者: 全フィールド
  admin: {
    id: true,
    title: true,
    content: true,
    author: true,
    createdAt: true,
    internalNotes: true,
    moderationStatus: true,
    revenue: true,
  },
};

function filterFields(data: Record<string, any>, role: string): Record<string, any> {
  const filter = articleFieldFilters[role] || articleFieldFilters.viewer;
  const filtered: Record<string, any> = {};

  for (const [key, value] of Object.entries(data)) {
    const allowed = filter[key];
    if (allowed === true || (typeof allowed === 'function' && allowed(data))) {
      filtered[key] = value;
    }
  }

  return filtered;
}
```

---

## 4. レート制限と API 認可

### 4.1 レート制限の設計

```
レート制限の設計:

  ┌──────────────────────────────────────────────────────────┐
  │                  レート制限の階層                          │
  │                                                          │
  │  ┌────────────────────────────────────────────────────┐  │
  │  │ Global Rate Limit: 10,000 req/min（全体）          │  │
  │  │                                                    │  │
  │  │  ┌──────────────────────────────────────────────┐  │  │
  │  │  │ Per-API-Key: 1,000 req/hour                  │  │  │
  │  │  │                                              │  │  │
  │  │  │  ┌────────────────────────────────────────┐  │  │  │
  │  │  │  │ Per-Endpoint: 100 req/min              │  │  │  │
  │  │  │  │                                        │  │  │  │
  │  │  │  │  ┌──────────────────────────────────┐  │  │  │  │
  │  │  │  │  │ Per-Resource: 10 req/min         │  │  │  │  │
  │  │  │  │  │ (DELETE等の破壊的操作)            │  │  │  │  │
  │  │  │  │  └──────────────────────────────────┘  │  │  │  │
  │  │  │  └────────────────────────────────────────┘  │  │  │
  │  │  └──────────────────────────────────────────────┘  │  │
  │  └────────────────────────────────────────────────────┘  │
  └──────────────────────────────────────────────────────────┘
```

### 4.2 Redis ベースのレート制限

```typescript
// Redis ベースのレート制限
import Redis from 'ioredis';

interface RateLimitConfig {
  windowMs: number;     // ウィンドウサイズ（ミリ秒）
  max: number;          // 最大リクエスト数
  keyPrefix?: string;   // Redis キープレフィックス
}

class RateLimiter {
  private redis: Redis;

  constructor(redis: Redis) {
    this.redis = redis;
  }

  async check(identifier: string, config: RateLimitConfig): Promise<{
    allowed: boolean;
    remaining: number;
    resetAt: Date;
    limit: number;
  }> {
    const key = `${config.keyPrefix || 'rl'}:${identifier}`;
    const now = Date.now();
    const windowStart = now - config.windowMs;

    // スライディングウィンドウ方式
    const pipeline = this.redis.pipeline();
    // 古いエントリを削除
    pipeline.zremrangebyscore(key, '-inf', windowStart);
    // 現在のリクエストを追加
    pipeline.zadd(key, now, `${now}-${Math.random()}`);
    // ウィンドウ内のリクエスト数を取得
    pipeline.zcard(key);
    // TTL を設定
    pipeline.expire(key, Math.ceil(config.windowMs / 1000));

    const results = await pipeline.exec();
    const count = results![2][1] as number;

    const allowed = count <= config.max;
    const remaining = Math.max(0, config.max - count);
    const resetAt = new Date(now + config.windowMs);

    return { allowed, remaining, resetAt, limit: config.max };
  }
}

// レート制限ミドルウェア
function rateLimitMiddleware(redis: Redis, config: RateLimitConfig) {
  const limiter = new RateLimiter(redis);

  return async (req: Request, res: Response, next: Function) => {
    // API キーベースの識別子
    const identifier = req.auth?.keyId || req.ip || 'anonymous';

    const result = await limiter.check(identifier, config);

    // レスポンスヘッダーの設定（RFC 6585 準拠）
    res.setHeader('X-RateLimit-Limit', result.limit);
    res.setHeader('X-RateLimit-Remaining', result.remaining);
    res.setHeader('X-RateLimit-Reset', Math.ceil(result.resetAt.getTime() / 1000));

    if (!result.allowed) {
      res.setHeader('Retry-After', Math.ceil(config.windowMs / 1000));
      return res.status(429).json({
        error: 'rate_limit_exceeded',
        message: `Too many requests. Limit: ${result.limit} per ${config.windowMs / 1000}s`,
        retry_after: Math.ceil(config.windowMs / 1000),
      });
    }

    next();
  };
}

// プランベースのレート制限
const PLAN_RATE_LIMITS: Record<string, RateLimitConfig> = {
  free: { windowMs: 60 * 60 * 1000, max: 100, keyPrefix: 'rl:free' },
  pro: { windowMs: 60 * 60 * 1000, max: 1000, keyPrefix: 'rl:pro' },
  enterprise: { windowMs: 60 * 60 * 1000, max: 10000, keyPrefix: 'rl:ent' },
};

function planBasedRateLimit(redis: Redis) {
  return async (req: Request, res: Response, next: Function) => {
    const plan = req.auth?.plan || 'free';
    const config = PLAN_RATE_LIMITS[plan] || PLAN_RATE_LIMITS.free;

    return rateLimitMiddleware(redis, config)(req, res, next);
  };
}

// 使用例
app.use('/api/', apiKeyAuth, planBasedRateLimit(redis));
```

### 4.3 エンドポイント別のレート制限

```typescript
// エンドポイント別のレート制限
const ENDPOINT_RATE_LIMITS: Record<string, RateLimitConfig> = {
  // 読み取り系: 緩い制限
  'GET:/api/articles': { windowMs: 60 * 1000, max: 100 },
  'GET:/api/users': { windowMs: 60 * 1000, max: 50 },

  // 書き込み系: 厳しい制限
  'POST:/api/articles': { windowMs: 60 * 1000, max: 10 },
  'PUT:/api/articles/:id': { windowMs: 60 * 1000, max: 20 },

  // 削除系: 非常に厳しい制限
  'DELETE:/api/articles/:id': { windowMs: 60 * 1000, max: 5 },

  // 認証系: 最も厳しい制限
  'POST:/api/auth/login': { windowMs: 15 * 60 * 1000, max: 5 },
  'POST:/api/auth/register': { windowMs: 60 * 60 * 1000, max: 3 },
};

function endpointRateLimit(redis: Redis) {
  const limiter = new RateLimiter(redis);

  return async (req: Request, res: Response, next: Function) => {
    const routeKey = `${req.method}:${req.route?.path || req.path}`;
    const config = ENDPOINT_RATE_LIMITS[routeKey];

    if (!config) return next(); // 制限なし

    const identifier = `${req.auth?.keyId || req.ip}:${routeKey}`;
    const result = await limiter.check(identifier, config);

    res.setHeader('X-RateLimit-Limit', result.limit);
    res.setHeader('X-RateLimit-Remaining', result.remaining);

    if (!result.allowed) {
      return res.status(429).json({
        error: 'rate_limit_exceeded',
        message: `Rate limit exceeded for ${routeKey}`,
      });
    }

    next();
  };
}
```

---

## 5. マルチテナント API 認可

### 5.1 テナント分離の設計

```
マルチテナント API のデータ分離:

  ┌──────────────────────────────────────────────────────────┐
  │                 テナント分離のパターン                      │
  │                                                          │
  │  パターン①: 行レベル分離（推奨）                           │
  │  ┌────────────────────────────────────────────────┐      │
  │  │ articles テーブル                               │      │
  │  │ ┌────┬──────────┬─────────┬──────────┐         │      │
  │  │ │ id │ title    │ content │ org_id   │         │      │
  │  │ ├────┼──────────┼─────────┼──────────┤         │      │
  │  │ │ 1  │ Article1 │ ...     │ org_001  │ ← Org A│      │
  │  │ │ 2  │ Article2 │ ...     │ org_001  │ ← Org A│      │
  │  │ │ 3  │ Article3 │ ...     │ org_002  │ ← Org B│      │
  │  │ └────┴──────────┴─────────┴──────────┘         │      │
  │  │ → 全テナントが同一テーブル                       │      │
  │  │ → WHERE org_id = ? で分離                       │      │
  │  │ → 最もシンプル、スケーラブル                     │      │
  │  └────────────────────────────────────────────────┘      │
  │                                                          │
  │  パターン②: スキーマ分離                                   │
  │  → テナントごとに DB スキーマを作成                        │
  │  → PostgreSQL の schema 機能を利用                        │
  │  → より強い分離、マイグレーションが複雑                     │
  │                                                          │
  │  パターン③: データベース分離                                │
  │  → テナントごとに DB インスタンスを分離                     │
  │  → 最も強い分離、コストが高い                              │
  │  → 金融・医療等のコンプライアンス要件向け                   │
  └──────────────────────────────────────────────────────────┘
```

### 5.2 テナント分離ミドルウェア

```typescript
// テナント分離ミドルウェア
function tenantIsolation() {
  return async (req: Request, res: Response, next: Function) => {
    const userId = req.auth.userId;
    const user = await db.user.findUnique({
      where: { id: userId },
      include: { organization: true },
    });

    if (!user?.orgId) {
      return res.status(403).json({
        error: 'no_organization',
        message: 'You must belong to an organization to access this resource.',
      });
    }

    // リクエストにテナントコンテキストを設定
    req.tenantId = user.orgId;
    req.tenantSlug = user.organization!.slug;

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
          async findUnique({ args, query }) {
            // findUnique は where に orgId を追加できないため
            // 結果を取得後にチェック
            const result = await query(args);
            if (result && 'orgId' in result && result.orgId !== req.tenantId) {
              return null; // テナント外のリソース
            }
            return result;
          },
          async create({ args, query }) {
            args.data = { ...args.data, orgId: req.tenantId };
            return query(args);
          },
          async update({ args, query }) {
            // update 前にテナントチェック
            const existing = await db[args.model as any].findFirst({
              where: { ...args.where, orgId: req.tenantId },
            });
            if (!existing) {
              throw new Error('Resource not found or access denied');
            }
            return query(args);
          },
          async delete({ args, query }) {
            // delete 前にテナントチェック
            const existing = await db[args.model as any].findFirst({
              where: { ...args.where, orgId: req.tenantId },
            });
            if (!existing) {
              throw new Error('Resource not found or access denied');
            }
            return query(args);
          },
        },
      },
    });

    next();
  };
}

// 使用例
app.use('/api/', apiKeyAuth, tenantIsolation());

// テナント分離されたエンドポイント
app.get('/api/articles', async (req, res) => {
  // req.prisma を使うとテナントフィルターが自動適用
  const articles = await req.prisma.article.findMany({
    orderBy: { createdAt: 'desc' },
    take: 20,
  });
  // → WHERE org_id = 'tenant_123' が自動追加される

  res.json(articles);
});
```

### 5.3 テナント間のデータ共有

```typescript
// テナント間でデータを共有する場合のパターン
// 例: マーケットプレイスでテナント A の記事をテナント B が閲覧

interface SharedResource {
  resourceType: string;
  resourceId: string;
  ownerOrgId: string;
  sharedWithOrgId: string;
  permissions: string[]; // ['read'] or ['read', 'write']
  expiresAt?: Date;
}

async function checkSharedAccess(
  orgId: string,
  resourceType: string,
  resourceId: string,
  action: string
): Promise<boolean> {
  const share = await db.sharedResource.findFirst({
    where: {
      sharedWithOrgId: orgId,
      resourceType,
      resourceId,
      permissions: { has: action },
      OR: [
        { expiresAt: null },
        { expiresAt: { gt: new Date() } },
      ],
    },
  });

  return !!share;
}
```

---

## 6. 認可レスポンスのベストプラクティス

### 6.1 HTTP ステータスコードの使い分け

```
API 認可エラーの設計:

  ┌──────────┬──────────────────────────────────────────────┐
  │ コード   │ 用途                                          │
  ├──────────┼──────────────────────────────────────────────┤
  │ 401      │ 認証エラー（未認証）                           │
  │          │ → API キー / トークンが未提供                  │
  │          │ → API キー / トークンが無効                    │
  │          │ → セッション期限切れ                           │
  ├──────────┼──────────────────────────────────────────────┤
  │ 403      │ 認可エラー（権限不足）                         │
  │          │ → スコープ不足                                │
  │          │ → ロール不足                                  │
  │          │ → リソースへのアクセス拒否                     │
  │          │ → IP ホワイトリスト外                         │
  ├──────────┼──────────────────────────────────────────────┤
  │ 404      │ リソース不在（セキュリティ上 403 の代替）      │
  │          │ → リソースの存在を隠したい場合                 │
  │          │ → 情報漏洩防止                                │
  ├──────────┼──────────────────────────────────────────────┤
  │ 429      │ レート制限超過                                │
  │          │ → Retry-After ヘッダーを付与                  │
  └──────────┴──────────────────────────────────────────────┘

  404 vs 403 の判断:
  ┌──────────────────────────────────────────────────┐
  │ リソースの存在を隠す必要がある場合 → 404          │
  │ → 例: 他ユーザーのプライベートリソース             │
  │ → 攻撃者にリソースの存在を知らせたくない          │
  │                                                   │
  │ 権限不足を明示する場合 → 403                      │
  │ → 例: アクセスを申請できるようにしたい場合         │
  │ → デバッグを容易にしたい場合                      │
  │                                                   │
  │ 一般的にはセキュリティ優先で 404 を推奨             │
  │ 内部 API や管理画面では 403 が適切                  │
  └──────────────────────────────────────────────────┘
```

### 6.2 エラーレスポンスの構造

```typescript
// 統一的なエラーレスポンス

// 401 Unauthorized
{
  "error": "unauthorized",
  "message": "Authentication required",
  "docs_url": "https://docs.myapi.com/auth"
}

// 403 Forbidden（スコープ不足）
{
  "error": "insufficient_scope",
  "message": "You do not have the required permissions",
  "required_scopes": ["articles:write"],
  "granted_scopes": ["articles:read"],
  "docs_url": "https://docs.myapi.com/scopes"
}

// 403 Forbidden（リソース認可）
{
  "error": "forbidden",
  "message": "You do not have permission to perform this action on this resource",
  "resource_type": "article",
  "action": "delete"
}

// 429 Too Many Requests
{
  "error": "rate_limit_exceeded",
  "message": "Too many requests",
  "limit": 1000,
  "remaining": 0,
  "reset_at": "2024-01-01T12:00:00Z",
  "retry_after": 3600
}
```

### 6.3 WWW-Authenticate ヘッダー

```typescript
// RFC 6750 に準拠した WWW-Authenticate ヘッダー
function setWWWAuthenticate(res: Response, options: {
  realm?: string;
  error?: string;
  errorDescription?: string;
  scope?: string;
}) {
  const parts = [`Bearer realm="${options.realm || 'api'}"`];

  if (options.error) {
    parts.push(`error="${options.error}"`);
  }
  if (options.errorDescription) {
    parts.push(`error_description="${options.errorDescription}"`);
  }
  if (options.scope) {
    parts.push(`scope="${options.scope}"`);
  }

  res.setHeader('WWW-Authenticate', parts.join(', '));
}

// 使用例
// 401: 認証なし
setWWWAuthenticate(res, {
  realm: 'api',
  error: 'invalid_token',
  errorDescription: 'The access token is invalid',
});

// 403: スコープ不足
setWWWAuthenticate(res, {
  realm: 'api',
  error: 'insufficient_scope',
  scope: 'articles:write',
});
```

---

## 7. エッジケースとセキュリティ

### 7.1 タイミング攻撃への対策

```typescript
// API キー検証でのタイミング攻撃対策
import crypto from 'crypto';

// 悪い例: 早期リターンでタイミング情報が漏洩
function unsafeValidate(providedKey: string, storedHash: string): boolean {
  const hash = crypto.createHash('sha256').update(providedKey).digest('hex');
  return hash === storedHash; // ← タイミング攻撃に脆弱
}

// 良い例: 一定時間比較
function safeValidate(providedKey: string, storedHash: string): boolean {
  const hash = crypto.createHash('sha256').update(providedKey).digest('hex');
  return crypto.timingSafeEqual(
    Buffer.from(hash, 'hex'),
    Buffer.from(storedHash, 'hex')
  );
}
```

### 7.2 横方向権限昇格（IDOR）の防止

```
IDOR（Insecure Direct Object Reference）:

  攻撃例:
  → GET /api/users/123/profile （自分）
  → GET /api/users/456/profile （他人）← ID を変えるだけ

  対策:
  ① リソース認可チェックを必ず実施
  ② シーケンシャル ID を避ける（UUID/CUID を使用）
  ③ 自分のリソースのみ返す設計
     → GET /api/me/profile（ID 不要）
     → GET /api/articles?mine=true（フィルタ）
```

```typescript
// IDOR 対策: 自分のリソースのみ返す
app.get('/api/me/profile', auth, async (req, res) => {
  const profile = await db.user.findUnique({
    where: { id: req.auth.userId }, // セッションの userId を使用
    select: { id: true, name: true, email: true, image: true },
  });
  res.json(profile);
});

// IDOR 対策: リソースの所有者チェック
app.get('/api/articles/:id', auth, async (req, res) => {
  const article = await db.article.findFirst({
    where: {
      id: req.params.id,
      // テナントフィルター
      orgId: req.auth.orgId,
    },
  });

  if (!article) {
    return res.status(404).json({ error: 'not_found' });
  }

  res.json(article);
});
```

### 7.3 Mass Assignment（一括代入）の防止

```typescript
// Mass Assignment 攻撃:
// クライアントが送信した body に予期しないフィールドが含まれる

// 悪い例
app.put('/api/articles/:id', async (req, res) => {
  await db.article.update({
    where: { id: req.params.id },
    data: req.body, // ← req.body に { role: 'admin' } が含まれている可能性
  });
});

// 良い例: 許可するフィールドを明示的に指定
app.put('/api/articles/:id', async (req, res) => {
  const allowedFields = ['title', 'content', 'tags', 'status'];
  const data: Record<string, any> = {};

  for (const field of allowedFields) {
    if (req.body[field] !== undefined) {
      data[field] = req.body[field];
    }
  }

  await db.article.update({
    where: { id: req.params.id },
    data,
  });
});

// さらに良い例: Zod でバリデーション
import { z } from 'zod';

const updateArticleSchema = z.object({
  title: z.string().min(1).max(200).optional(),
  content: z.string().optional(),
  tags: z.array(z.string()).optional(),
  status: z.enum(['draft', 'published', 'archived']).optional(),
});

app.put('/api/articles/:id', async (req, res) => {
  const parsed = updateArticleSchema.safeParse(req.body);
  if (!parsed.success) {
    return res.status(400).json({ error: 'validation_error', details: parsed.error.issues });
  }

  await db.article.update({
    where: { id: req.params.id },
    data: parsed.data,
  });
});
```

---

## 8. アンチパターン

```
API 認可のアンチパターン:

  ✗ アンチパターン①: フロントエンドのみで認可チェック
  ┌──────────────────────────────────────────────────┐
  │ // 危険: API を直接叩けばバイパスできる            │
  │ // フロントエンドで「削除ボタンを非表示」にしても   │
  │ // curl -X DELETE /api/articles/123 でアクセス可能 │
  │                                                   │
  │ → 全 API エンドポイントでサーバーサイド認可が必須   │
  └──────────────────────────────────────────────────┘

  ✗ アンチパターン②: API キーをクライアントサイドに埋め込む
  ┌──────────────────────────────────────────────────┐
  │ // 危険: Secret Key が漏洩する                     │
  │ const API_KEY = 'sk_live_xxxxx'; // JSバンドルに含む│
  │                                                   │
  │ → Secret Key はサーバーサイドでのみ使用            │
  │ → クライアントには Public Key のみ                 │
  └──────────────────────────────────────────────────┘

  ✗ アンチパターン③: ワイルドカードスコープ
  ┌──────────────────────────────────────────────────┐
  │ // 危険: 全権限を1つのスコープに                    │
  │ scopes: ['*'] // 全権限                           │
  │                                                   │
  │ → 最小権限原則に違反                               │
  │ → 必要なスコープのみ付与                           │
  └──────────────────────────────────────────────────┘
```

---

## 9. 演習

### 演習1: 基礎 - スコープベースの API 認可

```
【演習1】スコープベースの API 認可

目的: OAuth スコープの設計と実装を体験する

手順:
1. Express で REST API を構築
   - articles (CRUD)
   - users (Read, Update)

2. スコープを設計
   - articles:read, articles:write, articles:delete
   - users:read, users:write
   - admin

3. スコープ検証ミドルウェアを実装
   - 階層関係の考慮
   - エラーレスポンスに不足スコープを含める

4. テスト:
   - 各スコープでアクセスの可否を確認
   - admin スコープで全操作が可能なことを確認

評価基準:
  □ スコープ設計が最小権限原則を満たしている
  □ エラーレスポンスが informative
  □ 階層関係が正しく機能する
```

### 演習2: 応用 - API キー管理システム

```
【演習2】API キー管理システム

目的: API キーの生成・検証・ローテーションを実装する

手順:
1. API キー管理 API を構築
   - POST /api/keys（キー生成）
   - GET /api/keys（キー一覧）
   - POST /api/keys/:id/rotate（ローテーション）
   - DELETE /api/keys/:id（取り消し）

2. セキュリティ要件:
   - SHA-256 ハッシュ保存
   - プレフィックス付き（sk_live_ 等）
   - 有効期限設定
   - ローテーション時のグレースピリオド

3. API キー認証ミドルウェア:
   - 複数の送信方法に対応
   - レート制限の統合

評価基準:
  □ キーの平文がDBに保存されていない
  □ ローテーションが安全に動作する
  □ 取り消されたキーが即座に無効になる
```

### 演習3: 発展 - マルチテナント API

```
【演習3】マルチテナント API

目的: マルチテナント環境での API 認可を実装する

手順:
1. テナント（Organization）モデルを設計
2. テナント分離ミドルウェアを実装
3. Prisma Extension でテナントフィルターを自動適用
4. テナント間のデータ共有機能を実装
5. テナント管理 API（招待、ロール管理）

評価基準:
  □ テナント間でデータが完全に分離されている
  □ クロステナントアクセスが不可能
  □ 共有機能が正しく動作する
  □ テナント管理が安全
```

---

## 10. FAQ・トラブルシューティング

### Q1: API キーと OAuth トークンのどちらを使うべき？

```
A: ユースケースで判断:

  API キー:
  → サーバー間通信（M2M）
  → サードパーティ連携
  → シンプルな認証で十分な場合
  → 長期的なアクセス

  OAuth トークン:
  → ユーザー代理のアクセス
  → スコープベースの権限制御が必要
  → 短期的なアクセス（トークン更新あり）
  → 同意画面が必要な場合
```

### Q2: 404 と 403 のどちらを返すべき？

```
A: セキュリティ要件で判断:

  一般的なルール:
  → 外部向け API: 404（リソースの存在を隠す）
  → 内部向け API / 管理画面: 403（デバッグしやすい）
  → リソースの存在が秘密ではない場合: 403

  例:
  → GET /api/users/uuid-abc → 404（他人のプロフィール）
  → GET /api/admin/settings → 403（管理者権限が必要）
```

### Q3: レート制限のリセット時間はどう設計する？

```
A: 以下の基準で設計:

  読み取り API: 1分間ウィンドウ
  → 瞬間的なバーストを許容
  → ユーザー体験を損なわない

  書き込み API: 1時間ウィンドウ
  → 過度な書き込みを防止
  → 正常な使用パターンを許容

  認証 API: 15分間ウィンドウ
  → ブルートフォース攻撃を防止
  → 正規ユーザーの再試行を許容

  ヘッダーには必ず以下を含める:
  → X-RateLimit-Limit（上限）
  → X-RateLimit-Remaining（残り回数）
  → X-RateLimit-Reset（リセット時刻）
  → Retry-After（429 レスポンス時）
```

---

## 11. パフォーマンスに関する考察

```
API 認可のパフォーマンス:

  ┌───────────────────────┬───────────────┬──────────────────┐
  │ 操作                   │ レイテンシ     │ 最適化手法        │
  ├───────────────────────┼───────────────┼──────────────────┤
  │ JWT 検証               │ <1ms          │ 署名検証のみ      │
  │ API キー検証（DB）     │ 1-5ms         │ キャッシュ(Redis) │
  │ スコープチェック        │ <1ms          │ Set 操作          │
  │ リソース認可            │ 5-20ms        │ DB クエリ最適化   │
  │ テナント分離            │ 1-5ms         │ インデックス      │
  │ レート制限（Redis）    │ 1-3ms         │ パイプライン      │
  └───────────────────────┴───────────────┴──────────────────┘

  最適化のポイント:
  → API キーの検証結果を短期キャッシュ（30秒-5分）
  → リソース認可のクエリにインデックスを設定
  → レート制限は Redis パイプラインで複数操作を一括実行
  → テナントフィルターのカラムにインデックス必須
```

---

## まとめ

| 項目 | ポイント |
|------|---------|
| スコープ | resource:action 形式、最小権限、階層関係 |
| API キー | SHA-256 ハッシュ保存、プレフィックス付き、ローテーション |
| リソース認可 | ポリシーパターン、所有者チェック + ロールチェック |
| マルチテナント | テナント ID でデータ分離、Prisma Extension |
| レート制限 | 階層的な制限、プランベース、Redis スライディングウィンドウ |
| エラー設計 | 401 vs 403 vs 404 の使い分け、RFC 6750 準拠 |
| セキュリティ | タイミング攻撃対策、IDOR 防止、Mass Assignment 防止 |

---

## 次に読むべきガイド
→ [[03-frontend-authorization.md]] — フロントエンド認可

---

## 参考文献
1. OWASP. "Authorization Cheat Sheet." cheatsheetseries.owasp.org, 2024.
2. Stripe. "API Keys." stripe.com/docs, 2024.
3. RFC 6750 §3.1. "Insufficient Scope." IETF, 2012.
4. RFC 6585. "Additional HTTP Status Codes (429)." IETF, 2012.
5. OWASP. "IDOR Prevention Cheat Sheet." cheatsheetseries.owasp.org, 2024.
6. GitHub. "OAuth Scopes." docs.github.com, 2024.
7. Google. "OAuth 2.0 Scopes." developers.google.com, 2024.
