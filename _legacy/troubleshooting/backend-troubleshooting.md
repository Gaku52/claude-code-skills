# Backend トラブルシューティング

## 目次

1. [概要](#概要)
2. [HTTP・ネットワークエラー](#httpネットワークエラー)
3. [認証・認可エラー](#認証認可エラー)
4. [CORS・セキュリティエラー](#corsセキュリティエラー)
5. [データベースエラー](#データベースエラー)
6. [パフォーマンス問題](#パフォーマンス問題)
7. [サーバー設定エラー](#サーバー設定エラー)
8. [デプロイエラー](#デプロイエラー)

---

## 概要

このガイドは、バックエンド開発で頻繁に遭遇するエラーと解決策をまとめたトラブルシューティングデータベースです。

**収録エラー数:** 25個

**対象技術:** REST API, GraphQL, WebSocket, データベース

---

## HTTP・ネットワークエラー

### ❌ エラー1: 404 Not Found

```
HTTP 404 Not Found
```

**原因:**
- ルートが定義されていない
- エンドポイントのパスが間違っている

**解決策:**

```javascript
// ❌ ルートが定義されていない
// GET /api/users → 404

// ✅ Express - ルート定義
const express = require('express');
const app = express();

app.get('/api/users', (req, res) => {
  res.json({ users: [] });
});

// ✅ 404ハンドラー（すべてのルートの最後に配置）
app.use((req, res) => {
  res.status(404).json({
    error: 'Not Found',
    message: `Route ${req.method} ${req.path} not found`
  });
});
```

**FastAPI:**

```python
# ✅ FastAPI - ルート定義
from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.get('/api/users')
async def get_users():
    return {"users": []}

# 404エラーハンドラー
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": f"Route {request.method} {request.url.path} not found"
        }
    )
```

---

### ❌ エラー2: 500 Internal Server Error

```
HTTP 500 Internal Server Error
```

**原因:**
- サーバー側のコードエラー
- 例外がキャッチされていない

**解決策:**

```javascript
// ✅ Express - エラーハンドリング
const express = require('express');
const app = express();

app.get('/api/users/:id', async (req, res, next) => {
  try {
    const user = await getUserById(req.params.id);

    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }

    res.json(user);
  } catch (error) {
    next(error);  // エラーミドルウェアに渡す
  }
});

// グローバルエラーハンドラー（最後に配置）
app.use((err, req, res, next) => {
  console.error(err.stack);

  res.status(err.status || 500).json({
    error: process.env.NODE_ENV === 'production' ? 'Internal Server Error' : err.message,
    ...(process.env.NODE_ENV !== 'production' && { stack: err.stack })
  });
});
```

**FastAPI:**

```python
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get('/api/users/{user_id}')
async def get_user(user_id: int):
    try:
        user = await get_user_by_id(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# グローバル例外ハンドラー
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc) if DEBUG else "An error occurred"
        }
    )
```

---

### ❌ エラー3: 400 Bad Request (Invalid JSON)

```
HTTP 400 Bad Request
SyntaxError: Unexpected token < in JSON at position 0
```

**原因:**
- リクエストボディのJSONが不正

**解決策:**

```javascript
// ✅ Express - JSON解析エラーハンドリング
const express = require('express');
const app = express();

app.use(express.json());

// JSONパースエラーハンドラー
app.use((err, req, res, next) => {
  if (err instanceof SyntaxError && err.status === 400 && 'body' in err) {
    return res.status(400).json({
      error: 'Bad Request',
      message: 'Invalid JSON format'
    });
  }
  next(err);
});

app.post('/api/users', (req, res) => {
  const { name, email } = req.body;

  // バリデーション
  if (!name || !email) {
    return res.status(400).json({
      error: 'Bad Request',
      message: 'Name and email are required'
    });
  }

  res.json({ message: 'User created', user: { name, email } });
});
```

**クライアント側（正しいリクエスト）:**

```javascript
// ✅ 正しいJSON送信
const response = await fetch('/api/users', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    name: 'John Doe',
    email: 'john@example.com'
  })
});
```

---

### ❌ エラー4: 408 Request Timeout

```
HTTP 408 Request Timeout
```

**原因:**
- リクエスト処理に時間がかかりすぎている

**解決策:**

```javascript
// ✅ Express - タイムアウト設定
const express = require('express');
const app = express();

// サーバー全体のタイムアウト
const server = app.listen(3000);
server.timeout = 30000;  // 30秒

// または個別ルートでタイムアウト設定
const timeout = require('connect-timeout');

app.use('/api/slow', timeout('10s'));

app.get('/api/slow/operation', async (req, res) => {
  if (req.timedout) return;

  try {
    const result = await slowOperation();
    res.json(result);
  } catch (error) {
    if (!req.timedout) {
      res.status(500).json({ error: 'Operation failed' });
    }
  }
});

// タイムアウトハンドラー
app.use((req, res, next) => {
  if (req.timedout) {
    res.status(408).json({
      error: 'Request Timeout',
      message: 'Request took too long to process'
    });
  } else {
    next();
  }
});
```

**長時間処理の改善:**

```javascript
// ✅ バックグラウンドジョブで処理
const Queue = require('bull');
const exportQueue = new Queue('export');

app.post('/api/export', async (req, res) => {
  // ジョブをキューに追加
  const job = await exportQueue.add({
    userId: req.user.id,
    format: req.body.format
  });

  res.json({
    message: 'Export started',
    jobId: job.id
  });
});

// ジョブステータス確認
app.get('/api/export/:jobId', async (req, res) => {
  const job = await exportQueue.getJob(req.params.jobId);

  res.json({
    status: await job.getState(),
    progress: job.progress()
  });
});
```

---

### ❌ エラー5: 429 Too Many Requests

```
HTTP 429 Too Many Requests
```

**原因:**
- レート制限に引っかかった

**解決策:**

```javascript
// ✅ Express - レート制限実装
const rateLimit = require('express-rate-limit');

// API全体に適用
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000,  // 15分
  max: 100,  // 100リクエスト
  message: {
    error: 'Too Many Requests',
    message: 'Too many requests from this IP, please try again later.'
  },
  standardHeaders: true,
  legacyHeaders: false
});

app.use('/api', limiter);

// 特定エンドポイントに厳しい制限
const strictLimiter = rateLimit({
  windowMs: 60 * 1000,  // 1分
  max: 5,  // 5リクエスト
  message: 'Too many login attempts, please try again later.'
});

app.post('/api/login', strictLimiter, async (req, res) => {
  // ログイン処理
});

// IPベース + ユーザーIDベースの制限
const createAccountLimiter = rateLimit({
  windowMs: 60 * 60 * 1000,  // 1時間
  max: 3,
  keyGenerator: (req) => {
    return req.ip + req.body.email;  // IP + メールアドレス
  }
});

app.post('/api/register', createAccountLimiter, async (req, res) => {
  // ユーザー登録処理
});
```

---

## 認証・認可エラー

### ❌ エラー6: 401 Unauthorized

```
HTTP 401 Unauthorized
```

**原因:**
- 認証トークンが無効または期限切れ
- トークンが送信されていない

**解決策:**

```javascript
// ✅ Express - JWT認証ミドルウェア
const jwt = require('jsonwebtoken');

function authenticateToken(req, res, next) {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];  // "Bearer TOKEN"

  if (!token) {
    return res.status(401).json({
      error: 'Unauthorized',
      message: 'Access token is required'
    });
  }

  jwt.verify(token, process.env.JWT_SECRET, (err, user) => {
    if (err) {
      if (err.name === 'TokenExpiredError') {
        return res.status(401).json({
          error: 'Unauthorized',
          message: 'Access token has expired'
        });
      }

      return res.status(403).json({
        error: 'Forbidden',
        message: 'Invalid access token'
      });
    }

    req.user = user;
    next();
  });
}

// 保護されたルート
app.get('/api/profile', authenticateToken, (req, res) => {
  res.json({
    user: req.user
  });
});
```

**FastAPI:**

```python
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

app = FastAPI()
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

@app.get('/api/profile')
async def get_profile(user = Depends(verify_token)):
    return {"user": user}
```

---

### ❌ エラー7: 403 Forbidden

```
HTTP 403 Forbidden
```

**原因:**
- 認証は成功したが権限がない

**解決策:**

```javascript
// ✅ Express - ロールベースアクセス制御（RBAC）
function requireRole(...allowedRoles) {
  return (req, res, next) => {
    if (!req.user) {
      return res.status(401).json({
        error: 'Unauthorized',
        message: 'Authentication required'
      });
    }

    if (!allowedRoles.includes(req.user.role)) {
      return res.status(403).json({
        error: 'Forbidden',
        message: 'You do not have permission to access this resource'
      });
    }

    next();
  };
}

// 使用例
app.delete('/api/users/:id',
  authenticateToken,
  requireRole('admin'),
  async (req, res) => {
    // 管理者のみアクセス可能
    await deleteUser(req.params.id);
    res.json({ message: 'User deleted' });
  }
);

app.get('/api/admin/stats',
  authenticateToken,
  requireRole('admin', 'moderator'),
  async (req, res) => {
    // 管理者またはモデレーターがアクセス可能
    const stats = await getStats();
    res.json(stats);
  }
);
```

**リソースベースのアクセス制御:**

```javascript
// ✅ リソースの所有者チェック
app.put('/api/posts/:id',
  authenticateToken,
  async (req, res) => {
    const post = await getPostById(req.params.id);

    if (!post) {
      return res.status(404).json({ error: 'Post not found' });
    }

    // 投稿者または管理者のみ編集可能
    if (post.authorId !== req.user.id && req.user.role !== 'admin') {
      return res.status(403).json({
        error: 'Forbidden',
        message: 'You can only edit your own posts'
      });
    }

    await updatePost(req.params.id, req.body);
    res.json({ message: 'Post updated' });
  }
);
```

---

### ❌ エラー8: JWT Token Expired

```
JsonWebTokenError: jwt expired
```

**原因:**
- アクセストークンの有効期限切れ

**解決策:**

```javascript
// ✅ リフレッシュトークン実装
const jwt = require('jsonwebtoken');

// トークン生成
function generateTokens(userId) {
  const accessToken = jwt.sign(
    { userId },
    process.env.JWT_SECRET,
    { expiresIn: '15m' }  // 短い有効期限
  );

  const refreshToken = jwt.sign(
    { userId },
    process.env.JWT_REFRESH_SECRET,
    { expiresIn: '7d' }  // 長い有効期限
  );

  return { accessToken, refreshToken };
}

// ログイン
app.post('/api/login', async (req, res) => {
  const { email, password } = req.body;

  const user = await authenticateUser(email, password);

  if (!user) {
    return res.status(401).json({ error: 'Invalid credentials' });
  }

  const { accessToken, refreshToken } = generateTokens(user.id);

  // リフレッシュトークンをDBに保存
  await saveRefreshToken(user.id, refreshToken);

  res.json({ accessToken, refreshToken });
});

// トークン更新
app.post('/api/refresh', async (req, res) => {
  const { refreshToken } = req.body;

  if (!refreshToken) {
    return res.status(401).json({ error: 'Refresh token required' });
  }

  try {
    // リフレッシュトークン検証
    const payload = jwt.verify(refreshToken, process.env.JWT_REFRESH_SECRET);

    // DBに保存されているか確認
    const isValid = await isRefreshTokenValid(payload.userId, refreshToken);

    if (!isValid) {
      return res.status(401).json({ error: 'Invalid refresh token' });
    }

    // 新しいトークンを発行
    const tokens = generateTokens(payload.userId);

    // 古いトークンを削除し、新しいトークンを保存
    await replaceRefreshToken(payload.userId, refreshToken, tokens.refreshToken);

    res.json(tokens);
  } catch (error) {
    return res.status(401).json({ error: 'Invalid refresh token' });
  }
});

// ログアウト（リフレッシュトークンを無効化）
app.post('/api/logout', authenticateToken, async (req, res) => {
  const { refreshToken } = req.body;

  await revokeRefreshToken(req.user.id, refreshToken);

  res.json({ message: 'Logged out successfully' });
});
```

---

## CORS・セキュリティエラー

### ❌ エラー9: CORS Policy Blocked

```
Access to XMLHttpRequest at 'http://localhost:3000/api/users' from origin 'http://localhost:5173' has been blocked by CORS policy: No 'Access-Control-Allow-Origin' header is present on the requested resource.
```

**原因:**
- CORS設定が不足している

**解決策:**

```javascript
// ✅ Express - CORS設定
const cors = require('cors');
const express = require('express');
const app = express();

// 開発環境：すべてのオリジンを許可
if (process.env.NODE_ENV === 'development') {
  app.use(cors());
}

// 本番環境：特定のオリジンのみ許可
const allowedOrigins = [
  'https://example.com',
  'https://app.example.com'
];

app.use(cors({
  origin: function (origin, callback) {
    if (!origin || allowedOrigins.includes(origin)) {
      callback(null, true);
    } else {
      callback(new Error('Not allowed by CORS'));
    }
  },
  credentials: true,  // Cookieを許可
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization']
}));

// プリフライトリクエスト対応
app.options('*', cors());
```

**FastAPI:**

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS設定
origins = [
    "http://localhost:5173",
    "https://example.com",
    "https://app.example.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

### ❌ エラー10: CSRF Token Missing or Invalid

```
HTTP 403 Forbidden
CSRF token missing or invalid
```

**原因:**
- CSRF保護が有効だが、トークンが送信されていない

**解決策:**

```javascript
// ✅ Express - CSRF保護
const csrf = require('csurf');
const cookieParser = require('cookie-parser');

const app = express();

app.use(cookieParser());
app.use(csrf({ cookie: true }));

// CSRFトークンを取得
app.get('/api/csrf-token', (req, res) => {
  res.json({ csrfToken: req.csrfToken() });
});

// POSTリクエスト（CSRFトークン必須）
app.post('/api/users', (req, res) => {
  // CSRFトークンが自動検証される
  res.json({ message: 'User created' });
});

// CSRFエラーハンドラー
app.use((err, req, res, next) => {
  if (err.code === 'EBADCSRFTOKEN') {
    return res.status(403).json({
      error: 'Forbidden',
      message: 'Invalid CSRF token'
    });
  }
  next(err);
});
```

**クライアント側:**

```javascript
// ✅ CSRFトークンを取得して送信
async function createUser(userData) {
  // CSRFトークン取得
  const tokenResponse = await fetch('/api/csrf-token');
  const { csrfToken } = await tokenResponse.json();

  // リクエスト送信
  const response = await fetch('/api/users', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'CSRF-Token': csrfToken
    },
    body: JSON.stringify(userData)
  });

  return response.json();
}
```

---

### ❌ エラー11: SQL Injection Risk

**原因:**
- ユーザー入力を直接SQLクエリに埋め込んでいる

**間違った例:**

```javascript
// ❌ SQLインジェクションの危険
app.get('/api/users/:id', async (req, res) => {
  const userId = req.params.id;

  // 危険：SQLインジェクション可能
  const query = `SELECT * FROM users WHERE id = ${userId}`;
  const user = await db.query(query);

  res.json(user);
});
```

**解決策:**

```javascript
// ✅ プリペアドステートメント（パラメータ化クエリ）
const mysql = require('mysql2/promise');

app.get('/api/users/:id', async (req, res) => {
  const userId = req.params.id;

  // 安全：パラメータ化クエリ
  const [rows] = await pool.execute(
    'SELECT * FROM users WHERE id = ?',
    [userId]
  );

  if (rows.length === 0) {
    return res.status(404).json({ error: 'User not found' });
  }

  res.json(rows[0]);
});

// ✅ 検索クエリの安全な実装
app.get('/api/users/search', async (req, res) => {
  const { query } = req.query;

  // ワイルドカードをエスケープ
  const searchTerm = query.replace(/[%_]/g, '\\$&');

  const [rows] = await pool.execute(
    'SELECT * FROM users WHERE name LIKE ?',
    [`%${searchTerm}%`]
  );

  res.json(rows);
});
```

**ORM使用（Prisma）:**

```javascript
// ✅ PrismaはSQLインジェクション対策済み
const { PrismaClient } = require('@prisma/client');
const prisma = new PrismaClient();

app.get('/api/users/:id', async (req, res) => {
  const userId = parseInt(req.params.id);

  const user = await prisma.user.findUnique({
    where: { id: userId }
  });

  if (!user) {
    return res.status(404).json({ error: 'User not found' });
  }

  res.json(user);
});
```

---

## データベースエラー

### ❌ エラー12: Database Connection Pool Exhausted

```
Error: Too many connections
```

**原因:**
- コネクションプールの上限に達した
- コネクションがリリースされていない

**解決策:**

```javascript
// ✅ MySQL - コネクションプール設定
const mysql = require('mysql2/promise');

const pool = mysql.createPool({
  host: 'localhost',
  user: 'root',
  password: 'password',
  database: 'mydb',
  waitForConnections: true,
  connectionLimit: 10,
  queueLimit: 0,
  enableKeepAlive: true,
  keepAliveInitialDelay: 0
});

// コネクションを確実にリリース
async function executeQuery(sql, params) {
  const connection = await pool.getConnection();

  try {
    const [rows] = await connection.execute(sql, params);
    return rows;
  } catch (error) {
    console.error('Query failed:', error);
    throw error;
  } finally {
    connection.release();  // 必ずリリース
  }
}

// 使用例
app.get('/api/users', async (req, res) => {
  try {
    const users = await executeQuery('SELECT * FROM users', []);
    res.json(users);
  } catch (error) {
    res.status(500).json({ error: 'Database error' });
  }
});
```

**Prisma:**

```javascript
// ✅ Prismaは自動でコネクション管理
const { PrismaClient } = require('@prisma/client');
const prisma = new PrismaClient({
  datasources: {
    db: {
      url: process.env.DATABASE_URL
    }
  },
  log: ['query', 'error', 'warn']
});

// アプリケーション終了時にクリーンアップ
process.on('beforeExit', async () => {
  await prisma.$disconnect();
});
```

---

### ❌ エラー13: N+1 Query Problem

**原因:**
- ループ内でクエリを実行している

**間違った例:**

```javascript
// ❌ N+1 Query Problem
app.get('/api/users-with-posts', async (req, res) => {
  const users = await prisma.user.findMany();

  // 各ユーザーごとにクエリ実行（N+1問題）
  for (const user of users) {
    user.posts = await prisma.post.findMany({
      where: { authorId: user.id }
    });
  }

  res.json(users);
});
// 合計: 1 + N 回のクエリ実行
```

**解決策:**

```javascript
// ✅ Prismaでリレーション一括取得
app.get('/api/users-with-posts', async (req, res) => {
  const users = await prisma.user.findMany({
    include: {
      posts: true
    }
  });

  res.json(users);
});
// 1回のクエリで完了

// ✅ 必要なフィールドのみ取得
app.get('/api/users-with-posts', async (req, res) => {
  const users = await prisma.user.findMany({
    select: {
      id: true,
      name: true,
      posts: {
        select: {
          id: true,
          title: true,
          createdAt: true
        }
      }
    }
  });

  res.json(users);
});
```

**SQL JOIN:**

```javascript
// ✅ 生SQLでJOIN
const [rows] = await pool.execute(`
  SELECT
    u.id as user_id,
    u.name as user_name,
    p.id as post_id,
    p.title as post_title
  FROM users u
  LEFT JOIN posts p ON u.id = p.author_id
`);

// 結果を整形
const usersMap = new Map();

rows.forEach(row => {
  if (!usersMap.has(row.user_id)) {
    usersMap.set(row.user_id, {
      id: row.user_id,
      name: row.user_name,
      posts: []
    });
  }

  if (row.post_id) {
    usersMap.get(row.user_id).posts.push({
      id: row.post_id,
      title: row.post_title
    });
  }
});

const users = Array.from(usersMap.values());
res.json(users);
```

---

### ❌ エラー14: Database Deadlock

```
Error: Deadlock found when trying to get lock; try restarting transaction
```

**原因:**
- 複数のトランザクションが互いにロック待ちしている

**解決策:**

```javascript
// ✅ トランザクション順序を統一
async function transferMoney(fromUserId, toUserId, amount) {
  // 常にIDの小さい順にロック取得
  const [firstUserId, secondUserId] = [fromUserId, toUserId].sort((a, b) => a - b);

  await prisma.$transaction(async (tx) => {
    const firstUser = await tx.user.findUnique({
      where: { id: firstUserId }
    });

    const secondUser = await tx.user.findUnique({
      where: { id: secondUserId }
    });

    // 残高チェック
    if (fromUserId === firstUserId && firstUser.balance < amount) {
      throw new Error('Insufficient balance');
    }

    // 送金処理
    await tx.user.update({
      where: { id: fromUserId },
      data: { balance: { decrement: amount } }
    });

    await tx.user.update({
      where: { id: toUserId },
      data: { balance: { increment: amount } }
    });
  });
}

// ✅ デッドロック時の再試行
async function executeWithRetry(fn, maxRetries = 3) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await fn();
    } catch (error) {
      if (error.code === 'P2034' && i < maxRetries - 1) {
        // Prismaのデッドロックエラーコード
        await new Promise(resolve => setTimeout(resolve, 100 * (i + 1)));
        continue;
      }
      throw error;
    }
  }
}

app.post('/api/transfer', async (req, res) => {
  try {
    await executeWithRetry(() =>
      transferMoney(req.body.fromUserId, req.body.toUserId, req.body.amount)
    );

    res.json({ message: 'Transfer successful' });
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});
```

---

## パフォーマンス問題

### ❌ エラー15: Slow API Response

**原因:**
- データベースクエリが最適化されていない
- インデックスがない

**解決策:**

```javascript
// ✅ ページネーション実装
app.get('/api/users', async (req, res) => {
  const page = parseInt(req.query.page) || 1;
  const limit = parseInt(req.query.limit) || 10;
  const skip = (page - 1) * limit;

  const [users, total] = await Promise.all([
    prisma.user.findMany({
      skip,
      take: limit,
      orderBy: { createdAt: 'desc' }
    }),
    prisma.user.count()
  ]);

  res.json({
    data: users,
    pagination: {
      page,
      limit,
      total,
      totalPages: Math.ceil(total / limit)
    }
  });
});

// ✅ キャッシュ実装（Redis）
const Redis = require('ioredis');
const redis = new Redis();

async function getCachedUsers(page, limit) {
  const cacheKey = `users:page:${page}:limit:${limit}`;

  // キャッシュから取得
  const cached = await redis.get(cacheKey);

  if (cached) {
    return JSON.parse(cached);
  }

  // DBから取得
  const users = await prisma.user.findMany({
    skip: (page - 1) * limit,
    take: limit
  });

  // キャッシュに保存（60秒）
  await redis.setex(cacheKey, 60, JSON.stringify(users));

  return users;
}

app.get('/api/users', async (req, res) => {
  const page = parseInt(req.query.page) || 1;
  const limit = parseInt(req.query.limit) || 10;

  const users = await getCachedUsers(page, limit);
  res.json(users);
});
```

**データベースインデックス:**

```sql
-- インデックス作成
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_posts_author_id ON posts(author_id);
CREATE INDEX idx_posts_created_at ON posts(created_at DESC);

-- 複合インデックス
CREATE INDEX idx_posts_author_created ON posts(author_id, created_at DESC);
```

---

### ❌ エラー16: Memory Leak

**原因:**
- イベントリスナーが解除されていない
- グローバル変数にデータが蓄積

**解決策:**

```javascript
// ❌ メモリリーク
const usersCache = {};

app.get('/api/users/:id', async (req, res) => {
  const userId = req.params.id;

  if (usersCache[userId]) {
    return res.json(usersCache[userId]);
  }

  const user = await getUser(userId);
  usersCache[userId] = user;  // キャッシュが増え続ける

  res.json(user);
});

// ✅ LRUキャッシュ使用
const LRU = require('lru-cache');

const usersCache = new LRU({
  max: 500,  // 最大500件
  ttl: 1000 * 60 * 5  // 5分
});

app.get('/api/users/:id', async (req, res) => {
  const userId = req.params.id;

  let user = usersCache.get(userId);

  if (!user) {
    user = await getUser(userId);
    usersCache.set(userId, user);
  }

  res.json(user);
});

// ✅ イベントリスナーのクリーンアップ
const EventEmitter = require('events');

app.get('/api/stream', (req, res) => {
  const emitter = new EventEmitter();

  const handler = (data) => {
    res.write(JSON.stringify(data) + '\n');
  };

  emitter.on('data', handler);

  // クライアント切断時にリスナー削除
  req.on('close', () => {
    emitter.removeListener('data', handler);
  });

  // データストリーム開始
  startDataStream(emitter);
});
```

---

## サーバー設定エラー

### ❌ エラー17: Port Already in Use

```
Error: listen EADDRINUSE: address already in use :::3000
```

**解決策:**

```bash
# プロセスを確認・停止
lsof -ti :3000 | xargs kill -9

# または異なるポートを使用
PORT=3001 node server.js
```

```javascript
// ✅ 動的ポート設定
const PORT = process.env.PORT || 3000;

const server = app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});

// エラーハンドリング
server.on('error', (error) => {
  if (error.code === 'EADDRINUSE') {
    console.error(`Port ${PORT} is already in use`);
    process.exit(1);
  } else {
    throw error;
  }
});
```

---

### ❌ エラー18: Environment Variable Not Set

```
TypeError: Cannot read property 'DATABASE_URL' of undefined
```

**解決策:**

```javascript
// ✅ 環境変数チェック
require('dotenv').config();

const requiredEnvVars = [
  'DATABASE_URL',
  'JWT_SECRET',
  'PORT'
];

for (const envVar of requiredEnvVars) {
  if (!process.env[envVar]) {
    console.error(`Error: ${envVar} is not set`);
    process.exit(1);
  }
}

// 設定オブジェクト
const config = {
  database: {
    url: process.env.DATABASE_URL
  },
  jwt: {
    secret: process.env.JWT_SECRET,
    expiresIn: process.env.JWT_EXPIRES_IN || '7d'
  },
  server: {
    port: parseInt(process.env.PORT) || 3000,
    env: process.env.NODE_ENV || 'development'
  }
};

module.exports = config;
```

**.env.example:**

```
DATABASE_URL=postgresql://user:password@localhost:5432/mydb
JWT_SECRET=your-secret-key
JWT_REFRESH_SECRET=your-refresh-secret-key
PORT=3000
NODE_ENV=development
```

---

## デプロイエラー

### ❌ エラー19: Build Failed in Production

```
Error: Cannot find module './config/database'
```

**原因:**
- 開発依存のパッケージが本番環境にインストールされていない
- ビルドプロセスが失敗

**解決策:**

```bash
# ✅ 本番環境用インストール
npm ci --production

# TypeScriptビルド
npm run build

# package.json
{
  "scripts": {
    "build": "tsc",
    "start": "node dist/server.js",
    "dev": "ts-node-dev --respawn src/server.ts"
  },
  "dependencies": {
    "express": "^4.18.2",
    "prisma": "^5.0.0"
  },
  "devDependencies": {
    "@types/express": "^4.17.17",
    "typescript": "^5.0.0",
    "ts-node-dev": "^2.0.0"
  }
}
```

**Dockerfile:**

```dockerfile
# マルチステージビルド
FROM node:20-alpine AS builder

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

# 本番環境
FROM node:20-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci --production

COPY --from=builder /app/dist ./dist

EXPOSE 3000

CMD ["node", "dist/server.js"]
```

---

### ❌ エラー20: Connection Refused in Production

```
Error: connect ECONNREFUSED 127.0.0.1:5432
```

**原因:**
- データベースホストが間違っている
- ファイアウォールでブロックされている

**解決策:**

```javascript
// ✅ 環境別の設定
const config = {
  development: {
    database: {
      host: 'localhost',
      port: 5432
    }
  },
  production: {
    database: {
      host: process.env.DB_HOST,
      port: parseInt(process.env.DB_PORT)
    }
  }
};

const env = process.env.NODE_ENV || 'development';
module.exports = config[env];
```

**.env.production:**

```
DB_HOST=db.example.com
DB_PORT=5432
DATABASE_URL=postgresql://user:password@db.example.com:5432/mydb
```

**Docker Compose:**

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "3000:3000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/mydb
    depends_on:
      - db

  db:
    image: postgres:15
    environment:
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=mydb
    volumes:
      - postgres-data:/var/lib/postgresql/data

volumes:
  postgres-data:
```

---

### ❌ エラー21: SSL Certificate Error

```
Error: unable to verify the first certificate
```

**原因:**
- 自己署名証明書を使用している

**解決策:**

```javascript
// ❌ 本番環境では絶対に使わない
process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0';

// ✅ 開発環境のみSSL検証をスキップ
const https = require('https');

const agent = new https.Agent({
  rejectUnauthorized: process.env.NODE_ENV !== 'development'
});

const response = await fetch('https://api.example.com/data', {
  agent
});
```

**PostgreSQL SSL接続:**

```javascript
// ✅ Prisma SSL設定
// DATABASE_URL="postgresql://user:password@host:5432/db?sslmode=require"

const { PrismaClient } = require('@prisma/client');

const prisma = new PrismaClient({
  datasources: {
    db: {
      url: process.env.DATABASE_URL
    }
  }
});
```

---

### ❌ エラー22: Out of Memory in Production

```
FATAL ERROR: Reached heap limit Allocation failed - JavaScript heap out of memory
```

**解決策:**

```bash
# ✅ Node.jsメモリ上限を増やす
node --max-old-space-size=4096 server.js

# package.json
{
  "scripts": {
    "start": "node --max-old-space-size=4096 dist/server.js"
  }
}
```

**Docker:**

```dockerfile
FROM node:20-alpine

# メモリ制限
ENV NODE_OPTIONS="--max-old-space-size=4096"

WORKDIR /app
COPY . .

CMD ["npm", "start"]
```

---

### ❌ エラー23: Health Check Failed

**原因:**
- ヘルスチェックエンドポイントが未実装

**解決策:**

```javascript
// ✅ ヘルスチェックエンドポイント
app.get('/health', async (req, res) => {
  try {
    // データベース接続確認
    await prisma.$queryRaw`SELECT 1`;

    res.json({
      status: 'healthy',
      timestamp: new Date().toISOString(),
      uptime: process.uptime()
    });
  } catch (error) {
    res.status(503).json({
      status: 'unhealthy',
      error: error.message
    });
  }
});

// より詳細なヘルスチェック
app.get('/health/detailed', async (req, res) => {
  const checks = {
    database: false,
    redis: false,
    memory: false
  };

  try {
    // データベース
    await prisma.$queryRaw`SELECT 1`;
    checks.database = true;

    // Redis
    await redis.ping();
    checks.redis = true;

    // メモリ使用量
    const memUsage = process.memoryUsage();
    checks.memory = memUsage.heapUsed / memUsage.heapTotal < 0.9;

    const allHealthy = Object.values(checks).every(v => v);

    res.status(allHealthy ? 200 : 503).json({
      status: allHealthy ? 'healthy' : 'unhealthy',
      checks,
      memory: {
        heapUsed: `${Math.round(memUsage.heapUsed / 1024 / 1024)}MB`,
        heapTotal: `${Math.round(memUsage.heapTotal / 1024 / 1024)}MB`
      }
    });
  } catch (error) {
    res.status(503).json({
      status: 'unhealthy',
      checks,
      error: error.message
    });
  }
});
```

**Kubernetes:**

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: backend
spec:
  containers:
  - name: app
    image: myapp:latest
    ports:
    - containerPort: 3000
    livenessProbe:
      httpGet:
        path: /health
        port: 3000
      initialDelaySeconds: 30
      periodSeconds: 10
    readinessProbe:
      httpGet:
        path: /health
        port: 3000
      initialDelaySeconds: 5
      periodSeconds: 5
```

---

### ❌ エラー24: Rate Limit Exceeded (Third-party API)

```
Error: Rate limit exceeded. Retry after 60 seconds.
```

**解決策:**

```javascript
// ✅ 指数バックオフによるリトライ
async function fetchWithRetry(url, options = {}, maxRetries = 3) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      const response = await fetch(url, options);

      if (response.status === 429) {
        const retryAfter = parseInt(response.headers.get('Retry-After') || '60');
        const delay = Math.min(1000 * Math.pow(2, i), retryAfter * 1000);

        console.log(`Rate limited. Retrying after ${delay}ms`);
        await new Promise(resolve => setTimeout(resolve, delay));
        continue;
      }

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      if (i === maxRetries - 1) throw error;

      const delay = 1000 * Math.pow(2, i);
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }
}

// Bottleneckでレート制限
const Bottleneck = require('bottleneck');

const limiter = new Bottleneck({
  maxConcurrent: 5,
  minTime: 200  // 各リクエスト間に200ms
});

const fetchWithRateLimit = limiter.wrap(async (url) => {
  const response = await fetch(url);
  return response.json();
});
```

---

### ❌ エラー25: WebSocket Connection Failed

```
WebSocket connection to 'ws://localhost:3000' failed
```

**解決策:**

```javascript
// ✅ WebSocket実装（ws）
const express = require('express');
const { WebSocketServer } = require('ws');
const http = require('http');

const app = express();
const server = http.createServer(app);
const wss = new WebSocketServer({ server });

wss.on('connection', (ws, req) => {
  console.log('Client connected');

  ws.on('message', (message) => {
    console.log('Received:', message.toString());

    // メッセージをブロードキャスト
    wss.clients.forEach((client) => {
      if (client.readyState === WebSocket.OPEN) {
        client.send(message.toString());
      }
    });
  });

  ws.on('close', () => {
    console.log('Client disconnected');
  });

  ws.on('error', (error) => {
    console.error('WebSocket error:', error);
  });

  // 定期的なpingで接続維持
  const interval = setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.ping();
    }
  }, 30000);

  ws.on('close', () => {
    clearInterval(interval);
  });
});

server.listen(3000, () => {
  console.log('Server running on http://localhost:3000');
});
```

**Socket.io:**

```javascript
// ✅ Socket.io（より高機能）
const express = require('express');
const http = require('http');
const { Server } = require('socket.io');

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: {
    origin: 'http://localhost:5173',
    methods: ['GET', 'POST']
  }
});

io.on('connection', (socket) => {
  console.log('User connected:', socket.id);

  socket.on('message', (data) => {
    io.emit('message', data);  // 全クライアントに送信
  });

  socket.on('disconnect', () => {
    console.log('User disconnected:', socket.id);
  });
});

server.listen(3000);
```

---

## まとめ

### このガイドで学んだこと

- バックエンド開発における25の頻出エラー
- 各エラーの原因と解決策
- セキュリティとパフォーマンスのベストプラクティス

### エラー解決の基本手順

1. **ログを確認** - エラーメッセージとスタックトレース
2. **HTTPステータスコード** - 4xx（クライアントエラー）、5xx（サーバーエラー）
3. **データベースログ** - クエリエラーやパフォーマンス問題
4. **ネットワークトラフィック** - Chrome DevTools、Postman
5. **モニタリングツール** - Sentry、Datadog、New Relic

### デバッグツール

```bash
# cURLでAPIテスト
curl -X POST http://localhost:3000/api/users \
  -H "Content-Type: application/json" \
  -d '{"name":"John","email":"john@example.com"}'

# HTTPieでAPIテスト（より読みやすい）
http POST localhost:3000/api/users name=John email=john@example.com

# データベースクエリログ（Prisma）
DATABASE_URL="..." npx prisma studio
```

### さらに学ぶ

- **[Express公式ドキュメント](https://expressjs.com/)**
- **[FastAPI公式ドキュメント](https://fastapi.tiangolo.com/)**
- **[Prisma公式ドキュメント](https://www.prisma.io/docs/)**

---

**関連ガイド:**
- [Backend Development - バックエンド開発基礎](../backend-development/SKILL.md)
- [Node.js Development - Node.js開発](../nodejs-development/SKILL.md)

**親ガイド:** [トラブルシューティングDB](./README.md)
