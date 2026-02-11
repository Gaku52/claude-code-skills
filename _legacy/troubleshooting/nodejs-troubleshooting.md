# Node.js トラブルシューティング

## 目次

1. [概要](#概要)
2. [インストール・環境構築エラー](#インストール環境構築エラー)
3. [モジュール・依存関係エラー](#モジュール依存関係エラー)
4. [サーバー・ネットワークエラー](#サーバーネットワークエラー)
5. [データベース接続エラー](#データベース接続エラー)
6. [認証・認可エラー](#認証認可エラー)
7. [パフォーマンス・メモリエラー](#パフォーマンスメモリエラー)
8. [ビルド・デプロイエラー](#ビルドデプロイエラー)

---

## 概要

このガイドは、Node.js開発で頻繁に遭遇するエラーと解決策をまとめたトラブルシューティングデータベースです。

**収録エラー数:** 25個

**対象バージョン:** Node.js 18.x ~ 22.x

---

## インストール・環境構築エラー

### ❌ エラー1: node: command not found

```
bash: node: command not found
```

**原因:**
- Node.jsがインストールされていない
- PATHが通っていない

**解決策:**

```bash
# macOS (Homebrew)
brew install node

# Ubuntu/Debian
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# インストール確認
node --version
npm --version
```

**nvm使用（推奨）:**

```bash
# nvmインストール
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash

# Node.js最新LTSをインストール
nvm install --lts
nvm use --lts

# バージョン確認
node --version
```

---

### ❌ エラー2: npm ERR! code EACCES

```
npm ERR! code EACCES
npm ERR! syscall access
npm ERR! path /usr/local/lib/node_modules
npm ERR! errno -13
npm ERR! Error: EACCES: permission denied
```

**原因:**
- グローバルインストールの権限不足

**解決策:**

```bash
# ❌ sudoは使わない（非推奨）
# sudo npm install -g some-package

# ✅ npmのデフォルトディレクトリを変更
mkdir ~/.npm-global
npm config set prefix '~/.npm-global'

# ~/.bashrc または ~/.zshrc に追加
echo 'export PATH=~/.npm-global/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# グローバルインストール（権限不要）
npm install -g typescript
```

**または nvm を使用（推奨）:**

```bash
# nvmを使えば権限問題は発生しない
nvm install 20
npm install -g typescript
```

---

### ❌ エラー3: npm ERR! code ENOENT

```
npm ERR! code ENOENT
npm ERR! syscall open
npm ERR! path /path/to/package.json
npm ERR! errno -2
npm ERR! enoent ENOENT: no such file or directory, open '/path/to/package.json'
```

**原因:**
- `package.json`が存在しないディレクトリで`npm install`を実行

**解決策:**

```bash
# プロジェクトディレクトリに移動
cd /path/to/your/project

# package.jsonがあるか確認
ls package.json

# なければ新規作成
npm init -y
```

---

### ❌ エラー4: npm ERR! peer dependency conflict

```
npm ERR! code ERESOLVE
npm ERR! ERESOLVE unable to resolve dependency tree
npm ERR! Found: react@18.2.0
npm ERR! Could not resolve dependency:
npm ERR! peer react@"^17.0.0" from some-package@1.0.0
```

**原因:**
- パッケージのpeer dependencyのバージョンが合わない

**解決策:**

```bash
# ✅ 方法1: --legacy-peer-depsを使用
npm install --legacy-peer-deps

# ✅ 方法2: --forceを使用（非推奨）
npm install --force

# ✅ 方法3: package.jsonで.npmrcに設定
echo "legacy-peer-deps=true" > .npmrc

# ✅ 方法4: 互換性のあるバージョンを探す
npm info some-package peerDependencies
```

---

## モジュール・依存関係エラー

### ❌ エラー5: Error: Cannot find module 'express'

```
Error: Cannot find module 'express'
Require stack:
- /app/server.js
```

**原因:**
- モジュールがインストールされていない
- `node_modules`が削除された

**解決策:**

```bash
# 依存関係を再インストール
npm install

# 特定のモジュールをインストール
npm install express

# package.jsonとnode_modulesの同期を確認
npm ci  # クリーンインストール
```

**コード例:**

```javascript
// ✅ モジュールのインポート前に確認
try {
  const express = require('express');
  console.log('Express loaded successfully');
} catch (error) {
  console.error('Express not found. Run: npm install express');
  process.exit(1);
}
```

---

### ❌ エラー6: SyntaxError: Cannot use import statement outside a module

```
SyntaxError: Cannot use import statement outside a module
```

**原因:**
- CommonJS環境でESM構文を使用している

**解決策:**

```javascript
// ❌ CommonJS環境でimport
import express from 'express'; // エラー

// ✅ CommonJSではrequireを使用
const express = require('express');
```

**ESMを使用する場合（package.json）:**

```json
{
  "type": "module",
  "scripts": {
    "start": "node server.js"
  }
}
```

**TypeScript + Node.js:**

```json
// tsconfig.json
{
  "compilerOptions": {
    "module": "CommonJS",  // または "ESNext"
    "target": "ES2020",
    "esModuleInterop": true
  }
}
```

```bash
# TypeScriptをインストール
npm install --save-dev typescript @types/node

# ビルドして実行
npx tsc
node dist/server.js
```

---

### ❌ エラー7: Error [ERR_REQUIRE_ESM]: require() of ES Module not supported

```
Error [ERR_REQUIRE_ESM]: require() of ES Module /node_modules/chalk/source/index.js from /app/server.js not supported.
```

**原因:**
- ESM専用パッケージをrequireで読み込もうとしている

**解決策:**

```javascript
// ❌ ESMパッケージをrequire
const chalk = require('chalk'); // エラー

// ✅ 方法1: importを使用（package.jsonに "type": "module" を追加）
import chalk from 'chalk';

// ✅ 方法2: dynamic import
const loadChalk = async () => {
  const chalk = await import('chalk');
  console.log(chalk.default.blue('Hello'));
};
loadChalk();

// ✅ 方法3: 古いバージョンを使用
npm install chalk@4  // CommonJS対応バージョン
```

---

### ❌ エラー8: npm ERR! ENOLOCK: no lockfile found

```
npm ERR! code ENOLOCK
npm ERR! enolock ENOLOCK: no lockfile found
```

**原因:**
- `package-lock.json`が存在しない状態で`npm ci`を実行

**解決策:**

```bash
# ❌ npm ci（lockfileが必要）
npm ci

# ✅ npm install（lockfile生成）
npm install

# lockfile生成後にnpm ci
npm install
npm ci
```

---

## サーバー・ネットワークエラー

### ❌ エラー9: Error: listen EADDRINUSE: address already in use :::3000

```
Error: listen EADDRINUSE: address already in use :::3000
```

**原因:**
- ポート3000が既に使用中

**解決策:**

```bash
# macOS/Linux: ポートを使用しているプロセスを確認
lsof -ti :3000

# プロセスを停止
lsof -ti :3000 | xargs kill -9

# Windows (PowerShell)
netstat -ano | findstr :3000
taskkill /PID <PID> /F

# または別のポートを使用
PORT=3001 node server.js
```

**環境変数でポート設定:**

```javascript
// ✅ 環境変数でポート設定
const express = require('express');
const app = express();

const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
```

**.env:**

```
PORT=3001
```

---

### ❌ エラー10: Error: connect ECONNREFUSED 127.0.0.1:3000

```
Error: connect ECONNREFUSED 127.0.0.1:3000
```

**原因:**
- サーバーが起動していない
- 間違ったポートに接続しようとしている

**解決策:**

```bash
# サーバーが起動しているか確認
curl http://localhost:3000

# サーバーを起動
node server.js

# ポート番号を確認
echo $PORT
```

**クライアント側の設定:**

```javascript
// ✅ エラーハンドリング付きリクエスト
const axios = require('axios');

async function fetchData() {
  try {
    const response = await axios.get('http://localhost:3000/api/data');
    console.log(response.data);
  } catch (error) {
    if (error.code === 'ECONNREFUSED') {
      console.error('Server is not running on port 3000');
    } else {
      console.error('Request failed:', error.message);
    }
  }
}
```

---

### ❌ エラー11: Error: Request timeout

```
Error: timeout of 5000ms exceeded
```

**原因:**
- APIレスポンスが遅い
- タイムアウト設定が短すぎる

**解決策:**

```javascript
// ✅ タイムアウト設定を調整
const axios = require('axios');

const api = axios.create({
  baseURL: 'https://api.example.com',
  timeout: 10000  // 10秒
});

// リクエストごとにタイムアウト設定
api.get('/slow-endpoint', {
  timeout: 30000  // 30秒
});
```

**Express サーバー側:**

```javascript
const express = require('express');
const app = express();

// タイムアウト設定
const server = app.listen(3000);
server.timeout = 30000;  // 30秒

// または個別ルートで
app.get('/slow-operation', async (req, res) => {
  req.setTimeout(60000);  // このルートのみ60秒

  const result = await slowOperation();
  res.json(result);
});
```

---

### ❌ エラー12: CORS policy blocked

```
Access to XMLHttpRequest has been blocked by CORS policy: No 'Access-Control-Allow-Origin' header is present on the requested resource.
```

**原因:**
- CORS設定が不足

**解決策:**

```javascript
// ✅ Express + cors
const express = require('express');
const cors = require('cors');
const app = express();

// すべてのオリジンを許可（開発環境のみ）
app.use(cors());

// 特定のオリジンのみ許可（本番環境）
app.use(cors({
  origin: 'https://example.com',
  credentials: true
}));

// 複数のオリジンを許可
const allowedOrigins = ['https://example.com', 'https://app.example.com'];
app.use(cors({
  origin: function (origin, callback) {
    if (!origin || allowedOrigins.indexOf(origin) !== -1) {
      callback(null, true);
    } else {
      callback(new Error('Not allowed by CORS'));
    }
  },
  credentials: true
}));
```

**手動でCORSヘッダーを設定:**

```javascript
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, Authorization');

  if (req.method === 'OPTIONS') {
    return res.sendStatus(200);
  }

  next();
});
```

---

## データベース接続エラー

### ❌ エラー13: Error: Connection lost: The server closed the connection

```
Error: Connection lost: The server closed the connection.
```

**原因:**
- MySQLの接続タイムアウト
- データベースサーバーが停止

**解決策:**

```javascript
// ✅ MySQL接続プールを使用
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

// 使用例
async function query(sql, params) {
  let connection;
  try {
    connection = await pool.getConnection();
    const [rows] = await connection.execute(sql, params);
    return rows;
  } catch (error) {
    console.error('Database query failed:', error);
    throw error;
  } finally {
    if (connection) connection.release();
  }
}
```

---

### ❌ エラー14: Prisma Client Error: PrismaClientInitializationError

```
PrismaClientInitializationError:
Can't reach database server at `localhost:5432`
```

**原因:**
- データベースサーバーが起動していない
- DATABASE_URLが間違っている

**解決策:**

```bash
# PostgreSQLが起動しているか確認
pg_isready

# macOS (Homebrew)
brew services start postgresql

# Docker
docker run -d \
  --name postgres \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=mydb \
  -p 5432:5432 \
  postgres:15
```

**.env:**

```
DATABASE_URL="postgresql://username:password@localhost:5432/mydb?schema=public"
```

**Prisma再生成:**

```bash
# Prisma Clientを再生成
npx prisma generate

# マイグレーション実行
npx prisma migrate dev --name init

# 接続確認
npx prisma studio
```

---

### ❌ エラー15: MongoServerError: Authentication failed

```
MongoServerError: Authentication failed.
```

**原因:**
- MongoDB認証情報が間違っている

**解決策:**

```javascript
// ✅ MongoDB接続（Mongoose）
const mongoose = require('mongoose');

const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://localhost:27017/mydb';

mongoose.connect(MONGODB_URI, {
  useNewUrlParser: true,
  useUnifiedTopology: true,
  authSource: 'admin'  // 認証データベース指定
})
.then(() => console.log('MongoDB Connected'))
.catch(err => console.error('MongoDB Connection Error:', err));

// エラーハンドリング
mongoose.connection.on('error', err => {
  console.error('MongoDB error:', err);
});

mongoose.connection.on('disconnected', () => {
  console.log('MongoDB disconnected');
});
```

**.env:**

```
MONGODB_URI=mongodb://username:password@localhost:27017/mydb?authSource=admin
```

**Docker MongoDB:**

```bash
docker run -d \
  --name mongodb \
  -e MONGO_INITDB_ROOT_USERNAME=admin \
  -e MONGO_INITDB_ROOT_PASSWORD=password \
  -p 27017:27017 \
  mongo:6
```

---

### ❌ エラー16: Error: Prisma N+1 Query Problem

**原因:**
- リレーションデータを効率的にフェッチしていない

**間違った例:**

```javascript
// ❌ N+1 Query Problem
const users = await prisma.user.findMany();

for (const user of users) {
  const posts = await prisma.post.findMany({
    where: { authorId: user.id }
  });
  console.log(user.name, posts.length);
}
// 1 + N回のクエリ実行
```

**解決策:**

```javascript
// ✅ includeで一括取得
const users = await prisma.user.findMany({
  include: {
    posts: true
  }
});

users.forEach(user => {
  console.log(user.name, user.posts.length);
});
// 1回のクエリで完了

// ✅ selectで必要なフィールドのみ取得
const users = await prisma.user.findMany({
  select: {
    id: true,
    name: true,
    posts: {
      select: {
        id: true,
        title: true
      }
    }
  }
});
```

---

## 認証・認可エラー

### ❌ エラー17: JsonWebTokenError: invalid signature

```
JsonWebTokenError: invalid signature
```

**原因:**
- JWT署名検証に失敗
- 環境変数のJWT_SECRETが間違っている

**解決策:**

```javascript
// ✅ JWT署名・検証
const jwt = require('jsonwebtoken');

const JWT_SECRET = process.env.JWT_SECRET || 'your-secret-key';

// トークン生成
function generateToken(userId) {
  return jwt.sign(
    { userId },
    JWT_SECRET,
    { expiresIn: '7d' }
  );
}

// トークン検証
function verifyToken(token) {
  try {
    return jwt.verify(token, JWT_SECRET);
  } catch (error) {
    if (error.name === 'JsonWebTokenError') {
      throw new Error('Invalid token');
    }
    if (error.name === 'TokenExpiredError') {
      throw new Error('Token expired');
    }
    throw error;
  }
}

// Express middleware
function authMiddleware(req, res, next) {
  const authHeader = req.headers.authorization;

  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return res.status(401).json({ error: 'No token provided' });
  }

  const token = authHeader.substring(7);

  try {
    const decoded = verifyToken(token);
    req.userId = decoded.userId;
    next();
  } catch (error) {
    return res.status(401).json({ error: error.message });
  }
}
```

**.env:**

```
JWT_SECRET=your-very-secure-secret-key-change-this-in-production
```

---

### ❌ エラー18: TokenExpiredError: jwt expired

```
TokenExpiredError: jwt expired
```

**原因:**
- JWTの有効期限切れ

**解決策:**

```javascript
// ✅ リフレッシュトークン実装
const jwt = require('jsonwebtoken');

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

// リフレッシュエンドポイント
app.post('/api/refresh', (req, res) => {
  const { refreshToken } = req.body;

  if (!refreshToken) {
    return res.status(401).json({ error: 'No refresh token' });
  }

  try {
    const decoded = jwt.verify(refreshToken, process.env.JWT_REFRESH_SECRET);
    const { accessToken, refreshToken: newRefreshToken } = generateTokens(decoded.userId);

    res.json({ accessToken, refreshToken: newRefreshToken });
  } catch (error) {
    return res.status(401).json({ error: 'Invalid refresh token' });
  }
});
```

---

### ❌ エラー19: Error: bcrypt compare failed

```
Error: Illegal arguments: undefined, string
```

**原因:**
- bcrypt.compareでundefinedを渡している

**解決策:**

```javascript
// ✅ パスワードハッシュ化・検証
const bcrypt = require('bcrypt');

// ユーザー登録
async function registerUser(email, password) {
  if (!password) {
    throw new Error('Password is required');
  }

  const hashedPassword = await bcrypt.hash(password, 10);

  const user = await prisma.user.create({
    data: {
      email,
      password: hashedPassword
    }
  });

  return user;
}

// ログイン
async function loginUser(email, password) {
  const user = await prisma.user.findUnique({
    where: { email }
  });

  if (!user) {
    throw new Error('User not found');
  }

  if (!user.password) {
    throw new Error('Password not set for this user');
  }

  const isValid = await bcrypt.compare(password, user.password);

  if (!isValid) {
    throw new Error('Invalid password');
  }

  return user;
}
```

---

## パフォーマンス・メモリエラー

### ❌ エラー20: Error: FATAL ERROR: CALL_AND_RETRY_LAST Allocation failed - JavaScript heap out of memory

```
FATAL ERROR: CALL_AND_RETRY_LAST Allocation failed - JavaScript heap out of memory
```

**原因:**
- メモリリーク
- 大量データの一括処理

**解決策:**

```bash
# ✅ Node.jsのメモリ上限を増やす
node --max-old-space-size=4096 server.js

# package.json
{
  "scripts": {
    "start": "node --max-old-space-size=4096 server.js"
  }
}
```

**メモリリークを防ぐ:**

```javascript
// ❌ イベントリスナーのリーク
const EventEmitter = require('events');
const emitter = new EventEmitter();

function addListener() {
  emitter.on('event', () => {
    console.log('Event fired');
  });
}

// 呼ばれるたびにリスナーが増える
for (let i = 0; i < 1000; i++) {
  addListener();
}

// ✅ リスナーを削除
function addListener() {
  const handler = () => {
    console.log('Event fired');
  };

  emitter.on('event', handler);

  // クリーンアップ
  return () => emitter.removeListener('event', handler);
}

const cleanup = addListener();
cleanup();  // リスナーを削除
```

**大量データのストリーム処理:**

```javascript
// ❌ 一括読み込み（メモリ不足）
const fs = require('fs');
const data = fs.readFileSync('large-file.txt', 'utf8');
console.log(data);

// ✅ ストリーム処理
const fs = require('fs');
const readline = require('readline');

const fileStream = fs.createReadStream('large-file.txt');
const rl = readline.createInterface({
  input: fileStream,
  crlfDelay: Infinity
});

rl.on('line', (line) => {
  console.log(`Line: ${line}`);
});

rl.on('close', () => {
  console.log('File processing completed');
});
```

---

### ❌ エラー21: Event loop blocked warning

```
(node:12345) MaxListenersExceededWarning: Possible EventEmitter memory leak detected.
11 event listeners added to [EventEmitter]. Use emitter.setMaxListeners() to increase limit
```

**原因:**
- イベントリスナーが増え続けている

**解決策:**

```javascript
// ✅ 最大リスナー数を増やす
const EventEmitter = require('events');
const emitter = new EventEmitter();

emitter.setMaxListeners(20);

// ✅ または once() を使用（1回のみ実行）
emitter.once('event', () => {
  console.log('This will only run once');
});

// ✅ リスナーをクリーンアップ
const handler = () => console.log('Event');
emitter.on('event', handler);

// 不要になったら削除
emitter.removeListener('event', handler);
```

---

### ❌ エラー22: UnhandledPromiseRejectionWarning

```
(node:12345) UnhandledPromiseRejectionWarning: Error: Something went wrong
(Use `node --trace-warnings ...` to show where the warning was created)
(node:12345) UnhandledPromiseRejectionWarning: Unhandled promise rejection.
```

**原因:**
- Promiseのエラーをキャッチしていない

**解決策:**

```javascript
// ❌ catch がない
async function fetchData() {
  const response = await fetch('https://api.example.com/data');
  return response.json();
}

fetchData(); // エラー時に警告

// ✅ try-catch でエラーハンドリング
async function fetchData() {
  try {
    const response = await fetch('https://api.example.com/data');
    return await response.json();
  } catch (error) {
    console.error('Fetch failed:', error);
    throw error;
  }
}

// ✅ .catch() を使用
fetchData()
  .then(data => console.log(data))
  .catch(error => console.error(error));

// ✅ グローバルハンドラー（最終手段）
process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
  // アプリケーションを安全に終了
  process.exit(1);
});
```

---

### ❌ エラー23: EventEmitter memory leak detected

**原因:**
- イベントリスナーが適切に削除されていない

**解決策:**

```javascript
// ✅ Express でのメモリリーク防止
const express = require('express');
const app = express();

// ❌ ミドルウェア内でイベントリスナー追加
app.use((req, res, next) => {
  req.on('close', () => {
    console.log('Request closed');
  });
  next();
});

// ✅ once() を使用
app.use((req, res, next) => {
  req.once('close', () => {
    console.log('Request closed');
  });
  next();
});

// ✅ AbortController でクリーンアップ
app.get('/api/data', async (req, res) => {
  const controller = new AbortController();

  req.on('close', () => {
    controller.abort();
  });

  try {
    const response = await fetch('https://api.example.com/data', {
      signal: controller.signal
    });
    const data = await response.json();
    res.json(data);
  } catch (error) {
    if (error.name === 'AbortError') {
      console.log('Request cancelled');
    } else {
      console.error(error);
      res.status(500).json({ error: 'Internal server error' });
    }
  }
});
```

---

## ビルド・デプロイエラー

### ❌ エラー24: TypeError: Cannot read property 'version' of undefined

```
TypeError: Cannot read property 'version' of undefined
at Object.<anonymous> (/app/server.js:5:30)
```

**原因:**
- `package.json`の読み込み失敗
- 環境変数が設定されていない

**解決策:**

```javascript
// ❌ 危険な読み込み
const packageJson = require('./package.json');
console.log(packageJson.version);

// ✅ 安全な読み込み
const fs = require('fs');
const path = require('path');

function getAppVersion() {
  try {
    const packageJsonPath = path.join(__dirname, 'package.json');
    const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
    return packageJson.version || 'unknown';
  } catch (error) {
    console.error('Failed to read package.json:', error);
    return 'unknown';
  }
}

const version = getAppVersion();
console.log(`App version: ${version}`);
```

**環境変数のフォールバック:**

```javascript
// ✅ 環境変数に安全にアクセス
const PORT = process.env.PORT || 3000;
const NODE_ENV = process.env.NODE_ENV || 'development';
const DATABASE_URL = process.env.DATABASE_URL;

if (!DATABASE_URL) {
  console.error('DATABASE_URL is not set');
  process.exit(1);
}

// dotenvを使用
require('dotenv').config();

const config = {
  port: process.env.PORT || 3000,
  dbUrl: process.env.DATABASE_URL,
  jwtSecret: process.env.JWT_SECRET,
};

// 必須の環境変数をチェック
const requiredEnvVars = ['DATABASE_URL', 'JWT_SECRET'];
for (const envVar of requiredEnvVars) {
  if (!process.env[envVar]) {
    throw new Error(`${envVar} is not set`);
  }
}
```

---

### ❌ エラー25: Error: EMFILE: too many open files

```
Error: EMFILE: too many open files, open '/path/to/file'
```

**原因:**
- ファイルディスクリプタの上限に達した
- ファイルを閉じていない

**解決策:**

```bash
# macOS: ファイルディスクリプタ上限を増やす
ulimit -n 4096

# 永続化（~/.zshrc または ~/.bashrc）
echo 'ulimit -n 4096' >> ~/.zshrc
```

**コードでファイルを適切に閉じる:**

```javascript
// ❌ ファイルを閉じていない
const fs = require('fs');

for (let i = 0; i < 10000; i++) {
  fs.readFile(`file-${i}.txt`, 'utf8', (err, data) => {
    if (err) throw err;
    console.log(data);
  });
}

// ✅ ストリームを使用
const fs = require('fs');
const { pipeline } = require('stream/promises');

async function processFiles() {
  for (let i = 0; i < 10000; i++) {
    const readStream = fs.createReadStream(`file-${i}.txt`);

    readStream.on('data', (chunk) => {
      console.log(chunk.toString());
    });

    await new Promise((resolve, reject) => {
      readStream.on('end', resolve);
      readStream.on('error', reject);
    });
  }
}

// ✅ 同期処理で順次実行
const fs = require('fs').promises;

async function processFilesSequentially() {
  for (let i = 0; i < 10000; i++) {
    const data = await fs.readFile(`file-${i}.txt`, 'utf8');
    console.log(data);
  }
}

// ✅ 並列数を制限
const pLimit = require('p-limit');
const limit = pLimit(10);  // 最大10並列

const promises = [];
for (let i = 0; i < 10000; i++) {
  promises.push(
    limit(() => fs.readFile(`file-${i}.txt`, 'utf8'))
  );
}

const results = await Promise.all(promises);
```

---

## まとめ

### このガイドで学んだこと

- Node.js開発における25の頻出エラー
- 各エラーの原因と解決策
- ベストプラクティス

### エラー解決の基本手順

1. **エラーメッセージを読む** - エラーコードとメッセージを確認
2. **スタックトレースを確認** - エラーが発生したファイル・行番号
3. **公式ドキュメントを確認** - [Node.js Docs](https://nodejs.org/docs/)
4. **このガイドで検索** - よくあるエラーはここに記載
5. **ログを確認** - `console.log`や`debug`モジュールを活用

### デバッグツール

```bash
# Node.js組み込みデバッガー
node inspect server.js

# Chrome DevTools
node --inspect server.js

# VS Code デバッグ設定
{
  "type": "node",
  "request": "launch",
  "name": "Launch Program",
  "program": "${workspaceFolder}/server.js"
}
```

### さらに学ぶ

- **[Node.js公式ドキュメント](https://nodejs.org/docs/)**
- **[Express公式ドキュメント](https://expressjs.com/)**
- **[Prisma公式ドキュメント](https://www.prisma.io/docs/)**

---

**関連ガイド:**
- [Node.js Development - 基礎ガイド](../nodejs-development/SKILL.md)
- [Backend Development - バックエンド開発](../backend-development/SKILL.md)

**親ガイド:** [トラブルシューティングDB](./README.md)
