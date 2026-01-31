# Express基礎 - 完全初心者ガイド

## 目次

1. [概要](#概要)
2. [Expressとは](#expressとは)
3. [初めてのExpressアプリ](#初めてのexpressアプリ)
4. [ルーティング](#ルーティング)
5. [ミドルウェア](#ミドルウェア)
6. [リクエストとレスポンス](#リクエストとレスポンス)
7. [演習問題](#演習問題)
8. [次のステップ](#次のステップ)

---

## 概要

### 何を学ぶか

- Expressフレームワークの基礎
- ルーティングの基本
- ミドルウェアの概念
- リクエスト/レスポンスの処理

### 学習時間：1〜1.5時間

---

## Expressとは

### 定義

**Express**は、Node.jsで最も人気のあるWebフレームワークです。

**特徴**：
- シンプルで最小限
- 柔軟性が高い
- 豊富なミドルウェア
- 大規模なエコシステム

### インストール

```bash
# プロジェクト作成
mkdir express-app
cd express-app
npm init -y

# Expressインストール
npm install express
```

---

## 初めてのExpressアプリ

### Hello World

`index.js`を作成：

```javascript
const express = require('express')
const app = express()
const PORT = 3000

// ルート定義
app.get('/', (req, res) => {
  res.send('Hello, Express!')
})

// サーバー起動
app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`)
})
```

### 実行

```bash
node index.js
```

ブラウザで`http://localhost:3000`を開く

---

## ルーティング

### HTTPメソッド

```javascript
const express = require('express')
const app = express()

// GET
app.get('/users', (req, res) => {
  res.json({ users: ['太郎', '花子'] })
})

// POST
app.post('/users', (req, res) => {
  res.status(201).json({ message: 'User created' })
})

// PUT
app.put('/users/:id', (req, res) => {
  res.json({ message: `User ${req.params.id} updated` })
})

// DELETE
app.delete('/users/:id', (req, res) => {
  res.json({ message: `User ${req.params.id} deleted` })
})

app.listen(3000)
```

### パスパラメータ

```javascript
// /users/123
app.get('/users/:id', (req, res) => {
  const userId = req.params.id
  res.json({ userId })
})

// 複数のパラメータ
app.get('/users/:userId/posts/:postId', (req, res) => {
  const { userId, postId } = req.params
  res.json({ userId, postId })
})
```

### クエリパラメータ

```javascript
// /search?q=express&limit=10
app.get('/search', (req, res) => {
  const { q, limit } = req.query
  res.json({ query: q, limit: limit || 20 })
})
```

---

## ミドルウェア

### ミドルウェアとは

**ミドルウェア**は、リクエストとレスポンスの間で実行される関数です。

```
Request → Middleware 1 → Middleware 2 → Route Handler → Response
```

### 組み込みミドルウェア

```javascript
const express = require('express')
const app = express()

// JSONパーサー
app.use(express.json())

// URLエンコードされたデータ
app.use(express.urlencoded({ extended: true }))

// 静的ファイル配信
app.use(express.static('public'))
```

### カスタムミドルウェア

```javascript
// ログミドルウェア
const logger = (req, res, next) => {
  console.log(`${req.method} ${req.url}`)
  next()  // 次のミドルウェアへ
}

app.use(logger)

// 認証ミドルウェア
const authMiddleware = (req, res, next) => {
  const token = req.headers.authorization

  if (!token) {
    return res.status(401).json({ error: 'Unauthorized' })
  }

  // トークン検証
  req.user = { id: 1, name: '太郎' }
  next()
}

// 特定のルートに適用
app.get('/protected', authMiddleware, (req, res) => {
  res.json({ user: req.user })
})
```

---

## リクエストとレスポンス

### req（リクエスト）オブジェクト

```javascript
app.post('/api/users', (req, res) => {
  // ボディ
  const body = req.body

  // パラメータ
  const id = req.params.id

  // クエリ
  const query = req.query

  // ヘッダー
  const contentType = req.get('Content-Type')

  // メソッド
  const method = req.method

  // URL
  const url = req.url
  const path = req.path

  res.json({ received: true })
})
```

### res（レスポンス）オブジェクト

```javascript
app.get('/api/users', (req, res) => {
  // JSON
  res.json({ name: '太郎' })

  // テキスト
  res.send('Hello')

  // ステータスコード
  res.status(404).json({ error: 'Not Found' })

  // リダイレクト
  res.redirect('/home')

  // ヘッダー設定
  res.set('Content-Type', 'application/json')

  // ファイル送信
  res.sendFile('/path/to/file.pdf')
})
```

---

## 実践例

### ユーザーCRUD API

```javascript
const express = require('express')
const app = express()

app.use(express.json())

// ダミーデータ
let users = [
  { id: 1, name: '太郎', email: 'taro@example.com' },
  { id: 2, name: '花子', email: 'hanako@example.com' }
]

// GET /api/users - 一覧取得
app.get('/api/users', (req, res) => {
  res.json({ users })
})

// GET /api/users/:id - 詳細取得
app.get('/api/users/:id', (req, res) => {
  const id = parseInt(req.params.id)
  const user = users.find(u => u.id === id)

  if (!user) {
    return res.status(404).json({ error: 'User not found' })
  }

  res.json({ user })
})

// POST /api/users - 作成
app.post('/api/users', (req, res) => {
  const { name, email } = req.body

  if (!name || !email) {
    return res.status(400).json({ error: 'Name and email are required' })
  }

  const newUser = {
    id: users.length + 1,
    name,
    email
  }

  users.push(newUser)
  res.status(201).json({ user: newUser })
})

// PUT /api/users/:id - 更新
app.put('/api/users/:id', (req, res) => {
  const id = parseInt(req.params.id)
  const { name, email } = req.body
  const index = users.findIndex(u => u.id === id)

  if (index === -1) {
    return res.status(404).json({ error: 'User not found' })
  }

  users[index] = { id, name, email }
  res.json({ user: users[index] })
})

// DELETE /api/users/:id - 削除
app.delete('/api/users/:id', (req, res) => {
  const id = parseInt(req.params.id)
  const index = users.findIndex(u => u.id === id)

  if (index === -1) {
    return res.status(404).json({ error: 'User not found' })
  }

  users.splice(index, 1)
  res.json({ message: 'User deleted' })
})

app.listen(3000, () => {
  console.log('Server running on port 3000')
})
```

### テスト

```bash
# 一覧取得
curl http://localhost:3000/api/users

# 詳細取得
curl http://localhost:3000/api/users/1

# 作成
curl -X POST http://localhost:3000/api/users \
  -H "Content-Type: application/json" \
  -d '{"name":"次郎","email":"jiro@example.com"}'

# 更新
curl -X PUT http://localhost:3000/api/users/1 \
  -H "Content-Type: application/json" \
  -d '{"name":"山田太郎","email":"taro@example.com"}'

# 削除
curl -X DELETE http://localhost:3000/api/users/1
```

---

## エラーハンドリング

### エラーハンドラー

```javascript
// 404ハンドラー
app.use((req, res) => {
  res.status(404).json({ error: 'Not Found' })
})

// エラーハンドラー（最後に配置）
app.use((err, req, res, next) => {
  console.error(err.stack)
  res.status(500).json({ error: 'Internal Server Error' })
})
```

### try-catchパターン

```javascript
app.get('/api/users/:id', async (req, res, next) => {
  try {
    const id = parseInt(req.params.id)
    const user = await getUserById(id)

    if (!user) {
      return res.status(404).json({ error: 'User not found' })
    }

    res.json({ user })
  } catch (error) {
    next(error)  // エラーハンドラーに渡す
  }
})
```

---

## よくある間違い

### ❌ 間違い1：next()を忘れる

```javascript
app.use((req, res, next) => {
  console.log('Middleware')
  // next()を忘れると次に進まない
})
```

**✅ 正しい方法**：

```javascript
app.use((req, res, next) => {
  console.log('Middleware')
  next()
})
```

### ❌ 間違い2：レスポンスを複数回送る

```javascript
app.get('/', (req, res) => {
  res.send('Hello')
  res.send('World')  // エラー
})
```

**✅ 正しい方法**：

```javascript
app.get('/', (req, res) => {
  res.send('Hello World')
})
```

---

## 演習問題

### 問題：タスク管理API

以下の要件でAPIを作成してください：
- GET /api/tasks - タスク一覧
- POST /api/tasks - タスク作成
- DELETE /api/tasks/:id - タスク削除

**解答例**は次のガイドで実装します。

---

## 次のステップ

### このガイドで学んだこと

- ✅ Expressフレームワークの基礎
- ✅ ルーティングの基本
- ✅ ミドルウェアの概念
- ✅ リクエスト/レスポンスの処理

### 次に学ぶべきガイド

**次のガイド**：[05-async-programming.md](./05-async-programming.md) - 非同期プログラミング

---

**前のガイド**：[03-npm-basics.md](./03-npm-basics.md)

**親ガイド**：[Node.js Development - SKILL.md](../../SKILL.md)
