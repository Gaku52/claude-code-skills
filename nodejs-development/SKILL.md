---
name: nodejs-development
description: Node.js開発ガイド。Express、NestJS、Fastify、非同期処理、ストリーム、パフォーマンス最適化など、Node.jsアプリケーション開発のベストプラクティス。
---

# Node.js Development Skill

## 📋 目次

1. [概要](#概要)
2. [いつ使うか](#いつ使うか)
3. [Express](#express)
4. [NestJS](#nestjs)
5. [非同期処理](#非同期処理)
6. [ストリーム](#ストリーム)
7. [実践例](#実践例)
8. [Agent連携](#agent連携)

---

## 概要

このSkillは、Node.js開発をカバーします：

- **Express** - 軽量Webフレームワーク
- **NestJS** - エンタープライズ向けフレームワーク
- **非同期処理** - async/await, Promise
- **ストリーム** - 大容量ファイル処理
- **パフォーマンス** - クラスタリング、キャッシング
- **テスト** - Jest, Supertest

---

## いつ使うか

### 🎯 必須のタイミング

- [ ] 新規Node.jsプロジェクト作成時
- [ ] API開発時
- [ ] マイクロサービス構築時
- [ ] リアルタイム通信実装時（WebSocket）

---

## Express

### 基本セットアップ

```typescript
// src/index.ts
import express from 'express'
import cors from 'cors'
import helmet from 'helmet'
import morgan from 'morgan'

const app = express()

// ミドルウェア
app.use(express.json())
app.use(express.urlencoded({ extended: true }))
app.use(cors())
app.use(helmet())
app.use(morgan('dev'))

// ルート
app.get('/', (req, res) => {
  res.json({ message: 'API is running' })
})

// エラーハンドリング
app.use((err, req, res, next) => {
  console.error(err.stack)
  res.status(500).json({ error: 'Something went wrong' })
})

const PORT = process.env.PORT || 3000
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`)
})
```

### ルーティング

```typescript
// routes/users.ts
import express from 'express'

const router = express.Router()

router.get('/', async (req, res) => {
  const users = await prisma.user.findMany()
  res.json({ data: users })
})

router.get('/:id', async (req, res) => {
  const user = await prisma.user.findUnique({
    where: { id: req.params.id }
  })

  if (!user) {
    return res.status(404).json({ error: 'User not found' })
  }

  res.json({ data: user })
})

router.post('/', async (req, res) => {
  const user = await prisma.user.create({
    data: req.body
  })

  res.status(201).json({ data: user })
})

export default router

// src/index.ts
import userRoutes from './routes/users'
app.use('/api/users', userRoutes)
```

### ミドルウェア

```typescript
// middleware/auth.ts
import jwt from 'jsonwebtoken'

export async function authMiddleware(req, res, next) {
  const token = req.headers.authorization?.replace('Bearer ', '')

  if (!token) {
    return res.status(401).json({ error: 'No token provided' })
  }

  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET!)
    req.userId = decoded.userId
    next()
  } catch (error) {
    return res.status(401).json({ error: 'Invalid token' })
  }
}

// 使用例
router.get('/protected', authMiddleware, (req, res) => {
  res.json({ message: 'This is protected', userId: req.userId })
})
```

---

## NestJS

### プロジェクト作成

```bash
pnpm add -g @nestjs/cli
nest new my-project
```

### コントローラー

```typescript
// users.controller.ts
import { Controller, Get, Post, Body, Param, Delete } from '@nestjs/common'
import { UsersService } from './users.service'
import { CreateUserDto } from './dto/create-user.dto'

@Controller('users')
export class UsersController {
  constructor(private readonly usersService: UsersService) {}

  @Get()
  findAll() {
    return this.usersService.findAll()
  }

  @Get(':id')
  findOne(@Param('id') id: string) {
    return this.usersService.findOne(id)
  }

  @Post()
  create(@Body() createUserDto: CreateUserDto) {
    return this.usersService.create(createUserDto)
  }

  @Delete(':id')
  remove(@Param('id') id: string) {
    return this.usersService.remove(id)
  }
}
```

### サービス

```typescript
// users.service.ts
import { Injectable } from '@nestjs/common'
import { PrismaService } from '../prisma/prisma.service'
import { CreateUserDto } from './dto/create-user.dto'

@Injectable()
export class UsersService {
  constructor(private prisma: PrismaService) {}

  async findAll() {
    return this.prisma.user.findMany()
  }

  async findOne(id: string) {
    return this.prisma.user.findUnique({ where: { id } })
  }

  async create(createUserDto: CreateUserDto) {
    return this.prisma.user.create({ data: createUserDto })
  }

  async remove(id: string) {
    return this.prisma.user.delete({ where: { id } })
  }
}
```

### バリデーション

```typescript
// dto/create-user.dto.ts
import { IsEmail, IsString, MinLength } from 'class-validator'

export class CreateUserDto {
  @IsString()
  @MinLength(1)
  name: string

  @IsEmail()
  email: string

  @IsString()
  @MinLength(8)
  password: string
}

// main.ts
import { ValidationPipe } from '@nestjs/common'

async function bootstrap() {
  const app = await NestFactory.create(AppModule)
  app.useGlobalPipes(new ValidationPipe())
  await app.listen(3000)
}
```

---

## 非同期処理

### async/await

```typescript
// ✅ 良い例（async/await）
async function getUser(id: string) {
  try {
    const user = await prisma.user.findUnique({ where: { id } })
    const posts = await prisma.post.findMany({ where: { userId: id } })

    return { user, posts }
  } catch (error) {
    console.error(error)
    throw error
  }
}

// ✅ 並列実行
async function getUserWithPosts(id: string) {
  const [user, posts] = await Promise.all([
    prisma.user.findUnique({ where: { id } }),
    prisma.post.findMany({ where: { userId: id } })
  ])

  return { user, posts }
}
```

### Promise

```typescript
// ❌ 悪い例（Callback Hell）
function getData(callback) {
  getUser(userId, (user) => {
    getPosts(user.id, (posts) => {
      getComments(posts[0].id, (comments) => {
        callback({ user, posts, comments })
      })
    })
  })
}

// ✅ 良い例（Promise）
function getData() {
  return getUser(userId)
    .then(user => getPosts(user.id))
    .then(posts => getComments(posts[0].id))
    .then(comments => ({ user, posts, comments }))
}

// ✅ より良い例（async/await）
async function getData() {
  const user = await getUser(userId)
  const posts = await getPosts(user.id)
  const comments = await getComments(posts[0].id)

  return { user, posts, comments }
}
```

### エラーハンドリング

```typescript
// try-catch
async function createUser(data) {
  try {
    const user = await prisma.user.create({ data })
    return user
  } catch (error) {
    if (error.code === 'P2002') {
      throw new Error('Email already exists')
    }
    throw error
  }
}

// Promise.catch
getUserById(id)
  .then(user => console.log(user))
  .catch(error => console.error(error))
```

---

## ストリーム

### ファイル読み込み

```typescript
import fs from 'fs'

// ❌ 悪い例（大容量ファイルでメモリ不足）
const data = fs.readFileSync('large-file.txt', 'utf8')

// ✅ 良い例（ストリーム）
const stream = fs.createReadStream('large-file.txt', 'utf8')

stream.on('data', chunk => {
  console.log(chunk)
})

stream.on('end', () => {
  console.log('Done')
})

stream.on('error', error => {
  console.error(error)
})
```

### Express でファイルダウンロード

```typescript
router.get('/download', (req, res) => {
  const filePath = 'path/to/large-file.pdf'

  res.setHeader('Content-Disposition', 'attachment; filename=file.pdf')
  res.setHeader('Content-Type', 'application/pdf')

  const stream = fs.createReadStream(filePath)
  stream.pipe(res)
})
```

### CSV処理

```typescript
import csv from 'csv-parser'
import fs from 'fs'

async function processCSV(filePath: string) {
  const results: any[] = []

  return new Promise((resolve, reject) => {
    fs.createReadStream(filePath)
      .pipe(csv())
      .on('data', (data) => {
        // 各行を処理
        results.push(data)
      })
      .on('end', () => {
        resolve(results)
      })
      .on('error', reject)
  })
}
```

---

## 実践例

### Example 1: Express + Prisma CRUD API

```typescript
// src/index.ts
import express from 'express'
import cors from 'cors'
import helmet from 'helmet'
import { PrismaClient } from '@prisma/client'
import { z } from 'zod'

const app = express()
const prisma = new PrismaClient()

app.use(express.json())
app.use(cors())
app.use(helmet())

// GET /api/users
app.get('/api/users', async (req, res) => {
  const users = await prisma.user.findMany()
  res.json({ data: users })
})

// GET /api/users/:id
app.get('/api/users/:id', async (req, res) => {
  const user = await prisma.user.findUnique({
    where: { id: req.params.id }
  })

  if (!user) {
    return res.status(404).json({ error: 'User not found' })
  }

  res.json({ data: user })
})

// POST /api/users
app.post('/api/users', async (req, res) => {
  const schema = z.object({
    name: z.string().min(1),
    email: z.string().email()
  })

  try {
    const data = schema.parse(req.body)

    const user = await prisma.user.create({ data })
    res.status(201).json({ data: user })
  } catch (error) {
    if (error instanceof z.ZodError) {
      return res.status(400).json({ error: error.errors })
    }
    throw error
  }
})

app.listen(3000, () => {
  console.log('Server is running on port 3000')
})
```

### Example 2: WebSocket（Socket.io）

```typescript
import express from 'express'
import { createServer } from 'http'
import { Server } from 'socket.io'

const app = express()
const httpServer = createServer(app)
const io = new Server(httpServer, {
  cors: { origin: '*' }
})

io.on('connection', (socket) => {
  console.log('User connected:', socket.id)

  socket.on('message', (data) => {
    console.log('Message:', data)

    // 全クライアントにブロードキャスト
    io.emit('message', data)
  })

  socket.on('disconnect', () => {
    console.log('User disconnected:', socket.id)
  })
})

httpServer.listen(3000, () => {
  console.log('Server is running on port 3000')
})
```

---

## Agent連携

### 📖 Agentへの指示例

**Express API作成**
```
Express + Prismaで/api/postsのCRUD APIを作成してください。
Zodでバリデーションを含めてください。
```

**NestJS CRUD作成**
```
NestJSでPostsモジュールを作成してください。
Controller, Service, DTOを含めてください。
```

---

## まとめ

### Node.jsのベストプラクティス

1. **async/await** - 非同期処理を読みやすく
2. **ストリーム** - 大容量ファイル処理
3. **エラーハンドリング** - try-catch, Promise.catch
4. **型安全性** - TypeScript活用

---

_Last updated: 2025-12-24_
