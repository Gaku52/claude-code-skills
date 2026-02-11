# バックエンド実装ガイド

## 目次

1. [概要](#概要)
2. [環境構築](#環境構築)
3. [データベースセットアップ](#データベースセットアップ)
4. [プロジェクト構造](#プロジェクト構造)
5. [認証システム実装](#認証システム実装)
6. [タスクAPI実装](#タスクapi実装)
7. [エラーハンドリング](#エラーハンドリング)
8. [テスト](#テスト)
9. [トラブルシューティング](#トラブルシューティング)

---

## 概要

### このガイドで実装すること

- ✅ Express サーバーのセットアップ
- ✅ PostgreSQL + Prisma ORM の統合
- ✅ JWT 認証システム
- ✅ ユーザー登録・ログイン API
- ✅ タスク CRUD API
- ✅ ミドルウェア（認証、エラーハンドリング）
- ✅ バリデーション
- ✅ パスワードハッシュ化

### 学習時間：6-8時間

---

## 環境構築

### ステップ1：プロジェクト初期化

```bash
# プロジェクトディレクトリ作成
mkdir fullstack-task-app
cd fullstack-task-app
mkdir backend
cd backend

# package.json作成
npm init -y

# TypeScript設定
npm install -D typescript @types/node @types/express ts-node-dev
npx tsc --init
```

### ステップ2：依存関係インストール

```bash
# 本番依存関係
npm install express
npm install @prisma/client
npm install bcrypt jsonwebtoken
npm install dotenv cors
npm install zod

# 開発依存関係
npm install -D @types/bcrypt @types/jsonwebtoken @types/cors
npm install -D prisma
```

### ステップ3：tsconfig.json設定

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "commonjs",
    "lib": ["ES2020"],
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules"]
}
```

### ステップ4：package.json設定

```json
{
  "name": "task-app-backend",
  "version": "1.0.0",
  "scripts": {
    "dev": "ts-node-dev --respawn --transpile-only src/server.ts",
    "build": "tsc",
    "start": "node dist/server.js",
    "prisma:generate": "prisma generate",
    "prisma:migrate": "prisma migrate dev",
    "prisma:studio": "prisma studio"
  },
  "dependencies": {
    "@prisma/client": "^5.7.0",
    "bcrypt": "^5.1.1",
    "cors": "^2.8.5",
    "dotenv": "^16.3.1",
    "express": "^4.18.2",
    "jsonwebtoken": "^9.0.2",
    "zod": "^3.22.4"
  },
  "devDependencies": {
    "@types/bcrypt": "^5.0.2",
    "@types/cors": "^2.8.17",
    "@types/express": "^4.17.21",
    "@types/jsonwebtoken": "^9.0.5",
    "@types/node": "^20.10.5",
    "prisma": "^5.7.0",
    "ts-node-dev": "^2.0.0",
    "typescript": "^5.3.3"
  }
}
```

### ステップ5：環境変数設定

`.env`ファイルを作成：

```env
# サーバー設定
PORT=3001
NODE_ENV=development

# データベース
DATABASE_URL="postgresql://user:password@localhost:5432/taskapp?schema=public"

# JWT
JWT_SECRET=your-super-secret-jwt-key-change-this-in-production
JWT_EXPIRES_IN=7d

# CORS
CORS_ORIGIN=http://localhost:5173
```

**.env.example** も作成（Git用）：

```env
PORT=3001
NODE_ENV=development
DATABASE_URL="postgresql://user:password@localhost:5432/taskapp?schema=public"
JWT_SECRET=your-jwt-secret
JWT_EXPIRES_IN=7d
CORS_ORIGIN=http://localhost:5173
```

---

## データベースセットアップ

### ステップ1：Prisma初期化

```bash
npx prisma init
```

これで`prisma/schema.prisma`が作成されます。

### ステップ2：Prismaスキーマ定義

`prisma/schema.prisma`を編集：

```prisma
generator client {
  provider = "prisma-client-js"
}

datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")
}

model User {
  id        Int      @id @default(autoincrement())
  email     String   @unique
  password  String
  name      String
  tasks     Task[]
  createdAt DateTime @default(now())
  updatedAt DateTime @updatedAt

  @@map("users")
}

model Task {
  id          Int       @id @default(autoincrement())
  title       String
  description String?
  completed   Boolean   @default(false)
  priority    Priority  @default(MEDIUM)
  dueDate     DateTime?
  userId      Int
  user        User      @relation(fields: [userId], references: [id], onDelete: Cascade)
  createdAt   DateTime  @default(now())
  updatedAt   DateTime  @updatedAt

  @@index([userId])
  @@index([completed])
  @@index([priority])
  @@map("tasks")
}

enum Priority {
  LOW
  MEDIUM
  HIGH
}
```

### ステップ3：マイグレーション実行

```bash
# PostgreSQLが起動していることを確認

# マイグレーション作成・実行
npx prisma migrate dev --name init

# Prisma Clientの生成
npx prisma generate
```

### ステップ4：Prisma Studioで確認（オプション）

```bash
npx prisma studio
```

ブラウザで`http://localhost:5555`が開き、データベースをGUIで確認できます。

---

## プロジェクト構造

### ディレクトリ作成

```bash
mkdir -p src/{controllers,services,middleware,routes,types,utils,prisma}
```

### 最終的な構造

```
backend/
├── src/
│   ├── controllers/
│   │   ├── auth.controller.ts
│   │   └── task.controller.ts
│   ├── services/
│   │   ├── auth.service.ts
│   │   └── task.service.ts
│   ├── middleware/
│   │   ├── auth.middleware.ts
│   │   ├── error.middleware.ts
│   │   └── validation.middleware.ts
│   ├── routes/
│   │   ├── auth.routes.ts
│   │   └── task.routes.ts
│   ├── types/
│   │   └── index.ts
│   ├── utils/
│   │   ├── jwt.ts
│   │   ├── password.ts
│   │   └── validation.ts
│   ├── prisma/
│   │   └── client.ts
│   └── server.ts
├── prisma/
│   └── schema.prisma
├── .env
├── .env.example
├── .gitignore
├── package.json
└── tsconfig.json
```

---

## 認証システム実装

### ステップ1：Prisma Client設定

`src/prisma/client.ts`：

```typescript
import { PrismaClient } from '@prisma/client'

const prisma = new PrismaClient({
  log: process.env.NODE_ENV === 'development' ? ['query', 'error', 'warn'] : ['error'],
})

export default prisma
```

### ステップ2：型定義

`src/types/index.ts`：

```typescript
import { Request } from 'express'

export interface AuthRequest extends Request {
  userId?: number
}

export interface RegisterInput {
  email: string
  password: string
  name: string
}

export interface LoginInput {
  email: string
  password: string
}

export interface TaskInput {
  title: string
  description?: string
  priority?: 'LOW' | 'MEDIUM' | 'HIGH'
  dueDate?: string
}

export interface TaskUpdateInput {
  title?: string
  description?: string
  completed?: boolean
  priority?: 'LOW' | 'MEDIUM' | 'HIGH'
  dueDate?: string
}

export interface TaskQuery {
  completed?: string
  priority?: string
  sort?: string
  order?: 'asc' | 'desc'
}
```

### ステップ3：パスワードユーティリティ

`src/utils/password.ts`：

```typescript
import bcrypt from 'bcrypt'

const SALT_ROUNDS = 10

export async function hashPassword(password: string): Promise<string> {
  return bcrypt.hash(password, SALT_ROUNDS)
}

export async function comparePassword(
  password: string,
  hashedPassword: string
): Promise<boolean> {
  return bcrypt.compare(password, hashedPassword)
}
```

### ステップ4：JWTユーティリティ

`src/utils/jwt.ts`：

```typescript
import jwt from 'jsonwebtoken'

const JWT_SECRET = process.env.JWT_SECRET || 'fallback-secret-key'
const JWT_EXPIRES_IN = process.env.JWT_EXPIRES_IN || '7d'

export interface JwtPayload {
  userId: number
}

export function generateToken(userId: number): string {
  return jwt.sign({ userId } as JwtPayload, JWT_SECRET, {
    expiresIn: JWT_EXPIRES_IN,
  })
}

export function verifyToken(token: string): JwtPayload {
  try {
    return jwt.verify(token, JWT_SECRET) as JwtPayload
  } catch (error) {
    throw new Error('Invalid token')
  }
}
```

### ステップ5：バリデーション

`src/utils/validation.ts`：

```typescript
import { z } from 'zod'

export const registerSchema = z.object({
  email: z.string().email('Invalid email address'),
  password: z
    .string()
    .min(8, 'Password must be at least 8 characters')
    .regex(/[A-Za-z]/, 'Password must contain at least one letter')
    .regex(/[0-9]/, 'Password must contain at least one number'),
  name: z.string().min(1, 'Name is required').max(100, 'Name is too long'),
})

export const loginSchema = z.object({
  email: z.string().email('Invalid email address'),
  password: z.string().min(1, 'Password is required'),
})

export const taskSchema = z.object({
  title: z.string().min(1, 'Title is required').max(200, 'Title is too long'),
  description: z.string().max(1000, 'Description is too long').optional(),
  priority: z.enum(['LOW', 'MEDIUM', 'HIGH']).optional(),
  dueDate: z.string().datetime().optional(),
})

export const taskUpdateSchema = z.object({
  title: z.string().min(1).max(200).optional(),
  description: z.string().max(1000).optional(),
  completed: z.boolean().optional(),
  priority: z.enum(['LOW', 'MEDIUM', 'HIGH']).optional(),
  dueDate: z.string().datetime().optional(),
})
```

### ステップ6：認証サービス

`src/services/auth.service.ts`：

```typescript
import prisma from '../prisma/client'
import { hashPassword, comparePassword } from '../utils/password'
import { generateToken } from '../utils/jwt'
import { RegisterInput, LoginInput } from '../types'

export class AuthService {
  async register(input: RegisterInput) {
    // メール重複チェック
    const existingUser = await prisma.user.findUnique({
      where: { email: input.email },
    })

    if (existingUser) {
      throw new Error('Email already exists')
    }

    // パスワードハッシュ化
    const hashedPassword = await hashPassword(input.password)

    // ユーザー作成
    const user = await prisma.user.create({
      data: {
        email: input.email,
        password: hashedPassword,
        name: input.name,
      },
      select: {
        id: true,
        email: true,
        name: true,
        createdAt: true,
      },
    })

    // JWT生成
    const token = generateToken(user.id)

    return { user, token }
  }

  async login(input: LoginInput) {
    // ユーザー検索
    const user = await prisma.user.findUnique({
      where: { email: input.email },
    })

    if (!user) {
      throw new Error('Invalid credentials')
    }

    // パスワード検証
    const isValidPassword = await comparePassword(input.password, user.password)

    if (!isValidPassword) {
      throw new Error('Invalid credentials')
    }

    // JWT生成
    const token = generateToken(user.id)

    return {
      user: {
        id: user.id,
        email: user.email,
        name: user.name,
        createdAt: user.createdAt,
      },
      token,
    }
  }

  async getMe(userId: number) {
    const user = await prisma.user.findUnique({
      where: { id: userId },
      select: {
        id: true,
        email: true,
        name: true,
        createdAt: true,
      },
    })

    if (!user) {
      throw new Error('User not found')
    }

    return user
  }
}
```

### ステップ7：認証コントローラー

`src/controllers/auth.controller.ts`：

```typescript
import { Response } from 'express'
import { AuthService } from '../services/auth.service'
import { AuthRequest } from '../types'

const authService = new AuthService()

export class AuthController {
  async register(req: AuthRequest, res: Response) {
    try {
      const result = await authService.register(req.body)
      res.status(201).json(result)
    } catch (error) {
      if (error instanceof Error) {
        res.status(400).json({ error: error.message })
      } else {
        res.status(500).json({ error: 'Internal server error' })
      }
    }
  }

  async login(req: AuthRequest, res: Response) {
    try {
      const result = await authService.login(req.body)
      res.json(result)
    } catch (error) {
      if (error instanceof Error) {
        res.status(401).json({ error: error.message })
      } else {
        res.status(500).json({ error: 'Internal server error' })
      }
    }
  }

  async getMe(req: AuthRequest, res: Response) {
    try {
      if (!req.userId) {
        return res.status(401).json({ error: 'Unauthorized' })
      }

      const user = await authService.getMe(req.userId)
      res.json({ user })
    } catch (error) {
      if (error instanceof Error) {
        res.status(404).json({ error: error.message })
      } else {
        res.status(500).json({ error: 'Internal server error' })
      }
    }
  }

  async logout(req: AuthRequest, res: Response) {
    // JWT はステートレスなので、クライアント側でトークンを削除
    res.json({ message: 'Logged out successfully' })
  }
}
```

### ステップ8：認証ミドルウェア

`src/middleware/auth.middleware.ts`：

```typescript
import { Response, NextFunction } from 'express'
import { verifyToken } from '../utils/jwt'
import { AuthRequest } from '../types'

export function authenticate(req: AuthRequest, res: Response, next: NextFunction) {
  try {
    // Authorization ヘッダーからトークン取得
    const authHeader = req.headers.authorization

    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return res.status(401).json({ error: 'No token provided' })
    }

    const token = authHeader.substring(7) // "Bearer " を除去

    // トークン検証
    const payload = verifyToken(token)

    // ユーザーIDをリクエストに追加
    req.userId = payload.userId

    next()
  } catch (error) {
    res.status(401).json({ error: 'Invalid or expired token' })
  }
}
```

### ステップ9：バリデーションミドルウェア

`src/middleware/validation.middleware.ts`：

```typescript
import { Request, Response, NextFunction } from 'express'
import { z } from 'zod'

export function validate(schema: z.ZodSchema) {
  return (req: Request, res: Response, next: NextFunction) => {
    try {
      schema.parse(req.body)
      next()
    } catch (error) {
      if (error instanceof z.ZodError) {
        const errors = error.errors.map((err) => ({
          field: err.path.join('.'),
          message: err.message,
        }))
        return res.status(400).json({ errors })
      }
      next(error)
    }
  }
}
```

### ステップ10：認証ルート

`src/routes/auth.routes.ts`：

```typescript
import { Router } from 'express'
import { AuthController } from '../controllers/auth.controller'
import { authenticate } from '../middleware/auth.middleware'
import { validate } from '../middleware/validation.middleware'
import { registerSchema, loginSchema } from '../utils/validation'

const router = Router()
const authController = new AuthController()

router.post(
  '/register',
  validate(registerSchema),
  authController.register.bind(authController)
)

router.post(
  '/login',
  validate(loginSchema),
  authController.login.bind(authController)
)

router.get('/me', authenticate, authController.getMe.bind(authController))

router.post('/logout', authenticate, authController.logout.bind(authController))

export default router
```

---

## タスクAPI実装

### ステップ1：タスクサービス

`src/services/task.service.ts`：

```typescript
import prisma from '../prisma/client'
import { TaskInput, TaskUpdateInput, TaskQuery } from '../types'

export class TaskService {
  async getTasks(userId: number, query: TaskQuery) {
    const { completed, priority, sort = 'createdAt', order = 'desc' } = query

    // フィルター条件構築
    const where: any = { userId }

    if (completed !== undefined) {
      where.completed = completed === 'true'
    }

    if (priority) {
      where.priority = priority
    }

    // ソート条件
    const orderBy: any = {}
    orderBy[sort] = order

    // タスク取得
    const tasks = await prisma.task.findMany({
      where,
      orderBy,
      include: {
        user: {
          select: {
            id: true,
            name: true,
            email: true,
          },
        },
      },
    })

    return { tasks, total: tasks.length }
  }

  async getTaskById(userId: number, taskId: number) {
    const task = await prisma.task.findFirst({
      where: {
        id: taskId,
        userId,
      },
      include: {
        user: {
          select: {
            id: true,
            name: true,
            email: true,
          },
        },
      },
    })

    if (!task) {
      throw new Error('Task not found')
    }

    return task
  }

  async createTask(userId: number, input: TaskInput) {
    const task = await prisma.task.create({
      data: {
        title: input.title,
        description: input.description,
        priority: input.priority,
        dueDate: input.dueDate ? new Date(input.dueDate) : null,
        userId,
      },
      include: {
        user: {
          select: {
            id: true,
            name: true,
            email: true,
          },
        },
      },
    })

    return task
  }

  async updateTask(userId: number, taskId: number, input: TaskUpdateInput) {
    // タスク存在確認
    const existingTask = await prisma.task.findFirst({
      where: {
        id: taskId,
        userId,
      },
    })

    if (!existingTask) {
      throw new Error('Task not found')
    }

    // タスク更新
    const task = await prisma.task.update({
      where: { id: taskId },
      data: {
        ...(input.title !== undefined && { title: input.title }),
        ...(input.description !== undefined && { description: input.description }),
        ...(input.completed !== undefined && { completed: input.completed }),
        ...(input.priority !== undefined && { priority: input.priority }),
        ...(input.dueDate !== undefined && {
          dueDate: input.dueDate ? new Date(input.dueDate) : null,
        }),
      },
      include: {
        user: {
          select: {
            id: true,
            name: true,
            email: true,
          },
        },
      },
    })

    return task
  }

  async deleteTask(userId: number, taskId: number) {
    // タスク存在確認
    const existingTask = await prisma.task.findFirst({
      where: {
        id: taskId,
        userId,
      },
    })

    if (!existingTask) {
      throw new Error('Task not found')
    }

    // タスク削除
    await prisma.task.delete({
      where: { id: taskId },
    })

    return { message: 'Task deleted successfully' }
  }

  async getTaskStats(userId: number) {
    const [total, completed, highPriority, overdue] = await Promise.all([
      prisma.task.count({ where: { userId } }),
      prisma.task.count({ where: { userId, completed: true } }),
      prisma.task.count({ where: { userId, priority: 'HIGH' } }),
      prisma.task.count({
        where: {
          userId,
          completed: false,
          dueDate: {
            lt: new Date(),
          },
        },
      }),
    ])

    const pending = total - completed
    const completionRate = total > 0 ? Math.round((completed / total) * 100) : 0

    return {
      total,
      completed,
      pending,
      highPriority,
      overdue,
      completionRate,
    }
  }
}
```

### ステップ2：タスクコントローラー

`src/controllers/task.controller.ts`：

```typescript
import { Response } from 'express'
import { TaskService } from '../services/task.service'
import { AuthRequest } from '../types'

const taskService = new TaskService()

export class TaskController {
  async getTasks(req: AuthRequest, res: Response) {
    try {
      if (!req.userId) {
        return res.status(401).json({ error: 'Unauthorized' })
      }

      const result = await taskService.getTasks(req.userId, req.query)
      res.json(result)
    } catch (error) {
      if (error instanceof Error) {
        res.status(400).json({ error: error.message })
      } else {
        res.status(500).json({ error: 'Internal server error' })
      }
    }
  }

  async getTaskById(req: AuthRequest, res: Response) {
    try {
      if (!req.userId) {
        return res.status(401).json({ error: 'Unauthorized' })
      }

      const taskId = parseInt(req.params.id)
      const task = await taskService.getTaskById(req.userId, taskId)
      res.json({ task })
    } catch (error) {
      if (error instanceof Error) {
        res.status(404).json({ error: error.message })
      } else {
        res.status(500).json({ error: 'Internal server error' })
      }
    }
  }

  async createTask(req: AuthRequest, res: Response) {
    try {
      if (!req.userId) {
        return res.status(401).json({ error: 'Unauthorized' })
      }

      const task = await taskService.createTask(req.userId, req.body)
      res.status(201).json({ task })
    } catch (error) {
      if (error instanceof Error) {
        res.status(400).json({ error: error.message })
      } else {
        res.status(500).json({ error: 'Internal server error' })
      }
    }
  }

  async updateTask(req: AuthRequest, res: Response) {
    try {
      if (!req.userId) {
        return res.status(401).json({ error: 'Unauthorized' })
      }

      const taskId = parseInt(req.params.id)
      const task = await taskService.updateTask(req.userId, taskId, req.body)
      res.json({ task })
    } catch (error) {
      if (error instanceof Error) {
        res.status(404).json({ error: error.message })
      } else {
        res.status(500).json({ error: 'Internal server error' })
      }
    }
  }

  async deleteTask(req: AuthRequest, res: Response) {
    try {
      if (!req.userId) {
        return res.status(401).json({ error: 'Unauthorized' })
      }

      const taskId = parseInt(req.params.id)
      const result = await taskService.deleteTask(req.userId, taskId)
      res.json(result)
    } catch (error) {
      if (error instanceof Error) {
        res.status(404).json({ error: error.message })
      } else {
        res.status(500).json({ error: 'Internal server error' })
      }
    }
  }

  async getTaskStats(req: AuthRequest, res: Response) {
    try {
      if (!req.userId) {
        return res.status(401).json({ error: 'Unauthorized' })
      }

      const stats = await taskService.getTaskStats(req.userId)
      res.json({ stats })
    } catch (error) {
      res.status(500).json({ error: 'Internal server error' })
    }
  }
}
```

### ステップ3：タスクルート

`src/routes/task.routes.ts`：

```typescript
import { Router } from 'express'
import { TaskController } from '../controllers/task.controller'
import { authenticate } from '../middleware/auth.middleware'
import { validate } from '../middleware/validation.middleware'
import { taskSchema, taskUpdateSchema } from '../utils/validation'

const router = Router()
const taskController = new TaskController()

// 全てのルートに認証が必要
router.use(authenticate)

router.get('/', taskController.getTasks.bind(taskController))
router.get('/stats', taskController.getTaskStats.bind(taskController))
router.get('/:id', taskController.getTaskById.bind(taskController))
router.post(
  '/',
  validate(taskSchema),
  taskController.createTask.bind(taskController)
)
router.put(
  '/:id',
  validate(taskUpdateSchema),
  taskController.updateTask.bind(taskController)
)
router.delete('/:id', taskController.deleteTask.bind(taskController))

export default router
```

---

## エラーハンドリング

### グローバルエラーミドルウェア

`src/middleware/error.middleware.ts`：

```typescript
import { Request, Response, NextFunction } from 'express'

export function errorHandler(
  error: Error,
  req: Request,
  res: Response,
  next: NextFunction
) {
  console.error('Error:', error)

  if (res.headersSent) {
    return next(error)
  }

  res.status(500).json({
    error: 'Internal server error',
    message: process.env.NODE_ENV === 'development' ? error.message : undefined,
  })
}

export function notFoundHandler(req: Request, res: Response) {
  res.status(404).json({
    error: 'Not found',
    path: req.path,
  })
}
```

---

## サーバーセットアップ

### メインサーバーファイル

`src/server.ts`：

```typescript
import express from 'express'
import cors from 'cors'
import dotenv from 'dotenv'
import authRoutes from './routes/auth.routes'
import taskRoutes from './routes/task.routes'
import { errorHandler, notFoundHandler } from './middleware/error.middleware'

// 環境変数読み込み
dotenv.config()

const app = express()
const PORT = process.env.PORT || 3001

// ミドルウェア
app.use(cors({
  origin: process.env.CORS_ORIGIN || 'http://localhost:5173',
  credentials: true,
}))
app.use(express.json())
app.use(express.urlencoded({ extended: true }))

// ヘルスチェック
app.get('/', (req, res) => {
  res.json({
    message: 'Task App API',
    version: '1.0.0',
    status: 'healthy',
  })
})

// ルート
app.use('/api/auth', authRoutes)
app.use('/api/tasks', taskRoutes)

// エラーハンドリング
app.use(notFoundHandler)
app.use(errorHandler)

// サーバー起動
app.listen(PORT, () => {
  console.log(`🚀 Server running on http://localhost:${PORT}`)
  console.log(`📝 Environment: ${process.env.NODE_ENV}`)
})
```

---

## テスト

### サーバー起動

```bash
npm run dev
```

### curlでテスト

#### 1. ユーザー登録

```bash
curl -X POST http://localhost:3001/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "Test1234",
    "name": "テストユーザー"
  }'
```

**レスポンス:**
```json
{
  "user": {
    "id": 1,
    "email": "test@example.com",
    "name": "テストユーザー",
    "createdAt": "2024-12-24T10:00:00.000Z"
  },
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

#### 2. ログイン

```bash
curl -X POST http://localhost:3001/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "Test1234"
  }'
```

#### 3. タスク作成

```bash
# トークンを環境変数に設定
export TOKEN="your-jwt-token-here"

curl -X POST http://localhost:3001/api/tasks \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "title": "プロジェクト資料作成",
    "description": "Q1の報告書を作成する",
    "priority": "HIGH",
    "dueDate": "2024-12-31T00:00:00.000Z"
  }'
```

#### 4. タスク一覧取得

```bash
curl http://localhost:3001/api/tasks \
  -H "Authorization: Bearer $TOKEN"
```

#### 5. タスク統計取得

```bash
curl http://localhost:3001/api/tasks/stats \
  -H "Authorization: Bearer $TOKEN"
```

---

## トラブルシューティング

### ❌ 問題1：データベース接続エラー

```
Error: Can't reach database server
```

**解決策:**
```bash
# PostgreSQLが起動しているか確認
pg_isready

# 起動していない場合
brew services start postgresql@15  # macOS
sudo systemctl start postgresql    # Linux

# DATABASE_URLを確認
echo $DATABASE_URL
```

### ❌ 問題2：Prisma Client not generated

```
Error: @prisma/client did not initialize yet
```

**解決策:**
```bash
npx prisma generate
```

### ❌ 問題3：Port already in use

```
Error: listen EADDRINUSE: address already in use :::3001
```

**解決策:**
```bash
# 使用中のポートを確認
lsof -i :3001

# プロセスを終了
kill -9 <PID>

# または別のポートを使用
PORT=3002 npm run dev
```

### ❌ 問題4：JWT Secret not set

```
Error: JWT_SECRET is not defined
```

**解決策:**
```bash
# .envファイルに追加
echo "JWT_SECRET=$(openssl rand -base64 32)" >> .env
```

### ❌ 問題5：CORS エラー

```
Access to fetch has been blocked by CORS policy
```

**解決策:**

`.env`を確認：
```env
CORS_ORIGIN=http://localhost:5173
```

または`src/server.ts`で：
```typescript
app.use(cors({
  origin: ['http://localhost:5173', 'http://localhost:3000'],
  credentials: true,
}))
```

---

## まとめ

### このガイドで学んだこと

- ✅ Express + TypeScript プロジェクトのセットアップ
- ✅ Prisma ORM によるデータベース操作
- ✅ JWT 認証システムの実装
- ✅ レイヤードアーキテクチャ（Controller/Service）
- ✅ ミドルウェアパターン
- ✅ バリデーション（Zod）
- ✅ エラーハンドリング
- ✅ REST API 設計

### 次のステップ

**次のガイド:** [03-frontend-implementation.md](./03-frontend-implementation.md) - フロントエンド実装

フロントエンド実装ガイドでは、React + TypeScript でこのバックエンド API と連携する UI を構築します。

---

**前のガイド:** [01-project-overview.md](./01-project-overview.md)

**親ガイド:** [統合プロジェクト - README](../README.md)
