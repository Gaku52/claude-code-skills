# フロントエンド実装ガイド

## 目次

1. [概要](#概要)
2. [環境構築](#環境構築)
3. [プロジェクト構造](#プロジェクト構造)
4. [API通信設定](#api通信設定)
5. [認証システム実装](#認証システム実装)
6. [タスク管理機能実装](#タスク管理機能実装)
7. [ダッシュボード実装](#ダッシュボード実装)
8. [スタイリング](#スタイリング)
9. [テスト](#テスト)
10. [トラブルシューティング](#トラブルシューティング)

---

## 概要

### このガイドで実装すること

- ✅ React + TypeScript + Vite のセットアップ
- ✅ React Router によるルーティング
- ✅ React Query によるデータフェッチング
- ✅ 認証機能（ログイン・登録・ログアウト）
- ✅ タスク管理機能（CRUD）
- ✅ ダッシュボード（統計表示）
- ✅ Tailwind CSS によるスタイリング

### 学習時間：6-8時間

---

## 環境構築

### ステップ1：プロジェクト作成

```bash
# fullstack-task-app ディレクトリに移動
cd fullstack-task-app

# Vite + React + TypeScript プロジェクト作成
npm create vite@latest frontend -- --template react-ts
cd frontend
```

### ステップ2：依存関係インストール

```bash
# 基本パッケージ
npm install

# ルーティング
npm install react-router-dom

# データフェッチング
npm install @tanstack/react-query
npm install axios

# フォーム管理
npm install react-hook-form
npm install @hookform/resolvers zod

# スタイリング
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p

# アイコン
npm install lucide-react

# 日付ライブラリ
npm install date-fns
```

### ステップ3：Tailwind CSS設定

`tailwind.config.js`：

```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#f0f9ff',
          100: '#e0f2fe',
          200: '#bae6fd',
          300: '#7dd3fc',
          400: '#38bdf8',
          500: '#0ea5e9',
          600: '#0284c7',
          700: '#0369a1',
          800: '#075985',
          900: '#0c4a6e',
        },
      },
    },
  },
  plugins: [],
}
```

`src/index.css`：

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  body {
    @apply bg-gray-50 text-gray-900;
  }
}

@layer components {
  .btn {
    @apply px-4 py-2 rounded-lg font-medium transition-colors;
  }

  .btn-primary {
    @apply bg-primary-600 text-white hover:bg-primary-700;
  }

  .btn-secondary {
    @apply bg-gray-200 text-gray-800 hover:bg-gray-300;
  }

  .btn-danger {
    @apply bg-red-600 text-white hover:bg-red-700;
  }

  .input {
    @apply w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500;
  }

  .card {
    @apply bg-white rounded-lg shadow-md p-6;
  }
}
```

### ステップ4：環境変数設定

`.env.local`：

```env
VITE_API_URL=http://localhost:3001
```

`.env.production`：

```env
VITE_API_URL=https://your-backend-api.com
```

---

## プロジェクト構造

### ディレクトリ作成

```bash
mkdir -p src/{pages,components,hooks,services,types,utils,contexts}
```

### 最終的な構造

```
frontend/
├── src/
│   ├── pages/
│   │   ├── Login.tsx
│   │   ├── Register.tsx
│   │   ├── Dashboard.tsx
│   │   └── Tasks.tsx
│   ├── components/
│   │   ├── layout/
│   │   │   ├── Header.tsx
│   │   │   └── Layout.tsx
│   │   ├── tasks/
│   │   │   ├── TaskCard.tsx
│   │   │   ├── TaskForm.tsx
│   │   │   └── TaskList.tsx
│   │   └── common/
│   │       ├── Button.tsx
│   │       ├── Input.tsx
│   │       └── Loading.tsx
│   ├── hooks/
│   │   ├── useAuth.ts
│   │   └── useTasks.ts
│   ├── services/
│   │   └── api.ts
│   ├── types/
│   │   └── index.ts
│   ├── utils/
│   │   └── date.ts
│   ├── contexts/
│   │   └── AuthContext.tsx
│   ├── App.tsx
│   ├── main.tsx
│   └── index.css
├── .env.local
├── .env.production
├── package.json
├── tsconfig.json
├── vite.config.ts
└── tailwind.config.js
```

---

## 型定義

### `src/types/index.ts`

```typescript
export interface User {
  id: number
  email: string
  name: string
  createdAt: string
}

export interface AuthResponse {
  user: User
  token: string
}

export interface LoginInput {
  email: string
  password: string
}

export interface RegisterInput {
  email: string
  password: string
  name: string
}

export enum Priority {
  LOW = 'LOW',
  MEDIUM = 'MEDIUM',
  HIGH = 'HIGH',
}

export interface Task {
  id: number
  title: string
  description?: string
  completed: boolean
  priority: Priority
  dueDate?: string
  createdAt: string
  updatedAt: string
  user?: User
}

export interface TaskInput {
  title: string
  description?: string
  priority?: Priority
  dueDate?: string
}

export interface TaskUpdateInput {
  title?: string
  description?: string
  completed?: boolean
  priority?: Priority
  dueDate?: string
}

export interface TaskStats {
  total: number
  completed: number
  pending: number
  highPriority: number
  overdue: number
  completionRate: number
}

export interface TaskQuery {
  completed?: boolean
  priority?: Priority
  sort?: 'createdAt' | 'dueDate' | 'priority'
  order?: 'asc' | 'desc'
}
```

---

## API通信設定

### `src/services/api.ts`

```typescript
import axios from 'axios'
import {
  AuthResponse,
  LoginInput,
  RegisterInput,
  Task,
  TaskInput,
  TaskUpdateInput,
  TaskStats,
  TaskQuery,
  User,
} from '../types'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:3001'

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// リクエストインターセプター（トークンを自動付与）
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// レスポンスインターセプター（エラーハンドリング）
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // 認証エラー時、トークンを削除してログインページへ
      localStorage.removeItem('token')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

// 認証API
export const authAPI = {
  register: async (data: RegisterInput): Promise<AuthResponse> => {
    const response = await api.post<AuthResponse>('/api/auth/register', data)
    return response.data
  },

  login: async (data: LoginInput): Promise<AuthResponse> => {
    const response = await api.post<AuthResponse>('/api/auth/login', data)
    return response.data
  },

  getMe: async (): Promise<{ user: User }> => {
    const response = await api.get<{ user: User }>('/api/auth/me')
    return response.data
  },

  logout: async (): Promise<void> => {
    await api.post('/api/auth/logout')
  },
}

// タスクAPI
export const taskAPI = {
  getTasks: async (query?: TaskQuery): Promise<{ tasks: Task[]; total: number }> => {
    const response = await api.get<{ tasks: Task[]; total: number }>('/api/tasks', {
      params: query,
    })
    return response.data
  },

  getTaskById: async (id: number): Promise<{ task: Task }> => {
    const response = await api.get<{ task: Task }>(`/api/tasks/${id}`)
    return response.data
  },

  createTask: async (data: TaskInput): Promise<{ task: Task }> => {
    const response = await api.post<{ task: Task }>('/api/tasks', data)
    return response.data
  },

  updateTask: async (id: number, data: TaskUpdateInput): Promise<{ task: Task }> => {
    const response = await api.put<{ task: Task }>(`/api/tasks/${id}`, data)
    return response.data
  },

  deleteTask: async (id: number): Promise<{ message: string }> => {
    const response = await api.delete<{ message: string }>(`/api/tasks/${id}`)
    return response.data
  },

  getStats: async (): Promise<{ stats: TaskStats }> => {
    const response = await api.get<{ stats: TaskStats }>('/api/tasks/stats')
    return response.data
  },
}
```

---

## 認証システム実装

### `src/contexts/AuthContext.tsx`

```typescript
import React, { createContext, useState, useEffect, ReactNode } from 'react'
import { User, LoginInput, RegisterInput } from '../types'
import { authAPI } from '../services/api'

interface AuthContextType {
  user: User | null
  isLoading: boolean
  login: (data: LoginInput) => Promise<void>
  register: (data: RegisterInput) => Promise<void>
  logout: () => void
}

export const AuthContext = createContext<AuthContextType | undefined>(undefined)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  // 初回マウント時、トークンがあればユーザー情報取得
  useEffect(() => {
    const token = localStorage.getItem('token')
    if (token) {
      authAPI
        .getMe()
        .then((data) => setUser(data.user))
        .catch(() => {
          localStorage.removeItem('token')
        })
        .finally(() => setIsLoading(false))
    } else {
      setIsLoading(false)
    }
  }, [])

  const login = async (data: LoginInput) => {
    const response = await authAPI.login(data)
    localStorage.setItem('token', response.token)
    setUser(response.user)
  }

  const register = async (data: RegisterInput) => {
    const response = await authAPI.register(data)
    localStorage.setItem('token', response.token)
    setUser(response.user)
  }

  const logout = () => {
    authAPI.logout().catch(() => {}) // エラーは無視
    localStorage.removeItem('token')
    setUser(null)
    window.location.href = '/login'
  }

  return (
    <AuthContext.Provider value={{ user, isLoading, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  )
}
```

### `src/hooks/useAuth.ts`

```typescript
import { useContext } from 'react'
import { AuthContext } from '../contexts/AuthContext'

export function useAuth() {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider')
  }
  return context
}
```

### `src/pages/Login.tsx`

```typescript
import React, { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { useAuth } from '../hooks/useAuth'
import { LoginInput } from '../types'

export default function Login() {
  const navigate = useNavigate()
  const { login } = useAuth()
  const [formData, setFormData] = useState<LoginInput>({
    email: '',
    password: '',
  })
  const [error, setError] = useState('')
  const [isLoading, setIsLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setIsLoading(true)

    try {
      await login(formData)
      navigate('/dashboard')
    } catch (err: any) {
      setError(err.response?.data?.error || 'ログインに失敗しました')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8">
        <div>
          <h2 className="mt-6 text-center text-3xl font-extrabold text-gray-900">
            ログイン
          </h2>
          <p className="mt-2 text-center text-sm text-gray-600">
            アカウントをお持ちでない方は{' '}
            <Link to="/register" className="font-medium text-primary-600 hover:text-primary-500">
              新規登録
            </Link>
          </p>
        </div>

        <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
          {error && (
            <div className="rounded-md bg-red-50 p-4">
              <p className="text-sm text-red-800">{error}</p>
            </div>
          )}

          <div className="rounded-md shadow-sm -space-y-px">
            <div>
              <label htmlFor="email" className="sr-only">
                メールアドレス
              </label>
              <input
                id="email"
                name="email"
                type="email"
                required
                className="appearance-none rounded-none relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 rounded-t-md focus:outline-none focus:ring-primary-500 focus:border-primary-500 focus:z-10 sm:text-sm"
                placeholder="メールアドレス"
                value={formData.email}
                onChange={(e) => setFormData({ ...formData, email: e.target.value })}
              />
            </div>
            <div>
              <label htmlFor="password" className="sr-only">
                パスワード
              </label>
              <input
                id="password"
                name="password"
                type="password"
                required
                className="appearance-none rounded-none relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 rounded-b-md focus:outline-none focus:ring-primary-500 focus:border-primary-500 focus:z-10 sm:text-sm"
                placeholder="パスワード"
                value={formData.password}
                onChange={(e) => setFormData({ ...formData, password: e.target.value })}
              />
            </div>
          </div>

          <div>
            <button
              type="submit"
              disabled={isLoading}
              className="group relative w-full flex justify-center py-2 px-4 border border-transparent text-sm font-medium rounded-md text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 disabled:opacity-50"
            >
              {isLoading ? 'ログイン中...' : 'ログイン'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
```

### `src/pages/Register.tsx`

```typescript
import React, { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { useAuth } from '../hooks/useAuth'
import { RegisterInput } from '../types'

export default function Register() {
  const navigate = useNavigate()
  const { register } = useAuth()
  const [formData, setFormData] = useState<RegisterInput>({
    email: '',
    password: '',
    name: '',
  })
  const [error, setError] = useState('')
  const [isLoading, setIsLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setIsLoading(true)

    try {
      await register(formData)
      navigate('/dashboard')
    } catch (err: any) {
      setError(err.response?.data?.error || '登録に失敗しました')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8">
        <div>
          <h2 className="mt-6 text-center text-3xl font-extrabold text-gray-900">
            新規登録
          </h2>
          <p className="mt-2 text-center text-sm text-gray-600">
            アカウントをお持ちの方は{' '}
            <Link to="/login" className="font-medium text-primary-600 hover:text-primary-500">
              ログイン
            </Link>
          </p>
        </div>

        <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
          {error && (
            <div className="rounded-md bg-red-50 p-4">
              <p className="text-sm text-red-800">{error}</p>
            </div>
          )}

          <div className="rounded-md shadow-sm space-y-4">
            <div>
              <label htmlFor="name" className="block text-sm font-medium text-gray-700">
                名前
              </label>
              <input
                id="name"
                name="name"
                type="text"
                required
                className="input"
                placeholder="山田太郎"
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              />
            </div>
            <div>
              <label htmlFor="email" className="block text-sm font-medium text-gray-700">
                メールアドレス
              </label>
              <input
                id="email"
                name="email"
                type="email"
                required
                className="input"
                placeholder="email@example.com"
                value={formData.email}
                onChange={(e) => setFormData({ ...formData, email: e.target.value })}
              />
            </div>
            <div>
              <label htmlFor="password" className="block text-sm font-medium text-gray-700">
                パスワード（8文字以上、英数字含む）
              </label>
              <input
                id="password"
                name="password"
                type="password"
                required
                className="input"
                placeholder="********"
                value={formData.password}
                onChange={(e) => setFormData({ ...formData, password: e.target.value })}
              />
            </div>
          </div>

          <div>
            <button type="submit" disabled={isLoading} className="btn btn-primary w-full">
              {isLoading ? '登録中...' : '登録'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
```

---

## タスク管理機能実装

### `src/hooks/useTasks.ts`

```typescript
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { taskAPI } from '../services/api'
import { TaskInput, TaskUpdateInput, TaskQuery } from '../types'

export function useTasks(query?: TaskQuery) {
  return useQuery({
    queryKey: ['tasks', query],
    queryFn: () => taskAPI.getTasks(query),
  })
}

export function useTask(id: number) {
  return useQuery({
    queryKey: ['tasks', id],
    queryFn: () => taskAPI.getTaskById(id),
    enabled: !!id,
  })
}

export function useCreateTask() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (data: TaskInput) => taskAPI.createTask(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tasks'] })
    },
  })
}

export function useUpdateTask() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ id, data }: { id: number; data: TaskUpdateInput }) =>
      taskAPI.updateTask(id, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tasks'] })
    },
  })
}

export function useDeleteTask() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (id: number) => taskAPI.deleteTask(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tasks'] })
    },
  })
}

export function useTaskStats() {
  return useQuery({
    queryKey: ['task-stats'],
    queryFn: () => taskAPI.getStats(),
  })
}
```

### `src/components/tasks/TaskCard.tsx`

```typescript
import React from 'react'
import { Task, Priority } from '../../types'
import { format } from 'date-fns'
import { ja } from 'date-fns/locale'
import { CheckCircle2, Circle, Trash2, Edit } from 'lucide-react'

interface TaskCardProps {
  task: Task
  onToggle: (id: number, completed: boolean) => void
  onDelete: (id: number) => void
  onEdit: (task: Task) => void
}

const priorityColors = {
  [Priority.LOW]: 'bg-green-100 text-green-800',
  [Priority.MEDIUM]: 'bg-yellow-100 text-yellow-800',
  [Priority.HIGH]: 'bg-red-100 text-red-800',
}

const priorityLabels = {
  [Priority.LOW]: '低',
  [Priority.MEDIUM]: '中',
  [Priority.HIGH]: '高',
}

export default function TaskCard({ task, onToggle, onDelete, onEdit }: TaskCardProps) {
  return (
    <div className="card hover:shadow-lg transition-shadow">
      <div className="flex items-start gap-4">
        <button
          onClick={() => onToggle(task.id, !task.completed)}
          className="flex-shrink-0 mt-1"
        >
          {task.completed ? (
            <CheckCircle2 className="w-6 h-6 text-green-600" />
          ) : (
            <Circle className="w-6 h-6 text-gray-400" />
          )}
        </button>

        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-2">
            <h3
              className={`text-lg font-semibold ${
                task.completed ? 'line-through text-gray-500' : 'text-gray-900'
              }`}
            >
              {task.title}
            </h3>
            <span
              className={`flex-shrink-0 px-2 py-1 text-xs font-medium rounded ${
                priorityColors[task.priority]
              }`}
            >
              {priorityLabels[task.priority]}
            </span>
          </div>

          {task.description && (
            <p className="mt-2 text-sm text-gray-600">{task.description}</p>
          )}

          <div className="mt-3 flex items-center justify-between text-sm text-gray-500">
            <span>
              {task.dueDate &&
                `期限: ${format(new Date(task.dueDate), 'yyyy年MM月dd日', { locale: ja })}`}
            </span>
            <div className="flex gap-2">
              <button
                onClick={() => onEdit(task)}
                className="p-1 hover:bg-gray-100 rounded"
              >
                <Edit className="w-4 h-4" />
              </button>
              <button
                onClick={() => onDelete(task.id)}
                className="p-1 hover:bg-red-50 text-red-600 rounded"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
```

### `src/pages/Tasks.tsx`

```typescript
import React, { useState } from 'react'
import { useTasks, useUpdateTask, useDeleteTask, useCreateTask } from '../hooks/useTasks'
import TaskCard from '../components/tasks/TaskCard'
import { Plus } from 'lucide-react'
import { Task, TaskInput, Priority } from '../types'

export default function Tasks() {
  const [showForm, setShowForm] = useState(false)
  const [editingTask, setEditingTask] = useState<Task | null>(null)
  const [formData, setFormData] = useState<TaskInput>({
    title: '',
    description: '',
    priority: Priority.MEDIUM,
    dueDate: '',
  })

  const { data, isLoading } = useTasks()
  const createTask = useCreateTask()
  const updateTask = useUpdateTask()
  const deleteTask = useDeleteTask()

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (editingTask) {
      await updateTask.mutateAsync({
        id: editingTask.id,
        data: formData,
      })
    } else {
      await createTask.mutateAsync(formData)
    }

    setShowForm(false)
    setEditingTask(null)
    setFormData({
      title: '',
      description: '',
      priority: Priority.MEDIUM,
      dueDate: '',
    })
  }

  const handleEdit = (task: Task) => {
    setEditingTask(task)
    setFormData({
      title: task.title,
      description: task.description || '',
      priority: task.priority,
      dueDate: task.dueDate ? task.dueDate.split('T')[0] : '',
    })
    setShowForm(true)
  }

  const handleToggle = async (id: number, completed: boolean) => {
    await updateTask.mutateAsync({
      id,
      data: { completed },
    })
  }

  const handleDelete = async (id: number) => {
    if (confirm('このタスクを削除しますか？')) {
      await deleteTask.mutateAsync(id)
    }
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    )
  }

  return (
    <div className="max-w-4xl mx-auto">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold text-gray-900">タスク一覧</h1>
        <button
          onClick={() => setShowForm(true)}
          className="btn btn-primary flex items-center gap-2"
        >
          <Plus className="w-5 h-5" />
          新規作成
        </button>
      </div>

      {showForm && (
        <div className="card mb-6">
          <h2 className="text-xl font-semibold mb-4">
            {editingTask ? 'タスク編集' : '新規タスク'}
          </h2>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                タイトル
              </label>
              <input
                type="text"
                required
                className="input"
                value={formData.title}
                onChange={(e) => setFormData({ ...formData, title: e.target.value })}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                説明
              </label>
              <textarea
                className="input"
                rows={3}
                value={formData.description}
                onChange={(e) => setFormData({ ...formData, description: e.target.value })}
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  優先度
                </label>
                <select
                  className="input"
                  value={formData.priority}
                  onChange={(e) =>
                    setFormData({ ...formData, priority: e.target.value as Priority })
                  }
                >
                  <option value={Priority.LOW}>低</option>
                  <option value={Priority.MEDIUM}>中</option>
                  <option value={Priority.HIGH}>高</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  期限日
                </label>
                <input
                  type="date"
                  className="input"
                  value={formData.dueDate}
                  onChange={(e) => setFormData({ ...formData, dueDate: e.target.value })}
                />
              </div>
            </div>

            <div className="flex gap-2">
              <button type="submit" className="btn btn-primary">
                {editingTask ? '更新' : '作成'}
              </button>
              <button
                type="button"
                onClick={() => {
                  setShowForm(false)
                  setEditingTask(null)
                }}
                className="btn btn-secondary"
              >
                キャンセル
              </button>
            </div>
          </form>
        </div>
      )}

      <div className="space-y-4">
        {data?.tasks.map((task) => (
          <TaskCard
            key={task.id}
            task={task}
            onToggle={handleToggle}
            onDelete={handleDelete}
            onEdit={handleEdit}
          />
        ))}

        {data?.tasks.length === 0 && (
          <div className="text-center py-12 text-gray-500">
            タスクがありません。新規作成してください。
          </div>
        )}
      </div>
    </div>
  )
}
```

---

## ダッシュボード実装

### `src/pages/Dashboard.tsx`

```typescript
import React from 'react'
import { useTaskStats, useTasks } from '../hooks/useTasks'
import { CheckCircle2, Circle, AlertCircle, TrendingUp } from 'lucide-react'

export default function Dashboard() {
  const { data: statsData, isLoading: statsLoading } = useTaskStats()
  const { data: tasksData, isLoading: tasksLoading } = useTasks({
    sort: 'dueDate',
    order: 'asc',
  })

  if (statsLoading || tasksLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    )
  }

  const stats = statsData?.stats

  return (
    <div className="max-w-7xl mx-auto">
      <h1 className="text-3xl font-bold text-gray-900 mb-8">ダッシュボード</h1>

      {/* 統計カード */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">総タスク数</p>
              <p className="text-3xl font-bold text-gray-900">{stats?.total || 0}</p>
            </div>
            <Circle className="w-12 h-12 text-gray-400" />
          </div>
        </div>

        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">完了</p>
              <p className="text-3xl font-bold text-green-600">{stats?.completed || 0}</p>
            </div>
            <CheckCircle2 className="w-12 h-12 text-green-600" />
          </div>
        </div>

        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">未完了</p>
              <p className="text-3xl font-bold text-yellow-600">{stats?.pending || 0}</p>
            </div>
            <AlertCircle className="w-12 h-12 text-yellow-600" />
          </div>
        </div>

        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">完了率</p>
              <p className="text-3xl font-bold text-primary-600">
                {stats?.completionRate || 0}%
              </p>
            </div>
            <TrendingUp className="w-12 h-12 text-primary-600" />
          </div>
        </div>
      </div>

      {/* 最近のタスク */}
      <div className="card">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">期限が近いタスク</h2>
        <div className="space-y-3">
          {tasksData?.tasks.slice(0, 5).map((task) => (
            <div key={task.id} className="flex items-center justify-between py-2 border-b">
              <div className="flex items-center gap-3">
                {task.completed ? (
                  <CheckCircle2 className="w-5 h-5 text-green-600" />
                ) : (
                  <Circle className="w-5 h-5 text-gray-400" />
                )}
                <span className={task.completed ? 'line-through text-gray-500' : ''}>
                  {task.title}
                </span>
              </div>
              <span className="text-sm text-gray-500">
                {task.dueDate &&
                  new Date(task.dueDate).toLocaleDateString('ja-JP')}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
```

---

## ルーティング設定

### `src/App.tsx`

```typescript
import React from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { AuthProvider } from './contexts/AuthContext'
import { useAuth } from './hooks/useAuth'
import Layout from './components/layout/Layout'
import Login from './pages/Login'
import Register from './pages/Register'
import Dashboard from './pages/Dashboard'
import Tasks from './pages/Tasks'

const queryClient = new QueryClient()

function PrivateRoute({ children }: { children: React.ReactNode }) {
  const { user, isLoading } = useAuth()

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    )
  }

  return user ? <Layout>{children}</Layout> : <Navigate to="/login" />
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <AuthProvider>
          <Routes>
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<Register />} />
            <Route
              path="/dashboard"
              element={
                <PrivateRoute>
                  <Dashboard />
                </PrivateRoute>
              }
            />
            <Route
              path="/tasks"
              element={
                <PrivateRoute>
                  <Tasks />
                </PrivateRoute>
              }
            />
            <Route path="/" element={<Navigate to="/dashboard" />} />
          </Routes>
        </AuthProvider>
      </BrowserRouter>
    </QueryClientProvider>
  )
}

export default App
```

### `src/components/layout/Layout.tsx`

```typescript
import React from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../../hooks/useAuth'
import { LayoutDashboard, ListTodo, LogOut } from 'lucide-react'

export default function Layout({ children }: { children: React.ReactNode }) {
  const { user, logout } = useAuth()
  const navigate = useNavigate()

  const handleLogout = () => {
    logout()
    navigate('/login')
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex">
              <div className="flex-shrink-0 flex items-center">
                <h1 className="text-xl font-bold text-primary-600">TaskApp</h1>
              </div>
              <div className="hidden sm:ml-6 sm:flex sm:space-x-8">
                <Link
                  to="/dashboard"
                  className="inline-flex items-center px-1 pt-1 border-b-2 border-transparent text-sm font-medium text-gray-500 hover:border-gray-300 hover:text-gray-700"
                >
                  <LayoutDashboard className="w-4 h-4 mr-2" />
                  ダッシュボード
                </Link>
                <Link
                  to="/tasks"
                  className="inline-flex items-center px-1 pt-1 border-b-2 border-transparent text-sm font-medium text-gray-500 hover:border-gray-300 hover:text-gray-700"
                >
                  <ListTodo className="w-4 h-4 mr-2" />
                  タスク
                </Link>
              </div>
            </div>
            <div className="flex items-center">
              <span className="text-sm text-gray-700 mr-4">{user?.name}</span>
              <button
                onClick={handleLogout}
                className="inline-flex items-center px-3 py-2 border border-transparent text-sm leading-4 font-medium rounded-md text-gray-500 hover:text-gray-700"
              >
                <LogOut className="w-4 h-4 mr-2" />
                ログアウト
              </button>
            </div>
          </div>
        </div>
      </nav>

      <main className="py-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">{children}</div>
      </main>
    </div>
  )
}
```

---

## テスト

### サーバー起動

```bash
# バックエンド起動（別ターミナル）
cd backend
npm run dev

# フロントエンド起動
cd frontend
npm run dev
```

### テスト手順

1. http://localhost:5173 にアクセス
2. ユーザー登録
3. ダッシュボード確認
4. タスク作成
5. タスク編集・削除
6. ログアウト

---

## トラブルシューティング

### ❌ 問題1：CORS エラー

```
Access to fetch at 'http://localhost:3001/api/...' has been blocked by CORS policy
```

**解決策:**
バックエンドの`.env`を確認：
```env
CORS_ORIGIN=http://localhost:5173
```

### ❌ 問題2：トークンが保存されない

**解決策:**
`src/services/api.ts`のインターセプターを確認。

### ❌ 問題3：日付フォーマットエラー

**解決策:**
```bash
npm install date-fns
```

---

## まとめ

### このガイドで学んだこと

- ✅ React + TypeScript プロジェクトのセットアップ
- ✅ React Router によるルーティング
- ✅ React Query によるデータフェッチング
- ✅ Context API による認証管理
- ✅ カスタムフックの作成
- ✅ Tailwind CSS によるスタイリング

### 次のステップ

**次のガイド:** [04-deployment-guide.md](./04-deployment-guide.md) - デプロイガイド

---

**前のガイド:** [02-backend-implementation.md](./02-backend-implementation.md)

**親ガイド:** [統合プロジェクト - README](../README.md)
