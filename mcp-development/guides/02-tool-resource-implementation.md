# 🛠️ Tool & Resource 実装ガイド

> **目的**: MCP Server の Tool と Resource の詳細な実装方法と実践的なパターンを習得する

## 📚 目次

1. [Tool 実装](#tool-実装)
2. [Resource 実装](#resource-実装)
3. [Prompt 実装](#prompt-実装)
4. [実践パターン](#実践パターン)
5. [セキュリティ](#セキュリティ)
6. [エラーハンドリング](#エラーハンドリング)

---

## Tool 実装

### Tool の基本構造

```typescript
{
  name: string              // ツール名（一意）
  description: string       // ツールの説明
  inputSchema: JSONSchema   // 引数スキーマ
}
```

### シンプルな Tool

**計算ツール**:
```typescript
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from '@modelcontextprotocol/sdk/types.js'

// ツール一覧
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: 'calculate',
        description: 'Perform basic arithmetic operations',
        inputSchema: {
          type: 'object',
          properties: {
            operation: {
              type: 'string',
              enum: ['add', 'subtract', 'multiply', 'divide'],
              description: 'Arithmetic operation to perform',
            },
            a: {
              type: 'number',
              description: 'First number',
            },
            b: {
              type: 'number',
              description: 'Second number',
            },
          },
          required: ['operation', 'a', 'b'],
        },
      },
    ],
  }
})

// ツール実行
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params

  if (name === 'calculate') {
    const { operation, a, b } = args as {
      operation: 'add' | 'subtract' | 'multiply' | 'divide'
      a: number
      b: number
    }

    let result: number

    switch (operation) {
      case 'add':
        result = a + b
        break
      case 'subtract':
        result = a - b
        break
      case 'multiply':
        result = a * b
        break
      case 'divide':
        if (b === 0) {
          throw new Error('Division by zero')
        }
        result = a / b
        break
    }

    return {
      content: [
        {
          type: 'text',
          text: `${a} ${operation} ${b} = ${result}`,
        },
      ],
    }
  }

  throw new Error(`Unknown tool: ${name}`)
})
```

### ファイル操作 Tool

**read_file**:
```typescript
import fs from 'fs/promises'
import path from 'path'

{
  name: 'read_file',
  description: 'Read contents of a file',
  inputSchema: {
    type: 'object',
    properties: {
      path: {
        type: 'string',
        description: 'Absolute path to the file',
      },
    },
    required: ['path'],
  },
}

// 実装
if (name === 'read_file') {
  const filePath = String(args?.path)

  // セキュリティ: パストラバーサル防止
  if (filePath.includes('..')) {
    throw new Error('Path traversal detected')
  }

  // ファイル存在確認
  try {
    await fs.access(filePath)
  } catch {
    throw new Error(`File not found: ${filePath}`)
  }

  // ファイル読み込み
  const content = await fs.readFile(filePath, 'utf-8')

  return {
    content: [
      {
        type: 'text',
        text: content,
      },
    ],
  }
}
```

**write_file**:
```typescript
{
  name: 'write_file',
  description: 'Write content to a file',
  inputSchema: {
    type: 'object',
    properties: {
      path: {
        type: 'string',
        description: 'Absolute path to the file',
      },
      content: {
        type: 'string',
        description: 'Content to write',
      },
    },
    required: ['path', 'content'],
  },
}

// 実装
if (name === 'write_file') {
  const filePath = String(args?.path)
  const content = String(args?.content)

  // セキュリティチェック
  if (filePath.includes('..')) {
    throw new Error('Path traversal detected')
  }

  // ディレクトリ作成
  const dir = path.dirname(filePath)
  await fs.mkdir(dir, { recursive: true })

  // ファイル書き込み
  await fs.writeFile(filePath, content, 'utf-8')

  return {
    content: [
      {
        type: 'text',
        text: `File written successfully: ${filePath}`,
      },
    ],
  }
}
```

### HTTP API 呼び出し Tool

**get_weather**:
```typescript
import axios from 'axios'

{
  name: 'get_weather',
  description: 'Get current weather for a city',
  inputSchema: {
    type: 'object',
    properties: {
      city: {
        type: 'string',
        description: 'City name (e.g., Tokyo, New York)',
      },
    },
    required: ['city'],
  },
}

// 実装
if (name === 'get_weather') {
  const city = String(args?.city)

  // OpenWeatherMap API（例）
  const apiKey = process.env.OPENWEATHER_API_KEY
  if (!apiKey) {
    throw new Error('OPENWEATHER_API_KEY not set')
  }

  const response = await axios.get(
    `https://api.openweathermap.org/data/2.5/weather`,
    {
      params: {
        q: city,
        appid: apiKey,
        units: 'metric',
      },
    }
  )

  const weather = response.data

  return {
    content: [
      {
        type: 'text',
        text: `Weather in ${city}:
Temperature: ${weather.main.temp}°C
Feels like: ${weather.main.feels_like}°C
Humidity: ${weather.main.humidity}%
Description: ${weather.weather[0].description}`,
      },
    ],
  }
}
```

### データベースクエリ Tool

**search_users**:
```typescript
import { Pool } from 'pg'

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
})

{
  name: 'search_users',
  description: 'Search users in the database',
  inputSchema: {
    type: 'object',
    properties: {
      query: {
        type: 'string',
        description: 'Search query',
      },
      limit: {
        type: 'number',
        description: 'Maximum number of results',
        default: 10,
      },
    },
    required: ['query'],
  },
}

// 実装
if (name === 'search_users') {
  const query = String(args?.query)
  const limit = Number(args?.limit ?? 10)

  // SQL クエリ（パラメータ化）
  const result = await pool.query(
    `SELECT id, name, email
     FROM users
     WHERE name ILIKE $1 OR email ILIKE $1
     LIMIT $2`,
    [`%${query}%`, limit]
  )

  const users = result.rows.map(
    (row) => `${row.id}: ${row.name} (${row.email})`
  )

  return {
    content: [
      {
        type: 'text',
        text: `Found ${users.length} users:\n${users.join('\n')}`,
      },
    ],
  }
}
```

---

## Resource 実装

### Resource の基本構造

```typescript
{
  uri: string           // リソース URI（一意）
  name: string          // リソース名
  description?: string  // リソースの説明
  mimeType?: string     // MIME タイプ
}
```

### ファイルリソース

```typescript
import {
  ListResourcesRequestSchema,
  ReadResourceRequestSchema,
} from '@modelcontextprotocol/sdk/types.js'
import fs from 'fs/promises'
import path from 'path'

// Server 設定
const server = new Server(
  { name: 'file-server', version: '1.0.0' },
  {
    capabilities: {
      resources: {},  // リソース機能を有効化
    },
  }
)

// リソース一覧
server.setRequestHandler(ListResourcesRequestSchema, async () => {
  const dataDir = './data'
  const files = await fs.readdir(dataDir)

  const resources = files.map((file) => ({
    uri: `file:///${path.join(dataDir, file)}`,
    name: file,
    description: `File: ${file}`,
    mimeType: file.endsWith('.json')
      ? 'application/json'
      : 'text/plain',
  }))

  return { resources }
})

// リソース読み取り
server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
  const { uri } = request.params

  // URI パース
  if (!uri.startsWith('file:///')) {
    throw new Error('Invalid URI scheme')
  }

  const filePath = uri.replace('file:///', '')

  // セキュリティチェック
  if (filePath.includes('..')) {
    throw new Error('Path traversal detected')
  }

  // ファイル読み込み
  const content = await fs.readFile(filePath, 'utf-8')
  const mimeType = filePath.endsWith('.json')
    ? 'application/json'
    : 'text/plain'

  return {
    contents: [
      {
        uri,
        mimeType,
        text: content,
      },
    ],
  }
})
```

### 動的リソース

**API レスポンスをリソースとして公開**:
```typescript
// リソース一覧
server.setRequestHandler(ListResourcesRequestSchema, async () => {
  return {
    resources: [
      {
        uri: 'api://weather/tokyo',
        name: 'Tokyo Weather',
        description: 'Current weather in Tokyo',
        mimeType: 'application/json',
      },
      {
        uri: 'api://weather/osaka',
        name: 'Osaka Weather',
        description: 'Current weather in Osaka',
        mimeType: 'application/json',
      },
    ],
  }
})

// リソース読み取り
server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
  const { uri } = request.params

  if (uri.startsWith('api://weather/')) {
    const city = uri.replace('api://weather/', '')

    // API 呼び出し
    const weather = await fetchWeather(city)

    return {
      contents: [
        {
          uri,
          mimeType: 'application/json',
          text: JSON.stringify(weather, null, 2),
        },
      ],
    }
  }

  throw new Error(`Unknown resource: ${uri}`)
})
```

---

## Prompt 実装

### Prompt の基本構造

```typescript
{
  name: string              // プロンプト名
  description?: string      // プロンプトの説明
  arguments?: Array<{       // 引数定義
    name: string
    description?: string
    required?: boolean
  }>
}
```

### プロンプトテンプレート

```typescript
import {
  ListPromptsRequestSchema,
  GetPromptRequestSchema,
} from '@modelcontextprotocol/sdk/types.js'

// Server 設定
const server = new Server(
  { name: 'prompt-server', version: '1.0.0' },
  {
    capabilities: {
      prompts: {},  // プロンプト機能を有効化
    },
  }
)

// プロンプト一覧
server.setRequestHandler(ListPromptsRequestSchema, async () => {
  return {
    prompts: [
      {
        name: 'code-review',
        description: 'Review code for best practices',
        arguments: [
          {
            name: 'code',
            description: 'Code to review',
            required: true,
          },
          {
            name: 'language',
            description: 'Programming language',
            required: false,
          },
        ],
      },
      {
        name: 'bug-fix',
        description: 'Suggest bug fixes',
        arguments: [
          {
            name: 'error',
            description: 'Error message or description',
            required: true,
          },
        ],
      },
    ],
  }
})

// プロンプト取得
server.setRequestHandler(GetPromptRequestSchema, async (request) => {
  const { name, arguments: args } = request.params

  if (name === 'code-review') {
    const code = String(args?.code ?? '')
    const language = String(args?.language ?? 'unknown')

    return {
      messages: [
        {
          role: 'user',
          content: {
            type: 'text',
            text: `Please review the following ${language} code for best practices, potential bugs, and improvements:

\`\`\`${language}
${code}
\`\`\`

Focus on:
- Code quality and readability
- Performance issues
- Security vulnerabilities
- Best practices`,
          },
        },
      ],
    }
  }

  if (name === 'bug-fix') {
    const error = String(args?.error ?? '')

    return {
      messages: [
        {
          role: 'user',
          content: {
            type: 'text',
            text: `I'm encountering the following error:

${error}

Please help me:
1. Understand what's causing this error
2. Provide a step-by-step solution
3. Suggest how to prevent this in the future`,
          },
        },
      ],
    }
  }

  throw new Error(`Unknown prompt: ${name}`)
})
```

---

## 実践パターン

### Tool のモジュール化

**src/tools/calculator.ts**:
```typescript
import { Tool } from '@modelcontextprotocol/sdk/types.js'

export const calculatorTool: Tool = {
  name: 'calculate',
  description: 'Perform arithmetic operations',
  inputSchema: {
    type: 'object',
    properties: {
      operation: {
        type: 'string',
        enum: ['add', 'subtract', 'multiply', 'divide'],
      },
      a: { type: 'number' },
      b: { type: 'number' },
    },
    required: ['operation', 'a', 'b'],
  },
}

export async function executeCalculator(args: any) {
  const { operation, a, b } = args

  let result: number

  switch (operation) {
    case 'add':
      result = a + b
      break
    case 'subtract':
      result = a - b
      break
    case 'multiply':
      result = a * b
      break
    case 'divide':
      if (b === 0) throw new Error('Division by zero')
      result = a / b
      break
    default:
      throw new Error('Invalid operation')
  }

  return {
    content: [{ type: 'text', text: `Result: ${result}` }],
  }
}
```

**src/index.ts**:
```typescript
import { calculatorTool, executeCalculator } from './tools/calculator.js'
import { weatherTool, executeWeather } from './tools/weather.js'

// ツール一覧
const tools = [calculatorTool, weatherTool]

server.setRequestHandler(ListToolsRequestSchema, async () => {
  return { tools }
})

// ツール実行
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params

  switch (name) {
    case 'calculate':
      return await executeCalculator(args)
    case 'weather':
      return await executeWeather(args)
    default:
      throw new Error(`Unknown tool: ${name}`)
  }
})
```

### 環境変数管理

```typescript
import dotenv from 'dotenv'

dotenv.config()

// 必須環境変数チェック
const requiredEnvVars = ['OPENWEATHER_API_KEY', 'DATABASE_URL']

for (const envVar of requiredEnvVars) {
  if (!process.env[envVar]) {
    console.error(`Missing required environment variable: ${envVar}`)
    process.exit(1)
  }
}

// 環境変数アクセス
const config = {
  openWeatherApiKey: process.env.OPENWEATHER_API_KEY!,
  databaseUrl: process.env.DATABASE_URL!,
  port: parseInt(process.env.PORT ?? '3000'),
}
```

---

## セキュリティ

### 入力バリデーション

```typescript
function validateFilePath(filePath: string): void {
  // パストラバーサル防止
  if (filePath.includes('..')) {
    throw new Error('Path traversal detected')
  }

  // 絶対パスチェック
  if (!path.isAbsolute(filePath)) {
    throw new Error('Absolute path required')
  }

  // 許可されたディレクトリ内かチェック
  const allowedDir = '/path/to/allowed/directory'
  if (!filePath.startsWith(allowedDir)) {
    throw new Error('Access denied')
  }
}
```

### SQLインジェクション防止

```typescript
// ❌ 危険: SQL インジェクション
const query = `SELECT * FROM users WHERE name = '${userName}'`

// ✅ 安全: パラメータ化クエリ
const result = await pool.query(
  'SELECT * FROM users WHERE name = $1',
  [userName]
)
```

### API キー管理

```typescript
// ❌ ハードコード（絶対NG）
const apiKey = 'sk-xxxxxxxxxxxxx'

// ✅ 環境変数
const apiKey = process.env.OPENWEATHER_API_KEY

// さらに安全: キー検証
if (!apiKey || !apiKey.startsWith('sk-')) {
  throw new Error('Invalid API key')
}
```

---

## エラーハンドリング

### 構造化エラー

```typescript
class ToolError extends Error {
  constructor(
    message: string,
    public code: string,
    public details?: any
  ) {
    super(message)
    this.name = 'ToolError'
  }
}

// 使用例
throw new ToolError('File not found', 'FILE_NOT_FOUND', { path: filePath })
```

### エラーレスポンス

```typescript
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  try {
    // ツール実行
    return await executeTool(request.params.name, request.params.arguments)
  } catch (error) {
    console.error('Tool execution error:', error)

    return {
      content: [
        {
          type: 'text',
          text: `Error: ${error instanceof Error ? error.message : String(error)}`,
        },
      ],
      isError: true,
    }
  }
})
```

---

## まとめ

### Tool 実装チェックリスト

- [ ] 明確な `inputSchema` 定義
- [ ] 入力バリデーション
- [ ] エラーハンドリング
- [ ] セキュリティチェック（パストラバーサル、SQL インジェクションなど）
- [ ] 環境変数で機密情報管理

### Resource 実装チェックリスト

- [ ] 一意な URI
- [ ] 適切な MIME タイプ
- [ ] セキュリティチェック
- [ ] エラーハンドリング

### Prompt 実装チェックリスト

- [ ] 明確な引数定義
- [ ] 再利用可能なテンプレート
- [ ] わかりやすい説明

---

## 次のステップ

1. **03-claude-desktop-integration.md**: Claude Desktop 統合ガイド

---

*強力な Tool と Resource で Claude の能力を拡張しましょう。*
