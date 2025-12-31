# 🔌 MCP Server 基礎ガイド

> **目的**: Model Context Protocol（MCP）の基礎と、基本的な MCP Server の開発方法を習得する

## 📚 目次

1. [MCP とは](#mcp-とは)
2. [アーキテクチャ](#アーキテクチャ)
3. [開発環境セットアップ](#開発環境セットアップ)
4. [基本的な Server 実装](#基本的な-server-実装)
5. [Transport の理解](#transport-の理解)
6. [デバッグとログ](#デバッグとログ)

---

## MCP とは

### Model Context Protocol の概要

**MCP（Model Context Protocol）** は、AI モデルと外部ツールを接続するための標準プロトコルです。

**主な目的**:
- AI モデルに外部機能を提供
- ローカルリソースへのアクセス
- 外部 API との統合
- カスタムツールの提供

### MCP の 3 つの主要機能

#### 1. Tools（ツール）

AI が実行できる関数を提供します。

```typescript
// 例: 計算ツール
{
  name: 'calculate',
  description: 'Perform arithmetic operations',
  inputSchema: {
    type: 'object',
    properties: {
      operation: { type: 'string', enum: ['add', 'subtract'] },
      a: { type: 'number' },
      b: { type: 'number' }
    }
  }
}
```

**使用例（Claude Desktop）**:
```
User: Calculate 5 + 3
Claude: [calls calculate tool with {operation: 'add', a: 5, b: 3}]
Result: 8
```

#### 2. Resources（リソース）

AI がアクセスできるデータを公開します。

```typescript
// 例: ファイルリソース
{
  uri: 'file:///data/config.json',
  name: 'Configuration',
  mimeType: 'application/json',
  description: 'Application configuration'
}
```

**使用例**:
```
User: What's in the config?
Claude: [reads resource file:///data/config.json]
The configuration contains...
```

#### 3. Prompts（プロンプト）

再利用可能なプロンプトテンプレートを提供します。

```typescript
// 例: コードレビュープロンプト
{
  name: 'code-review',
  description: 'Review code for best practices',
  arguments: [
    { name: 'code', description: 'Code to review' }
  ]
}
```

---

## アーキテクチャ

### 全体構成

```
┌─────────────────┐
│  Claude Desktop │  ← AI クライアント
└────────┬────────┘
         │ MCP (stdio/http)
         ↓
┌─────────────────┐
│   MCP Server    │  ← あなたが開発するサーバー
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  External APIs  │  ← 外部リソース
│  Local Files    │
│  Databases      │
└─────────────────┘
```

### 通信フロー

```
1. Claude Desktop → MCP Server: "List available tools"
2. MCP Server → Claude Desktop: [calculate, weather, ...]

3. User: "Calculate 5 + 3"
4. Claude Desktop → MCP Server: Call tool "calculate"
5. MCP Server: Execute calculation
6. MCP Server → Claude Desktop: Result: 8
7. Claude Desktop → User: "The result is 8"
```

### Transport 層

MCP は 2 つの Transport をサポート:

**1. stdio（標準入出力）**:
- Claude Desktop のデフォルト
- プロセス間通信
- ローカル実行

**2. HTTP/SSE（Server-Sent Events）**:
- リモートサーバー
- Web 統合
- スケーラブル

---

## 開発環境セットアップ

### Node.js（TypeScript）

```bash
# プロジェクト作成
mkdir my-mcp-server
cd my-mcp-server

# package.json 初期化
npm init -y

# 依存関係インストール
npm install @modelcontextprotocol/sdk
npm install -D typescript @types/node ts-node

# TypeScript 初期化
npx tsc --init
```

**package.json**:
```json
{
  "name": "my-mcp-server",
  "version": "1.0.0",
  "type": "module",
  "main": "dist/index.js",
  "scripts": {
    "build": "tsc",
    "dev": "ts-node src/index.ts",
    "start": "node dist/index.js"
  },
  "dependencies": {
    "@modelcontextprotocol/sdk": "^0.5.0"
  },
  "devDependencies": {
    "@types/node": "^20.0.0",
    "typescript": "^5.3.0",
    "ts-node": "^10.9.0"
  }
}
```

**tsconfig.json**:
```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "Node16",
    "moduleResolution": "Node16",
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
```

### Python

```bash
# プロジェクト作成
mkdir my-mcp-server
cd my-mcp-server

# 仮想環境作成
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# MCP SDK インストール
pip install mcp
```

**pyproject.toml**:
```toml
[project]
name = "my-mcp-server"
version = "1.0.0"
requires-python = ">=3.10"
dependencies = [
    "mcp>=0.1.0",
]

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"
```

---

## 基本的な Server 実装

### 最小限の MCP Server（Node.js）

**src/index.ts**:
```typescript
#!/usr/bin/env node

import { Server } from '@modelcontextprotocol/sdk/server/index.js'
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js'
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from '@modelcontextprotocol/sdk/types.js'

// Server インスタンス作成
const server = new Server(
  {
    name: 'my-mcp-server',
    version: '1.0.0',
  },
  {
    capabilities: {
      tools: {},  // ツール機能を有効化
    },
  }
)

// ツール一覧を返す
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: 'hello',
        description: 'Say hello to someone',
        inputSchema: {
          type: 'object',
          properties: {
            name: {
              type: 'string',
              description: 'Name of the person to greet',
            },
          },
          required: ['name'],
        },
      },
    ],
  }
})

// ツール実行
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params

  if (name === 'hello') {
    const personName = String(args?.name ?? 'World')

    return {
      content: [
        {
          type: 'text',
          text: `Hello, ${personName}!`,
        },
      ],
    }
  }

  throw new Error(`Unknown tool: ${name}`)
})

// Server 起動
async function main() {
  const transport = new StdioServerTransport()
  await server.connect(transport)
  console.error('MCP Server running on stdio')
}

main().catch((error) => {
  console.error('Server error:', error)
  process.exit(1)
})
```

### 最小限の MCP Server（Python）

**server.py**:
```python
#!/usr/bin/env python3

import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Server インスタンス作成
app = Server("my-mcp-server")

# ツール一覧を返す
@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="hello",
            description="Say hello to someone",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the person to greet"
                    }
                },
                "required": ["name"]
            }
        )
    ]

# ツール実行
@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "hello":
        person_name = arguments.get("name", "World")

        return [
            TextContent(
                type="text",
                text=f"Hello, {person_name}!"
            )
        ]

    raise ValueError(f"Unknown tool: {name}")

# Server 起動
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
```

### ビルドと実行

**Node.js**:
```bash
# ビルド
npm run build

# ローカルテスト
npm start
```

**Python**:
```bash
# 実行
python server.py
```

---

## Transport の理解

### stdio Transport

**特徴**:
- 標準入出力を使用
- プロセス間通信
- Claude Desktop のデフォルト

**使用例（Node.js）**:
```typescript
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js'

const transport = new StdioServerTransport()
await server.connect(transport)
```

**通信フロー**:
```
Claude Desktop
    ↓ stdin (JSON-RPC request)
MCP Server
    ↓ stdout (JSON-RPC response)
Claude Desktop
```

**デバッグ注意**:
- `console.log()` は使わない（stdout を汚染）
- `console.error()` のみ使用（stderr に出力）

### HTTP/SSE Transport（将来サポート予定）

**特徴**:
- HTTP リクエスト/レスポンス
- Server-Sent Events でストリーミング
- リモートサーバーに配置可能

**使用例**:
```typescript
import { SSEServerTransport } from '@modelcontextprotocol/sdk/server/sse.js'
import express from 'express'

const app = express()

app.post('/mcp', async (req, res) => {
  const transport = new SSEServerTransport('/messages', res)
  await server.connect(transport)
})

app.listen(3000)
```

---

## デバッグとログ

### ログ出力

**❌ 間違い（stdout を使う）**:
```typescript
console.log('Processing request...')  // NG: JSON-RPC を壊す
```

**✅ 正しい（stderr を使う）**:
```typescript
console.error('Processing request...')  // OK: デバッグログ
```

### 構造化ログ

```typescript
function log(level: 'info' | 'error' | 'debug', message: string, data?: any) {
  const logEntry = {
    timestamp: new Date().toISOString(),
    level,
    message,
    data,
  }
  console.error(JSON.stringify(logEntry))
}

// 使用例
log('info', 'Tool called', { name: 'hello', args: { name: 'Alice' } })
log('error', 'Tool execution failed', { error: error.message })
```

### エラーハンドリング

```typescript
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params

  try {
    if (name === 'hello') {
      // バリデーション
      if (!args?.name || typeof args.name !== 'string') {
        throw new Error('Invalid argument: name must be a string')
      }

      const personName = String(args.name)

      return {
        content: [
          {
            type: 'text',
            text: `Hello, ${personName}!`,
          },
        ],
      }
    }

    throw new Error(`Unknown tool: ${name}`)
  } catch (error) {
    console.error('Tool execution error:', error)

    // エラーをクライアントに返す
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

### デバッグ用テストクライアント

**test-client.ts**:
```typescript
import { Client } from '@modelcontextprotocol/sdk/client/index.js'
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js'
import { spawn } from 'child_process'

async function testMCPServer() {
  // Server プロセス起動
  const serverProcess = spawn('node', ['dist/index.js'], {
    stdio: ['pipe', 'pipe', 'inherit'],
  })

  // Client 作成
  const transport = new StdioClientTransport({
    command: 'node',
    args: ['dist/index.js'],
  })

  const client = new Client(
    {
      name: 'test-client',
      version: '1.0.0',
    },
    {
      capabilities: {},
    }
  )

  await client.connect(transport)

  // ツール一覧取得
  const tools = await client.listTools()
  console.log('Available tools:', tools)

  // ツール実行
  const result = await client.callTool({
    name: 'hello',
    arguments: {
      name: 'Alice',
    },
  })
  console.log('Tool result:', result)

  await client.close()
  serverProcess.kill()
}

testMCPServer().catch(console.error)
```

**実行**:
```bash
ts-node test-client.ts
```

---

## まとめ

### MCP Server 開発の基本

**必須コンポーネント**:
1. Server インスタンス（名前、バージョン、capabilities）
2. Transport（stdio が基本）
3. Request Handler（ListTools、CallTool）

**開発フロー**:
1. プロジェクトセットアップ
2. Server 実装
3. ツール定義
4. ビルド
5. Claude Desktop で テスト

**デバッグのコツ**:
- stderr でログ出力（`console.error`）
- 構造化ログで追跡
- テストクライアントで動作確認

---

## 次のステップ

1. **02-tool-resource-implementation.md**: Tool と Resource の詳細実装ガイド
2. **03-claude-desktop-integration.md**: Claude Desktop 統合ガイド

---

*MCP Server で Claude に新しい能力を追加しましょう。*
