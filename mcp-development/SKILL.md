---
name: mcp-development
description: MCP（Model Context Protocol）Server開発ガイド。Claude Desktop統合、ツール定義、リソース公開、プロンプト実装、セキュリティなど、プロフェッショナルなMCPサーバー開発のベストプラクティス。
---

# MCP Development Skill

## 📋 目次

1. [概要](#概要)
2. [いつ使うか](#いつ使うか)
3. [MCP基礎](#mcp基礎)
4. [サーバー開発](#サーバー開発)
5. [ツール実装](#ツール実装)
6. [リソース公開](#リソース公開)
7. [Claude Desktop統合](#claude-desktop統合)
8. [実践例](#実践例)
9. [Agent連携](#agent連携)

---

## 概要

このSkillは、MCP Server開発をカバーします：

- **MCP基礎** - Model Context Protocol概要
- **ツール実装** - 関数定義、引数、実行
- **リソース公開** - ファイル、データ公開
- **プロンプト実装** - テンプレート定義
- **Claude Desktop統合** - 設定、テスト
- **セキュリティ** - 認証、バリデーション

---

## いつ使うか

### 🎯 必須のタイミング

- [ ] Claude Desktopに新機能追加時
- [ ] 外部API統合時
- [ ] ローカルファイルアクセス機能追加時
- [ ] カスタムツール提供時

---

## MCP基礎

### MCPとは

**Model Context Protocol（MCP）** は、AIアシスタントと外部ツールを接続するプロトコルです。

#### MCPの3つの主要機能

1. **Tools（ツール）** - 関数実行
2. **Resources（リソース）** - データ公開
3. **Prompts（プロンプト）** - テンプレート提供

### アーキテクチャ

```
Claude Desktop
    ↓ (MCP)
MCP Server (Node.js/Python)
    ↓
External APIs, Files, Databases
```

---

## サーバー開発

### Node.js（TypeScript）

```bash
# プロジェクト作成
mkdir my-mcp-server
cd my-mcp-server
pnpm init
pnpm add @modelcontextprotocol/sdk
pnpm add -D @types/node typescript
```

```typescript
// src/index.ts
import { Server } from '@modelcontextprotocol/sdk/server/index.js'
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js'
import {
  CallToolRequestSchema,
  ListToolsRequestSchema
} from '@modelcontextprotocol/sdk/types.js'

// サーバー作成
const server = new Server(
  {
    name: 'my-mcp-server',
    version: '1.0.0'
  },
  {
    capabilities: {
      tools: {}
    }
  }
)

// ツール一覧
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: 'get_weather',
        description: 'Get current weather for a city',
        inputSchema: {
          type: 'object',
          properties: {
            city: {
              type: 'string',
              description: 'City name'
            }
          },
          required: ['city']
        }
      }
    ]
  }
})

// ツール実行
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  if (request.params.name === 'get_weather') {
    const city = request.params.arguments?.city as string

    // 外部API呼び出し（例）
    const weather = await fetch(`https://api.weather.com/${city}`)
      .then(r => r.json())

    return {
      content: [
        {
          type: 'text',
          text: `Weather in ${city}: ${weather.description}`
        }
      ]
    }
  }

  throw new Error(`Unknown tool: ${request.params.name}`)
})

// サーバー起動
async function main() {
  const transport = new StdioServerTransport()
  await server.connect(transport)
  console.error('MCP Server running on stdio')
}

main().catch(console.error)
```

### Python

```python
# server.py
from mcp.server import Server
from mcp.types import Tool, TextContent
import httpx

app = Server("my-mcp-server")

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_weather",
            description="Get current weather for a city",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["city"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "get_weather":
        city = arguments["city"]

        async with httpx.AsyncClient() as client:
            response = await client.get(f"https://api.weather.com/{city}")
            weather = response.json()

        return [
            TextContent(
                type="text",
                text=f"Weather in {city}: {weather['description']}"
            )
        ]

    raise ValueError(f"Unknown tool: {name}")

if __name__ == "__main__":
    import asyncio
    from mcp.server.stdio import stdio_server

    async def main():
        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream)

    asyncio.run(main())
```

---

## ツール実装

### 基本的なツール

```typescript
// ツール定義
{
  name: 'calculate',
  description: 'Perform basic arithmetic',
  inputSchema: {
    type: 'object',
    properties: {
      operation: {
        type: 'string',
        enum: ['add', 'subtract', 'multiply', 'divide']
      },
      a: { type: 'number' },
      b: { type: 'number' }
    },
    required: ['operation', 'a', 'b']
  }
}

// ツール実行
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  if (request.params.name === 'calculate') {
    const { operation, a, b } = request.params.arguments as {
      operation: string
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
        result = a / b
        break
      default:
        throw new Error('Invalid operation')
    }

    return {
      content: [
        {
          type: 'text',
          text: `Result: ${result}`
        }
      ]
    }
  }
})
```

### ファイル操作ツール

```typescript
import fs from 'fs/promises'

{
  name: 'read_file',
  description: 'Read contents of a file',
  inputSchema: {
    type: 'object',
    properties: {
      path: {
        type: 'string',
        description: 'File path'
      }
    },
    required: ['path']
  }
}

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  if (request.params.name === 'read_file') {
    const { path } = request.params.arguments as { path: string }

    // セキュリティチェック
    if (path.includes('..')) {
      throw new Error('Invalid path')
    }

    const content = await fs.readFile(path, 'utf-8')

    return {
      content: [
        {
          type: 'text',
          text: content
        }
      ]
    }
  }
})
```

---

## リソース公開

### ファイルリソース

```typescript
import {
  ListResourcesRequestSchema,
  ReadResourceRequestSchema
} from '@modelcontextprotocol/sdk/types.js'

// リソース一覧
server.setRequestHandler(ListResourcesRequestSchema, async () => {
  return {
    resources: [
      {
        uri: 'file:///data/users.json',
        name: 'Users Data',
        mimeType: 'application/json',
        description: 'User database'
      }
    ]
  }
})

// リソース読み取り
server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
  const uri = request.params.uri

  if (uri === 'file:///data/users.json') {
    const data = await fs.readFile('./data/users.json', 'utf-8')

    return {
      contents: [
        {
          uri,
          mimeType: 'application/json',
          text: data
        }
      ]
    }
  }

  throw new Error(`Unknown resource: ${uri}`)
})
```

---

## Claude Desktop統合

### 設定ファイル

```json
// ~/Library/Application Support/Claude/claude_desktop_config.json (macOS)
// %APPDATA%/Claude/claude_desktop_config.json (Windows)
{
  "mcpServers": {
    "my-server": {
      "command": "node",
      "args": ["/path/to/my-mcp-server/dist/index.js"],
      "env": {
        "API_KEY": "your_api_key"
      }
    }
  }
}
```

### TypeScript ビルド

```json
// package.json
{
  "scripts": {
    "build": "tsc",
    "watch": "tsc --watch"
  }
}

// tsconfig.json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "Node16",
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true
  }
}
```

### テスト

```bash
# ビルド
pnpm build

# Claude Desktopを再起動

# Claudeで使用
# 「get_weather for Tokyo」
# → MCPサーバーのget_weatherツールが呼ばれる
```

---

## 実践例

### Example 1: Weather MCP Server

```typescript
// src/index.ts
import { Server } from '@modelcontextprotocol/sdk/server/index.js'
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js'
import {
  CallToolRequestSchema,
  ListToolsRequestSchema
} from '@modelcontextprotocol/sdk/types.js'

const server = new Server(
  { name: 'weather-server', version: '1.0.0' },
  { capabilities: { tools: {} } }
)

server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: 'get_weather',
        description: 'Get current weather for a city',
        inputSchema: {
          type: 'object',
          properties: {
            city: { type: 'string', description: 'City name' }
          },
          required: ['city']
        }
      },
      {
        name: 'get_forecast',
        description: 'Get 5-day weather forecast',
        inputSchema: {
          type: 'object',
          properties: {
            city: { type: 'string', description: 'City name' }
          },
          required: ['city']
        }
      }
    ]
  }
})

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params

  if (name === 'get_weather') {
    const city = args?.city as string
    // API呼び出し（実際にはOpenWeatherMap等を使用）
    return {
      content: [{ type: 'text', text: `Weather in ${city}: Sunny, 25°C` }]
    }
  }

  if (name === 'get_forecast') {
    const city = args?.city as string
    return {
      content: [{ type: 'text', text: `5-day forecast for ${city}: ...` }]
    }
  }

  throw new Error(`Unknown tool: ${name}`)
})

async function main() {
  const transport = new StdioServerTransport()
  await server.connect(transport)
}

main().catch(console.error)
```

---

## Agent連携

### 📖 Agentへの指示例

**MCP Server作成**
```
以下の機能を持つMCP Serverを作成してください：
- get_weather ツール（都市名から天気取得）
- get_forecast ツール（5日間予報取得）
- Claude Desktop設定ファイルも生成
```

**ファイル操作MCP作成**
```
ローカルファイルを操作するMCP Serverを作成してください：
- read_file ツール（ファイル読み込み）
- write_file ツール（ファイル書き込み）
- list_files ツール（ディレクトリ一覧）
セキュリティチェックを含めてください。
```

---

## まとめ

### MCP開発のベストプラクティス

1. **ツール定義** - 明確なinputSchema
2. **エラーハンドリング** - 適切なエラーメッセージ
3. **セキュリティ** - 入力バリデーション、パスチェック
4. **テスト** - Claude Desktopで動作確認

---

_Last updated: 2025-12-24_
