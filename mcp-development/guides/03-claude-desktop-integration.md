# 🖥️ Claude Desktop 統合ガイド

> **目的**: MCP Server を Claude Desktop に統合し、実際に動作させる方法を習得する

## 📚 目次

1. [Claude Desktop 設定](#claude-desktop-設定)
2. [Server の登録](#server-の登録)
3. [デバッグ](#デバッグ)
4. [実践例](#実践例)
5. [トラブルシューティング](#トラブルシューティング)
6. [配布](#配布)

---

## Claude Desktop 設定

### 設定ファイルの場所

**macOS**:
```
~/Library/Application Support/Claude/claude_desktop_config.json
```

**Windows**:
```
%APPDATA%\Claude\claude_desktop_config.json
```

**Linux**:
```
~/.config/Claude/claude_desktop_config.json
```

### 設定ファイルの作成

初回は手動で作成する必要があります。

```bash
# macOS
mkdir -p ~/Library/Application\ Support/Claude
touch ~/Library/Application\ Support/Claude/claude_desktop_config.json

# Windows (PowerShell)
New-Item -ItemType Directory -Force -Path "$env:APPDATA\Claude"
New-Item -ItemType File -Path "$env:APPDATA\Claude\claude_desktop_config.json"

# Linux
mkdir -p ~/.config/Claude
touch ~/.config/Claude/claude_desktop_config.json
```

---

## Server の登録

### 基本的な設定

**claude_desktop_config.json**:
```json
{
  "mcpServers": {
    "my-server": {
      "command": "node",
      "args": [
        "/absolute/path/to/my-mcp-server/dist/index.js"
      ]
    }
  }
}
```

**重要**:
- `command`: 実行するコマンド（`node`、`python`、バイナリパスなど）
- `args`: コマンドライン引数（**絶対パス**を使用）
- Server名（`my-server`）は任意（重複不可）

### Node.js Server

```json
{
  "mcpServers": {
    "file-manager": {
      "command": "node",
      "args": [
        "/Users/username/projects/file-manager-mcp/dist/index.js"
      ]
    }
  }
}
```

### Python Server

```json
{
  "mcpServers": {
    "data-analyzer": {
      "command": "python",
      "args": [
        "/Users/username/projects/data-analyzer-mcp/server.py"
      ]
    }
  }
}
```

### 複数 Server

```json
{
  "mcpServers": {
    "file-manager": {
      "command": "node",
      "args": ["/path/to/file-manager/dist/index.js"]
    },
    "weather": {
      "command": "node",
      "args": ["/path/to/weather-mcp/dist/index.js"]
    },
    "database": {
      "command": "python",
      "args": ["/path/to/database-mcp/server.py"]
    }
  }
}
```

### 環境変数の設定

```json
{
  "mcpServers": {
    "weather": {
      "command": "node",
      "args": ["/path/to/weather-mcp/dist/index.js"],
      "env": {
        "OPENWEATHER_API_KEY": "your_api_key_here",
        "LOG_LEVEL": "debug"
      }
    }
  }
}
```

**セキュリティ注意**:
- API キーを直接書かない方が安全
- 環境変数ファイル（`.env`）を使用

**推奨: .env ファイル**:
```json
{
  "mcpServers": {
    "weather": {
      "command": "node",
      "args": ["/path/to/weather-mcp/dist/index.js"],
      "env": {
        "DOTENV_CONFIG_PATH": "/path/to/weather-mcp/.env"
      }
    }
  }
}
```

**.env**:
```bash
OPENWEATHER_API_KEY=your_api_key_here
LOG_LEVEL=debug
```

**server.ts**:
```typescript
import dotenv from 'dotenv'

// .env 読み込み
dotenv.config({ path: process.env.DOTENV_CONFIG_PATH })

const apiKey = process.env.OPENWEATHER_API_KEY
```

---

## デバッグ

### Claude Desktop のログ確認

**macOS**:
```bash
tail -f ~/Library/Logs/Claude/mcp*.log
```

**Windows**:
```powershell
Get-Content "$env:APPDATA\Claude\Logs\mcp*.log" -Wait
```

**Linux**:
```bash
tail -f ~/.config/Claude/logs/mcp*.log
```

### Server のログ出力

**src/index.ts**:
```typescript
// stderr にログ出力（stdout は使わない）
console.error('[INFO] MCP Server starting...')
console.error('[DEBUG] Tool called:', toolName)
console.error('[ERROR] Failed to execute:', error)

// 構造化ログ
function log(level: string, message: string, data?: any) {
  const entry = {
    timestamp: new Date().toISOString(),
    level,
    message,
    data,
  }
  console.error(JSON.stringify(entry))
}

log('info', 'Server started')
log('debug', 'Tool called', { name: 'calculate', args: { a: 5, b: 3 } })
```

### 手動テスト

**test-server.sh**:
```bash
#!/bin/bash

# Server を直接実行してテスト
node dist/index.js <<EOF
{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}
EOF
```

**実行**:
```bash
chmod +x test-server.sh
./test-server.sh
```

### デバッグモード

**package.json**:
```json
{
  "scripts": {
    "dev": "LOG_LEVEL=debug ts-node src/index.ts",
    "debug": "node --inspect dist/index.js"
  }
}
```

**設定（デバッグモード）**:
```json
{
  "mcpServers": {
    "my-server": {
      "command": "npm",
      "args": ["run", "dev"],
      "cwd": "/path/to/my-mcp-server",
      "env": {
        "LOG_LEVEL": "debug"
      }
    }
  }
}
```

---

## 実践例

### Example 1: Weather MCP Server

**プロジェクト構造**:
```
weather-mcp/
├── src/
│   └── index.ts
├── dist/
│   └── index.js
├── .env
├── package.json
└── tsconfig.json
```

**src/index.ts**:
```typescript
#!/usr/bin/env node

import { Server } from '@modelcontextprotocol/sdk/server/index.js'
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js'
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from '@modelcontextprotocol/sdk/types.js'
import axios from 'axios'
import dotenv from 'dotenv'

dotenv.config()

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
            city: {
              type: 'string',
              description: 'City name (e.g., Tokyo, London)',
            },
          },
          required: ['city'],
        },
      },
    ],
  }
})

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params

  if (name === 'get_weather') {
    const city = String(args?.city)
    const apiKey = process.env.OPENWEATHER_API_KEY

    if (!apiKey) {
      throw new Error('OPENWEATHER_API_KEY not configured')
    }

    try {
      const response = await axios.get(
        'https://api.openweathermap.org/data/2.5/weather',
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
🌡️  Temperature: ${weather.main.temp}°C
🌡️  Feels like: ${weather.main.feels_like}°C
💧 Humidity: ${weather.main.humidity}%
☁️  Description: ${weather.weather[0].description}`,
          },
        ],
      }
    } catch (error) {
      console.error('Weather API error:', error)
      throw new Error(`Failed to fetch weather for ${city}`)
    }
  }

  throw new Error(`Unknown tool: ${name}`)
})

async function main() {
  const transport = new StdioServerTransport()
  await server.connect(transport)
  console.error('Weather MCP Server running')
}

main().catch(console.error)
```

**.env**:
```bash
OPENWEATHER_API_KEY=your_api_key_here
```

**ビルド**:
```bash
npm run build
```

**Claude Desktop 設定**:
```json
{
  "mcpServers": {
    "weather": {
      "command": "node",
      "args": ["/Users/username/projects/weather-mcp/dist/index.js"],
      "env": {
        "DOTENV_CONFIG_PATH": "/Users/username/projects/weather-mcp/.env"
      }
    }
  }
}
```

**Claude Desktop 再起動**:
```bash
# macOS
killall Claude
open -a Claude

# Windows
taskkill /IM Claude.exe /F
start claude://
```

**使用例（Claude Desktop で）**:
```
User: What's the weather in Tokyo?
Claude: [calls get_weather tool with {city: "Tokyo"}]

Weather in Tokyo:
🌡️  Temperature: 18°C
🌡️  Feels like: 16°C
💧 Humidity: 65%
☁️  Description: clear sky
```

### Example 2: File Manager MCP

**src/index.ts**:
```typescript
import fs from 'fs/promises'
import path from 'path'

// ... server setup ...

server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: 'read_file',
        description: 'Read contents of a file',
        inputSchema: {
          type: 'object',
          properties: {
            path: { type: 'string', description: 'File path' },
          },
          required: ['path'],
        },
      },
      {
        name: 'write_file',
        description: 'Write content to a file',
        inputSchema: {
          type: 'object',
          properties: {
            path: { type: 'string' },
            content: { type: 'string' },
          },
          required: ['path', 'content'],
        },
      },
      {
        name: 'list_directory',
        description: 'List files in a directory',
        inputSchema: {
          type: 'object',
          properties: {
            path: { type: 'string' },
          },
          required: ['path'],
        },
      },
    ],
  }
})

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params

  if (name === 'read_file') {
    const filePath = String(args?.path)
    const content = await fs.readFile(filePath, 'utf-8')
    return {
      content: [{ type: 'text', text: content }],
    }
  }

  if (name === 'write_file') {
    const filePath = String(args?.path)
    const content = String(args?.content)
    await fs.writeFile(filePath, content, 'utf-8')
    return {
      content: [{ type: 'text', text: `File written: ${filePath}` }],
    }
  }

  if (name === 'list_directory') {
    const dirPath = String(args?.path)
    const files = await fs.readdir(dirPath)
    return {
      content: [{ type: 'text', text: files.join('\n') }],
    }
  }

  throw new Error(`Unknown tool: ${name}`)
})
```

---

## トラブルシューティング

### Server が起動しない

**症状**: Claude Desktop で Tool が表示されない

**確認事項**:
1. **設定ファイルのパス**が正しいか
   ```bash
   # macOS
   cat ~/Library/Application\ Support/Claude/claude_desktop_config.json
   ```

2. **JSON が有効か**（構文エラーチェック）
   ```bash
   cat claude_desktop_config.json | jq .
   ```

3. **Server のパスが絶対パス**か
   ```json
   // ❌ 相対パス
   "args": ["./dist/index.js"]

   // ✅ 絶対パス
   "args": ["/Users/username/projects/my-mcp/dist/index.js"]
   ```

4. **Server がビルドされているか**
   ```bash
   npm run build
   ls dist/index.js
   ```

5. **ログを確認**
   ```bash
   tail -f ~/Library/Logs/Claude/mcp*.log
   ```

### Tool が実行できない

**症状**: Tool は表示されるが、実行時にエラー

**確認事項**:
1. **環境変数が設定されているか**
2. **ファイルパーミッション**
   ```bash
   chmod +x dist/index.js
   ```
3. **依存関係がインストールされているか**
   ```bash
   npm install
   ```
4. **Server のログ**（stderr）を確認

### Claude Desktop が反応しない

**解決策**: 完全再起動

```bash
# macOS
killall Claude
rm -rf ~/Library/Caches/Claude
open -a Claude

# Windows
taskkill /IM Claude.exe /F
Remove-Item -Recurse "$env:APPDATA\Claude\Cache"
start claude://
```

---

## 配布

### npm パッケージとして配布

**package.json**:
```json
{
  "name": "my-mcp-server",
  "version": "1.0.0",
  "bin": {
    "my-mcp": "./dist/index.js"
  },
  "files": ["dist"],
  "scripts": {
    "prepublishOnly": "npm run build"
  }
}
```

**公開**:
```bash
npm publish
```

**ユーザーがインストール**:
```bash
npm install -g my-mcp-server
```

**Claude Desktop 設定**:
```json
{
  "mcpServers": {
    "my-server": {
      "command": "my-mcp"
    }
  }
}
```

### GitHub で配布

**README.md**:
```markdown
# My MCP Server

## Installation

```bash
git clone https://github.com/username/my-mcp-server.git
cd my-mcp-server
npm install
npm run build
```

## Claude Desktop Configuration

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "my-server": {
      "command": "node",
      "args": ["/path/to/my-mcp-server/dist/index.js"]
    }
  }
}
```

## Usage

In Claude Desktop:
- "Read the file /path/to/file.txt"
- "Write 'Hello World' to /path/to/output.txt"
```

---

## まとめ

### Claude Desktop 統合チェックリスト

**設定**:
- [ ] 設定ファイルの場所確認
- [ ] JSON 構文が正しい
- [ ] 絶対パスを使用
- [ ] 環境変数設定（必要に応じて）

**デバッグ**:
- [ ] Server ビルド確認
- [ ] ログファイル確認
- [ ] 手動テスト実行
- [ ] Claude Desktop 再起動

**配布**:
- [ ] README.md 作成
- [ ] インストール手順記載
- [ ] 設定例を提供

---

## 実践: 完全なワークフロー

### 1. Server 開発

```bash
mkdir weather-mcp
cd weather-mcp
npm init -y
npm install @modelcontextprotocol/sdk axios dotenv
npm install -D typescript @types/node

# src/index.ts 作成
# tsconfig.json 作成

npm run build
```

### 2. ローカルテスト

```bash
# .env 作成
echo "OPENWEATHER_API_KEY=your_key" > .env

# Server 実行
node dist/index.js
```

### 3. Claude Desktop 設定

```json
{
  "mcpServers": {
    "weather": {
      "command": "node",
      "args": ["/Users/username/projects/weather-mcp/dist/index.js"],
      "env": {
        "DOTENV_CONFIG_PATH": "/Users/username/projects/weather-mcp/.env"
      }
    }
  }
}
```

### 4. Claude Desktop 再起動

```bash
killall Claude
open -a Claude
```

### 5. 使用

```
User: What's the weather in Tokyo?
Claude: [uses get_weather tool]
The weather in Tokyo is...
```

---

*Claude Desktop に MCP Server を統合して、新しい能力を追加しましょう。*

**🎉 これで全26スキルのガイドが完成しました！**
