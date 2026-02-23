# JavaScript から TypeScript への移行ガイド

> 既存 JS プロジェクトを段階的に TypeScript 化する実践的なロードマップと移行テクニック

## この章で学ぶこと

1. **段階的移行戦略** -- 全面書き換えではなく、ファイル単位で安全に TypeScript を導入する手順
2. **tsconfig の段階的厳密化** -- `allowJs` から始めて `strict: true` に到達するまでの設定チェーンの管理
3. **よくある移行パターン** -- 型定義の補完、any の排除、サードパーティ型の導入テクニック
4. **大規模プロジェクトでの移行戦略** -- 1万行以上のプロジェクトでの実践的なアプローチ
5. **移行後の品質維持** -- CI/CD での型チェック統合、チーム内の TypeScript 標準化

---

## 1. 移行ロードマップ

### 1-1. 全体フロー

```
段階的移行の 5 フェーズ:

Phase 1: 準備         Phase 2: 共存        Phase 3: 変換
+---------------+    +---------------+    +---------------+
| tsconfig 導入  |    | .js + .ts     |    | 主要ファイルを  |
| allowJs: true |    | 共存           |    | .ts に変換     |
| strict: false |    | checkJs: true |    | any を除去     |
+---------------+    +---------------+    +---------------+
       |                    |                    |
       v                    v                    v
Phase 4: 厳密化        Phase 5: 完了
+---------------+    +---------------+
| strict: true  |    | 全ファイル .ts  |
| 個別オプション |    | CI で型チェック |
| を順次有効化   |    | JSDoc 除去     |
+---------------+    +---------------+

期間の目安:
  ~1,000行:   1-2日
  ~10,000行:  1-2週間
  ~100,000行: 1-3ヶ月
  ~1,000,000行: 3-12ヶ月
```

### 1-2. Phase 1 -- 準備

```bash
# TypeScript と関連ツールをインストール
npm install -D typescript @types/node

# フレームワークの型もインストール
npm install -D @types/express @types/cors @types/lodash

# tsconfig.json を生成
npx tsc --init
```

```json
// Phase 1 の tsconfig.json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "allowJs": true,
    "checkJs": false,
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": false,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "noEmit": true,
    "resolveJsonModule": true,
    "isolatedModules": true
  },
  "include": ["src/**/*"]
}
```

```json
// package.json にスクリプト追加
{
  "scripts": {
    "typecheck": "tsc --noEmit",
    "typecheck:watch": "tsc --noEmit --watch"
  }
}
```

### 1-3. Phase 2 -- JS と TS の共存

```typescript
// JSDoc で既存 JS ファイルに型をつける（ファイル変換前）
// src/utils.js

/**
 * @param {string} name
 * @param {number} age
 * @returns {{ name: string, age: number, id: string }}
 */
function createUser(name, age) {
  return {
    name,
    age,
    id: Math.random().toString(36).slice(2),
  };
}

/**
 * @typedef {Object} Config
 * @property {string} apiUrl
 * @property {number} timeout
 * @property {boolean} [debug]
 */

/**
 * @param {Config} config
 * @returns {void}
 */
function initApp(config) {
  // ...
}

/**
 * @template T
 * @param {T[]} items
 * @param {(item: T) => boolean} predicate
 * @returns {T | undefined}
 */
function findItem(items, predicate) {
  return items.find(predicate);
}

/**
 * @param {unknown} value
 * @returns {value is string}
 */
function isString(value) {
  return typeof value === "string";
}

module.exports = { createUser, initApp, findItem, isString };
```

```json
// Phase 2 の tsconfig.json 更新
{
  "compilerOptions": {
    "allowJs": true,
    "checkJs": true,  // JS ファイルも型チェック
    "strict": false,
    // JSDoc の型チェックエラーを段階的に表示
    "noImplicitAny": false
  }
}
```

```
JSDoc の型アノテーション一覧:

  基本型:
  @param {string} name           -- 文字列
  @param {number} age            -- 数値
  @param {boolean} active        -- 真偽値
  @param {Date} createdAt        -- Date
  @param {any} data              -- any
  @param {unknown} input         -- unknown

  複合型:
  @param {string | number} id    -- ユニオン型
  @param {string[]} tags         -- 配列
  @param {{ name: string }} user -- オブジェクト
  @param {?string} name          -- nullable
  @param {string} [name]         -- optional

  ジェネリクス:
  @template T
  @param {T} value
  @returns {T}

  型定義:
  @typedef {Object} User
  @property {string} name
  @property {number} age

  型ガード:
  @param {unknown} value
  @returns {value is string}
```

### 1-4. Phase 3 -- ファイル変換

```
変換の手順（1ファイルにつき）:

  1. .js → .ts にリネーム
  2. require → import に変換
  3. module.exports → export に変換
  4. 型エラーを修正（暫定的に as any も許容）
  5. テストが通ることを確認
  6. コミット

  注意: 1ファイルごとにコミットすることで
  問題が発生した場合にロールバックが容易
```

### 1-5. Phase 4 -- 厳密化

```json
// 段階的に有効化
// Step 1: 暗黙の any を検出
{
  "compilerOptions": {
    "strict": false,
    "noImplicitAny": true
  }
}

// Step 2: null チェックを追加
{
  "compilerOptions": {
    "strict": false,
    "noImplicitAny": true,
    "strictNullChecks": true
  }
}

// Step 3: 関数型の厳密化
{
  "compilerOptions": {
    "strict": false,
    "noImplicitAny": true,
    "strictNullChecks": true,
    "strictFunctionTypes": true,
    "strictBindCallApply": true
  }
}

// Step 4: 完全な strict
{
  "compilerOptions": {
    "strict": true
  }
}

// Step 5: 追加の厳密性
{
  "compilerOptions": {
    "strict": true,
    "noUncheckedIndexedAccess": true,
    "exactOptionalPropertyTypes": true,
    "noImplicitOverride": true,
    "noFallthroughCasesInSwitch": true
  }
}
```

### 1-6. Phase 5 -- 完了と品質維持

```json
// 最終的な tsconfig.json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true,
    "noUncheckedIndexedAccess": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "noEmit": true,
    "verbatimModuleSyntax": true,
    "isolatedModules": true,
    // allowJs を false に（移行完了）
    "allowJs": false,
    "resolveJsonModule": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
```

---

## 2. ファイル変換テクニック

### 2-1. 変換の優先順位

```
変換優先順位（依存の葉から始める）:

  依存グラフ:
  index.ts ──→ routes.js ──→ controllers.js ──→ services.js
                                   |                |
                                   v                v
                              models.js        utils.js ← まずここから
                                   |
                                   v
                              database.js

  変換順序:
  1. utils.js → utils.ts      (依存なし)
  2. models.js → models.ts    (utils のみ依存)
  3. database.js → database.ts
  4. services.js → services.ts
  5. controllers.js → controllers.ts
  6. routes.js → routes.ts
  7. index.js → index.ts

  理由:
  - 依存の葉から変換すると、型エラーの連鎖が少ない
  - 変換済みファイルから型情報が伝播する
  - テストが段階的に通りやすい
```

### 2-2. 基本変換パターン

```typescript
// Before: src/user-service.js
const db = require("./database");

class UserService {
  constructor(database) {
    this.db = database;
  }

  async getUser(id) {
    const user = await this.db.query("SELECT * FROM users WHERE id = $1", [id]);
    if (!user) return null;
    return {
      id: user.id,
      name: user.name,
      email: user.email,
      createdAt: new Date(user.created_at),
    };
  }

  async createUser(data) {
    return this.db.query(
      "INSERT INTO users (name, email) VALUES ($1, $2) RETURNING *",
      [data.name, data.email]
    );
  }

  async getUsersByRole(role) {
    const users = await this.db.query(
      "SELECT * FROM users WHERE role = $1",
      [role]
    );
    return users.map(u => ({
      id: u.id,
      name: u.name,
      email: u.email,
      role: u.role,
    }));
  }
}

module.exports = { UserService };

// ─────────────────────────────────────────────

// After: src/user-service.ts
import { Database } from "./database";

// まず型定義を作成
interface User {
  id: string;
  name: string;
  email: string;
  createdAt: Date;
}

interface CreateUserDto {
  name: string;
  email: string;
}

interface UserRow {
  id: string;
  name: string;
  email: string;
  role: string;
  created_at: string;
}

type UserRole = "USER" | "ADMIN" | "MODERATOR";

class UserService {
  constructor(private readonly db: Database) {}

  async getUser(id: string): Promise<User | null> {
    const user = await this.db.query<UserRow>(
      "SELECT * FROM users WHERE id = $1",
      [id]
    );
    if (!user) return null;
    return {
      id: user.id,
      name: user.name,
      email: user.email,
      createdAt: new Date(user.created_at),
    };
  }

  async createUser(data: CreateUserDto): Promise<User> {
    const row = await this.db.query<UserRow>(
      "INSERT INTO users (name, email) VALUES ($1, $2) RETURNING *",
      [data.name, data.email]
    );
    return {
      id: row.id,
      name: row.name,
      email: row.email,
      createdAt: new Date(row.created_at),
    };
  }

  async getUsersByRole(role: UserRole): Promise<User[]> {
    const users = await this.db.queryMany<UserRow>(
      "SELECT * FROM users WHERE role = $1",
      [role]
    );
    return users.map((u) => ({
      id: u.id,
      name: u.name,
      email: u.email,
      createdAt: new Date(u.created_at),
    }));
  }
}

export { UserService, type User, type CreateUserDto, type UserRole };
```

### 2-3. require → import の変換

```typescript
// Before (CommonJS)
const express = require("express");
const { UserService } = require("./user-service");
const config = require("./config.json");
const path = require("path");
const fs = require("fs").promises;

// After (ESM)
import express from "express";
import { UserService } from "./user-service.js"; // NodeNext なら拡張子必要
import config from "./config.json" with { type: "json" };
import path from "node:path";
import { readFile, writeFile } from "node:fs/promises";

// ──────────────────────────────────────

// Before (module.exports)
module.exports = { UserService };
module.exports.default = app;

// After (export)
export { UserService };
export default app;

// ──────────────────────────────────────

// Before (条件付き require)
let sharp;
try {
  sharp = require("sharp");
} catch {
  sharp = null;
}

// After (動的 import)
let sharp: typeof import("sharp") | null;
try {
  sharp = await import("sharp");
} catch {
  sharp = null;
}

// ──────────────────────────────────────

// Before (require.resolve)
const packagePath = require.resolve("my-package/package.json");

// After (import.meta.resolve)
const packagePath = import.meta.resolve("my-package/package.json");
```

### 2-4. Express アプリケーションの変換例

```typescript
// Before: app.js
const express = require("express");
const cors = require("cors");
const { UserService } = require("./services/user-service");
const { authMiddleware } = require("./middleware/auth");

const app = express();
app.use(cors());
app.use(express.json());

app.get("/users", async (req, res) => {
  try {
    const users = await UserService.findAll();
    res.json(users);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.get("/users/:id", async (req, res) => {
  try {
    const user = await UserService.findById(req.params.id);
    if (!user) {
      return res.status(404).json({ error: "Not found" });
    }
    res.json(user);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post("/users", authMiddleware, async (req, res) => {
  try {
    const user = await UserService.create(req.body);
    res.status(201).json(user);
  } catch (err) {
    if (err.code === "VALIDATION_ERROR") {
      return res.status(400).json({ error: err.message });
    }
    res.status(500).json({ error: err.message });
  }
});

module.exports = app;

// ─────────────────────────────────────────────

// After: app.ts
import express, { type Request, type Response, type NextFunction } from "express";
import cors from "cors";
import { UserService } from "./services/user-service";
import { authMiddleware } from "./middleware/auth";
import type { User, CreateUserDto } from "./types";

const app = express();
app.use(cors());
app.use(express.json());

// エラーレスポンスの型
interface ErrorResponse {
  error: string;
  details?: unknown;
}

// 型付きリクエストハンドラ
app.get("/users", async (_req: Request, res: Response<User[] | ErrorResponse>) => {
  try {
    const users = await UserService.findAll();
    res.json(users);
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unknown error";
    res.status(500).json({ error: message });
  }
});

app.get(
  "/users/:id",
  async (req: Request<{ id: string }>, res: Response<User | ErrorResponse>) => {
    try {
      const user = await UserService.findById(req.params.id);
      if (!user) {
        return res.status(404).json({ error: "Not found" });
      }
      res.json(user);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Unknown error";
      res.status(500).json({ error: message });
    }
  }
);

app.post(
  "/users",
  authMiddleware,
  async (
    req: Request<unknown, unknown, CreateUserDto>,
    res: Response<User | ErrorResponse>
  ) => {
    try {
      const user = await UserService.create(req.body);
      res.status(201).json(user);
    } catch (err: unknown) {
      if (err instanceof Error && "code" in err && err.code === "VALIDATION_ERROR") {
        return res.status(400).json({ error: err.message });
      }
      const message = err instanceof Error ? err.message : "Unknown error";
      res.status(500).json({ error: message });
    }
  }
);

export default app;
```

### 2-5. React コンポーネントの変換

```typescript
// Before: UserCard.jsx
import React, { useState, useEffect } from "react";
import PropTypes from "prop-types";

function UserCard({ user, onEdit, onDelete }) {
  const [isEditing, setIsEditing] = useState(false);
  const [name, setName] = useState(user.name);

  useEffect(() => {
    setName(user.name);
  }, [user.name]);

  const handleSave = () => {
    onEdit(user.id, { name });
    setIsEditing(false);
  };

  return (
    <div className="user-card">
      {isEditing ? (
        <input value={name} onChange={(e) => setName(e.target.value)} />
      ) : (
        <span>{user.name}</span>
      )}
      <span>{user.email}</span>
      <button onClick={() => setIsEditing(!isEditing)}>
        {isEditing ? "Cancel" : "Edit"}
      </button>
      {isEditing && <button onClick={handleSave}>Save</button>}
      <button onClick={() => onDelete(user.id)}>Delete</button>
    </div>
  );
}

UserCard.propTypes = {
  user: PropTypes.shape({
    id: PropTypes.string.isRequired,
    name: PropTypes.string.isRequired,
    email: PropTypes.string.isRequired,
  }).isRequired,
  onEdit: PropTypes.func.isRequired,
  onDelete: PropTypes.func.isRequired,
};

export default UserCard;

// ─────────────────────────────────────────────

// After: UserCard.tsx
import { useState, useEffect, type FC } from "react";

interface User {
  id: string;
  name: string;
  email: string;
}

interface UserCardProps {
  user: User;
  onEdit: (id: string, data: { name: string }) => void;
  onDelete: (id: string) => void;
}

const UserCard: FC<UserCardProps> = ({ user, onEdit, onDelete }) => {
  const [isEditing, setIsEditing] = useState(false);
  const [name, setName] = useState(user.name);

  useEffect(() => {
    setName(user.name);
  }, [user.name]);

  const handleSave = () => {
    onEdit(user.id, { name });
    setIsEditing(false);
  };

  return (
    <div className="user-card">
      {isEditing ? (
        <input value={name} onChange={(e) => setName(e.target.value)} />
      ) : (
        <span>{user.name}</span>
      )}
      <span>{user.email}</span>
      <button onClick={() => setIsEditing(!isEditing)}>
        {isEditing ? "Cancel" : "Edit"}
      </button>
      {isEditing && <button onClick={handleSave}>Save</button>}
      <button onClick={() => onDelete(user.id)}>Delete</button>
    </div>
  );
};

export default UserCard;
// PropTypes は不要に → npm uninstall prop-types
```

---

## 3. any の段階的排除

### 3-1. any のトリアージ

```
any 排除の優先順位:

  +------------------+----------+-----------+-------------------+
  | カテゴリ          | 危険度   | 対応      | 推定作業量        |
  +------------------+----------+-----------+-------------------+
  | API レスポンス    | 高       | zod 導入  | 中                |
  | 関数パラメータ    | 高       | 型定義    | 小                |
  | イベントハンドラ  | 中       | 型定義    | 小                |
  | catch 変数       | 中       | unknown   | 小                |
  | JSON.parse 結果  | 中       | zod       | 中                |
  | サードパーティ    | 低       | @types/*  | 小〜中            |
  | 一時的な TODO     | 低       | コメント  | 将来              |
  +------------------+----------+-----------+-------------------+
```

```typescript
// Step 1: 明示的な any を unknown に置き換え
// Before
function processData(data: any): any {
  return data.map((item: any) => item.name);
}

// After
function processData(data: unknown): string[] {
  if (!Array.isArray(data)) {
    throw new Error("Expected array");
  }
  return data.map((item: unknown) => {
    if (typeof item === "object" && item !== null && "name" in item) {
      return String((item as { name: unknown }).name);
    }
    throw new Error("Invalid item");
  });
}

// Step 2: zod でバリデーション
import { z } from "zod";

const ItemSchema = z.object({ name: z.string() });
const DataSchema = z.array(ItemSchema);

function processData(data: unknown): string[] {
  const parsed = DataSchema.parse(data);
  return parsed.map((item) => item.name);
}

// Step 3: JSON.parse の安全な処理
// Before
function parseConfig(json: string): any {
  return JSON.parse(json);
}

// After
const ConfigSchema = z.object({
  apiUrl: z.string().url(),
  port: z.number().int().positive(),
  debug: z.boolean().default(false),
});
type Config = z.infer<typeof ConfigSchema>;

function parseConfig(json: string): Config {
  const raw: unknown = JSON.parse(json);
  return ConfigSchema.parse(raw);
}
```

### 3-2. @types/* の導入

```bash
# 型定義パッケージを検索・インストール
npm install -D @types/express @types/lodash @types/cors @types/compression

# 型定義が存在するか確認
npm info @types/some-package

# 複数パッケージを一括インストール
npm install -D @types/express @types/cors @types/morgan @types/cookie-parser
```

```typescript
// src/types/untyped-module.d.ts
// 型定義がないサードパーティモジュール用

// 最小限の型定義（一時的、段階的に充実させる）
declare module "untyped-lib" {
  export function doSomething(input: string): Promise<unknown>;
  export interface Config {
    apiKey: string;
    timeout?: number;
  }
}

// 型定義を段階的に充実させる
declare module "legacy-lib" {
  export interface Options {
    format: "json" | "csv" | "xml";
    encoding?: BufferEncoding;
    strict?: boolean;
  }

  export function parse(input: string, options?: Options): Record<string, unknown>;
  export function stringify(data: Record<string, unknown>, options?: Options): string;

  export class Parser {
    constructor(options?: Options);
    parse(input: string): Record<string, unknown>;
    on(event: "data", handler: (chunk: unknown) => void): this;
    on(event: "error", handler: (error: Error) => void): this;
    on(event: "end", handler: () => void): this;
  }

  export default Parser;
}

// CSS / 画像モジュールの型定義（Vite / webpack 用）
declare module "*.css" {
  const classes: Record<string, string>;
  export default classes;
}

declare module "*.module.css" {
  const classes: Record<string, string>;
  export default classes;
}

declare module "*.svg" {
  import type { FC, SVGProps } from "react";
  const component: FC<SVGProps<SVGSVGElement>>;
  export default component;
}

declare module "*.png" {
  const url: string;
  export default url;
}

// グローバル型の拡張
declare global {
  interface Window {
    __APP_CONFIG__: {
      apiUrl: string;
      version: string;
    };
  }

  namespace NodeJS {
    interface ProcessEnv {
      NODE_ENV: "development" | "production" | "test";
      DATABASE_URL: string;
      API_KEY: string;
      PORT?: string;
    }
  }
}

export {}; // モジュールとして認識させるために必要
```

### 3-3. any を使わない型安全なユーティリティ

```typescript
// any を使わずに柔軟な型を定義するパターン

// 1. unknown + 型ガード
function isError(value: unknown): value is Error {
  return value instanceof Error;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function getErrorMessage(error: unknown): string {
  if (isError(error)) return error.message;
  if (typeof error === "string") return error;
  return "Unknown error";
}

// 2. ジェネリクスで柔軟性を確保
function safeJsonParse<T>(json: string, schema: z.ZodType<T>): T | null {
  try {
    const raw: unknown = JSON.parse(json);
    return schema.parse(raw);
  } catch {
    return null;
  }
}

// 3. satisfies でオブジェクトの型チェック
const routes = {
  home: "/",
  about: "/about",
  user: "/users/:id",
} satisfies Record<string, string>;

// 4. Record<string, unknown> で動的オブジェクト
function filterObject(
  obj: Record<string, unknown>,
  predicate: (key: string, value: unknown) => boolean
): Record<string, unknown> {
  return Object.fromEntries(
    Object.entries(obj).filter(([key, value]) => predicate(key, value))
  );
}
```

---

## 4. strict 化のロードマップ

### 4-1. noImplicitAny 対応パターン

```typescript
// エラー: Parameter 'x' implicitly has an 'any' type
// 対応パターン集

// 1. コールバック引数
// Before
array.forEach(function (item) { /* ... */ });
// After
array.forEach(function (item: ItemType) { /* ... */ });
// もしくはアロー関数（型推論が効く場合）
array.forEach((item) => { /* ... */ });

// 2. オブジェクトの動的アクセス
// Before
function getValue(obj, key) { return obj[key]; }
// After
function getValue<T extends Record<string, unknown>>(
  obj: T,
  key: keyof T
): T[keyof T] {
  return obj[key];
}

// 3. イベントハンドラ
// Before
element.addEventListener("click", function (e) { /* ... */ });
// After（DOM の型定義から自動推論）
element.addEventListener("click", (e: MouseEvent) => { /* ... */ });

// 4. デストラクチャリング
// Before
function processResponse({ data, status }) { /* ... */ }
// After
interface ApiResponse {
  data: unknown;
  status: number;
}
function processResponse({ data, status }: ApiResponse) { /* ... */ }

// 5. 関数のオーバーロード
// Before
function format(value) {
  if (typeof value === "number") return value.toFixed(2);
  if (typeof value === "string") return value.trim();
  return String(value);
}
// After
function format(value: number): string;
function format(value: string): string;
function format(value: unknown): string;
function format(value: unknown): string {
  if (typeof value === "number") return value.toFixed(2);
  if (typeof value === "string") return value.trim();
  return String(value);
}

// 6. reduce のアキュムレータ
// Before
const total = items.reduce((sum, item) => sum + item.price, 0);
// After
const total = items.reduce<number>((sum, item) => sum + item.price, 0);
// もしくは初期値から推論される場合はそのまま
```

### 4-2. strictNullChecks 対応パターン

```typescript
// エラー: Object is possibly 'null'
// 対応パターン集

// 1. 早期リターン（Guard Clause）
function getUser(id: string): User | null {
  const user = findById(id);
  if (!user) return null; // または throw
  // ここ以降 user は User 型
  return user;
}

// 2. Optional Chaining + Nullish Coalescing
const name = user?.name ?? "Unknown";
const city = user?.address?.city ?? "N/A";

// 3. 非 null アサーション（確実な場合のみ）
const element = document.getElementById("app")!;
// ↑ element が null の可能性がある場合は使わない

// 4. Map / Set の型安全なアクセス
const map = new Map<string, User>();
const user = map.get("key"); // User | undefined
if (user) {
  console.log(user.name); // OK
}

// 5. 配列の find
const found = users.find((u) => u.id === targetId);
if (!found) {
  throw new Error(`User ${targetId} not found`);
}
// found は User 型

// 6. Promise の結果
async function fetchUser(id: string): Promise<User | null> {
  const response = await fetch(`/api/users/${id}`);
  if (!response.ok) return null;
  return response.json() as Promise<User>;
}
```

---

## 5. 大規模プロジェクトでの移行戦略

### 5-1. モジュール境界での移行

```
大規模プロジェクトでは、モジュール境界で区切って移行する:

  +------------------+     +------------------+     +------------------+
  | Authentication   |     | User Management  |     | Order System     |
  | (TypeScript化済) | --> | (移行中)          | --> | (JavaScript)     |
  +------------------+     +------------------+     +------------------+
         |                        |                        |
         v                        v                        v
  +------------------+     +------------------+     +------------------+
  | 型定義ファイル     |     | 部分的に型付き    |     | .d.ts で橋渡し   |
  | 完全に型安全      |     | allowJs + checkJs |     | 将来移行         |
  +------------------+     +------------------+     +------------------+

  ポイント:
  1. ドメイン単位でモジュールを分割
  2. モジュール間のインターフェースに .d.ts を定義
  3. 高リスク/高頻度変更のモジュールから優先移行
  4. 新機能は常に TypeScript で開発
```

### 5-2. any カウンターの導入

```typescript
// scripts/count-any.ts -- any の数をカウントするスクリプト
import { Project } from "ts-morph";

const project = new Project({
  tsConfigFilePath: "./tsconfig.json",
});

let totalAny = 0;
const fileStats: { file: string; count: number }[] = [];

for (const sourceFile of project.getSourceFiles()) {
  const filePath = sourceFile.getFilePath();
  const text = sourceFile.getFullText();

  // 明示的な any をカウント
  const anyCount = (text.match(/:\s*any\b/g) || []).length;
  // as any をカウント
  const asAnyCount = (text.match(/as\s+any\b/g) || []).length;

  const total = anyCount + asAnyCount;
  if (total > 0) {
    fileStats.push({ file: filePath, count: total });
    totalAny += total;
  }
}

console.log(`Total any count: ${totalAny}`);
console.log("\nTop 10 files with most 'any':");
fileStats
  .sort((a, b) => b.count - a.count)
  .slice(0, 10)
  .forEach(({ file, count }) => {
    console.log(`  ${count} any: ${file}`);
  });
```

```json
// package.json
{
  "scripts": {
    "count-any": "tsx scripts/count-any.ts",
    "migration-status": "tsx scripts/migration-status.ts"
  }
}
```

### 5-3. 移行進捗の可視化

```typescript
// scripts/migration-status.ts
import * as fs from "fs";
import * as path from "path";

function countFiles(dir: string): { js: number; ts: number; jsx: number; tsx: number } {
  const result = { js: 0, ts: 0, jsx: 0, tsx: 0 };

  function walk(currentDir: string) {
    const entries = fs.readdirSync(currentDir, { withFileTypes: true });
    for (const entry of entries) {
      if (entry.isDirectory()) {
        if (entry.name === "node_modules" || entry.name === "dist") continue;
        walk(path.join(currentDir, entry.name));
      } else {
        const ext = path.extname(entry.name);
        if (ext === ".js") result.js++;
        if (ext === ".ts" && !entry.name.endsWith(".d.ts")) result.ts++;
        if (ext === ".jsx") result.jsx++;
        if (ext === ".tsx") result.tsx++;
      }
    }
  }

  walk(dir);
  return result;
}

const counts = countFiles("./src");
const total = counts.js + counts.ts + counts.jsx + counts.tsx;
const tsTotal = counts.ts + counts.tsx;
const jsTotal = counts.js + counts.jsx;
const percentage = ((tsTotal / total) * 100).toFixed(1);

console.log("=== Migration Status ===");
console.log(`TypeScript files: ${tsTotal} (${percentage}%)`);
console.log(`JavaScript files: ${jsTotal} (${(100 - Number(percentage)).toFixed(1)}%)`);
console.log(`  .ts:  ${counts.ts}`);
console.log(`  .tsx: ${counts.tsx}`);
console.log(`  .js:  ${counts.js}`);
console.log(`  .jsx: ${counts.jsx}`);
console.log(`Total:  ${total}`);
console.log(`\nProgress: [${"█".repeat(Math.floor(Number(percentage) / 5))}${"░".repeat(20 - Math.floor(Number(percentage) / 5))}] ${percentage}%`);
```

---

## 6. よくある移行パターン

### 6-1. コールバック地獄から async/await へ

```typescript
// Before: コールバック (JavaScript)
function getUser(id, callback) {
  db.query("SELECT * FROM users WHERE id = ?", [id], (err, rows) => {
    if (err) return callback(err);
    if (rows.length === 0) return callback(new Error("Not found"));
    callback(null, rows[0]);
  });
}

// After: async/await (TypeScript)
async function getUser(id: string): Promise<User> {
  const rows = await db.query<UserRow[]>(
    "SELECT * FROM users WHERE id = ?",
    [id]
  );
  if (rows.length === 0) {
    throw new NotFoundError(`User ${id} not found`);
  }
  return mapUserRow(rows[0]);
}
```

### 6-2. 設定オブジェクトの型定義

```typescript
// Before: 動的な設定オブジェクト
const config = {
  database: {
    host: process.env.DB_HOST || "localhost",
    port: parseInt(process.env.DB_PORT || "5432"),
    name: process.env.DB_NAME || "myapp",
  },
  redis: {
    url: process.env.REDIS_URL || "redis://localhost:6379",
  },
  server: {
    port: parseInt(process.env.PORT || "3000"),
    cors: {
      origin: process.env.CORS_ORIGIN || "*",
    },
  },
};

// After: zod による型安全な設定
import { z } from "zod";

const ConfigSchema = z.object({
  database: z.object({
    host: z.string().default("localhost"),
    port: z.coerce.number().int().positive().default(5432),
    name: z.string().default("myapp"),
    ssl: z.boolean().default(false),
  }),
  redis: z.object({
    url: z.string().url().default("redis://localhost:6379"),
  }),
  server: z.object({
    port: z.coerce.number().int().positive().default(3000),
    cors: z.object({
      origin: z.string().default("*"),
    }),
  }),
});

type Config = z.infer<typeof ConfigSchema>;

export const config: Config = ConfigSchema.parse({
  database: {
    host: process.env.DB_HOST,
    port: process.env.DB_PORT,
    name: process.env.DB_NAME,
  },
  redis: {
    url: process.env.REDIS_URL,
  },
  server: {
    port: process.env.PORT,
    cors: {
      origin: process.env.CORS_ORIGIN,
    },
  },
});
```

---

## 比較表

### 移行アプローチの比較

| アプローチ | 期間 | リスク | チーム影響 | 推奨プロジェクト規模 |
|-----------|------|--------|-----------|-------------------|
| ビッグバン（全ファイル一括変換） | 短 | 高 | 大 | 小（~50ファイル） |
| 段階的移行（ファイル単位） | 長 | 低 | 小 | 中〜大 |
| 新機能のみ TS | 最長 | 最低 | 最小 | 大（レガシー） |
| 別ブランチで並行 | 中 | 中 | 中 | 中 |
| モジュール境界で分割 | 中〜長 | 低 | 小 | 大（マイクロサービス） |

### any 排除ツールの比較

| ツール | 用途 | 自動修正 | 精度 |
|--------|------|---------|------|
| `tsc --strict` | 暗黙 any 検出 | なし | 高 |
| `@typescript-eslint/no-explicit-any` | 明示 any 検出 | なし | 高 |
| `ts-prune` | 未使用 export 検出 | なし | 中 |
| zod | ランタイム型検証 | 型推論 | 最高 |
| TypeStat | 自動型追加 | あり | 中 |
| ts-morph | プログラム的な型操作 | あり | 高 |

---

## アンチパターン

### AP-1: 全ファイルを一括変換

```bash
# NG: 一括で .js → .ts にリネーム
find src -name "*.js" -exec bash -c 'mv "$0" "${0%.js}.ts"' {} \;
# → 数百のコンパイルエラーが一度に発生し、対応不能に

# OK: 1ファイルずつ変換、各変換後にテスト
# 1. utils.js → utils.ts (型エラー修正、テスト実行)
# 2. models.js → models.ts (型エラー修正、テスト実行)
# 3. ...（1日 3-5 ファイルのペースで）
```

### AP-2: as any でエラーを黙らせる

```typescript
// NG: as any で型エラーを握りつぶす
const result = someFunction(data as any) as any;

// OK: TODO コメント付きで一時的に許容
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const result = someFunction(data as any); // TODO: #123 で型を修正

// さらに OK: 正しい型を定義
interface InputData {
  name: string;
  values: number[];
}
const result = someFunction(data as InputData);
```

### AP-3: 型定義ファイルを書きすぎる

```typescript
// NG: 全てのサードパーティモジュールに詳細な .d.ts を書く
// → 保守コストが膨大

// OK: 段階的なアプローチ
// Step 1: 最小限の型定義
declare module "old-lib" {
  export function process(data: unknown): unknown;
}

// Step 2: 使用頻度の高い関数のみ型を充実
declare module "old-lib" {
  export function process<T>(data: T): ProcessResult<T>;
  export interface ProcessResult<T> {
    data: T;
    metadata: Record<string, string>;
  }
}

// Step 3: DefinitelyTyped に PR を出すことも検討
```

### AP-4: 移行中にリファクタリングも同時進行

```
NG:
  .js → .ts 変換 + ロジック変更 + リファクタリング

  → 変更が多すぎてレビューが困難
  → バグが入り込んでも原因特定が困難
  → テストが壊れた時に何が原因かわからない

OK:
  Step 1: .js → .ts（型のみ追加、ロジック変更なし）
  Step 2: テスト確認、コミット
  Step 3: リファクタリング（別のコミットで）
```

---

## 移行チェックリスト

```
Phase 1 準備:
  [ ] TypeScript インストール
  [ ] tsconfig.json 作成 (allowJs: true, strict: false)
  [ ] ビルドパイプラインに tsc --noEmit 追加
  [ ] @types/* パッケージインストール
  [ ] ESLint を typescript-eslint に設定
  [ ] CI で型チェックを実行

Phase 2 共存:
  [ ] checkJs: true 有効化
  [ ] JSDoc で主要関数に型アノテーション
  [ ] 共通の型定義ファイル (types/) 作成
  [ ] 型なしサードパーティの .d.ts 作成
  [ ] エディタの TypeScript 設定確認

Phase 3 変換:
  [ ] ユーティリティファイルから .ts 変換開始
  [ ] require → import 変換
  [ ] module.exports → export 変換
  [ ] any を具体的な型に置換
  [ ] テスト実行・CI で型チェック
  [ ] 1ファイルずつコミット

Phase 4 厳密化:
  [ ] noImplicitAny: true
  [ ] strictNullChecks: true
  [ ] strictFunctionTypes: true
  [ ] strict: true
  [ ] noUncheckedIndexedAccess: true

Phase 5 完了:
  [ ] 全ファイル .ts 化
  [ ] allowJs: false
  [ ] checkJs 削除
  [ ] JSDoc 型アノテーション除去
  [ ] prop-types 削除（React）
  [ ] CI で strict ビルドを必須に
  [ ] any カウントが 0 であることを確認
  [ ] チームの TypeScript コーディングガイドライン作成
```

---

## FAQ

### Q1: 移行にどのくらいの期間がかかりますか？

規模によります。1万行程度なら 1〜2 週間、10 万行なら 1〜3 ヶ月が目安です。重要なのは「完全な移行」を待たずに、Phase 2（共存）の時点で既に型チェックの恩恵を受けられることです。移行は「終わり」がある作業ではなく、型の品質を継続的に改善するプロセスです。

### Q2: 既存のテストは動き続けますか？

はい。`allowJs: true` の状態では既存の .js ファイルはそのまま動きます。ファイルを .ts に変換しても、テストランナーが TypeScript をサポートしていれば（Vitest, Jest + ts-jest など）テストは継続して動作します。

### Q3: チームメンバーが TypeScript を知らない場合はどうすべきですか？

Phase 2（JSDoc 型アノテーション）から始めることで、TypeScript の構文を学ばずに型の恩恵を受けられます。並行して TypeScript の基本を学ぶ学習時間を確保し、新機能の開発は TypeScript で行う方針にすると自然に習熟していきます。ペアプログラミングで知識を共有することも効果的です。

### Q4: 移行中に新機能の開発はどうすべきですか？

新機能は最初から TypeScript で開発してください。新しいファイルは .ts で作成し、strict: true の設定で書きます。既存の JS ファイルとの連携は JSDoc や .d.ts ファイルで橋渡しします。これにより「新しいコードは常に型安全」という基準を維持できます。

### Q5: monorepo の場合、どのパッケージから移行すべきですか？

1. 共有ライブラリ（shared パッケージ）から始める -- 他のパッケージに型情報が伝播する
2. 次にバックエンド（型が最も重要な箇所）
3. 最後にフロントエンド（PropTypes → TypeScript 型への移行が必要）

---

## まとめ表

| 概念 | 要点 |
|------|------|
| 段階的移行 | ファイル単位で変換、依存の葉から開始 |
| allowJs | JS と TS の共存を可能にする設定 |
| checkJs | JS ファイルの型チェックを有効化 |
| JSDoc 型 | .ts 変換前に JSDoc で型を追加 |
| any 排除 | unknown + zod で段階的に型安全化 |
| strict 化 | noImplicitAny → strictNullChecks → strict |
| 移行進捗 | スクリプトで .ts/.js の比率を可視化 |
| 新機能は TS | 移行中でも新しいコードは TypeScript で |

---

## 次に読むべきガイド

- [tsconfig.json](./00-tsconfig.md) -- 移行各フェーズの推奨設定詳細
- [ESLint + TypeScript](./04-eslint-typescript.md) -- 移行中の lint ルール設定
- [Zod バリデーション](../04-ecosystem/00-zod-validation.md) -- any 排除の強力なツール

---

## 参考文献

1. **TypeScript - Migrating from JavaScript**
   https://www.typescriptlang.org/docs/handbook/migrating-from-javascript.html

2. **Total TypeScript - Migrating to TypeScript**
   https://www.totaltypescript.com/tutorials/migrating-to-typescript

3. **Airbnb の TypeScript 移行記** -- Brie Bunge, JSConf 2019
   大規模 JS→TS 移行の実践レポート

4. **ts-morph** -- TypeScript コードのプログラム的な操作
   https://github.com/dsherret/ts-morph
