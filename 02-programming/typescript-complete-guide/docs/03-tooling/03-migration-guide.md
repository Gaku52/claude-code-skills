# JavaScript から TypeScript への移行ガイド

> 既存 JS プロジェクトを段階的に TypeScript 化する実践的なロードマップと移行テクニック

## この章で学ぶこと

1. **段階的移行戦略** -- 全面書き換えではなく、ファイル単位で安全に TypeScript を導入する手順
2. **tsconfig の段階的厳密化** -- `allowJs` から始めて `strict: true` に到達するまでの設定チェーンの管理
3. **よくある移行パターン** -- 型定義の補完、any の排除、サードパーティ型の導入テクニック

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
```

### 1-2. Phase 1 -- 準備

```bash
# TypeScript と関連ツールをインストール
npm install -D typescript @types/node

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
    "noEmit": true
  },
  "include": ["src/**/*"]
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

module.exports = { createUser, initApp };
```

```json
// Phase 2 の tsconfig.json 更新
{
  "compilerOptions": {
    "allowJs": true,
    "checkJs": true,  // JS ファイルも型チェック
    "strict": false
  }
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
}

module.exports = { UserService };

// ─────────────────────────────────────────────

// After: src/user-service.ts
import { Database } from "./database";

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
  created_at: string;
}

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
}

export { UserService, type User, type CreateUserDto };
```

### 2-3. require → import の変換

```typescript
// Before (CommonJS)
const express = require("express");
const { UserService } = require("./user-service");
const config = require("./config.json");

// After (ESM)
import express from "express";
import { UserService } from "./user-service.js"; // NodeNext なら拡張子必要
import config from "./config.json" with { type: "json" };

// ──────────────────────────────────────

// Before (module.exports)
module.exports = { UserService };
module.exports.default = app;

// After (export)
export { UserService };
export default app;
```

---

## 3. any の段階的排除

### 3-1. any のトリアージ

```
any 排除の優先順位:

  +------------------+----------+-----------+
  | カテゴリ          | 危険度   | 対応      |
  +------------------+----------+-----------+
  | API レスポンス    | 高       | zod 導入  |
  | 関数パラメータ    | 高       | 型定義    |
  | イベントハンドラ  | 中       | 型定義    |
  | catch 変数       | 中       | unknown   |
  | サードパーティ    | 低       | @types/*  |
  | 一時的な TODO     | 低       | コメント  |
  +------------------+----------+-----------+
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
```

### 3-2. @types/* の導入

```bash
# 型定義パッケージを検索・インストール
npm install -D @types/express @types/lodash @types/cors

# 型定義がないパッケージには declaration ファイルを作成
```

```typescript
// src/types/untyped-module.d.ts
// 型定義がないサードパーティモジュール用

// 最小限の型定義（一時的）
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
  }
}
```

---

## 4. strict 化のロードマップ

### 4-1. 段階的に有効化

```json
// Phase 3: 基本の strict オプション
{
  "compilerOptions": {
    "strict": false,
    "noImplicitAny": true,        // Step 1
    "strictNullChecks": true,      // Step 2
    "strictFunctionTypes": true    // Step 3
  }
}

// Phase 4: 完全な strict
{
  "compilerOptions": {
    "strict": true,                // 全て有効
    "noUncheckedIndexedAccess": true, // 追加の厳密性
    "exactOptionalPropertyTypes": true
  }
}
```

### 4-2. noImplicitAny 対応パターン

```typescript
// エラー: Parameter 'x' implicitly has an 'any' type
// 対応パターン集

// 1. コールバック引数
// Before
array.forEach(function (item) { ... });
// After
array.forEach(function (item: ItemType) { ... });
// もしくはアロー関数（型推論が効く場合）
array.forEach((item) => { ... });

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
element.addEventListener("click", function (e) { ... });
// After（DOM の型定義から自動推論）
element.addEventListener("click", (e: MouseEvent) => { ... });
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

### any 排除ツールの比較

| ツール | 用途 | 自動修正 | 精度 |
|--------|------|---------|------|
| `tsc --strict` | 暗黙 any 検出 | なし | 高 |
| `@typescript-eslint/no-explicit-any` | 明示 any 検出 | なし | 高 |
| `ts-prune` | 未使用 export 検出 | なし | 中 |
| zod | ランタイム型検証 | 型推論 | 最高 |
| TypeStat | 自動型追加 | あり | 中 |

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
# 3. ...
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

---

## 移行チェックリスト

```
Phase 1 準備:
  [ ] TypeScript インストール
  [ ] tsconfig.json 作成 (allowJs: true, strict: false)
  [ ] ビルドパイプラインに tsc --noEmit 追加
  [ ] @types/* パッケージインストール

Phase 2 共存:
  [ ] checkJs: true 有効化
  [ ] JSDoc で主要関数に型アノテーション
  [ ] 共通の型定義ファイル (types/) 作成

Phase 3 変換:
  [ ] ユーティリティファイルから .ts 変換開始
  [ ] require → import 変換
  [ ] any を具体的な型に置換
  [ ] テスト実行・CI で型チェック

Phase 4 厳密化:
  [ ] noImplicitAny: true
  [ ] strictNullChecks: true
  [ ] strict: true
  [ ] noUncheckedIndexedAccess: true

Phase 5 完了:
  [ ] 全ファイル .ts 化
  [ ] allowJs: false
  [ ] JSDoc 型アノテーション除去
  [ ] CI で strict ビルドを必須に
```

---

## FAQ

### Q1: 移行にどのくらいの期間がかかりますか？

規模によります。1万行程度なら 1〜2 週間、10 万行なら 1〜3 ヶ月が目安です。重要なのは「完全な移行」を待たずに、Phase 2（共存）の時点で既に型チェックの恩恵を受けられることです。

### Q2: 既存のテストは動き続けますか？

はい。`allowJs: true` の状態では既存の .js ファイルはそのまま動きます。ファイルを .ts に変換しても、テストランナーが TypeScript をサポートしていれば（Vitest, Jest + ts-jest など）テストは継続して動作します。

### Q3: チームメンバーが TypeScript を知らない場合はどうすべきですか？

Phase 2（JSDoc 型アノテーション）から始めることで、TypeScript の構文を学ばずに型の恩恵を受けられます。並行して TypeScript の基本を学ぶ学習時間を確保し、新機能の開発は TypeScript で行う方針にすると自然に習熟していきます。

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
