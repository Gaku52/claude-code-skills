# npmパッケージ開発

> npmパッケージの設計から公開までの全工程。package.jsonの設計、ESM/CJSデュアルパッケージ、TypeScript設定、ビルドパイプライン、バージョニング、公開ワークフローまで、プロフェッショナルなパッケージ開発を習得する。

## この章で学ぶこと

- [ ] package.jsonの設計とexportsフィールドを理解する
- [ ] ESM/CJSデュアルパッケージの構築方法を把握する
- [ ] セマンティックバージョニングと公開ワークフローを学ぶ

---

## 1. package.json設計

```json
{
  "name": "@example/sdk",
  "version": "1.0.0",
  "description": "Official SDK for Example API",
  "license": "MIT",

  "type": "module",

  "exports": {
    ".": {
      "import": {
        "types": "./dist/index.d.ts",
        "default": "./dist/index.js"
      },
      "require": {
        "types": "./dist/index.d.cts",
        "default": "./dist/index.cjs"
      }
    },
    "./users": {
      "import": {
        "types": "./dist/users/index.d.ts",
        "default": "./dist/users/index.js"
      },
      "require": {
        "types": "./dist/users/index.d.cts",
        "default": "./dist/users/index.cjs"
      }
    }
  },

  "main": "./dist/index.cjs",
  "module": "./dist/index.js",
  "types": "./dist/index.d.ts",

  "files": ["dist", "README.md", "LICENSE"],

  "engines": { "node": ">=18.0.0" },

  "sideEffects": false,

  "keywords": ["api", "sdk", "example"],
  "repository": { "type": "git", "url": "https://github.com/example/sdk" },
  "homepage": "https://example.com/docs/sdk",
  "bugs": { "url": "https://github.com/example/sdk/issues" },

  "scripts": {
    "build": "tsup",
    "test": "vitest run",
    "test:watch": "vitest",
    "lint": "eslint src/",
    "typecheck": "tsc --noEmit",
    "prepublishOnly": "npm run build && npm run test && npm run typecheck",
    "release": "changeset publish"
  },

  "devDependencies": {
    "tsup": "^8.0.0",
    "typescript": "^5.4.0",
    "vitest": "^2.0.0",
    "@changesets/cli": "^2.27.0",
    "eslint": "^9.0.0"
  },

  "peerDependencies": {},
  "dependencies": {}
}
```

```
package.json の重要フィールド:

  "type": "module"
  → パッケージのデフォルトをESMに

  "exports"（Node.js 12+）:
  → パッケージのエントリポイントを明示的に定義
  → サブパス: import { Users } from '@example/sdk/users'
  → 条件分岐: import/require で異なるファイル
  → types は各条件の先頭に置く

  "files":
  → npm publish に含めるファイル
  → dist, README.md, LICENSE のみ（ソースは含めない）

  "sideEffects": false
  → Tree-shaking 可能であることを宣言
  → バンドラーが未使用コードを除去可能

  "engines":
  → 動作するNode.jsバージョンを明示
  → npm install 時に警告
```

---

## 2. tsup によるビルド

```typescript
// tsup.config.ts
import { defineConfig } from 'tsup';

export default defineConfig({
  entry: ['src/index.ts', 'src/users/index.ts'],
  format: ['esm', 'cjs'],    // ESM と CJS を両方出力
  dts: true,                  // 型定義ファイルを生成
  splitting: true,            // コード分割
  sourcemap: true,            // ソースマップ
  clean: true,                // ビルド前にdistをクリア
  minify: false,              // SDKはminifyしない（デバッグのため）
  target: 'es2022',
  outDir: 'dist',
  external: [],               // 外部依存（バンドルしない）
});
```

```
ビルドツールの比較:

  tsup:
  → esbuild ベース、高速
  → ESM + CJS + DTS を1コマンド
  → 設定が少なく、SDK開発に最適

  rollup:
  → 柔軟な設定
  → Tree-shaking が優秀
  → プラグインエコシステムが豊富

  esbuild:
  → 最速のビルド
  → 型定義の生成は別途必要
  → CJSの出力に制限あり

  tsc:
  → TypeScript公式
  → ESMとCJSを別々にビルド
  → 遅いが最も正確

推奨: tsup（バランスが良い）
```

---

## 3. TypeScript設定

```json
// tsconfig.json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "lib": ["ES2022"],
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,
    "outDir": "dist",
    "rootDir": "src",
    "isolatedModules": true,
    "resolveJsonModule": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "forceConsistentCasingInFileNames": true
  },
  "include": ["src"],
  "exclude": ["node_modules", "dist", "**/*.test.ts"]
}
```

---

## 4. ゼロ依存設計

```
SDK依存関係の原則:

  理想: ゼロ依存（dependencies が空）
  → インストールサイズが小さい
  → バージョン競合が起きない
  → セキュリティリスクが低い

  fetch():
  → Node.js 18+ のグローバルfetchを使用
  → 古い環境では undici / node-fetch を peerDependencies に

  避けるべき依存:
  ✗ axios（fetch で代替可能）
  ✗ lodash（必要な関数だけ実装）
  ✗ moment.js（Intl API / date-fns で代替）

  許容される依存:
  ○ 暗号系（crypto / jose）
  ○ WebSocket（ws）— Node.js環境用
  ○ Protocol Buffers（protobuf.js）— gRPC用

依存を減らすテクニック:
  → URLSearchParams（qsの代替）
  → structuredClone（deep cloneの代替）
  → AbortController（キャンセル処理）
  → crypto.randomUUID()（uuidの代替）
```

---

## 5. セマンティックバージョニング

```
SemVer: MAJOR.MINOR.PATCH

  MAJOR（1.0.0 → 2.0.0）: 破壊的変更
  → 公開メソッドの削除/リネーム
  → 引数の型変更
  → デフォルト動作の変更

  MINOR（1.0.0 → 1.1.0）: 後方互換の機能追加
  → 新しいメソッド/クラスの追加
  → オプショナルパラメータの追加
  → 新しいイベントの追加

  PATCH（1.0.0 → 1.0.1）: バグ修正
  → バグ修正
  → パフォーマンス改善
  → 依存の更新

プレリリース:
  1.0.0-alpha.1  → アルファ版
  1.0.0-beta.1   → ベータ版
  1.0.0-rc.1     → リリース候補

Changesets（推奨）:
  → モノレポ対応のバージョン管理
  → PRごとに変更の種類を記録
  → リリース時に自動でバージョンアップ + CHANGELOG生成
```

```bash
# Changesets ワークフロー
npx changeset init

# 変更を記録
npx changeset
# → パッケージ選択
# → 変更の種類（major / minor / patch）
# → 変更の説明

# バージョンアップ + CHANGELOG更新
npx changeset version

# 公開
npx changeset publish
```

---

## 6. テスト戦略

```typescript
// vitest.config.ts
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    coverage: {
      provider: 'v8',
      thresholds: { branches: 80, functions: 80, lines: 80 },
    },
  },
});

// --- ユニットテスト ---
// src/users/__tests__/users.test.ts
import { describe, it, expect, beforeAll, afterAll, afterEach } from 'vitest';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import { ExampleClient } from '../../index';

const server = setupServer();

beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

describe('UsersResource', () => {
  const client = new ExampleClient({ apiKey: 'test-key' });

  it('should get a user by id', async () => {
    server.use(
      http.get('https://api.example.com/v1/users/123', () => {
        return HttpResponse.json({
          id: '123', name: 'Taro', email: 'taro@example.com', role: 'user',
        });
      }),
    );

    const user = await client.users.get('123');
    expect(user.name).toBe('Taro');
    expect(user.id).toBe('123');
  });

  it('should handle 404 errors', async () => {
    server.use(
      http.get('https://api.example.com/v1/users/999', () => {
        return HttpResponse.json(
          { code: 'NOT_FOUND', detail: 'User not found' },
          { status: 404 },
        );
      }),
    );

    await expect(client.users.get('999')).rejects.toThrow('User not found');
  });

  it('should retry on 500 errors', async () => {
    let attempts = 0;
    server.use(
      http.get('https://api.example.com/v1/users/123', () => {
        attempts++;
        if (attempts < 3) {
          return HttpResponse.json({}, { status: 500 });
        }
        return HttpResponse.json({ id: '123', name: 'Taro' });
      }),
    );

    const user = await client.users.get('123');
    expect(user.name).toBe('Taro');
    expect(attempts).toBe(3);
  });
});
```

---

## 7. 公開ワークフロー

```
公開チェックリスト:
  □ テストが全て通る
  □ 型チェックが通る
  □ lint エラーがない
  □ CHANGELOG が更新されている
  □ README が最新
  □ package.json の version が正しい
  □ npm pack で内容を確認
  □ .npmignore または files フィールドが正しい

公開手順:
  1. npm login
  2. npm pack --dry-run  ← 含まれるファイルの確認
  3. npm publish --access public  ← スコープ付きパッケージの場合

GitHub Actions での自動公開:
  → main ブランチへのマージで自動リリース
  → changeset-bot がPRにラベル付け
  → changeset version でバージョン更新
  → changeset publish で npm に公開
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| package.json | exports で ESM/CJS 対応 |
| ビルド | tsup が SDK開発に最適 |
| 依存 | ゼロ依存を目指す |
| バージョン | SemVer + Changesets |
| テスト | MSW でHTTPレベルのモック |
| 公開 | GitHub Actions + Changesets |

---

## 次に読むべきガイド
→ [[02-api-documentation.md]] — APIドキュメンテーション

---

## 参考文献
1. npm. "package.json documentation." docs.npmjs.com, 2024.
2. tsup. "Bundle your TypeScript library." github.com/egoist/tsup, 2024.
3. Changesets. "A way to manage your versioning." github.com/changesets, 2024.
