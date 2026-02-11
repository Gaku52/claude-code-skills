# プロジェクト構成

> プロジェクト構成は開発チームの生産性を決定づける。Feature-based構成、レイヤードアーキテクチャ、モノレポ設計まで、スケーラブルで保守しやすいディレクトリ設計の原則とパターンを習得する。

## この章で学ぶこと

- [ ] Feature-basedなディレクトリ設計を理解する
- [ ] レイヤードアーキテクチャの適用方法を把握する
- [ ] モノレポとパッケージ分割の設計を学ぶ

---

## 1. ディレクトリ設計の原則

```
悪いパターン（型ベース）:
  src/
  ├── components/         ← 全コンポーネントが混在
  │   ├── Button.tsx
  │   ├── UserList.tsx
  │   ├── OrderTable.tsx
  │   └── ... (100+)
  ├── hooks/              ← 全hooksが混在
  ├── utils/              ← 全ユーティリティが混在
  └── types/              ← 全型が混在

  問題:
  ✗ ファイル数が増えると見通しが悪い
  ✗ 関連するコードが分散（User関連が3箇所に）
  ✗ 依存関係が不明確
  ✗ 削除・リファクタが困難

良いパターン（Feature-based）:
  src/
  ├── features/           ← 機能ごとにまとめる
  │   ├── users/
  │   │   ├── components/
  │   │   │   ├── UserList.tsx
  │   │   │   └── UserCard.tsx
  │   │   ├── hooks/
  │   │   │   └── useUsers.ts
  │   │   ├── api/
  │   │   │   └── users.ts
  │   │   ├── types/
  │   │   │   └── user.ts
  │   │   └── index.ts    ← 公開APIの定義
  │   ├── orders/
  │   │   ├── components/
  │   │   ├── hooks/
  │   │   ├── api/
  │   │   └── index.ts
  │   └── auth/
  │       ├── components/
  │       ├── hooks/
  │       └── index.ts
  ├── shared/             ← 共有コンポーネント
  │   ├── components/
  │   │   ├── Button.tsx
  │   │   ├── Modal.tsx
  │   │   └── Table.tsx
  │   ├── hooks/
  │   ├── utils/
  │   └── types/
  └── app/                ← ルーティング・レイアウト
      ├── layout.tsx
      └── (routes)/
```

---

## 2. Next.js App Router のプロジェクト構成

```
推奨構成（Next.js 14+）:

  project-root/
  ├── src/
  │   ├── app/                    ← ルーティング（App Router）
  │   │   ├── layout.tsx
  │   │   ├── page.tsx
  │   │   ├── error.tsx
  │   │   ├── loading.tsx
  │   │   ├── not-found.tsx
  │   │   ├── (marketing)/       ← ルートグループ
  │   │   │   ├── page.tsx       ← /
  │   │   │   ├── about/page.tsx ← /about
  │   │   │   └── pricing/page.tsx
  │   │   ├── (app)/             ← 認証必要エリア
  │   │   │   ├── layout.tsx     ← 認証チェック
  │   │   │   ├── dashboard/page.tsx
  │   │   │   └── settings/page.tsx
  │   │   └── api/               ← API Routes
  │   │       └── webhooks/
  │   ├── features/              ← 機能モジュール
  │   │   ├── users/
  │   │   ├── orders/
  │   │   └── auth/
  │   ├── shared/                ← 共有リソース
  │   │   ├── components/
  │   │   │   └── ui/            ← shadcn/ui等
  │   │   ├── hooks/
  │   │   ├── lib/               ← ユーティリティ
  │   │   │   ├── api-client.ts
  │   │   │   ├── auth.ts
  │   │   │   └── utils.ts
  │   │   └── types/
  │   └── styles/
  │       └── globals.css
  ├── public/
  ├── prisma/                    ← Prisma スキーマ
  │   └── schema.prisma
  ├── tests/
  ├── next.config.js
  ├── tailwind.config.ts
  └── tsconfig.json

ルール:
  ① app/ にはルーティングとページコンポーネントのみ
  ② ビジネスロジックは features/ に配置
  ③ 2つ以上の feature で使うものは shared/ に
  ④ features間の直接import は禁止（shared 経由）
```

---

## 3. Featureモジュールの設計

```typescript
// features/users/index.ts — 公開API
// ← このファイルからのみ外部アクセス可能

// Components
export { UserList } from './components/UserList';
export { UserCard } from './components/UserCard';
export { UserAvatar } from './components/UserAvatar';

// Hooks
export { useUsers } from './hooks/useUsers';
export { useUser } from './hooks/useUser';

// Types
export type { User, CreateUserInput } from './types/user';

// 内部ファイル構成
// features/users/
// ├── components/
// │   ├── UserList.tsx          ← Server Component
// │   ├── UserCard.tsx
// │   ├── UserAvatar.tsx
// │   └── UserSearchInput.tsx   ← Client Component
// ├── hooks/
// │   ├── useUsers.ts
// │   └── useUser.ts
// ├── api/
// │   ├── queries.ts            ← TanStack Query のキー定義
// │   └── actions.ts            ← Server Actions
// ├── types/
// │   └── user.ts
// ├── utils/
// │   └── format.ts             ← feature固有のユーティリティ
// └── index.ts                  ← 公開API

// --- 依存ルール ---
// features/users/ → shared/     ✓ OK
// features/users/ → features/orders/  ✗ NG（直接参照禁止）
// app/ → features/users/        ✓ OK
// app/ → shared/                ✓ OK
```

---

## 4. パスエイリアス

```json
// tsconfig.json
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@/*": ["./src/*"],
      "@features/*": ["./src/features/*"],
      "@shared/*": ["./src/shared/*"],
      "@app/*": ["./src/app/*"]
    }
  }
}
```

```typescript
// 使用例
import { UserList } from '@features/users';
import { Button } from '@shared/components/ui/Button';
import { formatDate } from '@shared/lib/utils';
```

---

## 5. モノレポ設計

```
モノレポ構成（Turborepo）:

  monorepo/
  ├── apps/
  │   ├── web/            ← Next.js フロントエンド
  │   ├── admin/          ← 管理画面
  │   ├── api/            ← バックエンドAPI
  │   └── docs/           ← ドキュメント
  ├── packages/
  │   ├── ui/             ← 共有UIコンポーネント
  │   ├── db/             ← Prismaスキーマ + クライアント
  │   ├── config/         ← ESLint, TSConfig共有
  │   │   ├── eslint/
  │   │   └── typescript/
  │   ├── types/          ← 共有型定義
  │   └── utils/          ← 共有ユーティリティ
  ├── turbo.json
  ├── package.json
  └── pnpm-workspace.yaml

ツール比較:
  Turborepo:  キャッシュが強力、設定が少ない（推奨）
  Nx:         プラグイン豊富、大規模向け
  Lerna:      古い、Nx に統合
  pnpm workspace: シンプル、最小構成

モノレポの利点:
  ✓ コードの共有が容易
  ✓ 依存のバージョンを統一
  ✓ 横断的なリファクタリング
  ✓ 統一されたCI/CD

モノレポの注意点:
  → パッケージ間の依存方向を管理
  → ビルドキャッシュの活用
  → 影響範囲を限定したテスト実行
```

---

## 6. 設計チェックリスト

```
プロジェクト構成チェックリスト:

  構造:
  □ Feature-based でモジュール分割されているか
  □ 各featureが index.ts で公開APIを定義しているか
  □ features間の直接importが禁止されているか
  □ 共有コンポーネントが shared/ にあるか
  □ app/ にはルーティングのみか

  命名:
  □ ファイル名が一貫しているか（PascalCase or kebab-case）
  □ パスエイリアスが設定されているか
  □ テストファイルが対象の近くにあるか

  スケーラビリティ:
  □ 新機能追加が既存コードに影響しないか
  □ featureの削除が容易か
  □ チームメンバーが迷わず配置できるか
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| Feature-based | 機能ごとにディレクトリを分割 |
| 公開API | index.ts で外部向けインターフェースを定義 |
| 依存ルール | features間は直接参照禁止 |
| モノレポ | Turborepo + pnpm workspace |
| パスエイリアス | @features/, @shared/ で可読性向上 |

---

## 次に読むべきガイド
→ [[02-component-architecture.md]] — コンポーネント設計

---

## 参考文献
1. Bulletproof React. "Project Structure." github.com/alan2207/bulletproof-react, 2024.
2. Next.js. "Project Organization." nextjs.org/docs, 2024.
3. Turborepo. "Getting Started." turbo.build, 2024.
