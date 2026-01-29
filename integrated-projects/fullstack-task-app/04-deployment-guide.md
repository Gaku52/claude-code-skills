# デプロイガイド

## 目次

1. [概要](#概要)
2. [事前準備](#事前準備)
3. [バックエンドデプロイ（Render）](#バックエンドデプロイrender)
4. [フロントエンドデプロイ（Vercel）](#フロントエンドデプロイvercel)
5. [環境変数設定](#環境変数設定)
6. [動作確認](#動作確認)
7. [トラブルシューティング](#トラブルシューティング)
8. [CI/CD設定](#cicd設定)

---

## 概要

### デプロイ先

| サービス | 用途 | 理由 |
|---------|------|------|
| **Render** | バックエンド + データベース | 無料プラン、PostgreSQL統合、簡単デプロイ |
| **Vercel** | フロントエンド | 無料プラン、React最適化、自動デプロイ |

### デプロイフロー

```mermaid
graph TB
    A[GitHub Repository] -->|Push| B[Render - Backend]
    A -->|Push| C[Vercel - Frontend]
    B --> D[PostgreSQL Database]
    C -->|API Request| B
```

### 所要時間：1-2時間

---

## 事前準備

### 1. GitHubリポジトリ作成

```bash
# プロジェクトルートで
git init
git add .
git commit -m "Initial commit"

# GitHubでリポジトリ作成後
git remote add origin https://github.com/your-username/fullstack-task-app.git
git push -u origin main
```

### 2. .gitignoreファイル確認

**プロジェクトルート `.gitignore`:**

```
# フロントエンド
frontend/node_modules
frontend/dist
frontend/.env.local
frontend/.env.production

# バックエンド
backend/node_modules
backend/dist
backend/.env

# OS
.DS_Store
Thumbs.db
```

### 3. アカウント作成

- [Render](https://render.com/) - GitHub連携でサインアップ
- [Vercel](https://vercel.com/) - GitHub連携でサインアップ

---

## バックエンドデプロイ（Render）

### ステップ1：PostgreSQLデータベース作成

1. Renderダッシュボードにログイン
2. **New** → **PostgreSQL** をクリック
3. 設定入力：
   ```
   Name: taskapp-db
   Region: Singapore (最寄りのリージョン)
   PostgreSQL Version: 15
   Plan: Free
   ```
4. **Create Database** をクリック
5. **Internal Database URL** をコピー（後で使用）

### ステップ2：バックエンドのビルド設定

`backend/package.json`に以下を追加：

```json
{
  "scripts": {
    "dev": "ts-node-dev --respawn --transpile-only src/server.ts",
    "build": "tsc",
    "start": "node dist/server.js",
    "postinstall": "prisma generate",
    "prisma:migrate": "prisma migrate deploy"
  }
}
```

### ステップ3：Prisma設定更新

`backend/prisma/schema.prisma`：

```prisma
generator client {
  provider = "prisma-client-js"
  binaryTargets = ["native", "debian-openssl-3.0.x"]
}

datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")
}

// ... モデル定義 ...
```

### ステップ4：Web Service作成

1. Renderダッシュボードで **New** → **Web Service** をクリック
2. GitHubリポジトリを選択
3. 設定入力：

```
Name: taskapp-api
Environment: Node
Region: Singapore
Branch: main
Root Directory: backend
Build Command: npm install && npm run build && npx prisma migrate deploy
Start Command: npm start
Plan: Free
```

### ステップ5：環境変数設定

Render Web Serviceの設定で **Environment** タブを開き、以下を追加：

```env
NODE_ENV=production
PORT=10000
DATABASE_URL=[先ほどコピーしたInternal Database URL]
JWT_SECRET=[ランダムな文字列を生成]
JWT_EXPIRES_IN=7d
CORS_ORIGIN=[後でVercelのURLを設定]
```

**JWT_SECRET生成方法:**

```bash
# ターミナルで実行
openssl rand -base64 32
# 出力された文字列をコピー
```

### ステップ6：デプロイ

1. **Create Web Service** をクリック
2. ビルド開始（5-10分かかる）
3. デプロイ完了後、URLをコピー（例：`https://taskapp-api.onrender.com`）

### ステップ7：動作確認

```bash
# ヘルスチェック
curl https://taskapp-api.onrender.com

# レスポンス例
{
  "message": "Task App API",
  "version": "1.0.0",
  "status": "healthy"
}
```

---

## フロントエンドデプロイ（Vercel）

### ステップ1：ビルド設定

`frontend/package.json`確認：

```json
{
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview"
  }
}
```

### ステップ2：環境変数ファイル作成

`frontend/.env.production`：

```env
VITE_API_URL=https://taskapp-api.onrender.com
```

**重要:** このファイルは.gitignoreに追加してGitHubにプッシュしない。代わりにVercel側で設定します。

### ステップ3：Vercelプロジェクト作成

1. [Vercel Dashboard](https://vercel.com/dashboard) にログイン
2. **Add New** → **Project** をクリック
3. GitHubリポジトリを選択
4. 設定入力：

```
Project Name: taskapp-frontend
Framework Preset: Vite
Root Directory: frontend
Build Command: npm run build
Output Directory: dist
Install Command: npm install
```

### ステップ4：環境変数設定

Vercelプロジェクト設定で **Settings** → **Environment Variables** を開き、以下を追加：

```
VITE_API_URL = https://taskapp-api.onrender.com
```

**Environment:** Production にチェック

### ステップ5：デプロイ

1. **Deploy** をクリック
2. ビルド開始（2-3分）
3. デプロイ完了後、URLをコピー（例：`https://taskapp-frontend.vercel.app`）

### ステップ6：バックエンドCORS設定更新

Renderに戻り、バックエンドの環境変数を更新：

```env
CORS_ORIGIN=https://taskapp-frontend.vercel.app
```

**Manual Deploy** → **Deploy latest commit** でバックエンドを再デプロイ。

---

## 環境変数設定

### バックエンド（Render）

| 変数名 | 値 | 説明 |
|--------|-----|------|
| NODE_ENV | production | 本番環境フラグ |
| PORT | 10000 | Renderデフォルトポート |
| DATABASE_URL | [PostgreSQL URL] | データベース接続URL |
| JWT_SECRET | [ランダム文字列] | JWT署名キー |
| JWT_EXPIRES_IN | 7d | トークン有効期限 |
| CORS_ORIGIN | [Vercel URL] | CORS許可オリジン |

### フロントエンド（Vercel）

| 変数名 | 値 | 説明 |
|--------|-----|------|
| VITE_API_URL | [Render API URL] | バックエンドAPI URL |

---

## 動作確認

### 1. バックエンド確認

```bash
# ヘルスチェック
curl https://taskapp-api.onrender.com

# ユーザー登録テスト
curl -X POST https://taskapp-api.onrender.com/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "Test1234",
    "name": "テストユーザー"
  }'
```

### 2. フロントエンド確認

1. ブラウザで `https://taskapp-frontend.vercel.app` にアクセス
2. ユーザー登録
3. ログイン
4. タスク作成
5. ダッシュボード確認

### 3. ログ確認

#### Render
- ダッシュボード → Web Service → **Logs** タブ

#### Vercel
- プロジェクト → **Deployments** → 最新デプロイ → **Function Logs**

---

## トラブルシューティング

### ❌ 問題1：バックエンドビルドエラー

```
Error: Cannot find module '@prisma/client'
```

**解決策:**

`package.json`に追加：
```json
{
  "scripts": {
    "postinstall": "prisma generate"
  }
}
```

### ❌ 問題2：データベース接続エラー

```
Error: Can't reach database server
```

**解決策:**
1. Render PostgreSQLの **Internal Database URL** を使用していることを確認
2. DATABASE_URLフォーマット確認：
   ```
   postgresql://user:password@host:port/database
   ```

### ❌ 問題3：Prismaマイグレーションエラー

```
Error: Prisma schema loaded from prisma/schema.prisma is invalid
```

**解決策:**

`schema.prisma`の`binaryTargets`を追加：
```prisma
generator client {
  provider = "prisma-client-js"
  binaryTargets = ["native", "debian-openssl-3.0.x"]
}
```

### ❌ 問題4：CORS エラー

```
Access to fetch has been blocked by CORS policy
```

**解決策:**
1. バックエンドの`CORS_ORIGIN`環境変数を確認
2. 正しいVercel URLを設定
3. バックエンドを再デプロイ

### ❌ 問題5：環境変数が反映されない

**解決策（Vercel）:**
1. Settings → Environment Variables で確認
2. 変更後、**Deployments** → **Redeploy**

**解決策（Render）:**
1. Environment タブで確認
2. 変更後、**Manual Deploy**

### ❌ 問題6：Renderサービスがスリープする

**症状:** Free Planでは15分アクセスがないとスリープ、次回アクセス時に起動まで遅延

**解決策:**
1. Paid Planにアップグレード
2. または、定期的にヘルスチェックを送信（例：UptimeRobot）

---

## CI/CD設定

### 自動デプロイ設定

#### Render（自動）
- `main`ブランチへのプッシュで自動デプロイ
- 設定済み

#### Vercel（自動）
- `main`ブランチへのプッシュで自動デプロイ
- プルリクエストごとにプレビューデプロイ
- 設定済み

### デプロイフロー

```
1. ローカルで開発
   ↓
2. git push origin main
   ↓
3. 自動デプロイ開始
   ├─ Render: バックエンド + DB
   └─ Vercel: フロントエンド
   ↓
4. デプロイ完了通知
```

### プレビューデプロイ（Vercel）

1. フィーチャーブランチ作成
   ```bash
   git checkout -b feature/new-feature
   ```
2. コミット & プッシュ
   ```bash
   git add .
   git commit -m "Add new feature"
   git push origin feature/new-feature
   ```
3. GitHubでプルリクエスト作成
4. Vercelが自動でプレビューURL生成
5. レビュー後、マージで本番デプロイ

---

## カスタムドメイン設定（オプション）

### フロントエンド（Vercel）

1. Vercelプロジェクト → **Settings** → **Domains**
2. ドメインを入力（例：`taskapp.com`）
3. DNS設定を追加：
   ```
   Type: A
   Name: @
   Value: 76.76.21.21

   Type: CNAME
   Name: www
   Value: cname.vercel-dns.com
   ```

### バックエンド（Render）

1. Render Web Service → **Settings** → **Custom Domain**
2. ドメインを入力（例：`api.taskapp.com`）
3. DNS設定を追加：
   ```
   Type: CNAME
   Name: api
   Value: [Render提供のCNAME]
   ```

---

## セキュリティチェックリスト

デプロイ前に以下を確認：

- [ ] **環境変数**: `.env`ファイルをGitHubにプッシュしていない
- [ ] **JWT_SECRET**: 強力なランダム文字列を使用
- [ ] **パスワード**: bcryptでハッシュ化されている
- [ ] **CORS**: 必要なオリジンのみ許可
- [ ] **HTTPS**: 全ての通信がHTTPSで暗号化されている
- [ ] **データベース**: Renderの Internal URL を使用
- [ ] **バリデーション**: Zodでユーザー入力を検証
- [ ] **エラーメッセージ**: 本番環境で詳細を表示しない

---

## パフォーマンス最適化

### フロントエンド

```javascript
// vite.config.ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom', 'react-router-dom'],
          query: ['@tanstack/react-query'],
        },
      },
    },
  },
})
```

### バックエンド

```typescript
// src/server.ts
import compression from 'compression'

app.use(compression()) // Gzip圧縮
```

---

## モニタリング設定（オプション）

### Sentry統合

#### バックエンド

```bash
npm install @sentry/node
```

```typescript
// src/server.ts
import * as Sentry from '@sentry/node'

Sentry.init({
  dsn: process.env.SENTRY_DSN,
  environment: process.env.NODE_ENV,
})

app.use(Sentry.Handlers.errorHandler())
```

#### フロントエンド

```bash
npm install @sentry/react
```

```typescript
// src/main.tsx
import * as Sentry from '@sentry/react'

Sentry.init({
  dsn: import.meta.env.VITE_SENTRY_DSN,
  environment: import.meta.env.MODE,
})
```

---

## まとめ

### このガイドで学んだこと

- ✅ Renderへのバックエンドデプロイ
- ✅ PostgreSQLデータベースのセットアップ
- ✅ Vercelへのフロントエンドデプロイ
- ✅ 環境変数の管理
- ✅ CORS設定
- ✅ CI/CD自動デプロイ
- ✅ トラブルシューティング

### デプロイ完了後のURL

- **フロントエンド**: https://taskapp-frontend.vercel.app
- **バックエンド**: https://taskapp-api.onrender.com
- **データベース**: Render PostgreSQL（Internal）

### 次のステップ

1. **カスタムドメイン設定**
2. **モニタリング導入**（Sentry等）
3. **パフォーマンス最適化**
4. **セキュリティ監査**
5. **ユーザーフィードバック収集**

---

## デプロイ後のメンテナンス

### 定期的なタスク

| タスク | 頻度 | 方法 |
|--------|------|------|
| ログ確認 | 毎日 | Render/Vercelダッシュボード |
| 依存関係更新 | 月1回 | `npm outdated` → `npm update` |
| セキュリティ監査 | 月1回 | `npm audit fix` |
| データベースバックアップ | 週1回 | Render PostgreSQL自動バックアップ確認 |
| パフォーマンス測定 | 月1回 | Lighthouse、Web Vitals |

### アップデート手順

```bash
# 1. ローカルで開発
git checkout -b feature/update

# 2. 変更を実装

# 3. テスト
npm run dev  # ローカルテスト

# 4. コミット
git add .
git commit -m "Update: description"

# 5. プッシュ
git push origin feature/update

# 6. プルリクエスト作成

# 7. レビュー後、マージ

# 8. 自動デプロイ完了を確認
```

---

## リソース

### 公式ドキュメント

- [Render Documentation](https://render.com/docs)
- [Vercel Documentation](https://vercel.com/docs)
- [Prisma Deployment](https://www.prisma.io/docs/guides/deployment)

### サポート

- Render Status: https://status.render.com/
- Vercel Status: https://www.vercel-status.com/

---

**前のガイド:** [03-frontend-implementation.md](./03-frontend-implementation.md)

**親ガイド:** [統合プロジェクト - README](../README.md)

**おめでとうございます！** フルスタックアプリケーションのデプロイが完了しました。これでポートフォリオに追加できる実務レベルのアプリケーションが完成です！
