# ポートフォリオ作成ガイド

## 目次

1. [なぜポートフォリオが必要か](#なぜポートフォリオが必要か)
2. [GitHubプロフィールの最適化](#githubプロフィールの最適化)
3. [プロジェクトの選び方](#プロジェクトの選び方)
4. [README.mdの書き方](#readmemdの書き方)
5. [デモサイトの作成と公開](#デモサイトの作成と公開)
6. [ポートフォリオサイトの作成](#ポートフォリオサイトの作成)
7. [採用担当者が見るポイント](#採用担当者が見るポイント)
8. [履歴書・職務経歴書への記載方法](#履歴書職務経歴書への記載方法)
9. [実例紹介：統合プロジェクトの活用](#実例紹介統合プロジェクトの活用)
10. [良い例・悪い例の比較](#良い例悪い例の比較)

---

## なぜポートフォリオが必要か

### ポートフォリオの重要性

エンジニアの採用において、ポートフォリオは**実力を証明する最も効果的な手段**です。

**ポートフォリオが重要な理由：**

1. **実力の可視化**
   - 「できます」ではなく「作りました」を示せる
   - 実際のコードを見てもらえる
   - 技術レベルが一目瞭然

2. **差別化**
   - 未経験者でも経験者と同じ土俵に立てる
   - 他の応募者との差別化
   - 本気度のアピール

3. **面接での話題**
   - 具体的な技術の話ができる
   - 問題解決のプロセスを説明できる
   - 自己PRの根拠になる

4. **継続的な学習の証明**
   - GitHubの草（Contribution Graph）が成長を示す
   - 定期的なコミットが学習意欲を証明
   - 技術トレンドへの対応力を示せる

### ポートフォリオがないとどうなるか

```
❌ ポートフォリオなし
「Reactができます」
→ 採用担当者：「本当に？どのくらいのレベル？」

✅ ポートフォリオあり
「Reactでタスク管理アプリを作りました。こちらがデモサイトとコードです」
→ 採用担当者：「なるほど、この技術レベルなら即戦力だ」
```

### 本ガイドで作成するポートフォリオ

以下の3つの要素を整備します：

1. **GitHub プロフィール** - 開発者としての顔
2. **プロジェクトリポジトリ** - 技術力の証明
3. **ポートフォリオサイト** - 総合的なアピール

---

## GitHubプロフィールの最適化

### プロフィール写真

**良い例：**
- 顔がはっきり見える写真
- 明るく清潔感のある印象
- プロフェッショナルな雰囲気

**悪い例：**
- デフォルトのアイコン
- 暗い・ぼやけた写真
- アニメキャラクター（企業によってはNG）

### プロフィール情報の設定

```markdown
Name: 山田太郎
Bio: フロントエンドエンジニア志望 | React / TypeScript / Next.js
Location: 東京都
Company: -
Website: https://my-portfolio.vercel.app
Twitter: @yamada_dev
```

**ポイント：**
- **Name**: 本名または活動名
- **Bio**: 志望職種 + 得意技術（簡潔に）
- **Location**: 勤務希望地
- **Website**: ポートフォリオサイトのURL
- **Twitter**: 技術情報を発信している場合

### READMEプロフィールの作成

GitHubプロフィールページにREADMEを表示できます。

**作成手順：**

1. 自分のユーザー名と同じ名前のリポジトリを作成
   - 例：ユーザー名が `yamada-taro` なら `yamada-taro` リポジトリを作成
2. `README.md` を作成

**テンプレート：**

```markdown
# 👋 こんにちは！山田太郎です

## 🚀 About Me

フロントエンドエンジニアを目指して学習中です。
React / TypeScript を使ったWebアプリケーション開発が得意です。

## 💻 Tech Stack

**Languages:**
![JavaScript](https://img.shields.io/badge/-JavaScript-F7DF1E?style=flat-square&logo=javascript&logoColor=black)
![TypeScript](https://img.shields.io/badge/-TypeScript-3178C6?style=flat-square&logo=typescript&logoColor=white)
![HTML5](https://img.shields.io/badge/-HTML5-E34F26?style=flat-square&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/-CSS3-1572B6?style=flat-square&logo=css3&logoColor=white)

**Frameworks & Libraries:**
![React](https://img.shields.io/badge/-React-61DAFB?style=flat-square&logo=react&logoColor=black)
![Next.js](https://img.shields.io/badge/-Next.js-000000?style=flat-square&logo=next.js&logoColor=white)
![Node.js](https://img.shields.io/badge/-Node.js-339933?style=flat-square&logo=node.js&logoColor=white)

**Tools:**
![Git](https://img.shields.io/badge/-Git-F05032?style=flat-square&logo=git&logoColor=white)
![VS Code](https://img.shields.io/badge/-VS%20Code-007ACC?style=flat-square&logo=visual-studio-code&logoColor=white)
![Figma](https://img.shields.io/badge/-Figma-F24E1E?style=flat-square&logo=figma&logoColor=white)

## 📌 Featured Projects

### [📝 フルスタックタスク管理アプリ](https://github.com/yamada-taro/fullstack-task-app)
Next.js + Supabase で作成したタスク管理アプリケーション。
認証、CRUD操作、リアルタイム更新を実装。

**Tech:** Next.js, TypeScript, Supabase, Tailwind CSS

[🔗 Live Demo](https://my-task-app.vercel.app)

### [🎨 ポートフォリオサイト](https://github.com/yamada-taro/portfolio)
自己紹介とプロジェクト紹介を兼ねたポートフォリオサイト。

**Tech:** React, TypeScript, Tailwind CSS

[🔗 Live Demo](https://my-portfolio.vercel.app)

## 📊 GitHub Stats

![GitHub Stats](https://github-readme-stats.vercel.app/api?username=yamada-taro&show_icons=true&theme=radical)

## 📫 Contact

- Email: yamada@example.com
- Twitter: [@yamada_dev](https://twitter.com/yamada_dev)
- Zenn: [https://zenn.dev/yamada](https://zenn.dev/yamada)

## 🌱 Currently Learning

- Next.js App Router
- GraphQL
- Docker
```

**カスタマイズのポイント：**

1. **バッジの追加**
   - [Shields.io](https://shields.io/) で技術スタックのバッジを作成
   - 学習中の技術も含める

2. **GitHub Stats の追加**
   - [github-readme-stats](https://github.com/anuraghazra/github-readme-stats) を活用
   - コントリビューション数、使用言語を可視化

3. **注目プロジェクトの紹介**
   - 2〜3個の代表作を紹介
   - Live Demo のリンクを必ず含める

### Pinned Repositories の設定

GitHubプロフィールページに表示されるピン留めリポジトリを設定しましょう。

**設定方法：**
1. GitHubプロフィールページにアクセス
2. "Customize your pins" をクリック
3. 代表的なプロジェクトを6つまで選択

**選ぶべきリポジトリ：**
- フルスタックタスク管理アプリ
- ポートフォリオサイト
- 技術ブログ（あれば）
- オリジナルプロジェクト
- 貢献度の高いOSSプロジェクト

---

## プロジェクトの選び方

### ポートフォリオに含めるべきプロジェクト

**最低限必要なプロジェクト数：**
- **3〜5個**が目安
- 質を重視（1個でも完成度の高いものがあればOK）

### プロジェクトの種類

#### 1. フルスタックアプリケーション（必須）

**例：タスク管理アプリ、ブログシステム、ECサイト**

```
推奨技術スタック：
- フロントエンド：React / Next.js + TypeScript
- バックエンド：Next.js API Routes / Node.js / Supabase
- データベース：PostgreSQL / Supabase
- デプロイ：Vercel / Netlify
```

**実装すべき機能：**
- ユーザー認証（サインアップ、ログイン）
- CRUD操作（作成、読み取り、更新、削除）
- バリデーション
- エラーハンドリング
- レスポンシブデザイン

#### 2. フロントエンド特化プロジェクト

**例：ポートフォリオサイト、ランディングページ、UI コンポーネント集**

```
推奨技術スタック：
- React / Next.js
- TypeScript
- Tailwind CSS / Styled Components
- Framer Motion（アニメーション）
```

**実装すべき要素：**
- モダンなUIデザイン
- スムーズなアニメーション
- SEO対策
- パフォーマンス最適化
- アクセシビリティ対応

#### 3. バックエンド特化プロジェクト

**例：REST API、GraphQL API、CLI ツール**

```
推奨技術スタック：
- Node.js / Express / NestJS
- TypeScript
- PostgreSQL / MongoDB
- JWT認証
- Swagger（API ドキュメント）
```

**実装すべき機能：**
- RESTful設計
- 認証・認可
- データベース設計
- エラーハンドリング
- ユニットテスト

#### 4. オリジナル企画プロジェクト（推奨）

**独自性のあるアイデアで作成したプロジェクト**

**例：**
- 趣味や興味に基づいたアプリ
- 日常の問題を解決するツール
- ニッチな分野のサービス

**ポイント：**
- 「なぜ作ったか」を明確に説明できる
- 自分らしさが出る
- 面接で話が盛り上がる

### プロジェクト選びのチェックリスト

```markdown
✅ 技術スタックは履歴書と一致しているか
✅ 実務で使われている技術を採用しているか
✅ コードの品質は十分か（リンター、フォーマッター導入）
✅ デモサイトは正常に動作するか
✅ README.md は充実しているか
✅ レスポンシブデザインに対応しているか
✅ エラーハンドリングは適切か
✅ セキュリティ対策は施されているか
✅ デプロイされているか
✅ 継続的にメンテナンスしているか
```

---

## README.mdの書き方

### README.md の重要性

README.md は**プロジェクトの顔**です。採用担当者が最初に見る場所であり、プロジェクトの理解度と説明能力を示す重要な要素です。

### 必須セクション

1. **プロジェクト名と概要**
2. **デモサイトのリンク**
3. **スクリーンショット**
4. **技術スタック**
5. **主な機能**
6. **セットアップ方法**
7. **工夫した点・学んだこと**

### README.md テンプレート（完全版）

```markdown
# 📝 フルスタックタスク管理アプリ

Next.js と Supabase を使用したモダンなタスク管理アプリケーションです。
ユーザー認証、リアルタイム更新、ドラッグ&ドロップによる直感的な操作を実装しています。

## 🚀 デモ

**Live Demo:** [https://my-task-app.vercel.app](https://my-task-app.vercel.app)

**テストアカウント:**
- Email: demo@example.com
- Password: demo1234

## 📸 スクリーンショット

### ダッシュボード
![Dashboard](./docs/images/dashboard.png)

### タスク編集
![Task Edit](./docs/images/task-edit.png)

### レスポンシブデザイン
![Responsive](./docs/images/responsive.png)

## 🛠️ 技術スタック

### フロントエンド
- **Framework:** Next.js 14 (App Router)
- **Language:** TypeScript
- **Styling:** Tailwind CSS
- **State Management:** Zustand
- **Form Handling:** React Hook Form + Zod
- **Drag & Drop:** dnd-kit
- **Icons:** Lucide React

### バックエンド
- **BaaS:** Supabase
  - Authentication
  - PostgreSQL Database
  - Real-time Subscriptions
  - Row Level Security (RLS)

### デプロイ・ツール
- **Hosting:** Vercel
- **Version Control:** Git / GitHub
- **Package Manager:** pnpm
- **Linter:** ESLint
- **Formatter:** Prettier

## ✨ 主な機能

### 認証機能
- [x] メールアドレスでのサインアップ
- [x] ログイン / ログアウト
- [x] セッション管理
- [x] パスワードリセット（メール送信）

### タスク管理機能
- [x] タスクの作成・編集・削除
- [x] タスクのステータス管理（未着手 / 進行中 / 完了）
- [x] 優先度設定（高 / 中 / 低）
- [x] 期限設定
- [x] タグ付け機能
- [x] ドラッグ&ドロップでのステータス変更

### UI/UX
- [x] レスポンシブデザイン（PC / タブレット / スマホ対応）
- [x] ダークモード対応
- [x] スムーズなアニメーション
- [x] リアルタイム更新（他のブラウザでの変更が即座に反映）
- [x] ローディング状態の表示
- [x] エラーハンドリング

### その他
- [x] フィルタリング機能（ステータス、優先度、タグ）
- [x] 検索機能
- [x] ソート機能（作成日、期限、優先度）
- [x] ページネーション

## 📦 セットアップ

### 必要な環境
- Node.js 18.0以上
- pnpm（または npm / yarn）
- Supabaseアカウント

### インストール手順

1. **リポジトリのクローン**
```bash
git clone https://github.com/your-username/fullstack-task-app.git
cd fullstack-task-app
```

2. **依存関係のインストール**
```bash
pnpm install
```

3. **環境変数の設定**

`.env.local.example` をコピーして `.env.local` を作成：
```bash
cp .env.local.example .env.local
```

`.env.local` を編集：
```env
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key
```

4. **Supabaseのセットアップ**

`supabase/schema.sql` をSupabase SQLエディタで実行してテーブルを作成。

5. **開発サーバーの起動**
```bash
pnpm dev
```

http://localhost:3000 でアプリケーションが起動します。

### ビルド

```bash
pnpm build
pnpm start
```

## 📂 プロジェクト構成

```
fullstack-task-app/
├── app/                    # Next.js App Router
│   ├── (auth)/            # 認証関連ページ
│   ├── (dashboard)/       # ダッシュボード
│   ├── api/               # API Routes
│   └── layout.tsx         # ルートレイアウト
├── components/            # Reactコンポーネント
│   ├── ui/               # 共通UIコンポーネント
│   ├── features/         # 機能別コンポーネント
│   └── layouts/          # レイアウトコンポーネント
├── lib/                   # ユーティリティ
│   ├── supabase/         # Supabase関連
│   ├── utils/            # ヘルパー関数
│   └── hooks/            # カスタムフック
├── types/                 # TypeScript型定義
├── supabase/             # Supabaseスキーマ
└── public/               # 静的ファイル
```

## 💡 工夫した点

### 1. TypeScript + Zod による型安全性

フロントエンドからバックエンドまで完全に型付けし、実行時のバリデーションも Zod で実装しました。

```typescript
// タスクのスキーマ定義
export const taskSchema = z.object({
  title: z.string().min(1, "タイトルは必須です").max(100),
  description: z.string().optional(),
  status: z.enum(["todo", "in_progress", "done"]),
  priority: z.enum(["low", "medium", "high"]),
  due_date: z.string().optional(),
});
```

### 2. Supabase RLS によるセキュリティ

Row Level Security を活用し、ユーザーは自分のデータのみアクセス可能に設定。

```sql
CREATE POLICY "Users can only access their own tasks"
ON tasks FOR ALL
USING (auth.uid() = user_id);
```

### 3. リアルタイム更新

Supabase Realtime を使用し、複数デバイスでの同期を実現。

```typescript
useEffect(() => {
  const channel = supabase
    .channel('tasks')
    .on('postgres_changes',
      { event: '*', schema: 'public', table: 'tasks' },
      (payload) => {
        // タスクリストを更新
        refreshTasks();
      }
    )
    .subscribe();

  return () => { channel.unsubscribe(); };
}, []);
```

### 4. パフォーマンス最適化

- Next.js の Image コンポーネントで画像最適化
- React.memo でコンポーネントの再レンダリング抑制
- useMemo / useCallback で不要な計算を削減

### 5. アクセシビリティ対応

- セマンティックHTML の使用
- ARIA属性の適切な設定
- キーボード操作対応

## 📖 学んだこと

### 技術的な学び
1. **Next.js App Router の理解**
   - Server Components と Client Components の使い分け
   - ストリーミングとサスペンス

2. **Supabase の活用**
   - PostgreSQL のクエリ最適化
   - RLS によるセキュリティ設計
   - Realtime の実装

3. **状態管理**
   - Zustand によるグローバル状態管理
   - Server State と Client State の分離

### 設計の学び
1. **コンポーネント設計**
   - Atomic Design の考え方
   - 再利用可能なコンポーネントの作成

2. **エラーハンドリング**
   - ユーザーフレンドリーなエラーメッセージ
   - エッジケースへの対応

3. **UI/UX の重要性**
   - ユーザー視点でのデザイン
   - フィードバックの明確化

## 🚧 今後の改善予定

- [ ] タスクのカテゴリ機能追加
- [ ] コメント機能
- [ ] ファイル添付機能
- [ ] 通知機能（期限が近いタスク）
- [ ] チーム機能（タスクの共有）
- [ ] カレンダービュー
- [ ] パフォーマンスモニタリング（Sentry導入）
- [ ] E2Eテスト（Playwright導入）

## 📝 ライセンス

MIT License

## 👤 作成者

**山田太郎**
- GitHub: [@yamada-taro](https://github.com/yamada-taro)
- Portfolio: [https://my-portfolio.vercel.app](https://my-portfolio.vercel.app)
- Email: yamada@example.com

---

このプロジェクトが気に入ったら⭐️をお願いします！
```

### README.md 作成のチェックリスト

```markdown
✅ プロジェクト名と一行説明がある
✅ デモサイトのリンクがある
✅ スクリーンショットが含まれている
✅ 技術スタックが明記されている
✅ 主な機能がリスト化されている
✅ セットアップ手順が詳細に書かれている
✅ 工夫した点が具体的に説明されている
✅ コードスニペットが含まれている
✅ 今後の改善予定が書かれている
✅ 連絡先情報がある
```

---

## デモサイトの作成と公開

### デプロイサービスの選択

| サービス | 特徴 | 推奨用途 |
|---------|------|---------|
| **Vercel** | Next.js に最適化、自動デプロイ | Next.js プロジェクト |
| **Netlify** | 静的サイトに強い、フォーム機能 | React SPA、静的サイト |
| **GitHub Pages** | 無料、シンプル | ポートフォリオサイト |
| **Render** | バックエンドも対応 | フルスタックアプリ |
| **Railway** | データベース込みでデプロイ可能 | フルスタックアプリ |

### Vercel でのデプロイ（Next.js 推奨）

**手順：**

1. **Vercel アカウント作成**
   - [https://vercel.com](https://vercel.com) にアクセス
   - GitHub アカウントで連携

2. **プロジェクトをインポート**
   - "New Project" をクリック
   - GitHub リポジトリを選択
   - "Import" をクリック

3. **環境変数の設定**
   - "Environment Variables" セクションで設定
   ```
   NEXT_PUBLIC_SUPABASE_URL=your_value
   NEXT_PUBLIC_SUPABASE_ANON_KEY=your_value
   ```

4. **デプロイ**
   - "Deploy" をクリック
   - 数分で完了

5. **カスタムドメインの設定（オプション）**
   - Settings > Domains
   - 独自ドメインを追加可能

**メリット：**
- Git push で自動デプロイ
- プレビューデプロイ（PR ごとに環境が作られる）
- 無料で十分な機能
- Next.js に最適化されたパフォーマンス

### Netlify でのデプロイ（React SPA 推奨）

**手順：**

1. **Netlify アカウント作成**
   - [https://www.netlify.com](https://www.netlify.com)
   - GitHub で連携

2. **サイトを追加**
   - "Add new site" > "Import an existing project"
   - GitHub リポジトリを選択

3. **ビルド設定**
   ```
   Build command: npm run build
   Publish directory: dist (または build)
   ```

4. **環境変数の設定**
   - Site settings > Environment variables
   - 必要な環境変数を追加

5. **デプロイ**
   - "Deploy site" をクリック

**メリット：**
- フォーム機能が標準搭載
- サーバーレス関数が使える
- リダイレクト設定が簡単

### GitHub Pages でのデプロイ（静的サイト）

**手順：**

1. **リポジトリ設定**
   - Settings > Pages
   - Source: "GitHub Actions" を選択

2. **GitHub Actions ワークフロー作成**

`.github/workflows/deploy.yml`:
```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches: [main]

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'

      - name: Install dependencies
        run: npm install

      - name: Build
        run: npm run build

      - name: Deploy
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./dist
```

3. **ベースパスの設定（必要な場合）**

`vite.config.ts`:
```typescript
export default defineConfig({
  base: '/repository-name/',
  // ...
});
```

**メリット：**
- 完全無料
- GitHub リポジトリと統合
- シンプルで理解しやすい

### デプロイ時の注意点

#### 環境変数の管理

**❌ 悪い例：**
```javascript
const API_KEY = "sk_live_12345abcde"; // コードに直接記述
```

**✅ 良い例：**
```javascript
const API_KEY = process.env.NEXT_PUBLIC_API_KEY;
```

**セキュリティチェックリスト：**
```markdown
✅ .env ファイルを .gitignore に追加
✅ .env.example を用意（値は空欄）
✅ デプロイサービスの環境変数設定を使用
✅ API キーなどの機密情報をコードに含めない
✅ フロントエンドでは NEXT_PUBLIC_ などのプレフィックスを使用
```

#### パフォーマンス最適化

**チェックリスト：**
```markdown
✅ 画像の最適化（WebP、適切なサイズ）
✅ コード分割（Dynamic Import）
✅ 不要な依存関係の削除
✅ Lighthouse スコア 90 以上
✅ モバイルでの動作確認
```

#### SEO対策（ポートフォリオサイト向け）

**必須項目：**
```html
<!-- pages/_document.tsx または app/layout.tsx -->
<head>
  <title>山田太郎 - フロントエンドエンジニア ポートフォリオ</title>
  <meta name="description" content="React / Next.js を使ったWebアプリケーション開発が得意なフロントエンドエンジニアのポートフォリオサイトです。" />
  <meta property="og:title" content="山田太郎 - フロントエンドエンジニア" />
  <meta property="og:description" content="React / Next.js を使ったWebアプリケーション開発が得意です。" />
  <meta property="og:image" content="https://my-portfolio.vercel.app/og-image.png" />
  <meta name="twitter:card" content="summary_large_image" />
</head>
```

---

## ポートフォリオサイトの作成

### ポートフォリオサイトの目的

1. **第一印象の向上**
2. **プロジェクトの整理された紹介**
3. **スキルセットの可視化**
4. **連絡先の提供**

### 必須セクション

1. **ヒーローセクション**（自己紹介）
2. **スキルセクション**（技術スタック）
3. **プロジェクトセクション**（作品紹介）
4. **経歴・学習歴**
5. **お問い合わせ**

### ポートフォリオサイトのテンプレート

#### 推奨技術スタック

```
- Next.js 14 + TypeScript
- Tailwind CSS
- Framer Motion（アニメーション）
- React Hook Form（お問い合わせフォーム）
```

#### セクション構成例

**1. ヒーローセクション**

```tsx
export default function Hero() {
  return (
    <section className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="text-center">
        <h1 className="text-5xl font-bold text-gray-900 mb-4">
          山田太郎
        </h1>
        <p className="text-2xl text-gray-700 mb-8">
          Frontend Engineer
        </p>
        <p className="text-lg text-gray-600 max-w-2xl mx-auto mb-8">
          React / Next.js を使ったWebアプリケーション開発が得意です。
          ユーザー体験を重視した、美しく使いやすいUIの実装を心がけています。
        </p>
        <div className="flex gap-4 justify-center">
          <a
            href="#projects"
            className="px-8 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700"
          >
            プロジェクトを見る
          </a>
          <a
            href="#contact"
            className="px-8 py-3 border-2 border-indigo-600 text-indigo-600 rounded-lg hover:bg-indigo-50"
          >
            お問い合わせ
          </a>
        </div>
      </div>
    </section>
  );
}
```

**2. スキルセクション**

```tsx
const skills = {
  "Languages": ["JavaScript", "TypeScript", "HTML", "CSS"],
  "Frameworks": ["React", "Next.js", "Node.js"],
  "Styling": ["Tailwind CSS", "Styled Components", "Sass"],
  "Tools": ["Git", "GitHub", "VS Code", "Figma"],
  "Others": ["REST API", "Supabase", "Vercel"]
};

export default function Skills() {
  return (
    <section className="py-20 bg-white">
      <div className="container mx-auto px-4">
        <h2 className="text-4xl font-bold text-center mb-12">Skills</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {Object.entries(skills).map(([category, items]) => (
            <div key={category} className="bg-gray-50 p-6 rounded-lg">
              <h3 className="text-xl font-semibold mb-4">{category}</h3>
              <div className="flex flex-wrap gap-2">
                {items.map((skill) => (
                  <span
                    key={skill}
                    className="px-3 py-1 bg-indigo-100 text-indigo-700 rounded-full text-sm"
                  >
                    {skill}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
```

**3. プロジェクトセクション**

```tsx
const projects = [
  {
    title: "フルスタックタスク管理アプリ",
    description: "Next.js + Supabase で作成したタスク管理アプリ。認証、CRUD操作、リアルタイム更新を実装。",
    image: "/projects/task-app.png",
    tags: ["Next.js", "TypeScript", "Supabase", "Tailwind CSS"],
    demoUrl: "https://my-task-app.vercel.app",
    githubUrl: "https://github.com/yamada-taro/task-app",
  },
  // ...他のプロジェクト
];

export default function Projects() {
  return (
    <section className="py-20 bg-gray-50">
      <div className="container mx-auto px-4">
        <h2 className="text-4xl font-bold text-center mb-12">Projects</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {projects.map((project) => (
            <div key={project.title} className="bg-white rounded-lg overflow-hidden shadow-lg hover:shadow-xl transition">
              <img src={project.image} alt={project.title} className="w-full h-48 object-cover" />
              <div className="p-6">
                <h3 className="text-xl font-semibold mb-2">{project.title}</h3>
                <p className="text-gray-600 mb-4">{project.description}</p>
                <div className="flex flex-wrap gap-2 mb-4">
                  {project.tags.map((tag) => (
                    <span key={tag} className="px-2 py-1 bg-gray-100 text-gray-700 rounded text-sm">
                      {tag}
                    </span>
                  ))}
                </div>
                <div className="flex gap-4">
                  <a href={project.demoUrl} className="text-indigo-600 hover:underline">
                    Live Demo
                  </a>
                  <a href={project.githubUrl} className="text-gray-600 hover:underline">
                    GitHub
                  </a>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
```

### デザインのポイント

**良いポートフォリオサイトの特徴：**

1. **シンプルで読みやすい**
   - 余白を十分に取る
   - フォントサイズは適切に
   - コントラストを意識

2. **レスポンシブ**
   - スマホでも快適に閲覧できる
   - Tailwind CSS のブレークポイントを活用

3. **パフォーマンス**
   - 画像の最適化
   - 高速なページロード

4. **アクセシビリティ**
   - セマンティックHTML
   - alt属性の設定

### 無料で使えるデザインリソース

**カラーパレット：**
- [Coolors](https://coolors.co/)
- [Adobe Color](https://color.adobe.com/)

**アイコン：**
- [Lucide Icons](https://lucide.dev/)
- [Heroicons](https://heroicons.com/)

**フォント：**
- [Google Fonts](https://fonts.google.com/)
  - Inter（モダン、読みやすい）
  - Noto Sans JP（日本語）

**画像：**
- [Unsplash](https://unsplash.com/)
- [Pexels](https://www.pexels.com/)

---

## 採用担当者が見るポイント

### 技術力の評価項目

#### 1. コードの品質

**チェックされる項目：**
- コードの可読性（命名、コメント、構造）
- 設計パターンの理解
- エラーハンドリング
- セキュリティ対策

**良いコードの例：**

```typescript
// ❌ 悪い例
function f(x: any) {
  let y = x * 2;
  return y;
}

// ✅ 良い例
/**
 * 数値を2倍にして返す
 * @param value - 計算対象の数値
 * @returns 2倍にした値
 */
function doubleValue(value: number): number {
  const result = value * 2;
  return result;
}
```

#### 2. Git の使い方

**チェックされる項目：**
- コミットメッセージの質
- コミット頻度
- ブランチ戦略

**良いコミットメッセージ：**

```bash
# ❌ 悪い例
git commit -m "update"
git commit -m "fix bug"
git commit -m "変更"

# ✅ 良い例
git commit -m "feat: タスク作成機能を追加"
git commit -m "fix: ログイン時のバリデーションエラーを修正"
git commit -m "refactor: タスクリストコンポーネントを分割"
```

#### 3. README.md の完成度

**チェック項目：**
- プロジェクトの説明が明確か
- セットアップ手順が詳細か
- スクリーンショットがあるか
- 技術選定の理由が書かれているか

#### 4. デモサイトの動作

**チェック項目：**
- デモサイトが正常に動作するか
- エラーが発生しないか
- レスポンシブ対応しているか
- ローディング速度は適切か

### 非技術的な評価項目

#### 1. 問題解決能力

**評価される要素：**
- 「なぜこの機能を実装したか」が説明できる
- 技術選定の理由が明確
- 工夫した点を具体的に説明できる

**面接での質問例：**
```
Q: このプロジェクトで一番苦労したことは何ですか？
A: リアルタイム更新の実装です。最初は useEffect で
   ポーリングを実装しましたが、パフォーマンスが悪かったため、
   Supabase Realtime を導入しました。結果、ネットワーク負荷を
   大幅に削減できました。
```

#### 2. 学習意欲

**評価される要素：**
- 新しい技術への取り組み
- 継続的な開発（GitHub の草）
- 技術記事の執筆（Zenn、Qiita）

#### 3. コミュニケーション能力

**評価される要素：**
- ドキュメントの分かりやすさ
- 技術説明の明確さ
- フィードバックへの対応

---

## 履歴書・職務経歴書への記載方法

### ポートフォリオセクションの記載

**職務経歴書の例：**

```markdown
## ポートフォリオ

### GitHub
https://github.com/yamada-taro

### ポートフォリオサイト
https://my-portfolio.vercel.app

### 代表的なプロジェクト

#### 1. フルスタックタスク管理アプリ
**期間:** 2024年10月 - 2024年12月（3ヶ月）
**技術スタック:** Next.js, TypeScript, Supabase, Tailwind CSS
**URL:** https://my-task-app.vercel.app
**GitHub:** https://github.com/yamada-taro/task-app

**概要:**
ユーザー認証、CRUD操作、リアルタイム更新を実装したタスク管理アプリケーション。

**実装した機能:**
- メール認証によるユーザー登録・ログイン
- タスクの作成・編集・削除（CRUD操作）
- ドラッグ&ドロップによるステータス変更
- リアルタイム更新（Supabase Realtime）
- レスポンシブデザイン（PC / タブレット / スマホ対応）
- ダークモード対応

**工夫した点:**
- TypeScript + Zod による型安全な実装
- Supabase RLS によるセキュリティ対策
- React.memo によるパフォーマンス最適化
- アクセシビリティ対応（ARIA属性、キーボード操作）

**学んだこと:**
- Next.js App Router の理解を深めた
- Supabase を使った認証・データベース設計
- リアルタイム通信の実装
```

### 技術スキル欄の記載

```markdown
## 技術スキル

### 言語
- JavaScript (2年)
- TypeScript (1年)
- HTML / CSS (2年)

### フレームワーク・ライブラリ
- React (1年6ヶ月) ★★★★☆
- Next.js (1年) ★★★★☆
- Node.js (1年) ★★★☆☆

### データベース
- PostgreSQL (Supabase) (6ヶ月) ★★★☆☆

### ツール・その他
- Git / GitHub (2年) ★★★★☆
- Vercel / Netlify (1年) ★★★★☆
- Figma (1年) ★★★☆☆

### 得意分野
- フロントエンド開発（React / Next.js）
- レスポンシブデザインの実装
- TypeScript による型安全な実装
- REST API の設計・実装
```

---

## 実例紹介：統合プロジェクトの活用

### フルスタックタスク管理アプリをポートフォリオに

本カリキュラムで作成したフルスタックタスク管理アプリは、**最高のポートフォリオ材料**です。

**アピールポイント：**

1. **フルスタック開発の経験**
   - フロントエンド（React / Next.js）
   - バックエンド（Supabase / PostgreSQL）
   - 認証・認可

2. **実務レベルの機能**
   - ユーザー認証
   - CRUD操作
   - リアルタイム更新
   - エラーハンドリング

3. **モダンな技術スタック**
   - TypeScript
   - Next.js 14 (App Router)
   - Tailwind CSS
   - Supabase

### 面接での説明例

**質問: このプロジェクトについて説明してください**

```
【回答例】

このプロジェクトは、Next.js と Supabase を使用したタスク管理アプリケーションです。

【背景】
フルスタック開発のスキルを習得するために、
実務で使われる技術スタックを採用して作成しました。

【主な機能】
1. ユーザー認証（メールアドレスでのサインアップ・ログイン）
2. タスクのCRUD操作
3. ドラッグ&ドロップによる直感的な操作
4. リアルタイム更新

【工夫した点】
特に力を入れたのは、セキュリティとパフォーマンスです。

セキュリティ面では、Supabase の Row Level Security を活用し、
ユーザーは自分のデータのみアクセスできるよう設計しました。

パフォーマンス面では、React.memo や useMemo を使って
不要な再レンダリングを抑制し、快適な操作感を実現しました。

【学んだこと】
このプロジェクトを通じて、Next.js App Router の理解が深まりました。
特に Server Components と Client Components の使い分けや、
Suspense を使ったストリーミングの実装が学べました。

また、Supabase を使った認証・データベース設計の経験を積むことができ、
バックエンド開発の基礎も身につきました。
```

---

## 良い例・悪い例の比較

### GitHubプロフィール

#### ❌ 悪い例

```
- プロフィール写真：デフォルトのアイコン
- Bio：空欄
- Pinned Repositories：なし
- README：なし
- リポジトリ：1〜2個、最終コミットが半年前
- コミットメッセージ：「update」「fix」のみ
```

#### ✅ 良い例

```
- プロフィール写真：顔がはっきり見える写真
- Bio：「フロントエンドエンジニア志望 | React / TypeScript」
- Pinned Repositories：6個（代表作を厳選）
- README：技術スタック、プロジェクト紹介、連絡先を記載
- リポジトリ：10個以上、直近1週間以内にコミット
- コミットメッセージ：「feat: タスク作成機能を追加」など詳細
```

### README.md

#### ❌ 悪い例

```markdown
# Task App

タスク管理アプリです。

## インストール
npm install
npm run dev
```

#### ✅ 良い例

```markdown
# 📝 フルスタックタスク管理アプリ

Next.js と Supabase を使用したモダンなタスク管理アプリケーションです。

## 🚀 デモ
**Live Demo:** https://my-task-app.vercel.app

## 📸 スクリーンショット
（画像を複数枚掲載）

## 🛠️ 技術スタック
（詳細な技術リストと選定理由）

## ✨ 主な機能
（チェックリスト形式で機能を列挙）

## 📦 セットアップ
（詳細な手順を番号付きで説明）

## 💡 工夫した点
（具体的なコード例と説明）

## 📖 学んだこと
（技術的な学びと設計の学び）
```

### ポートフォリオサイト

#### ❌ 悪い例

```
- デザイン：派手すぎる、読みにくい
- コンテンツ：自己紹介のみ、プロジェクトへのリンクなし
- パフォーマンス：画像が重い、ロードに10秒以上
- レスポンシブ：スマホで見ると崩れる
- 更新頻度：作成後、放置
```

#### ✅ 良い例

```
- デザイン：シンプルで読みやすい、プロフェッショナル
- コンテンツ：自己紹介、スキル、プロジェクト、連絡先が明確
- パフォーマンス：Lighthouse スコア 90 以上
- レスポンシブ：PC / タブレット / スマホで完璧に表示
- 更新頻度：月に1回は新しいプロジェクトを追加
```

### プロジェクトの選択

#### ❌ 悪い例

```
- チュートリアルをそのまま写しただけ
- 機能が少なすぎる（Hello World レベル）
- デモサイトがない
- エラーが頻発する
- 古い技術スタック（jQuery、CRA など）
```

#### ✅ 良い例

```
- オリジナリティがある（独自の機能追加）
- 実務で使えるレベルの機能（認証、CRUD、検索など）
- デモサイトが正常に動作する
- エラーハンドリングが適切
- モダンな技術スタック（Next.js、TypeScript など）
```

---

## チェックリスト

### ポートフォリオ完成チェックリスト

```markdown
## GitHubプロフィール
- [ ] プロフィール写真を設定
- [ ] Bio を記載（志望職種 + 得意技術）
- [ ] Pinned Repositories を設定（3〜6個）
- [ ] README.md を作成（技術スタック、プロジェクト紹介）
- [ ] GitHub Stats を表示

## プロジェクト（各リポジトリ）
- [ ] README.md が充実している
- [ ] デモサイトのリンクがある
- [ ] スクリーンショットを掲載
- [ ] 技術スタックを明記
- [ ] セットアップ手順が詳細
- [ ] 工夫した点を記載
- [ ] コミットメッセージが適切
- [ ] .gitignore を設定（.env を含む）
- [ ] ライセンスを設定

## デモサイト
- [ ] デプロイ済み（Vercel / Netlify）
- [ ] 正常に動作する
- [ ] エラーが発生しない
- [ ] レスポンシブ対応
- [ ] Lighthouse スコア 80 以上
- [ ] 環境変数が適切に設定されている

## ポートフォリオサイト
- [ ] ヒーローセクション（自己紹介）
- [ ] スキルセクション
- [ ] プロジェクトセクション
- [ ] 連絡先セクション
- [ ] レスポンシブデザイン
- [ ] パフォーマンス最適化
- [ ] SEO対策（meta タグ）

## 履歴書・職務経歴書
- [ ] ポートフォリオサイトのURLを記載
- [ ] GitHubのURLを記載
- [ ] 代表的なプロジェクトを詳細に説明
- [ ] 技術スタックを明記
- [ ] 実装した機能を列挙
- [ ] 工夫した点を記載

## その他
- [ ] 技術記事を執筆（Zenn / Qiita）
- [ ] Twitterで技術情報を発信
- [ ] 継続的に開発（週に数回はコミット）
```

---

## まとめ

### ポートフォリオ作成の重要ポイント

1. **GitHubプロフィールの最適化**
   - プロフィール写真、Bio、README を充実
   - Pinned Repositories で代表作をアピール

2. **質の高いプロジェクト**
   - フルスタックアプリケーションを最低1つ
   - README.md を詳細に記載
   - デモサイトを公開

3. **ポートフォリオサイトの作成**
   - シンプルで読みやすいデザイン
   - プロジェクトへのリンクを明記
   - 連絡先を記載

4. **継続的な更新**
   - 定期的にコミット
   - 新しいプロジェクトを追加
   - 技術記事を執筆

### 次のステップ

ポートフォリオが完成したら、次は実際の転職活動です。

次のガイド：
- [就職活動ガイド](./job-application-guide.md)
- [フリーランスガイド](./freelance-guide.md)

---

**作成者:** Claude Code Curriculum Team
**最終更新:** 2026-01-29
