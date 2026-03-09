# AI 開発ツール

> GitHub Copilot、Claude Code CLI、Cursor を活用し、AI を開発ワークフローに統合してコーディング効率を飛躍的に向上させるためのガイド。

## この章で学ぶこと

1. GitHub Copilot の設定と効果的なプロンプトテクニック
2. Claude Code CLI のセットアップと実践的な活用方法
3. Cursor エディタの導入と AI エディタの比較選定
4. AI ツールのセキュリティ・プライバシー管理
5. チーム導入時のガバナンスとベストプラクティス


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解
- [Git 設定](./02-git-config.md) の内容を理解していること

---

## 1. AI 開発ツールの全体像

### 1.1 主要ツール比較

| 特徴 | GitHub Copilot | Claude Code CLI | Cursor | Cody (Sourcegraph) | Windsurf | Aider |
|------|---------------|-----------------|--------|---------------------|----------|-------|
| 形態 | VS Code 拡張 | CLI ツール | エディタ | VS Code 拡張 | エディタ | CLI ツール |
| AI モデル | GPT-4o / Claude | Claude | GPT-4o / Claude | 複数対応 | Claude / GPT | 複数対応 |
| インライン補完 | あり | なし | あり | あり | あり | なし |
| チャット | あり | あり | あり | あり | あり | あり |
| エージェント | あり | あり | あり | 一部 | あり | あり |
| ファイル編集 | あり | あり | あり | 限定的 | あり | あり |
| マルチファイル | あり | あり | あり | 限定的 | あり | あり |
| MCP サポート | あり | あり | あり | なし | あり | なし |
| 料金 (月額) | $10-39 | 従量制 | $20 | 無料枠あり | $15 | 無料(API費用) |
| オフライン | 不可 | 不可 | 不可 | 不可 | 不可 | 不可 |

### 1.2 AI ツールの役割分担

```
AI 開発ツールの活用レイヤー:

┌─────────────────────────────────────────────┐
│            開発ワークフロー                    │
├──────────────┬──────────────┬───────────────┤
│   コード記述  │  コードレビュー │  アーキテクチャ  │
│              │              │               │
│  Copilot     │  Claude Code │  Claude Code  │
│  Cursor      │  Copilot Chat│  Cursor       │
│  (補完中心)   │  (分析中心)   │  (設計中心)    │
├──────────────┼──────────────┼───────────────┤
│   デバッグ    │  テスト生成   │  ドキュメント   │
│              │              │               │
│  Copilot Chat│  Claude Code │  Claude Code  │
│  Cursor      │  Copilot     │  Copilot Chat │
│  (対話中心)   │  (生成中心)   │  (生成中心)    │
├──────────────┼──────────────┼───────────────┤
│  CI/CD 設定   │  マイグレーション│  学習・調査    │
│              │              │               │
│  Claude Code │  Claude Code │  Copilot Chat │
│  Cursor      │  Cursor      │  Claude Code  │
│  (生成中心)   │  (分析+生成)  │  (対話中心)    │
└──────────────┴──────────────┴───────────────┘
```

### 1.3 AI 開発ツールの進化 (2024-2026)

```
AI 開発ツールのパラダイムシフト:

  2023: 補完 (Autocomplete)
  ├── 1行～数行のコード補完
  ├── Tab キーで受け入れ
  └── コンテキスト: 現在のファイル

  2024: チャット + エージェント
  ├── マルチファイル編集
  ├── ターミナルコマンド実行
  ├── テスト生成・実行
  └── コンテキスト: プロジェクト全体

  2025-2026: 自律エージェント
  ├── タスク全体の自律実行
  ├── Plan → Code → Test → Fix サイクル
  ├── MCP でツール連携
  ├── CI/CD パイプライン統合
  └── コンテキスト: リポジトリ + 外部知識
```

---

## 2. GitHub Copilot

### 2.1 セットアップ

```bash
# VS Code に拡張機能をインストール
code --install-extension GitHub.copilot
code --install-extension GitHub.copilot-chat

# GitHub アカウントでサインイン
# VS Code 左下のアカウントアイコン → Sign in with GitHub

# プラン確認
# GitHub → Settings → Copilot → ライセンスタイプ確認
# Individual: $10/月
# Business: $19/月 (組織管理、IP保護)
# Enterprise: $39/月 (カスタマイズ、監査ログ)
```

### 2.2 VS Code 設定

```jsonc
// .vscode/settings.json
{
  // Copilot の基本設定
  "github.copilot.enable": {
    "*": true,
    "plaintext": false,
    "markdown": true,
    "yaml": true
  },

  // インライン候補の表示
  "editor.inlineSuggest.enabled": true,

  // 言語ごとの無効化 (機密ファイル)
  "github.copilot.enable": {
    "dotenv": false,
    "properties": false,
    "ini": false
  },

  // Copilot Chat の設定
  "github.copilot.chat.localeOverride": "ja",

  // エディタ内での候補表示設定
  "github.copilot.editor.enableAutoCompletions": true,

  // Next Edit Suggestions (NES) の有効化
  "github.copilot.nextEditSuggestions.enabled": true
}
```

### 2.3 効果的なプロンプトテクニック

```typescript
// ─── テクニック 1: 関数シグネチャ + コメントで誘導 ───

// ユーザーの年齢を検証する。18歳未満は不可。
// エラー時は具体的なメッセージを返す。
function validateAge(age: unknown): { valid: boolean; message: string } {
  // ← Copilot が適切な実装を提案
}

// ─── テクニック 2: テストケースを先に書く (TDD) ───

describe('calculateDiscount', () => {
  it('通常会員は5%割引', () => {
    expect(calculateDiscount(1000, 'normal')).toBe(950);
  });
  it('プレミアム会員は15%割引', () => {
    expect(calculateDiscount(1000, 'premium')).toBe(850);
  });
  it('割引後の金額は0未満にならない', () => {
    expect(calculateDiscount(10, 'premium')).toBe(0);
  });
});
// → テストから実装を自動生成させる

// ─── テクニック 3: 型定義から実装を生成 ───

type SortDirection = 'asc' | 'desc';

interface SortOptions<T> {
  data: T[];
  key: keyof T;
  direction: SortDirection;
}

function sortBy<T>(options: SortOptions<T>): T[] {
  // ← 型情報から正確な実装が提案される
}

// ─── テクニック 4: 例示パターン (Few-shot) ───

// 既にある関数の隣に同様の関数を書くと、パターンを学習して提案
function getUserById(id: string): Promise<User> {
  return db.users.findUnique({ where: { id } });
}

function getPostById(id: string): Promise<Post> {
  // ← 上の関数のパターンから正確に推論
}

// ─── テクニック 5: JSDoc で意図を明示 ───

/**
 * CSV ファイルを読み込み、指定されたカラムでグループ化する
 * @param filePath CSVファイルのパス
 * @param groupBy グループ化するカラム名
 * @returns グループ化されたデータ
 * @throws FileNotFoundError ファイルが存在しない場合
 * @example
 * const result = await groupCsvByColumn('./data.csv', 'department');
 * // { 'engineering': [...], 'marketing': [...] }
 */
async function groupCsvByColumn(
  filePath: string,
  groupBy: string
): Promise<Record<string, any[]>> {
  // ← JSDoc の情報から正確な実装が提案される
}
```

### 2.4 Copilot Chat の活用

```
Copilot Chat の主要コマンド:

┌──────────────────────────────────────────┐
│ /explain   → コードの説明                 │
│ /fix       → バグ修正の提案               │
│ /tests     → テストコード生成             │
│ /doc       → ドキュメント生成             │
│ /optimize  → パフォーマンス最適化提案     │
│ /new       → 新規ファイル/プロジェクト生成│
│ @workspace → ワークスペース全体を文脈に   │
│ @terminal  → ターミナル出力を文脈に       │
│ @vscode    → VS Code の設定に関する質問   │
│ @github    → GitHub 関連の操作            │
│ #file      → 特定ファイルを文脈に追加     │
│ #selection → 選択中のコードを文脈に       │
│ #codebase  → コードベース全体を検索       │
│ #terminalSelection → ターミナル選択を文脈に│
└──────────────────────────────────────────┘

効果的な使い方の例:
  "@workspace このプロジェクトの認証フローを説明して"
  "#file:src/auth/middleware.ts /fix セキュリティの問題を修正して"
  "@terminal このエラーの原因と解決策を教えて"
  "#selection /tests このコードのユニットテストを生成して"
  "/explain この正規表現は何をしているか"
```

### 2.5 Copilot Agent モード (Copilot Edits)

```
Copilot Agent モード (VS Code):

  ┌─────────────────────────────────────────┐
  │ Cmd+Shift+I (macOS) / Ctrl+Shift+I      │
  │                                           │
  │ 機能:                                    │
  │ - 複数ファイルにまたがる変更を一括実行   │
  │ - ファイルの作成・削除も可能             │
  │ - 変更のプレビュー付き                   │
  │ - Undo/Redo 対応                         │
  │                                           │
  │ 使用例:                                  │
  │ "Userモデルにemailフィールドを追加して、  │
  │  バリデーション、テスト、マイグレーション │
  │  も一緒に更新して"                       │
  │                                           │
  │ → 4-5ファイルを同時に正確に変更          │
  └─────────────────────────────────────────┘
```

---

## 3. Claude Code CLI

### 3.1 インストール

```bash
# npm でインストール
npm install -g @anthropic-ai/claude-code

# 認証
claude auth login

# バージョン確認
claude --version

# アップデート
npm update -g @anthropic-ai/claude-code
```

### 3.2 基本的な使い方

```bash
# インタラクティブモードで起動
claude

# ワンショットコマンド
claude "このプロジェクトの構造を説明して"

# パイプで入力
cat error.log | claude "このエラーの原因を分析して"

# 特定ファイルを指定
claude "この関数のテストを書いて" --file src/utils/validate.ts

# 出力形式の指定
claude --output-format json "package.json の依存関係を分析して"

# ─── セッション管理 ───
claude --resume         # 前回のセッションを再開
claude --session-id abc  # 特定セッションを再開

# ─── モデル指定 ───
# デフォルトは最新の Claude モデル
# API キーに応じたモデルが使用される
```

### 3.3 プロジェクト設定

```markdown
# CLAUDE.md (プロジェクトルートに配置)

## プロジェクト概要
TypeScript + React のWebアプリケーション
バックエンドは Express + Prisma

## 技術スタック
- Frontend: React 19, TypeScript, Tailwind CSS, Zustand
- Backend: Express, Prisma, PostgreSQL
- Testing: Vitest, React Testing Library, Playwright
- CI/CD: GitHub Actions
- Deploy: Vercel (Frontend), Railway (Backend)

## コーディング規約
- 関数コンポーネントのみ使用 (クラスコンポーネント禁止)
- 状態管理は Zustand を使用
- スタイリングは Tailwind CSS
- テストは Vitest + Testing Library
- エラーハンドリングは Result 型パターン
- `any` 型の使用禁止
- console.log の使用禁止 (logger を使用)

## ディレクトリ構造
```
src/
├── components/    # UIコンポーネント
│   ├── ui/        # 汎用コンポーネント (Button, Input等)
│   ├── features/  # 機能別コンポーネント
│   └── layouts/   # レイアウトコンポーネント
├── hooks/         # カスタムフック
├── stores/        # Zustand ストア
├── utils/         # ユーティリティ関数
├── types/         # 型定義
├── lib/           # 外部ライブラリのラッパー
├── api/           # API クライアント
└── __tests__/     # テスト
```

## コマンド
- `npm run dev` - 開発サーバー起動 (port 3000)
- `npm test` - テスト実行
- `npm run test:watch` - テスト監視モード
- `npm run build` - ビルド
- `npm run lint` - ESLint 実行
- `npm run lint:fix` - ESLint 自動修正
- `npm run format` - Prettier 実行
- `npm run typecheck` - TypeScript 型チェック
- `npm run db:migrate` - Prisma マイグレーション実行
- `npm run db:seed` - シードデータ投入

## やってはいけないこと
- テストなしで機能を追加しない
- Prisma の raw query を使わない (ORM メソッドを使用)
- 環境変数をハードコードしない
- node_modules 内のファイルを変更しない
```

### 3.4 CLAUDE.md の階層構造

```bash
# CLAUDE.md は複数の場所に配置可能
# 優先順位: ローカル > プロジェクト > グローバル

~/.claude/CLAUDE.md              # グローバル設定 (全プロジェクト共通)
~/projects/my-app/CLAUDE.md      # プロジェクトルート
~/projects/my-app/src/CLAUDE.md  # サブディレクトリ (追加ルール)

# グローバル CLAUDE.md の例
# ~/.claude/CLAUDE.md
# ---
# ## 共通ルール
# - 日本語でコメントを書くこと
# - TypeScript を使用する場合は strict モードを有効にする
# - テストは必ずユニットテストと統合テストを含める
# - コミットメッセージは Conventional Commits に従う
```

### 3.5 実践的なワークフロー

```
Claude Code の典型的な活用パターン:

┌─────────────────────────────────────────┐
│ 1. コードレビュー                        │
│    claude "src/api/の変更をレビューして"   │
│    → セキュリティ・パフォーマンス指摘     │
│    → 具体的な改善コード提案              │
│                                           │
│ 2. リファクタリング                      │
│    claude "この関数を小さく分割して"       │
│    → ファイル分割とテスト更新を一括実行  │
│    → import パスの自動更新              │
│                                           │
│ 3. バグ修正                              │
│    cat error.log | claude "修正して"      │
│    → エラー分析 → 原因特定 → 修正適用   │
│    → テストで修正を検証                  │
│                                           │
│ 4. テスト生成                            │
│    claude "src/utils/のテストを追加して"   │
│    → カバレッジの低い部分を特定して生成  │
│    → エッジケースの網羅                  │
│                                           │
│ 5. ドキュメント更新                      │
│    claude "API変更に合わせてREADME更新"    │
│    → コード変更を検出して自動反映        │
│                                           │
│ 6. マイグレーション                      │
│    claude "React Router v6 → v7 に移行"   │
│    → Breaking Changes の検出と修正       │
│    → テストの更新                        │
│                                           │
│ 7. 新機能実装                            │
│    claude "ユーザー招待機能を実装して"     │
│    → DB設計 → API → UI → テスト         │
│    → 段階的にファイルを作成・編集        │
│                                           │
│ 8. CI/CD 構築                            │
│    claude "GitHub Actions の CI を設定して"│
│    → lint → test → build → deploy       │
│    → キャッシュ最適化も自動で            │
└─────────────────────────────────────────┘
```

### 3.6 Claude Code のスラッシュコマンド

```bash
# インタラクティブモード内で使用
/help           # ヘルプ表示
/clear          # 会話履歴をクリア
/compact        # コンテキストを圧縮 (メモリ節約)
/config         # 設定の確認・変更
/cost           # 現在のセッションのコスト表示
/doctor         # 環境診断
/init           # CLAUDE.md の初期化
/review         # コードレビューモード
/terminal       # ターミナルコマンド実行

# MCP (Model Context Protocol) ツール
/mcp            # MCP サーバーの管理
```

### 3.7 Claude Code の MCP 統合

```json
// ~/.claude/claude_desktop_config.json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_..."
      }
    },
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres"],
      "env": {
        "DATABASE_URL": "postgresql://localhost:5432/mydb"
      }
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/docs"]
    },
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "env": {
        "BRAVE_API_KEY": "..."
      }
    }
  }
}
```

```
MCP (Model Context Protocol) の活用:

  ┌─────────────────────────────────────────┐
  │  Claude Code                             │
  │       ↕ MCP                              │
  │  ┌─────────┐ ┌──────────┐ ┌──────────┐ │
  │  │ GitHub  │ │ Database │ │  Search  │ │
  │  │ Server  │ │ Server   │ │  Server  │ │
  │  └─────────┘ └──────────┘ └──────────┘ │
  │                                           │
  │  活用例:                                 │
  │  - GitHub Issues から仕様を取得して実装   │
  │  - DB スキーマを確認しながらクエリ作成   │
  │  - ドキュメントを検索して最新の API 確認 │
  │  - PR のコメントを分析してコード修正     │
  └─────────────────────────────────────────┘
```

---

## 4. Cursor

### 4.1 セットアップ

```bash
# macOS
brew install --cask cursor

# VS Code 設定のインポート
# Cursor 初回起動時に "Import VS Code Settings" を選択
# → 拡張機能、設定、キーバインドが自動移行
```

### 4.2 Cursor 固有の機能

```
Cursor の AI 機能:

  ┌─────────────────────────────────────────┐
  │ Cmd+K (Inline Edit)                      │
  │   コード内で直接 AI に編集を依頼          │
  │   例: "この関数にエラーハンドリングを追加" │
  │   選択範囲がある場合はその部分を編集     │
  │   選択なしの場合は新しいコードを生成     │
  ├─────────────────────────────────────────┤
  │ Cmd+L (Chat)                             │
  │   サイドパネルで AI と対話               │
  │   @ファイル名 で特定ファイルを参照       │
  │   @codebase でプロジェクト全体を検索     │
  │   @docs で公式ドキュメントを参照         │
  │   @web でウェブ検索結果を参照            │
  ├─────────────────────────────────────────┤
  │ Cmd+I (Composer)                         │
  │   複数ファイルにまたがる変更を一括実行   │
  │   プロジェクト全体の文脈を理解           │
  │   Agent モード: 自律的にコード変更       │
  │   ファイル作成・削除も可能               │
  ├─────────────────────────────────────────┤
  │ Tab (Autocomplete)                       │
  │   次の編集位置を予測して提案             │
  │   複数行の変更を一度に適用               │
  │   カーソル位置から文脈を推測             │
  │   diff に基づく編集予測                  │
  ├─────────────────────────────────────────┤
  │ Cmd+Shift+K (Terminal Cmd+K)             │
  │   ターミナル内でAIにコマンドを生成させる │
  │   自然言語 → シェルコマンドに変換       │
  └─────────────────────────────────────────┘
```

### 4.3 .cursorrules 設定

```markdown
# .cursorrules (プロジェクトルートに配置)

You are an expert TypeScript developer working on a React + Express application.

## Code Style
- Use functional programming patterns
- Prefer immutable data structures
- Always use explicit return types for functions
- Use descriptive variable names (no abbreviations)
- Maximum function length: 30 lines
- Maximum file length: 300 lines

## Framework Conventions
- React: Use functional components with hooks
- State: Zustand for global, useState for local
- Styling: Tailwind CSS utility classes
- Testing: Vitest + React Testing Library
- API: tRPC for type-safe API calls

## Error Handling
- Always use Result pattern for error handling
- Never use try-catch in business logic (only in infrastructure layer)
- Log errors with structured logging (pino)
- Return user-friendly error messages

## File Naming
- Components: PascalCase (UserProfile.tsx)
- Hooks: camelCase with use prefix (useAuth.ts)
- Utils: camelCase (formatDate.ts)
- Types: PascalCase (User.ts)
- Tests: *.test.ts or *.spec.ts

## Do Not
- Never use `any` type
- Never use `console.log` in production code
- Never mutate function arguments
- Never use `var` (use `const` or `let`)
- Never commit TODO comments without issue reference
- Never use inline styles (use Tailwind)

## When Writing Tests
- Use describe/it blocks with clear descriptions in Japanese
- Mock external dependencies
- Test edge cases and error scenarios
- Aim for >80% code coverage
- Use factory functions for test data

## Commit Messages
- Follow Conventional Commits format
- Subject in English, body in Japanese is OK
```

### 4.4 Cursor の @docs 機能

```bash
# Cursor の @docs で外部ドキュメントを参照可能
# Cursor Settings → Features → Docs → Add

# 追加推奨ドキュメント:
# - React: https://react.dev
# - Next.js: https://nextjs.org/docs
# - Tailwind CSS: https://tailwindcss.com/docs
# - Prisma: https://www.prisma.io/docs
# - tRPC: https://trpc.io/docs
# - Vitest: https://vitest.dev/guide/

# 使い方:
# Chat で "@docs React の useEffect の正しい使い方を教えて"
# → React 公式ドキュメントの内容を踏まえた回答
```

---

## 5. AI ツールの効率的な使い分け

### 5.1 タスク別最適ツール

| タスク | 推奨ツール | 理由 |
|--------|-----------|------|
| 1行のコード補完 | Copilot / Cursor Tab | リアルタイム補完が最速 |
| 関数の実装 | Copilot + コメント誘導 | シグネチャから推論 |
| バグ修正 | Claude Code / Cursor | コンテキスト理解が深い |
| リファクタリング | Claude Code | 複数ファイル一括変更 |
| テスト生成 | Claude Code / Copilot | 既存コードから推論 |
| コードレビュー | Claude Code | セキュリティ分析に強い |
| アーキテクチャ設計 | Claude Code / Cursor | 大局的な判断が必要 |
| 学習・理解 | Copilot Chat / Claude | 対話で深掘り |
| DB マイグレーション | Claude Code | スキーマ理解+SQL生成 |
| CI/CD 構築 | Claude Code / Cursor | YAML 生成が正確 |
| ドキュメント生成 | Claude Code | 構造化された出力 |
| 依存関係更新 | Claude Code | Breaking Changes 検出 |

### 5.2 プロンプトエンジニアリング原則

```
効果的なプロンプトの構造:

┌─────────────────────────────────────────┐
│ 1. コンテキスト (何のプロジェクトか)      │
│    "TypeScript + Express の REST API で"  │
│                                           │
│ 2. タスク (何をしてほしいか)              │
│    "ユーザー認証のミドルウェアを作成して"   │
│                                           │
│ 3. 制約 (守るべきルール)                  │
│    "JWT を使い、エラーは AppError で統一"   │
│                                           │
│ 4. 出力形式 (どう返してほしいか)          │
│    "型定義とテストも含めて"               │
│                                           │
│ 5. 例示 (具体的な入出力例)               │
│    "入力: { email, password }             │
│     出力: { token, user }"                │
└─────────────────────────────────────────┘

❌ "認証を作って"

✅ "Express + TypeScript で JWT ベースの認証ミドルウェアを作成して。
    トークンの検証失敗時は 401 を返し、AppError クラスを使って
    エラーハンドリングすること。リフレッシュトークンの仕組みも含める。
    Vitest のテストも含めて。"

さらに良い:
✅ "Express + TypeScript で JWT 認証ミドルウェアを実装してください。

    要件:
    1. アクセストークン (15分) とリフレッシュトークン (7日) の2トークン方式
    2. リフレッシュトークンは HTTP-only Cookie に保存
    3. トークン検証失敗時は AppError(401, 'UNAUTHORIZED') を throw
    4. req.user に { id, email, role } をセット

    既存のコード:
    - AppError クラス: src/errors/AppError.ts
    - User 型: src/types/User.ts
    - 環境変数: JWT_SECRET, JWT_REFRESH_SECRET

    テストは Vitest で、正常系と異常系の両方を含めてください。"
```

### 5.3 反復的な改善プロセス

```
AI との効果的な反復プロセス:

  Step 1: 初回指示
  ├── 要件を明確に伝える
  ├── 技術スタックを指定
  └── 出力形式を指定

  Step 2: レビューと修正指示
  ├── 生成されたコードをレビュー
  ├── 問題点を具体的にフィードバック
  └── "〇〇の部分を △△ に変更して"

  Step 3: テストで検証
  ├── テストを実行
  ├── 失敗するテストを報告
  └── "このテストが失敗する。修正して"

  Step 4: エッジケース対処
  ├── "空配列の場合はどうなる？"
  ├── "並行処理でのレースコンディションは？"
  └── "100万件のデータでのパフォーマンスは？"

  避けるべきパターン:
  ❌ 一度で完璧を求める
  ❌ 曖昧な指示で "いい感じに" 頼む
  ❌ 生成結果を読まずに次の指示を出す
  ✅ 小さく分割して段階的に改善する
```

---

## 6. セキュリティとプライバシー

### 6.1 注意すべき設定

```jsonc
// .vscode/settings.json
{
  // 機密ファイルでの Copilot 無効化
  "github.copilot.enable": {
    "dotenv": false,
    "properties": false,
    "ini": false
  },

  // Telemetry の制限
  "github.copilot.advanced": {
    // 公開コードと類似する提案をブロック
    "duplicationDetection": "block"
  }
}
```

```bash
# .gitignore に AI 関連の設定ファイルを追加
echo '.cursorrules' >> .gitignore  # 必要に応じて
# CLAUDE.md はコミットする (チーム共有のため)

# ─── .aiignore (Claude Code 用) ───
# Claude Code に読ませたくないファイルを指定
cat << 'EOF' > .aiignore
# 機密ファイル
.env
.env.*
secrets/
credentials/
*.pem
*.key

# 大きなバイナリ
*.zip
*.tar.gz
node_modules/
dist/
build/

# 生成ファイル
coverage/
.next/
EOF
```

### 6.2 データ送信の範囲

```
各ツールが送信するデータ:

┌──────────────┬─────────────────────────────┐
│ ツール        │ 送信されるデータ             │
├──────────────┼─────────────────────────────┤
│ Copilot      │ 開いているファイルの一部      │
│ Individual   │ 隣接ファイルのコンテキスト    │
│              │ (コード断片は学習に使用される) │
├──────────────┼─────────────────────────────┤
│ Copilot      │ 同上だがデータ保持なし        │
│ Business     │ (学習にデータは使用されない)  │
│              │ 監査ログ利用可能             │
├──────────────┼─────────────────────────────┤
│ Claude Code  │ 指定ファイルの内容           │
│              │ コマンド出力                 │
│              │ (30日後に削除)               │
│              │ .aiignore で除外可能         │
├──────────────┼─────────────────────────────┤
│ Cursor       │ アクティブファイル           │
│              │ @参照ファイル                │
│              │ (Privacy Mode で送信制限可)  │
│              │ SOC 2 Type II 認証済み       │
├──────────────┼─────────────────────────────┤
│ Aider        │ 指定ファイルの内容           │
│              │ API キー経由で直接送信       │
│              │ (ツール側でのデータ保持なし)  │
└──────────────┴─────────────────────────────┘
```

### 6.3 企業導入時のセキュリティチェックリスト

```
AI ツール企業導入チェックリスト:

  □ データ保持ポリシーの確認
    - 送信されたコードは学習に使用されるか？
    - データの保持期間は？
    - データの保存場所 (リージョン) は？

  □ アクセス制御
    - SSO/SAML 連携は可能か？
    - チーム/ロールベースのアクセス制御は？
    - 監査ログは取得可能か？

  □ コンプライアンス
    - SOC 2 Type II 認証済みか？
    - GDPR 対応しているか？
    - IP (知的財産) 保護の条項は？

  □ 技術的制限
    - 特定のリポジトリ/ファイルの除外設定は？
    - VPN/プロキシ経由での利用は？
    - オンプレミスデプロイの選択肢は？

  □ ライセンス
    - AI 生成コードの著作権は？
    - オープンソースライセンス侵害のリスクは？
    - 公開コード類似検出機能は？
```

---

## 7. AI コード品質の担保

### 7.1 AI 生成コードのレビューチェックリスト

```
AI 生成コードレビューの重点項目:

  □ セキュリティ
    - SQL インジェクション / XSS の脆弱性
    - 認証・認可の欠落
    - 機密情報のハードコード
    - 入力バリデーションの不足
    - CORS 設定の緩さ

  □ ロジック
    - エッジケースの処理
    - null/undefined の安全な処理
    - off-by-one エラー
    - レースコンディション
    - リソースリーク (close 忘れ等)

  □ パフォーマンス
    - N+1 クエリ問題
    - 不要なリレンダリング (React)
    - メモリリーク
    - 非効率なアルゴリズム
    - 大量データの同期処理

  □ 保守性
    - 過度な複雑さ
    - マジックナンバー / マジックストリング
    - DRY 原則違反
    - テスト可能性
    - ドキュメントの正確性

  □ 依存関係
    - 非推奨 API の使用
    - 古いライブラリバージョン
    - ライセンス問題
    - 不要な依存の追加
```

### 7.2 自動検証パイプライン

```yaml
# .github/workflows/ai-code-quality.yml
name: AI Code Quality Check
on: [pull_request]

jobs:
  quality-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Type Check
        run: npx tsc --noEmit

      - name: Lint
        run: npx eslint . --max-warnings 0

      - name: Security Audit
        run: npm audit --audit-level=moderate

      - name: Test
        run: npm test -- --coverage

      - name: Coverage Check
        run: |
          COVERAGE=$(npx coverage-summary | grep 'All files' | awk '{print $NF}')
          if (( $(echo "$COVERAGE < 80" | bc -l) )); then
            echo "Coverage is below 80%: $COVERAGE"
            exit 1
          fi

      - name: License Check
        run: npx license-checker --failOn 'GPL'

      - name: Bundle Size
        run: npx size-limit
```

---

## 8. アンチパターン

### 8.1 AI の出力を検証せずに受け入れる

```
❌ アンチパターン: AI が生成したコードをそのままコミット

問題:
  - セキュリティ脆弱性 (SQL インジェクション等)
  - 非推奨 API の使用
  - エッジケースの未処理
  - ライセンス問題のあるコードの混入
  - 不正確なロジック

✅ 正しいアプローチ:
  - AI 生成コードは「ドラフト」として扱う
  - 必ず自分でレビューしてから採用
  - テストを書いて動作を検証
  - セキュリティスキャン (npm audit, Snyk) を実行
  - CI で自動品質チェックを実施
```

### 8.2 コンテキストを与えずにプロンプトを書く

```
❌ アンチパターン:
  "ソートする関数を書いて"

問題:
  - 言語が不明
  - 何をソートするか不明
  - パフォーマンス要件が不明
  - エラー処理の方針が不明

✅ 正しいアプローチ:
  "TypeScript で、ユーザーオブジェクトの配列を
   最終ログイン日時の降順でソートする関数を書いて。
   配列が空の場合は空配列を返すこと。
   型: User[] を受け取り User[] を返す。
   User 型は { id: string, name: string, lastLoginAt: Date }。"
```

### 8.3 AI に依存しすぎて学習しない

```
❌ アンチパターン: 理解せずに AI の出力をコピペし続ける

問題:
  - 基礎的な理解が不足する
  - AI が使えない環境で作業できなくなる
  - デバッグ能力が育たない
  - コードレビューで問題を見抜けない

✅ 正しいアプローチ:
  - AI の出力を読んで理解してから採用する
  - 「なぜこう書いたのか」を AI に説明させる
  - 基礎的なアルゴリズム・データ構造は自分で学ぶ
  - AI 生成コードを改善・最適化する練習をする
  - 時には AI なしでコードを書いてみる
```

### 8.4 全てを1つのプロンプトで解決しようとする

```
❌ アンチパターン:
  "ECサイトの商品管理機能を全部作って"

問題:
  - コンテキストが大きすぎてAI が混乱
  - 生成されるコードの品質が低下
  - レビューが困難

✅ 正しいアプローチ:
  1. "商品の型定義を作って"
  2. "商品のCRUD APIエンドポイントを作って"
  3. "商品一覧コンポーネントを作って"
  4. "商品検索・フィルタ機能を追加して"
  5. "各機能のテストを書いて"
  → 小さなタスクに分割して段階的に進める
```


---

## 実践演習

### 演習1: 基本的な実装

以下の要件を満たすコードを実装してください。

**要件:**
- 入力データの検証を行うこと
- エラーハンドリングを適切に実装すること
- テストコードも作成すること

```python
# 演習1: 基本実装のテンプレート
class Exercise1:
    """基本的な実装パターンの演習"""

    def __init__(self):
        self.data = []

    def validate_input(self, value):
        """入力値の検証"""
        if value is None:
            raise ValueError("入力値がNoneです")
        return True

    def process(self, value):
        """データ処理のメインロジック"""
        self.validate_input(value)
        self.data.append(value)
        return self.data

    def get_results(self):
        """処理結果の取得"""
        return {
            'count': len(self.data),
            'data': self.data
        }

# テスト
def test_exercise1():
    ex = Exercise1()
    assert ex.process(1) == [1]
    assert ex.process(2) == [1, 2]
    assert ex.get_results()['count'] == 2

    try:
        ex.process(None)
        assert False, "例外が発生するべき"
    except ValueError:
        pass

    print("全テスト合格!")

test_exercise1()
```

### 演習2: 応用パターン

基本実装を拡張して、以下の機能を追加してください。

```python
# 演習2: 応用パターン
from typing import List, Dict, Optional
from datetime import datetime

class AdvancedExercise:
    """応用パターンの演習"""

    def __init__(self, max_size: int = 100):
        self._items: List[Dict] = []
        self._max_size = max_size
        self._created_at = datetime.now()

    def add(self, key: str, value: any) -> bool:
        """アイテムの追加（サイズ制限付き）"""
        if len(self._items) >= self._max_size:
            return False
        self._items.append({
            'key': key,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
        return True

    def find(self, key: str) -> Optional[Dict]:
        """キーによる検索"""
        for item in reversed(self._items):
            if item['key'] == key:
                return item
        return None

    def remove(self, key: str) -> bool:
        """キーによる削除"""
        for i, item in enumerate(self._items):
            if item['key'] == key:
                self._items.pop(i)
                return True
        return False

    def stats(self) -> Dict:
        """統計情報"""
        return {
            'total_items': len(self._items),
            'max_size': self._max_size,
            'usage_percent': len(self._items) / self._max_size * 100,
            'uptime': str(datetime.now() - self._created_at)
        }

# テスト
def test_advanced():
    ex = AdvancedExercise(max_size=3)
    assert ex.add("a", 1) == True
    assert ex.add("b", 2) == True
    assert ex.add("c", 3) == True
    assert ex.add("d", 4) == False  # サイズ制限
    assert ex.find("b")['value'] == 2
    assert ex.remove("b") == True
    assert ex.find("b") is None
    stats = ex.stats()
    assert stats['total_items'] == 2
    print("応用テスト全合格!")

test_advanced()
```

### 演習3: パフォーマンス最適化

以下のコードのパフォーマンスを改善してください。

```python
# 演習3: パフォーマンス最適化
import time
from functools import lru_cache

# 最適化前（O(n^2)）
def slow_search(data: list, target: int) -> int:
    """非効率な検索"""
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i] + data[j] == target:
                return (i, j)
    return (-1, -1)

# 最適化後（O(n)）
def fast_search(data: list, target: int) -> tuple:
    """ハッシュマップを使った効率的な検索"""
    seen = {}
    for i, num in enumerate(data):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
    return (-1, -1)

# ベンチマーク
def benchmark():
    import random
    data = list(range(5000))
    random.shuffle(data)
    target = data[100] + data[4000]

    start = time.time()
    result1 = slow_search(data, target)
    slow_time = time.time() - start

    start = time.time()
    result2 = fast_search(data, target)
    fast_time = time.time() - start

    print(f"非効率版: {slow_time:.4f}秒")
    print(f"効率版:   {fast_time:.6f}秒")
    print(f"高速化率: {slow_time/fast_time:.0f}倍")

benchmark()
```

**ポイント:**
- アルゴリズムの計算量を意識する
- 適切なデータ構造を選択する
- ベンチマークで効果を測定する

---

## トラブルシューティング

### よくあるエラーと解決策

| エラー | 原因 | 解決策 |
|--------|------|--------|
| 初期化エラー | 設定ファイルの不備 | 設定ファイルのパスと形式を確認 |
| タイムアウト | ネットワーク遅延/リソース不足 | タイムアウト値の調整、リトライ処理の追加 |
| メモリ不足 | データ量の増大 | バッチ処理の導入、ページネーションの実装 |
| 権限エラー | アクセス権限の不足 | 実行ユーザーの権限確認、設定の見直し |
| データ不整合 | 並行処理の競合 | ロック機構の導入、トランザクション管理 |

### デバッグの手順

1. **エラーメッセージの確認**: スタックトレースを読み、発生箇所を特定する
2. **再現手順の確立**: 最小限のコードでエラーを再現する
3. **仮説の立案**: 考えられる原因をリストアップする
4. **段階的な検証**: ログ出力やデバッガを使って仮説を検証する
5. **修正と回帰テスト**: 修正後、関連する箇所のテストも実行する

```python
# デバッグ用ユーティリティ
import logging
import traceback
from functools import wraps

# ロガーの設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def debug_decorator(func):
    """関数の入出力をログ出力するデコレータ"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"呼び出し: {func.__name__}(args={args}, kwargs={kwargs})")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"戻り値: {func.__name__} -> {result}")
            return result
        except Exception as e:
            logger.error(f"例外発生: {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

@debug_decorator
def process_data(items):
    """データ処理（デバッグ対象）"""
    if not items:
        raise ValueError("空のデータ")
    return [item * 2 for item in items]
```

### パフォーマンス問題の診断

パフォーマンス問題が発生した場合の診断手順:

1. **ボトルネックの特定**: プロファイリングツールで計測
2. **メモリ使用量の確認**: メモリリークの有無をチェック
3. **I/O待ちの確認**: ディスクやネットワークI/Oの状況を確認
4. **同時接続数の確認**: コネクションプールの状態を確認

| 問題の種類 | 診断ツール | 対策 |
|-----------|-----------|------|
| CPU負荷 | cProfile, py-spy | アルゴリズム改善、並列化 |
| メモリリーク | tracemalloc, objgraph | 参照の適切な解放 |
| I/Oボトルネック | strace, iostat | 非同期I/O、キャッシュ |
| DB遅延 | EXPLAIN, slow query log | インデックス、クエリ最適化 |

---

## 設計判断ガイド

### 選択基準マトリクス

技術選択を行う際の判断基準を以下にまとめます。

| 判断基準 | 重視する場合 | 妥協できる場合 |
|---------|------------|-------------|
| パフォーマンス | リアルタイム処理、大規模データ | 管理画面、バッチ処理 |
| 保守性 | 長期運用、チーム開発 | プロトタイプ、短期プロジェクト |
| スケーラビリティ | 成長が見込まれるサービス | 社内ツール、固定ユーザー |
| セキュリティ | 個人情報、金融データ | 公開データ、社内利用 |
| 開発速度 | MVP、市場投入スピード | 品質重視、ミッションクリティカル |

### アーキテクチャパターンの選択

```
┌─────────────────────────────────────────────────┐
│              アーキテクチャ選択フロー              │
├─────────────────────────────────────────────────┤
│                                                 │
│  ① チーム規模は？                                │
│    ├─ 小規模（1-5人）→ モノリス                   │
│    └─ 大規模（10人+）→ ②へ                       │
│                                                 │
│  ② デプロイ頻度は？                               │
│    ├─ 週1回以下 → モノリス + モジュール分割         │
│    └─ 毎日/複数回 → ③へ                          │
│                                                 │
│  ③ チーム間の独立性は？                            │
│    ├─ 高い → マイクロサービス                      │
│    └─ 中程度 → モジュラーモノリス                   │
│                                                 │
└─────────────────────────────────────────────────┘
```

### トレードオフの分析

技術的な判断には必ずトレードオフが伴います。以下の観点で分析を行いましょう:

**1. 短期 vs 長期のコスト**
- 短期的に速い方法が長期的には技術的負債になることがある
- 逆に、過剰な設計は短期的なコストが高く、プロジェクトの遅延を招く

**2. 一貫性 vs 柔軟性**
- 統一された技術スタックは学習コストが低い
- 多様な技術の採用は適材適所が可能だが、運用コストが増加

**3. 抽象化のレベル**
- 高い抽象化は再利用性が高いが、デバッグが困難になる場合がある
- 低い抽象化は直感的だが、コードの重複が発生しやすい

```python
# 設計判断の記録テンプレート
class ArchitectureDecisionRecord:
    """ADR (Architecture Decision Record) の作成"""

    def __init__(self, title: str):
        self.title = title
        self.context = ""
        self.decision = ""
        self.consequences = []
        self.alternatives = []

    def set_context(self, context: str):
        """背景と課題の記述"""
        self.context = context
        return self

    def set_decision(self, decision: str):
        """決定内容の記述"""
        self.decision = decision
        return self

    def add_consequence(self, consequence: str, positive: bool = True):
        """結果の追加"""
        self.consequences.append({
            'description': consequence,
            'type': 'positive' if positive else 'negative'
        })
        return self

    def add_alternative(self, name: str, reason_rejected: str):
        """却下した代替案の追加"""
        self.alternatives.append({
            'name': name,
            'reason_rejected': reason_rejected
        })
        return self

    def to_markdown(self) -> str:
        """Markdown形式で出力"""
        md = f"# ADR: {self.title}\n\n"
        md += f"## 背景\n{self.context}\n\n"
        md += f"## 決定\n{self.decision}\n\n"
        md += "## 結果\n"
        for c in self.consequences:
            icon = "✅" if c['type'] == 'positive' else "⚠️"
            md += f"- {icon} {c['description']}\n"
        md += "\n## 却下した代替案\n"
        for a in self.alternatives:
            md += f"- **{a['name']}**: {a['reason_rejected']}\n"
        return md
```

---

## 実務での適用シナリオ

### シナリオ1: スタートアップでのMVP開発

**状況:** 限られたリソースで素早くプロダクトをリリースする必要がある

**アプローチ:**
- シンプルなアーキテクチャを選択
- 必要最小限の機能に集中
- 自動テストはクリティカルパスのみ
- モニタリングは早期から導入

**学んだ教訓:**
- 完璧を求めすぎない（YAGNI原則）
- ユーザーフィードバックを早期に取得
- 技術的負債は意識的に管理する

### シナリオ2: レガシーシステムのモダナイゼーション

**状況:** 10年以上運用されているシステムを段階的に刷新する

**アプローチ:**
- Strangler Fig パターンで段階的に移行
- 既存のテストがない場合はCharacterization Testを先に作成
- APIゲートウェイで新旧システムを共存
- データ移行は段階的に実施

| フェーズ | 作業内容 | 期間目安 | リスク |
|---------|---------|---------|--------|
| 1. 調査 | 現状分析、依存関係の把握 | 2-4週間 | 低 |
| 2. 基盤 | CI/CD構築、テスト環境 | 4-6週間 | 低 |
| 3. 移行開始 | 周辺機能から順次移行 | 3-6ヶ月 | 中 |
| 4. コア移行 | 中核機能の移行 | 6-12ヶ月 | 高 |
| 5. 完了 | 旧システム廃止 | 2-4週間 | 中 |

### シナリオ3: 大規模チームでの開発

**状況:** 50人以上のエンジニアが同一プロダクトを開発する

**アプローチ:**
- ドメイン駆動設計で境界を明確化
- チームごとにオーナーシップを設定
- 共通ライブラリはInner Source方式で管理
- APIファーストで設計し、チーム間の依存を最小化

```python
# チーム間のAPI契約定義
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class APIContract:
    """チーム間のAPI契約"""
    endpoint: str
    method: str
    owner_team: str
    consumers: List[str]
    sla_ms: int  # レスポンスタイムSLA
    priority: Priority

    def validate_sla(self, actual_ms: int) -> bool:
        """SLA準拠の確認"""
        return actual_ms <= self.sla_ms

    def to_openapi(self) -> dict:
        """OpenAPI形式で出力"""
        return {
            'path': self.endpoint,
            'method': self.method,
            'x-owner': self.owner_team,
            'x-consumers': self.consumers,
            'x-sla-ms': self.sla_ms
        }

# 使用例
contracts = [
    APIContract(
        endpoint="/api/v1/users",
        method="GET",
        owner_team="user-team",
        consumers=["order-team", "notification-team"],
        sla_ms=200,
        priority=Priority.HIGH
    ),
    APIContract(
        endpoint="/api/v1/orders",
        method="POST",
        owner_team="order-team",
        consumers=["payment-team", "inventory-team"],
        sla_ms=500,
        priority=Priority.CRITICAL
    )
]
```

### シナリオ4: パフォーマンスクリティカルなシステム

**状況:** ミリ秒単位のレスポンスが求められるシステム

**最適化ポイント:**
1. キャッシュ戦略（L1: インメモリ、L2: Redis、L3: CDN）
2. 非同期処理の活用
3. コネクションプーリング
4. クエリ最適化とインデックス設計

| 最適化手法 | 効果 | 実装コスト | 適用場面 |
|-----------|------|-----------|---------|
| インメモリキャッシュ | 高 | 低 | 頻繁にアクセスされるデータ |
| CDN | 高 | 低 | 静的コンテンツ |
| 非同期処理 | 中 | 中 | I/O待ちが多い処理 |
| DB最適化 | 高 | 高 | クエリが遅い場合 |
| コード最適化 | 低-中 | 高 | CPU律速の場合 |

---

## チーム開発での活用

### コードレビューのチェックリスト

このトピックに関連するコードレビューで確認すべきポイント:

- [ ] 命名規則が一貫しているか
- [ ] エラーハンドリングが適切か
- [ ] テストカバレッジは十分か
- [ ] パフォーマンスへの影響はないか
- [ ] セキュリティ上の問題はないか
- [ ] ドキュメントは更新されているか

### ナレッジ共有のベストプラクティス

| 方法 | 頻度 | 対象 | 効果 |
|------|------|------|------|
| ペアプログラミング | 随時 | 複雑なタスク | 即時のフィードバック |
| テックトーク | 週1回 | チーム全体 | 知識の水平展開 |
| ADR (設計記録) | 都度 | 将来のメンバー | 意思決定の透明性 |
| 振り返り | 2週間ごと | チーム全体 | 継続的改善 |
| モブプログラミング | 月1回 | 重要な設計 | 合意形成 |

### 技術的負債の管理

```
優先度マトリクス:

        影響度 高
          │
    ┌─────┼─────┐
    │ 計画 │ 即座 │
    │ 的に │ に   │
    │ 対応 │ 対応 │
    ├─────┼─────┤
    │ 記録 │ 次の │
    │ のみ │ Sprint│
    │     │ で   │
    └─────┼─────┘
          │
        影響度 低
    発生頻度 低  発生頻度 高
```

---

## マイグレーションガイド

### バージョンアップ時の注意点

| バージョン | 主な変更点 | 移行作業 | 影響範囲 |
|-----------|-----------|---------|---------|
| v1.x → v2.x | API設計の刷新 | エンドポイント変更 | 全クライアント |
| v2.x → v3.x | 認証方式の変更 | トークン形式更新 | 認証関連 |
| v3.x → v4.x | データモデル変更 | マイグレーションスクリプト実行 | DB関連 |

### 段階的移行の手順

```python
# マイグレーションスクリプトのテンプレート
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Callable

logger = logging.getLogger(__name__)

class MigrationRunner:
    """段階的マイグレーション実行エンジン"""

    def __init__(self, migration_dir: str):
        self.migration_dir = Path(migration_dir)
        self.migrations: List[Dict] = []
        self.completed: List[str] = []

    def register(self, version: str, description: str,
                 up: Callable, down: Callable):
        """マイグレーションの登録"""
        self.migrations.append({
            'version': version,
            'description': description,
            'up': up,
            'down': down,
            'registered_at': datetime.now().isoformat()
        })

    def run_up(self, target_version: str = None):
        """マイグレーションの実行（アップグレード）"""
        for migration in self.migrations:
            if migration['version'] in self.completed:
                continue
            logger.info(f"実行中: {migration['version']} - "
                       f"{migration['description']}")
            try:
                migration['up']()
                self.completed.append(migration['version'])
                logger.info(f"完了: {migration['version']}")
            except Exception as e:
                logger.error(f"失敗: {migration['version']}: {e}")
                raise
            if target_version and migration['version'] == target_version:
                break

    def run_down(self, target_version: str):
        """マイグレーションのロールバック"""
        for migration in reversed(self.migrations):
            if migration['version'] not in self.completed:
                continue
            if migration['version'] == target_version:
                break
            logger.info(f"ロールバック: {migration['version']}")
            migration['down']()
            self.completed.remove(migration['version'])

    def status(self) -> Dict:
        """マイグレーション状態の確認"""
        return {
            'total': len(self.migrations),
            'completed': len(self.completed),
            'pending': len(self.migrations) - len(self.completed),
            'versions': {
                m['version']: 'completed'
                if m['version'] in self.completed else 'pending'
                for m in self.migrations
            }
        }
```

### ロールバック計画

移行作業には必ずロールバック計画を準備してください:

1. **データのバックアップ**: 移行前に完全バックアップを取得
2. **テスト環境での検証**: 本番と同等の環境で事前検証
3. **段階的なロールアウト**: カナリアリリースで段階的に展開
4. **監視の強化**: 移行中はメトリクスの監視間隔を短縮
5. **判断基準の明確化**: ロールバックを判断する基準を事前に定義
---

## 9. FAQ

### Q1: Copilot と Cursor、両方契約すべき？

**A:** 基本的にはどちらか一方で十分。VS Code ベースのワークフローを変えたくないなら Copilot、AI ファーストの体験を求めるなら Cursor。Cursor は VS Code の fork なので、Copilot 拡張も併用可能だが、補完が競合する場合がある。予算に余裕があれば Claude Code CLI + Copilot の組み合わせが最もカバー範囲が広い。Cursor の Composer 機能はマルチファイル編集に強く、Claude Code のエージェント機能は自律的なタスク実行に強い。

### Q2: AI ツールで生成したコードの著作権は？

**A:** 2026年時点で法的にはグレーゾーンが多いが、主要ツールの利用規約では生成コードの権利はユーザーに帰属するとされている。ただし、既存のOSSコードに酷似した出力には注意が必要。Copilot Business/Enterprise の `duplicationDetection: "block"` 設定で公開コードに類似する提案をフィルタリングできる。企業利用では法務確認を推奨する。

### Q3: Claude Code の CLAUDE.md は何を書くべき？

**A:** 以下の要素を含める。
1. プロジェクト概要（技術スタック、アーキテクチャ）
2. コーディング規約（命名規則、パターン）
3. ディレクトリ構造の説明
4. よく使うコマンド（ビルド、テスト、デプロイ）
5. やってはいけないこと（禁止パターン）
6. 環境変数の説明（値は書かない）
7. DB スキーマの概要

チームで共有する情報なのでリポジトリにコミットすべき。機密情報（APIキー等）は絶対に書かない。

### Q4: AI ツールのコストを抑えるには？

**A:** 以下の戦略が有効。
1. CLAUDE.md / .cursorrules をしっかり書いて、やり直しの回数を減らす
2. 小さなタスクに分割して、1回のプロンプトで確実に成果を得る
3. Copilot (定額) でインライン補完、Claude Code (従量) で複雑なタスクと使い分ける
4. `/compact` コマンドでコンテキストを圧縮し、トークン消費を抑える
5. 不要なファイルを .aiignore で除外し、コンテキスト汚染を防ぐ

### Q5: MCP とは何か？導入すべき？

**A:** MCP (Model Context Protocol) は AI モデルが外部ツール・データソースと連携するためのオープンプロトコル。Anthropic が策定し、GitHub、Cursor、Windsurf 等が採用。データベース、GitHub Issues、ドキュメント検索等を AI が直接参照できるようになる。開発ワークフローの自動化が大幅に進むため、チーム開発では積極的に導入を推奨する。ただし、データベースへの書き込み権限等はセキュリティリスクがあるため、読み取り専用からスタートするのが安全。

---


## FAQ

### Q1: このトピックを学ぶ上で最も重要なポイントは何ですか？

実践的な経験を積むことが最も重要です。理論だけでなく、実際にコードを書いて動作を確認することで理解が深まります。

### Q2: 初心者がよく陥る間違いは何ですか？

基礎を飛ばして応用に進むことです。このガイドで説明している基本概念をしっかり理解してから、次のステップに進むことをお勧めします。

### Q3: 実務ではどのように活用されていますか？

このトピックの知識は、日常的な開発業務で頻繁に活用されます。特にコードレビューやアーキテクチャ設計の際に重要になります。

---

## 10. まとめ

| ツール | 主な用途 | 導入コスト | 効果 |
|--------|---------|-----------|------|
| GitHub Copilot | インライン補完・チャット | $10-39/月 | コード記述速度 2-3倍 |
| Claude Code | エージェント・複雑タスク | 従量制 | リファクタ・レビュー自動化 |
| Cursor | AI統合エディタ | $20/月 | ファイル横断編集 |
| CLAUDE.md | プロジェクト文脈共有 | 無料 | AI 出力品質向上 |
| .cursorrules | Cursor 文脈設定 | 無料 | Cursor 出力品質向上 |
| MCP | ツール連携プロトコル | 無料 | AI の行動範囲拡大 |
| Aider | CLI エージェント | 無料(API費用) | Git 統合が優秀 |

---

## 次に読むべきガイド

- [00-vscode-setup.md](./00-vscode-setup.md) -- VS Code の詳細設定
- [../01-runtime-and-package/03-linter-formatter.md](../01-runtime-and-package/03-linter-formatter.md) -- AI 生成コードの品質チェック
- [../03-team-setup/00-project-standards.md](../03-team-setup/00-project-standards.md) -- チーム標準の設定

---

## 参考文献

1. **GitHub Copilot Documentation** -- https://docs.github.com/en/copilot -- Copilot の公式ドキュメント。設定から活用法まで。
2. **Claude Code CLI** -- https://docs.anthropic.com/en/docs/claude-code -- Claude Code の公式ドキュメント。
3. **Cursor Documentation** -- https://docs.cursor.com -- Cursor エディタの公式ドキュメントと設定ガイド。
4. **Pragmatic AI-Assisted Development** -- https://martinfowler.com/articles/exploring-gen-ai.html -- Martin Fowler による AI 開発ツールの実践的考察。
5. **Model Context Protocol (MCP)** -- https://modelcontextprotocol.io/ -- MCP の公式仕様ドキュメント。
6. **Aider** -- https://aider.chat/ -- Aider の公式サイト。Git 統合 AI ペアプログラミング。
7. **AI Code Review Best Practices** -- https://github.blog/developer-skills/github/how-to-review-code-generated-by-ai/ -- GitHub による AI 生成コードのレビュー手法。
8. **OWASP AI Security** -- https://owasp.org/www-project-ai-security-and-privacy-guide/ -- AI 活用時のセキュリティガイドライン。
