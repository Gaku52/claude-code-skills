# Cursor / Windsurf ── AI IDE、コンテキスト管理

> AIをエディタの中核に据えた次世代IDE「Cursor」と「Windsurf」の特徴・機能・活用法を比較し、プロジェクトに最適なAI IDEを選定する判断基準を身につける。

---

## この章で学ぶこと

1. **AI IDEの設計思想** ── 従来のIDEとAI IDEの根本的な違いを理解する
2. **CursorとWindsurfの機能詳細** ── 各ツールの操作方法、コンテキスト管理、差別化ポイントを把握する
3. **最適なAI IDE選定** ── プロジェクト特性やチーム規模に応じた選定基準を学ぶ
4. **実践的な活用パターン** ── 日常開発からチーム運用まで、AI IDEの効果を最大化するテクニックを習得する


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解
- [Claude Code ── CLI、エージェント、MCP](./01-claude-code.md) の内容を理解していること

---

## 1. AI IDEの設計思想

### 1.1 従来のIDEとAI IDEの違い

```
従来のIDE (VSCode等)                  AI IDE (Cursor/Windsurf)
┌─────────────────────┐              ┌─────────────────────┐
│  エディタ           │              │  エディタ           │
│  ┌───────────────┐  │              │  ┌───────────────┐  │
│  │ シンタックス   │  │              │  │ AI補完エンジン │  │
│  │ ハイライト     │  │              │  │ (ネイティブ)   │  │
│  ├───────────────┤  │              │  ├───────────────┤  │
│  │ LSP          │  │              │  │ コンテキスト   │  │
│  │ (言語サーバー) │  │              │  │ インデクサー   │  │
│  ├───────────────┤  │              │  ├───────────────┤  │
│  │ 拡張機能      │  │              │  │ AIチャット     │  │
│  │ (プラグイン)  │  │  AI は       │  │ (組み込み)     │  │
│  │  ┌─────────┐ │  │  後付け      │  ├───────────────┤  │
│  │  │Copilot  │ │  │  ─────►     │  │ Agent Mode    │  │
│  │  │(後付け)  │ │  │              │  │ (自律実行)     │  │
│  │  └─────────┘ │  │              │  ├───────────────┤  │
│  └───────────────┘  │              │  │ マルチファイル │  │
│                     │              │  │ 同時編集       │  │
│                     │              │  └───────────────┘  │
└─────────────────────┘              └─────────────────────┘
```

### 1.2 AI IDEのコンテキスト管理

```
┌──────────────────────────────────────────────────┐
│          AI IDEのコンテキスト階層                   │
│                                                  │
│  レベル1: カーソル位置                             │
│  ┌──────────────────────────────────┐            │
│  │ 現在の行の前後数十行              │            │
│  │ → インライン補完に使用            │            │
│  └──────────────────────────────────┘            │
│                                                  │
│  レベル2: ファイル                                │
│  ┌──────────────────────────────────┐            │
│  │ 開いているファイル全体            │            │
│  │ → 関数補完、リファクタリング      │            │
│  └──────────────────────────────────┘            │
│                                                  │
│  レベル3: プロジェクト                             │
│  ┌──────────────────────────────────┐            │
│  │ コードベース全体のインデックス     │            │
│  │ → Codebase検索、@記法で参照       │            │
│  └──────────────────────────────────┘            │
│                                                  │
│  レベル4: 外部知識                                │
│  ┌──────────────────────────────────┐            │
│  │ Docs、Web検索、MCP接続            │            │
│  │ → 最新APIドキュメント等           │            │
│  └──────────────────────────────────┘            │
└──────────────────────────────────────────────────┘
```

### 1.3 AI IDEのアーキテクチャ比較

```
┌──────────────────────────────────────────────────────┐
│                 AI IDEのアーキテクチャ                  │
│                                                      │
│  Cursor                          Windsurf             │
│  ┌────────────────────┐         ┌─────────────────┐  │
│  │ VSCode Fork        │         │ VSCode Fork     │  │
│  │ ┌────────────────┐ │         │ ┌─────────────┐ │  │
│  │ │ Copilot++      │ │         │ │Supercomplete│ │  │
│  │ │ (Tab補完)      │ │         │ │(ブロック補完)│ │  │
│  │ ├────────────────┤ │         │ ├─────────────┤ │  │
│  │ │ Chat (Cmd+L)   │ │         │ │ Cascade     │ │  │
│  │ │ 対話型チャット  │ │         │ │ (AIアシスタ │ │  │
│  │ ├────────────────┤ │         │ │  ント)      │ │  │
│  │ │ Composer       │ │         │ ├─────────────┤ │  │
│  │ │ (Cmd+I)        │ │         │ │ Flows       │ │  │
│  │ │ マルチファイル  │ │         │ │ (再利用可能 │ │  │
│  │ ├────────────────┤ │         │ │  ワークフロ │ │  │
│  │ │ @記法         │ │         │ │  ー)        │ │  │
│  │ │ コンテキスト   │ │         │ ├─────────────┤ │  │
│  │ │ 指定          │ │         │ │ 自動インデ  │ │  │
│  │ ├────────────────┤ │         │ │ ックス      │ │  │
│  │ │ Agent Mode     │ │         │ ├─────────────┤ │  │
│  │ │ 自律タスク実行 │ │         │ │ Agent Mode  │ │  │
│  │ └────────────────┘ │         │ │ 自律タスク  │ │  │
│  └────────────────────┘         │ └─────────────┘ │  │
│                                 └─────────────────┘  │
└──────────────────────────────────────────────────────┘
```

---

## 2. Cursor詳細

### コード例1: Cursorの基本操作

```
# Cursor キーボードショートカット

## コード生成
Cmd+K          : インラインコード生成（選択範囲を変換）
Cmd+L          : チャットパネルを開く
Cmd+Shift+L    : チャットに選択コードを追加
Cmd+I          : Composer（マルチファイル編集）

## コンテキスト追加（@記法）
@file          : 特定ファイルをコンテキストに追加
@folder        : フォルダ内全ファイルを追加
@code          : コードベース検索結果を追加
@web           : Web検索結果を追加
@docs          : ドキュメント検索結果を追加
@git           : Gitの差分・履歴を追加

## Agent Mode
Cmd+.          : Agent Modeの切り替え
                 → ファイル作成、コマンド実行、テストまで自律実行
```

### コード例2: .cursorrules でプロジェクト規約を設定

```markdown
# .cursorrules

## 技術スタック
- Next.js 14 (App Router)
- TypeScript 5.4 (strict mode)
- Tailwind CSS 3.4
- Prisma (ORM)
- tRPC (API)

## コーディング規約
- コンポーネントは function 宣言で書く（アロー関数禁止）
- CSS はTailwind のみ使用（CSS Modules禁止）
- 状態管理は Zustand を使用
- フォームは React Hook Form + Zod
- データフェッチは TanStack Query

## ファイル命名
- コンポーネント: PascalCase (UserProfile.tsx)
- ユーティリティ: camelCase (formatDate.ts)
- 定数: SCREAMING_SNAKE_CASE
- テスト: *.test.ts / *.test.tsx

## 禁止事項
- any型の使用
- console.logの本番コードへの残存
- useEffect内でのデータフェッチ
- 200行を超えるコンポーネント
```

### コード例3: Cursor Composerの活用

```typescript
// Cmd+I でComposerを開き、マルチファイル編集を指示

// プロンプト例:
// "ユーザープロファイルページを作成して。
//  以下のファイルを生成:
//  1. app/profile/page.tsx - プロファイル表示ページ
//  2. components/ProfileCard.tsx - プロファイルカード
//  3. hooks/useProfile.ts - プロファイル取得フック
//  4. lib/api/profile.ts - API呼び出し
//  5. __tests__/ProfileCard.test.tsx - テスト"

// → Composerが5ファイルを同時に生成・編集
// → 各ファイルの差分をプレビューで確認可能
// → Accept / Reject を選択
```

### コード例4: Cursor @記法の高度な活用

```
# @記法のパワーユーザー活用例

## コードベース検索で関連コードを発見
@codebase "認証に関する関数を全て検索して、
          セッション管理の脆弱性がないか確認して"

## 特定ファイルをピンポイントで参照
@file src/types/user.ts
@file src/schemas/user.schema.ts
"UserTypeとUserSchemaの整合性を確認して。
 不整合があれば修正して"

## Gitの差分をコンテキストに含める
@git diff main
"mainブランチからの変更をレビューして。
 パフォーマンスへの影響がある変更を指摘して"

## 公式ドキュメントを参照して実装
@docs Next.js App Router
"Server Actionsを使ったフォーム送信を実装して。
 最新のNext.js App Routerの仕様に準拠して"

## Web検索で最新情報を取得
@web "React 19 use hook"
"React 19のuseフックを使ったデータフェッチに
 リファクタリングして"

## フォルダ全体を参照
@folder src/components/ui/
"このUIコンポーネントライブラリに
 Tooltipコンポーネントを追加して。
 既存コンポーネントのスタイルに合わせて"
```

### コード例5: Cursor Agent Modeの実践

```typescript
// Agent Modeでの開発フロー
// Cmd+. で Agent Mode を有効化

// === プロンプト例 1: 機能実装 ===
// "通知システムを実装して:
//  1. NotificationServiceクラスを作成
//  2. WebSocket接続でリアルタイム通知
//  3. 通知の永続化（DBに保存）
//  4. 未読カウントのAPI
//  5. テストを含める
//  6. テストを実行して全て通ることを確認"

// Agent Modeが実行する内容:
// Step 1: 関連ファイルの調査
//   → プロジェクト構造の把握
//   → 既存のサービスパターンの確認
//
// Step 2: ファイル生成・編集
//   → src/services/notification.ts
//   → src/api/notifications/route.ts
//   → src/hooks/useNotifications.ts
//   → prisma/schema.prisma (スキーマ追加)
//
// Step 3: テスト実行
//   → npm test -- --watch=false
//   → 失敗したテストの自動修正
//
// Step 4: 結果レポート
//   → 変更したファイルの一覧
//   → テスト結果のサマリー

// === プロンプト例 2: バグ修正 ===
// "ユーザーがプロフィール画像をアップロードすると
//  500エラーが発生する。原因を調査して修正して。
//  1. エラーログを確認
//  2. 関連コードを調査
//  3. 原因を特定
//  4. 修正を実装
//  5. テストを追加して確認"
```

### コード例6: Cursor設定の最適化

```jsonc
// .vscode/settings.json（Cursorでも有効）
{
  // AI補完の設定
  "cursor.cpp.enablePartialAccepts": true,  // 部分的な補完の受け入れ
  "cursor.chat.defaultModel": "claude-sonnet-4-20250514",

  // コンテキスト設定
  "cursor.chat.alwaysSearchWeb": false,      // 必要時のみWeb検索
  "cursor.general.enableShadowWorkspace": true, // バックグラウンドインデックス

  // 補完の挙動
  "editor.inlineSuggest.enabled": true,
  "editor.suggest.preview": true,

  // AI関連のファイル除外（パフォーマンス最適化）
  "cursor.general.ignoredPaths": [
    "node_modules",
    ".next",
    "dist",
    "coverage",
    "*.min.js",
    "*.bundle.js"
  ]
}
```

---

## 3. Windsurf詳細

### コード例7: Windsurf（Cascade）の特徴

```
# Windsurf の特徴的機能

## Cascade（AIアシスタント）
- コードベース全体を自動でインデックス
- 自然言語でファイル操作・コード生成
- マルチステップの変更を自動実行
- 変更のプレビューと承認フロー

## Flows
- AIとのやり取りを「Flow」として保存
- 過去のFlowを再実行可能
- チームでFlowを共有

## Supercomplete
- Copilotより高度な補完
- 行単位ではなくブロック単位の補完
- 直前の編集パターンを学習して適応
```

### コード例8: Windsurf Cascade の実践例

```typescript
// Windsurf Cascade でのリファクタリング例

// プロンプト: "このファイルをClean Architectureに分割して"

// Before: 単一ファイルに全てが混在
// src/features/todo.ts (200行)

// After: Cascadeが自動分割
// src/features/todo/
// ├── domain/
// │   ├── Todo.ts          (エンティティ)
// │   └── TodoRepository.ts (リポジトリインターフェース)
// ├── application/
// │   ├── CreateTodoUseCase.ts
// │   ├── CompleteTodoUseCase.ts
// │   └── ListTodosUseCase.ts
// ├── infrastructure/
// │   └── PrismaTodoRepository.ts
// └── presentation/
//     ├── TodoController.ts
//     └── todoRouter.ts

// Cascadeは既存のimportパスも全て自動更新する
```

### コード例9: Windsurf Flowsの活用

```
# Windsurf Flows の実践活用

## Flow 1: コンポーネント生成テンプレート
保存名: "React Component Generator"
再利用可能なFlowとして保存:

プロンプト:
"以下のパターンでReactコンポーネントを生成して:
1. コンポーネント本体（function宣言）
2. Props型定義
3. Storybookファイル
4. テストファイル
5. CSSモジュール

コンポーネント名: {name}
Props: {props}
ディレクトリ: src/components/{name}/"

## Flow 2: API エンドポイント追加
保存名: "API Endpoint Scaffold"
プロンプト:
"新しいAPIエンドポイントを追加して:
1. ルートハンドラ (app/api/{resource}/route.ts)
2. バリデーションスキーマ (schemas/{resource}.ts)
3. サービスクラス (services/{resource}Service.ts)
4. Prismaモデル追加 (prisma/schema.prisma)
5. APIテスト (__tests__/api/{resource}.test.ts)

リソース名: {resource}
CRUD: {operations}"

## Flow 3: バグ修正ワークフロー
保存名: "Bug Investigation"
プロンプト:
"以下の手順でバグを調査して修正して:
1. エラーメッセージから関連コードを特定
2. 原因を分析
3. 修正を実装
4. 回帰テストを追加
5. 変更のサマリーを出力

エラー: {error_description}"
```

### コード例10: Windsurf Supercomplete の効果

```typescript
// Supercompleteの動作例

// ユーザーが入力し始めると、Supercompleteがブロック単位で補完を提案

// 入力: "async function fetchUser"
// Supercomplete提案（ブロック単位）:
async function fetchUserById(userId: string): Promise<User | null> {
  try {
    const response = await prisma.user.findUnique({
      where: { id: userId },
      include: {
        profile: true,
        preferences: true,
      },
    });

    if (!response) {
      return null;
    }

    return mapToUser(response);
  } catch (error) {
    logger.error('Failed to fetch user', { userId, error });
    throw new DatabaseError('User fetch failed', { cause: error });
  }
}

// 特徴:
// - 行単位ではなく関数全体を提案
// - プロジェクトの既存パターン（prisma, logger, エラー型）を学習
// - try-catchパターンやエラーハンドリングも含む
// - 直前の編集内容（例: 別のfetch関数）のパターンを踏襲
```

---

## 4. 機能比較

### 4.1 Cursor vs Windsurf vs 従来IDE+Copilot

| 機能 | Cursor | Windsurf | VSCode+Copilot |
|------|--------|----------|----------------|
| インライン補完 | Tab (高品質) | Supercomplete | Copilot |
| チャット | Cmd+L | Cascade | Copilot Chat |
| マルチファイル編集 | Composer | Cascade | 非対応 |
| Agent Mode | あり | あり | 限定的 |
| @記法コンテキスト | 豊富 | 基本的 | 限定的 |
| Codebase検索 | @codebase | 自動 | 非対応 |
| モデル選択 | Claude/GPT/等 | Claude/GPT | GPT系のみ |
| 料金 | $20/月 (Pro) | $15/月 (Pro) | $10/月 |
| ベースエディタ | VSCode fork | VSCode fork | VSCode本体 |
| 拡張機能互換 | ほぼ完全 | ほぼ完全 | 完全 |
| オフライン | 不可 | 不可 | Copilot不可 |
| ワークフロー保存 | 不可 | Flows | 不可 |
| カスタムDocs | @docs | 限定的 | 不可 |
| プロジェクト規約 | .cursorrules | Cascade設定 | 拡張機能依存 |

### 4.2 ユースケース別推奨

| ユースケース | 推奨ツール | 理由 |
|-------------|-----------|------|
| フロントエンド開発 | Cursor | Composerでのマルチファイル生成が強力 |
| バックエンドAPI | Claude Code | CLIでテスト実行→修正ループが効率的 |
| スタートアップ | Windsurf | 低コストで高機能 |
| エンタープライズ | VSCode+Copilot | セキュリティ・コンプライアンス対応 |
| データサイエンス | Cursor | Notebook対応 + @docs でライブラリ参照 |
| インフラ/DevOps | Claude Code | Bash実行 + 設定ファイル操作が得意 |
| モバイル開発 | Cursor | React Native/Flutter対応が良好 |
| OSS開発 | Windsurf | Flowsでコントリビューションガイドを共有 |

### 4.3 詳細機能比較マトリクス

```
┌──────────────────────────────────────────────────────┐
│            詳細機能比較マトリクス                       │
│                                                      │
│  ★★★ = 優秀  ★★☆ = 良好  ★☆☆ = 基本的               │
│                                                      │
│  機能               Cursor    Windsurf    VSCode     │
│  ─────────────────  ────────  ──────────  ────────   │
│  インライン補完      ★★★      ★★★        ★★☆        │
│  チャット精度        ★★★      ★★☆        ★★☆        │
│  マルチファイル      ★★★      ★★★        ★☆☆        │
│  コードベース理解    ★★★      ★★★        ★☆☆        │
│  @記法の柔軟性      ★★★      ★★☆        ★☆☆        │
│  Agent Mode         ★★★      ★★★        ★★☆        │
│  起動速度           ★★☆      ★★★        ★★★        │
│  メモリ使用量       ★★☆      ★★★        ★★★        │
│  拡張機能互換性     ★★★      ★★☆        ★★★        │
│  ドキュメント充実度 ★★★      ★★☆        ★★★        │
│  コスパ             ★★☆      ★★★        ★★★        │
│  学習コスト         ★★☆      ★★★        ★★★        │
│  エンタープライズ   ★★☆      ★☆☆        ★★★        │
│  コミュニティ       ★★★      ★★☆        ★★★        │
└──────────────────────────────────────────────────────┘
```

---

## 5. コンテキスト管理のベストプラクティス

```
┌────────────────────────────────────────────────────┐
│        コンテキスト管理のベストプラクティス           │
│                                                    │
│  1. 必要なファイルだけをコンテキストに含める          │
│     ┌───────────────────────┐                     │
│     │ @file schema.prisma  │ ← DBスキーマ         │
│     │ @file types.ts       │ ← 型定義            │
│     │ @folder api/         │ ← 関連API           │
│     └───────────────────────┘                     │
│     ❌ @codebase (全体は重すぎる)                   │
│                                                    │
│  2. ルールファイルで暗黙知を明示化                   │
│     .cursorrules / CLAUDE.md で規約を定義           │
│                                                    │
│  3. 段階的にコンテキストを追加                      │
│     最初は最小限 → 不足なら追加                     │
│     → AIの応答速度と精度のバランスを保つ             │
│                                                    │
│  4. Docsインデックスを活用                          │
│     @docs Next.js → 公式ドキュメントを参照          │
│     → 最新APIの正確な使い方を生成                   │
└────────────────────────────────────────────────────┘
```

### 5.1 コンテキスト管理の詳細戦略

```
┌────────────────────────────────────────────────────┐
│        コンテキスト管理 ── 実践的な戦略              │
│                                                    │
│  戦略1: 段階的コンテキスト拡大                      │
│  ┌──────────────────────────────────────┐         │
│  │ Level 1: 現在のファイルのみ（自動）   │         │
│  │    ↓ AIの回答が不十分                 │         │
│  │ Level 2: @file で関連ファイルを追加   │         │
│  │    ↓ まだ不十分                       │         │
│  │ Level 3: @folder で関連モジュール追加 │         │
│  │    ↓ 依然として不十分                 │         │
│  │ Level 4: @codebase で全体検索         │         │
│  └──────────────────────────────────────┘         │
│                                                    │
│  戦略2: タスク別コンテキスト設計                    │
│  ┌──────────────────────────────────────┐         │
│  │ バグ修正:                             │         │
│  │   @file エラーが出るファイル           │         │
│  │   @file テストファイル                │         │
│  │   @git log (最近の変更)               │         │
│  │                                      │         │
│  │ 新機能:                               │         │
│  │   @file 型定義ファイル                │         │
│  │   @folder 類似機能のディレクトリ       │         │
│  │   @docs フレームワーク公式            │         │
│  │                                      │         │
│  │ リファクタリング:                      │         │
│  │   @file 対象ファイル                  │         │
│  │   @codebase 依存しているコード        │         │
│  │   @file テスト全般                    │         │
│  └──────────────────────────────────────┘         │
│                                                    │
│  戦略3: インデックス最適化                          │
│  ┌──────────────────────────────────────┐         │
│  │ .cursorignore でインデックス除外:     │         │
│  │   node_modules/                      │         │
│  │   dist/                              │         │
│  │   .next/                             │         │
│  │   coverage/                          │         │
│  │   *.min.js                           │         │
│  │   vendor/                            │         │
│  └──────────────────────────────────────┘         │
└────────────────────────────────────────────────────┘
```

---

## 6. AI IDE導入のチーム運用

### 6.1 チーム導入のロードマップ

```
┌──────────────────────────────────────────────────────┐
│          AI IDE チーム導入ロードマップ                  │
│                                                      │
│  Phase 1: パイロット（1-2週間）                       │
│  ┌──────────────────────────────────────┐            │
│  │ - 2-3名の技術リードで試験導入        │            │
│  │ - .cursorrules の初版を作成          │            │
│  │ - 効果測定の基準を定義              │            │
│  │ - セキュリティ設定の確立            │            │
│  └──────────────────────────────────────┘            │
│                                                      │
│  Phase 2: チーム展開（2-4週間）                       │
│  ┌──────────────────────────────────────┐            │
│  │ - チーム全員にライセンス配布         │            │
│  │ - 操作研修（2時間のワークショップ）   │            │
│  │ - ペアプロでの実践（先輩＋新人）      │            │
│  │ - .cursorrules のチームレビュー      │            │
│  └──────────────────────────────────────┘            │
│                                                      │
│  Phase 3: 最適化（継続的）                            │
│  ┌──────────────────────────────────────┐            │
│  │ - 効果的なプロンプトの共有Wiki        │            │
│  │ - 月次振り返り（活用度・コスト分析）  │            │
│  │ - .cursorrules の継続的改善          │            │
│  │ - 新機能のキャッチアップ             │            │
│  └──────────────────────────────────────┘            │
└──────────────────────────────────────────────────────┘
```

### 6.2 チーム共通の .cursorrules テンプレート

```markdown
# .cursorrules - チーム共通テンプレート

## プロジェクト情報
- プロジェクト名: {project_name}
- リポジトリ: {repo_url}
- 主要な技術スタック: {tech_stack}

## アーキテクチャ
- パターン: {architecture_pattern}
- ディレクトリ構成の説明:
  - src/domain/  : ビジネスロジック（外部依存なし）
  - src/app/     : アプリケーション層
  - src/infra/   : インフラストラクチャ（DB、外部API）
  - src/ui/      : プレゼンテーション層

## コーディング規約
- 言語バージョン: {language_version}
- リンター: {linter_config}
- フォーマッター: {formatter_config}
- テストフレームワーク: {test_framework}
- カバレッジ目標: {coverage_target}

## AI生成コードの品質基準
- 型安全性: strict mode必須、any禁止
- エラーハンドリング: Result型 or 明示的な例外
- テスト: 生成コードには必ずテストを含める
- ドキュメント: public関数にはJSDoc/docstring必須
- 命名: ドメイン用語集（glossary.md）に従う

## 禁止事項
- 機密情報のハードコード
- console.log/printの本番コードへの残存
- テストなしのビジネスロジック
- 200行を超えるファイル
- 循環依存の導入

## レビュー注意事項
- AI生成コードは必ず人間がレビューする
- セキュリティ関連は2名以上でレビュー
- パフォーマンス影響がある変更はベンチマーク必須
```

### 6.3 効果測定の指標

```
┌────────────────────────────────────────────────────┐
│          AI IDE 効果測定指標                         │
│                                                    │
│  定量指標:                                          │
│  ┌──────────────────────────────────────┐          │
│  │ 指標                 目標値          │          │
│  │ ──────────────────── ──────────────  │          │
│  │ PRリードタイム       30%削減         │          │
│  │ コードレビュー時間    20%削減         │          │
│  │ バグ修正時間          40%削減         │          │
│  │ テストカバレッジ      10%向上         │          │
│  │ 新機能実装時間        25%削減         │          │
│  │ ドキュメント更新頻度  50%向上         │          │
│  └──────────────────────────────────────┘          │
│                                                    │
│  定性指標:                                          │
│  ┌──────────────────────────────────────┐          │
│  │ - 開発者満足度（月次アンケート）      │          │
│  │ - コードの可読性・保守性評価          │          │
│  │ - 学習曲線の傾き（新メンバー）       │          │
│  │ - AI活用の習熟度（5段階評価）        │          │
│  └──────────────────────────────────────┘          │
│                                                    │
│  コスト指標:                                        │
│  ┌──────────────────────────────────────┐          │
│  │ - ライセンス費用 / 開発者あたり      │          │
│  │ - 生産性向上による人件費削減額       │          │
│  │ - ROI（投資対効果）                 │          │
│  │ - API使用量（Claude Code併用時）    │          │
│  └──────────────────────────────────────┘          │
└────────────────────────────────────────────────────┘
```

---

## 7. トラブルシューティング

### 7.1 よくある問題と解決策

```
┌───────────────────────────────────────────────────────┐
│          AI IDE トラブルシューティング                   │
│                                                       │
│  問題1: 補完の質が低い                                 │
│  ┌─────────────────────────────────────────────┐      │
│  │ 原因:                                        │      │
│  │   - .cursorrules が設定されていない           │      │
│  │   - インデックスが古い                        │      │
│  │   - コンテキストが不足している                │      │
│  │ 対処:                                        │      │
│  │   - .cursorrules にプロジェクト規約を追加     │      │
│  │   - Cmd+Shift+P → "Reindex" を実行          │      │
│  │   - @file で関連ファイルを明示的に追加        │      │
│  └─────────────────────────────────────────────┘      │
│                                                       │
│  問題2: AIが古い情報を使う                             │
│  ┌─────────────────────────────────────────────┐      │
│  │ 原因:                                        │      │
│  │   - AIの学習データが古い                      │      │
│  │   - @docs のインデックスが更新されていない    │      │
│  │ 対処:                                        │      │
│  │   - @web で最新情報を検索して提供             │      │
│  │   - @docs のURLを最新版に更新                 │      │
│  │   - プロンプトに"2025年最新の仕様で"と明記    │      │
│  └─────────────────────────────────────────────┘      │
│                                                       │
│  問題3: エディタが重い                                 │
│  ┌─────────────────────────────────────────────┐      │
│  │ 原因:                                        │      │
│  │   - プロジェクトが大きすぎる                  │      │
│  │   - インデックス対象が多すぎる                │      │
│  │   - 拡張機能の競合                            │      │
│  │ 対処:                                        │      │
│  │   - .cursorignore で不要ファイルを除外        │      │
│  │   - 大規模プロジェクトはCLI（Claude Code）を  │      │
│  │     検討                                     │      │
│  │   - AI系拡張機能の重複を解消                  │      │
│  └─────────────────────────────────────────────┘      │
│                                                       │
│  問題4: Composerの変更が意図と異なる                   │
│  ┌─────────────────────────────────────────────┐      │
│  │ 原因:                                        │      │
│  │   - プロンプトが曖昧                          │      │
│  │   - 既存コードのパターンと乖離                │      │
│  │ 対処:                                        │      │
│  │   - プロンプトに既存コードの例を含める        │      │
│  │   - @file で参考にすべきファイルを指定         │      │
│  │   - 段階的に変更（一度に多くを変更しない）    │      │
│  │   - Reject して具体的なフィードバックを追加    │      │
│  └─────────────────────────────────────────────┘      │
│                                                       │
│  問題5: チーム間でAIの応答品質にバラつき               │
│  ┌─────────────────────────────────────────────┐      │
│  │ 原因:                                        │      │
│  │   - .cursorrules の品質が不十分               │      │
│  │   - メンバーのプロンプト力に差がある          │      │
│  │ 対処:                                        │      │
│  │   - .cursorrules をチームで精緻化             │      │
│  │   - プロンプトテンプレートを共有              │      │
│  │   - ペアプロでの知識共有を実施                │      │
│  └─────────────────────────────────────────────┘      │
└───────────────────────────────────────────────────────┘
```

### 7.2 AI IDEのセキュリティ設定チェックリスト

```markdown
# AI IDE セキュリティチェックリスト

## 1. ファイル除外設定
- [ ] .cursorignore に機密ファイルを追加
  - .env, .env.*, .env.local
  - *.pem, *.key, *.cert
  - credentials.json, secrets.yaml
  - .aws/, .ssh/

## 2. プライバシー設定
- [ ] Privacy Mode を有効化（学習データ非利用）
- [ ] テレメトリ設定を確認
- [ ] チームのセキュリティポリシーに準拠

## 3. コードの送信範囲
- [ ] どのファイルがAIに送信されるか把握
- [ ] 機密コード（認証、暗号化）の扱いを定義
- [ ] 顧客データを含むファイルの取り扱い

## 4. 拡張機能の管理
- [ ] 信頼できるソースの拡張機能のみ使用
- [ ] AI系拡張機能の重複がないか確認
- [ ] 不要な拡張機能を無効化

## 5. ネットワーク
- [ ] VPN経由での利用を確認
- [ ] プロキシ設定が適切か確認
- [ ] ファイアウォールルールの確認
```

---

## 8. AI IDE × 他ツールの連携パターン

### 8.1 Cursor + Claude Code の最強連携

```
┌────────────────────────────────────────────────────┐
│        Cursor + Claude Code 連携ワークフロー        │
│                                                    │
│  日常的なコーディング（Cursor）                      │
│  ┌──────────────────────────────────────┐          │
│  │ - インライン補完でコードを高速記述    │          │
│  │ - Cmd+K で局所的なコード変換          │          │
│  │ - Cmd+L で疑問点を対話的に解決        │          │
│  └────────────────┬─────────────────────┘          │
│                   │                                │
│                   │ 複雑なタスクが発生               │
│                   ▼                                │
│  エージェントタスク（Claude Code）                   │
│  ┌──────────────────────────────────────┐          │
│  │ - マルチファイルのリファクタリング     │          │
│  │ - テスト生成→実行→修正ループ         │          │
│  │ - GitHub PR作成・レビュー             │          │
│  │ - CI/CDパイプラインとの連携           │          │
│  └────────────────┬─────────────────────┘          │
│                   │                                │
│                   │ 結果をCursorで確認               │
│                   ▼                                │
│  レビューと仕上げ（Cursor）                         │
│  ┌──────────────────────────────────────┐          │
│  │ - Gitの差分を視覚的に確認             │          │
│  │ - 細かい調整をインラインで実施         │          │
│  │ - テストの追加確認                    │          │
│  └──────────────────────────────────────┘          │
└────────────────────────────────────────────────────┘
```

### 8.2 AI IDE + 従来ツールの併用マトリクス

| 作業 | AI IDE (Cursor/Windsurf) | Claude Code (CLI) | 従来ツール |
|------|--------------------------|-------------------|-----------|
| コード記述 | メイン | - | サブ |
| デバッグ（IDE内） | メイン | - | ブレークポイント |
| デバッグ（ログ分析） | サブ | メイン | grep/awk |
| テスト作成 | メイン | サブ | 手動 |
| テスト実行 | IDE上 | CLI上 | CI/CD |
| リファクタリング | 小規模 | 大規模 | - |
| PR作成 | - | メイン | GitHub UI |
| コードレビュー | 差分表示 | 自動レビュー | GitHub UI |
| ドキュメント | 初稿生成 | 自動メンテ | 手動編集 |
| DB操作 | - | MCP経由 | クライアント |
| デプロイ | - | CI統合 | CI/CD |

---

## アンチパターン

### アンチパターン 1: コンテキスト過剰投入

```
❌ BAD: 全てのファイルをコンテキストに含める
   @codebase "全ファイルを読んでからリファクタリングして"
   → トークン制限超過、応答遅延、品質低下

✅ GOOD: 関連ファイルだけを選択
   @file src/services/auth.ts
   @file src/types/user.ts
   "この認証サービスにOAuth対応を追加して"
   → 高速で高品質な応答
```

### アンチパターン 2: AI IDEのロックイン

```
❌ BAD: 特定AI IDEの固有機能に依存しすぎる
   - .cursorrules にビジネスロジックを埋め込む
   - Cursor固有のAPIに依存した開発フロー
   → ツール変更時に大きなコストが発生

✅ GOOD: 標準的な構成を維持しつつAI IDEを活用
   - 設定は .editorconfig / ESLint / Prettier で管理
   - AI固有設定は薄いレイヤーで分離
   - どのエディタでも開発できる状態を維持
```

### アンチパターン 3: AI IDEの盲信

```
❌ BAD: AIの補完を無条件に受け入れる
   - Tabキーを連打して全ての提案を受け入れる
   - Composerの出力をレビューせずにAccept
   → バグの混入、セキュリティ脆弱性、技術的負債の蓄積

✅ GOOD: 批判的にレビューしながら活用
   - 補完内容を読んでから受け入れる
   - Composerの変更は差分を確認してからAccept
   - セキュリティ・パフォーマンスに影響する変更は特に注意
   - テストを実行して動作確認してからコミット
```

### アンチパターン 4: AI IDE導入のトップダウン強制

```
❌ BAD: 管理層がAI IDEの使用を一方的に強制
   - "全員Cursorを使え"と指示を出すだけ
   - 研修なし、サポートなし
   - 効果測定なし
   → 反発、非効率な使い方、形骸化

✅ GOOD: 段階的な導入とサポート
   - パイロットチームでの検証から始める
   - ハンズオン研修とペアプロの実施
   - 定期的なフィードバック収集
   - 効果測定と改善サイクル
   - 強制ではなく推奨（個人の裁量を尊重）
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

## セキュリティの考慮事項

### 一般的な脆弱性と対策

| 脆弱性 | リスクレベル | 対策 | 検出方法 |
|--------|------------|------|---------|
| インジェクション攻撃 | 高 | 入力値のバリデーション・パラメータ化クエリ | SAST/DAST |
| 認証の不備 | 高 | 多要素認証・セッション管理の強化 | ペネトレーションテスト |
| 機密データの露出 | 高 | 暗号化・アクセス制御 | セキュリティ監査 |
| 設定の不備 | 中 | セキュリティヘッダー・最小権限の原則 | 構成スキャン |
| ログの不足 | 中 | 構造化ログ・監査証跡 | ログ分析 |

### セキュアコーディングのベストプラクティス

```python
# セキュアコーディング例
import hashlib
import secrets
import hmac
from typing import Optional

class SecurityUtils:
    """セキュリティユーティリティ"""

    @staticmethod
    def generate_token(length: int = 32) -> str:
        """暗号学的に安全なトークン生成"""
        return secrets.token_urlsafe(length)

    @staticmethod
    def hash_password(password: str, salt: Optional[str] = None) -> tuple:
        """パスワードのハッシュ化"""
        if salt is None:
            salt = secrets.token_hex(16)
        hashed = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            iterations=100000
        )
        return hashed.hex(), salt

    @staticmethod
    def verify_password(password: str, hashed: str, salt: str) -> bool:
        """パスワードの検証"""
        new_hash, _ = SecurityUtils.hash_password(password, salt)
        return hmac.compare_digest(new_hash, hashed)

    @staticmethod
    def sanitize_input(value: str) -> str:
        """入力値のサニタイズ"""
        dangerous_chars = ['<', '>', '"', "'", '&', '\\']
        result = value
        for char in dangerous_chars:
            result = result.replace(char, '')
        return result.strip()

# 使用例
token = SecurityUtils.generate_token()
hashed, salt = SecurityUtils.hash_password("my_password")
is_valid = SecurityUtils.verify_password("my_password", hashed, salt)
```

### セキュリティチェックリスト

- [ ] 全ての入力値がバリデーションされている
- [ ] 機密情報がログに出力されていない
- [ ] HTTPS が強制されている
- [ ] CORS ポリシーが適切に設定されている
- [ ] 依存パッケージの脆弱性スキャンが実施されている
- [ ] エラーメッセージに内部情報が含まれていない

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

## FAQ

### Q1: CursorはVSCodeの拡張機能と互換性があるか？

CursorはVSCodeのフォークであるため、ほぼ全ての拡張機能がそのまま動作する。ただし、一部のAI系拡張機能（Copilot等）はCursorのネイティブ機能と競合する場合がある。ESLint、Prettier、GitLens等の開発ツール系拡張は問題なく利用可能。

### Q2: WindsurfとCursorのどちらが初心者に向いているか？

WindsurfのCascade機能は操作が直感的で、AIとの対話が自然なため初心者に向いている。CursorはComposerや@記法など機能が豊富だが、使いこなすには学習コストがかかる。料金もWindsurfの方が安い。まずWindsurfで慣れてから、より高度な機能が必要になったらCursorに移行するパスが合理的。

### Q3: AI IDEのセキュリティリスクはどう管理すべきか？

主なリスクは「コードが外部サーバーに送信される」こと。対策として(1) .cursorignoreや設定で機密ファイルを除外、(2) プライベートモードを有効化（学習データとして使用されない設定）、(3) SOC2認証を取得しているツールを選択、(4) 企業のセキュリティチームと連携してポリシーを策定する。

### Q4: .cursorrules と CLAUDE.md の使い分けは？

.cursorrules はCursorでの対話時に自動的に適用されるプロジェクト規約ファイル。CLAUDE.md はClaude Code（CLI）で参照される設定ファイル。両方を維持するのがベスト。共通する内容（コーディング規約、アーキテクチャ等）は一方をSource of Truthとし、他方から参照する。

### Q5: CursorのProプランは個人開発者にとってコスパが良いか？

個人開発者がプロフェッショナルレベルで開発する場合、Cursor Pro（$20/月）は十分にコスパが良い。特にComposerによるマルチファイル生成と@docs による公式ドキュメント参照は、1日の生産性向上が1-2時間に相当する。月額$20は時給換算で30分以下の節約で元が取れる計算になる。

### Q6: AI IDEでの開発はペアプログラミングと併用できるか？

むしろ相性が良い。ペアプログラミングで「ナビゲーター」がAI IDEを活用してコード提案やドキュメント検索を行い、「ドライバー」が実装に集中するパターンが効果的。AIを「第三のペア」として扱う「トリオプログラミング」という概念も広がりつつある。

### Q7: Cursor/Windsurf の学習リソースとしておすすめは？

各ツールの公式ドキュメントが最も信頼性が高い。Cursorは公式YouTube チャンネルのチュートリアルが充実している。Windsurfは公式ブログとChangelog が有用。いずれもDiscord コミュニティでの質疑応答が活発。実践的な学習には、小規模なサイドプロジェクトでAgent Modeを積極的に使うのが効果的である。

---

## まとめ

| 項目 | 要点 |
|------|------|
| AI IDEの本質 | AIを後付けではなく中核に据えた開発環境 |
| Cursorの強み | Composer、@記法、モデル選択の柔軟性 |
| Windsurfの強み | Cascade、Flows、低コスト、直感的操作 |
| コンテキスト管理 | 必要最小限のファイルを選択的に提供 |
| 選定基準 | チーム規模、予算、開発領域で判断 |
| チーム導入 | パイロット→展開→最適化の3フェーズ |
| 併用戦略 | Cursor（日常コーディング）+ Claude Code（複雑タスク） |
| 注意点 | ロックイン回避、セキュリティ設定、盲信禁止 |

---

## 次に読むべきガイド

- [03-ai-coding-best-practices.md](./03-ai-coding-best-practices.md) ── AIコーディングのベストプラクティス
- [../02-workflow/00-ai-testing.md](../02-workflow/00-ai-testing.md) ── AI IDEでのテスト自動化
- [../03-team/00-ai-team-practices.md](../03-team/00-ai-team-practices.md) ── チームでのAI IDE導入

---

## 参考文献

1. Cursor, "Cursor Documentation," 2025. https://docs.cursor.com/
2. Codeium, "Windsurf Documentation," 2025. https://docs.codeium.com/windsurf
3. Pragmatic Engineer, "AI coding tools compared: Copilot vs Cursor vs Windsurf," 2025. https://blog.pragmaticengineer.com/
4. Cursor, "Cursor Changelog and Blog," 2025. https://changelog.cursor.com/
5. Codeium, "Windsurf Blog," 2025. https://codeium.com/blog
