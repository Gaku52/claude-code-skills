# Cursor / Windsurf ── AI IDE、コンテキスト管理

> AIをエディタの中核に据えた次世代IDE「Cursor」と「Windsurf」の特徴・機能・活用法を比較し、プロジェクトに最適なAI IDEを選定する判断基準を身につける。

---

## この章で学ぶこと

1. **AI IDEの設計思想** ── 従来のIDEとAI IDEの根本的な違いを理解する
2. **CursorとWindsurfの機能詳細** ── 各ツールの操作方法、コンテキスト管理、差別化ポイントを把握する
3. **最適なAI IDE選定** ── プロジェクト特性やチーム規模に応じた選定基準を学ぶ

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

---

## 3. Windsurf詳細

### コード例4: Windsurf（Cascade）の特徴

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

### コード例5: Windsurf Cascade の実践例

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

### 4.2 ユースケース別推奨

| ユースケース | 推奨ツール | 理由 |
|-------------|-----------|------|
| フロントエンド開発 | Cursor | Composerでのマルチファイル生成が強力 |
| バックエンドAPI | Claude Code | CLIでテスト実行→修正ループが効率的 |
| スタートアップ | Windsurf | 低コストで高機能 |
| エンタープライズ | VSCode+Copilot | セキュリティ・コンプライアンス対応 |
| データサイエンス | Cursor | Notebook対応 + @docs でライブラリ参照 |
| インフラ/DevOps | Claude Code | Bash実行 + 設定ファイル操作が得意 |

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

---

## FAQ

### Q1: CursorはVSCodeの拡張機能と互換性があるか？

CursorはVSCodeのフォークであるため、ほぼ全ての拡張機能がそのまま動作する。ただし、一部のAI系拡張機能（Copilot等）はCursorのネイティブ機能と競合する場合がある。ESLint、Prettier、GitLens等の開発ツール系拡張は問題なく利用可能。

### Q2: WindsurfとCursorのどちらが初心者に向いているか？

WindsurfのCascade機能は操作が直感的で、AIとの対話が自然なため初心者に向いている。CursorはComposerや@記法など機能が豊富だが、使いこなすには学習コストがかかる。料金もWindsurfの方が安い。まずWindsurfで慣れてから、より高度な機能が必要になったらCursorに移行するパスが合理的。

### Q3: AI IDEのセキュリティリスクはどう管理すべきか？

主なリスクは「コードが外部サーバーに送信される」こと。対策として(1) .cursorignoreや設定で機密ファイルを除外、(2) プライベートモードを有効化（学習データとして使用されない設定）、(3) SOC2認証を取得しているツールを選択、(4) 企業のセキュリティチームと連携してポリシーを策定する。

---

## まとめ

| 項目 | 要点 |
|------|------|
| AI IDEの本質 | AIを後付けではなく中核に据えた開発環境 |
| Cursorの強み | Composer、@記法、モデル選択の柔軟性 |
| Windsurfの強み | Cascade、低コスト、直感的操作 |
| コンテキスト管理 | 必要最小限のファイルを選択的に提供 |
| 選定基準 | チーム規模、予算、開発領域で判断 |
| 注意点 | ロックイン回避、セキュリティ設定の重要性 |

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
