# AI 開発ツール

> GitHub Copilot、Claude Code CLI、Cursor を活用し、AI を開発ワークフローに統合してコーディング効率を飛躍的に向上させるためのガイド。

## この章で学ぶこと

1. GitHub Copilot の設定と効果的なプロンプトテクニック
2. Claude Code CLI のセットアップと実践的な活用方法
3. Cursor エディタの導入と AI エディタの比較選定

---

## 1. AI 開発ツールの全体像

### 1.1 主要ツール比較

| 特徴 | GitHub Copilot | Claude Code CLI | Cursor | Cody (Sourcegraph) |
|------|---------------|-----------------|--------|---------------------|
| 形態 | VS Code 拡張 | CLI ツール | エディタ | VS Code 拡張 |
| AI モデル | GPT-4o / Claude | Claude | GPT-4o / Claude | 複数対応 |
| インライン補完 | あり | なし | あり | あり |
| チャット | あり | あり | あり | あり |
| エージェント | あり | あり | あり | 一部 |
| ファイル編集 | あり | あり | あり | 限定的 |
| 料金 (月額) | $10-39 | 従量制 | $20 | 無料枠あり |
| オフライン | 不可 | 不可 | 不可 | 不可 |

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
└──────────────┴──────────────┴───────────────┘
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
    "dotenv": false
  }
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

// ─── テクニック 2: テストケースを先に書く ───

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
```

### 2.4 Copilot Chat の活用

```
Copilot Chat の主要コマンド:

┌──────────────────────────────────────────┐
│ /explain   → コードの説明                 │
│ /fix       → バグ修正の提案               │
│ /tests     → テストコード生成             │
│ /doc       → ドキュメント生成             │
│ @workspace → ワークスペース全体を文脈に   │
│ @terminal  → ターミナル出力を文脈に       │
│ @vscode    → VS Code の設定に関する質問   │
│ #file      → 特定ファイルを文脈に追加     │
│ #selection → 選択中のコードを文脈に       │
└──────────────────────────────────────────┘
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
```

### 3.3 プロジェクト設定

```markdown
# CLAUDE.md (プロジェクトルートに配置)

## プロジェクト概要
TypeScript + React のWebアプリケーション

## コーディング規約
- 関数コンポーネントのみ使用 (クラスコンポーネント禁止)
- 状態管理は Zustand を使用
- スタイリングは Tailwind CSS
- テストは Vitest + Testing Library

## ディレクトリ構造
- src/components/ - UIコンポーネント
- src/hooks/ - カスタムフック
- src/stores/ - Zustand ストア
- src/utils/ - ユーティリティ関数
- src/types/ - 型定義

## コマンド
- npm run dev - 開発サーバー起動
- npm test - テスト実行
- npm run build - ビルド
- npm run lint - リント実行
```

### 3.4 実践的なワークフロー

```
Claude Code の典型的な活用パターン:

┌─────────────────────────────────────────┐
│ 1. コードレビュー                        │
│    claude "src/api/の変更をレビューして"   │
│    → セキュリティ・パフォーマンス指摘     │
│                                           │
│ 2. リファクタリング                      │
│    claude "この関数を小さく分割して"       │
│    → ファイル分割とテスト更新を一括実行  │
│                                           │
│ 3. バグ修正                              │
│    cat error.log | claude "修正して"      │
│    → エラー分析 → 原因特定 → 修正適用   │
│                                           │
│ 4. テスト生成                            │
│    claude "src/utils/のテストを追加して"   │
│    → カバレッジの低い部分を特定して生成  │
│                                           │
│ 5. ドキュメント更新                      │
│    claude "API変更に合わせてREADME更新"    │
│    → コード変更を検出して自動反映        │
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
  ├─────────────────────────────────────────┤
  │ Cmd+L (Chat)                             │
  │   サイドパネルで AI と対話               │
  │   @ファイル名 で特定ファイルを参照       │
  ├─────────────────────────────────────────┤
  │ Cmd+I (Composer)                         │
  │   複数ファイルにまたがる変更を一括実行   │
  │   プロジェクト全体の文脈を理解           │
  ├─────────────────────────────────────────┤
  │ Tab (Autocomplete)                       │
  │   次の編集位置を予測して提案             │
  │   複数行の変更を一度に適用               │
  └─────────────────────────────────────────┘
```

### 4.3 .cursorrules 設定

```markdown
# .cursorrules (プロジェクトルートに配置)

You are an expert TypeScript developer.

## Code Style
- Use functional programming patterns
- Prefer immutable data structures
- Always use explicit return types for functions
- Use descriptive variable names (no abbreviations)

## Framework Conventions
- React: Use functional components with hooks
- State: Zustand for global, useState for local
- Styling: Tailwind CSS utility classes
- Testing: Vitest + React Testing Library

## Error Handling
- Always use Result pattern for error handling
- Never use try-catch in business logic
- Log errors with structured logging

## Do Not
- Never use `any` type
- Never use `console.log` in production code
- Never mutate function arguments
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
└─────────────────────────────────────────┘

❌ "認証を作って"
✅ "Express + TypeScript で JWT ベースの認証ミドルウェアを作成して。
    トークンの検証失敗時は 401 を返し、AppError クラスを使って
    エラーハンドリングすること。Vitest のテストも含めて。"
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
    "properties": false
  }
}
```

```bash
# .gitignore に AI 関連の設定ファイルを追加
echo '.cursorrules' >> .gitignore  # 必要に応じて
# CLAUDE.md はコミットする (チーム共有のため)
```

### 6.2 データ送信の範囲

```
各ツールが送信するデータ:

┌──────────────┬─────────────────────────────┐
│ ツール        │ 送信されるデータ             │
├──────────────┼─────────────────────────────┤
│ Copilot      │ 開いているファイルの一部      │
│              │ 隣接ファイルのコンテキスト    │
│              │ (Business プランはデータ保持なし) │
├──────────────┼─────────────────────────────┤
│ Claude Code  │ 指定ファイルの内容           │
│              │ コマンド出力                 │
│              │ (30日後に削除)               │
├──────────────┼─────────────────────────────┤
│ Cursor       │ アクティブファイル           │
│              │ @参照ファイル                │
│              │ (Privacy Mode で送信制限可)  │
└──────────────┴─────────────────────────────┘
```

---

## 7. アンチパターン

### 7.1 AI の出力を検証せずに受け入れる

```
❌ アンチパターン: AI が生成したコードをそのままコミット

問題:
  - セキュリティ脆弱性 (SQL インジェクション等)
  - 非推奨 API の使用
  - エッジケースの未処理
  - ライセンス問題のあるコードの混入

✅ 正しいアプローチ:
  - AI 生成コードは「ドラフト」として扱う
  - 必ず自分でレビューしてから採用
  - テストを書いて動作を検証
  - セキュリティスキャン (npm audit, Snyk) を実行
```

### 7.2 コンテキストを与えずにプロンプトを書く

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
   型: User[] を受け取り User[] を返す。"
```

---

## 8. FAQ

### Q1: Copilot と Cursor、両方契約すべき？

**A:** 基本的にはどちらか一方で十分。VS Code ベースのワークフローを変えたくないなら Copilot、AI ファーストの体験を求めるなら Cursor。Cursor は VS Code の fork なので、Copilot 拡張も併用可能だが、補完が競合する場合がある。予算に余裕があれば Claude Code CLI + Copilot の組み合わせが最もカバー範囲が広い。

### Q2: AI ツールで生成したコードの著作権は？

**A:** 2025年時点で法的にはグレーゾーンだが、GitHub Copilot の利用規約では、生成コードの権利はユーザーに帰属するとされている。ただし、既存のOSSコードに酷似した出力には注意が必要。Copilot の `"github.copilot.advanced"` 設定で公開コードに類似する提案をフィルタリングできる。

### Q3: Claude Code のCLAUDE.md は何を書くべき？

**A:** 以下の要素を含める。
1. プロジェクト概要（技術スタック、アーキテクチャ）
2. コーディング規約（命名規則、パターン）
3. ディレクトリ構造の説明
4. よく使うコマンド（ビルド、テスト、デプロイ）
5. やってはいけないこと（禁止パターン）

チームで共有する情報なのでリポジトリにコミットすべき。

---

## 9. まとめ

| ツール | 主な用途 | 導入コスト | 効果 |
|--------|---------|-----------|------|
| GitHub Copilot | インライン補完・チャット | $10-39/月 | コード記述速度 2-3倍 |
| Claude Code | エージェント・複雑タスク | 従量制 | リファクタ・レビュー自動化 |
| Cursor | AI統合エディタ | $20/月 | ファイル横断編集 |
| CLAUDE.md | プロジェクト文脈共有 | 無料 | AI 出力品質向上 |
| .cursorrules | Cursor 文脈設定 | 無料 | Cursor 出力品質向上 |

---

## 次に読むべきガイド

- [00-vscode-setup.md](./00-vscode-setup.md) — VS Code の詳細設定
- [../01-runtime-and-package/03-linter-formatter.md](../01-runtime-and-package/03-linter-formatter.md) — AI 生成コードの品質チェック
- [../03-team-setup/00-project-standards.md](../03-team-setup/00-project-standards.md) — チーム標準の設定

---

## 参考文献

1. **GitHub Copilot Documentation** — https://docs.github.com/en/copilot — Copilot の公式ドキュメント。設定から活用法まで。
2. **Claude Code CLI** — https://docs.anthropic.com/en/docs/claude-code — Claude Code の公式ドキュメント。
3. **Cursor Documentation** — https://docs.cursor.com — Cursor エディタの公式ドキュメントと設定ガイド。
4. **Pragmatic AI-Assisted Development** — https://martinfowler.com/articles/exploring-gen-ai.html — Martin Fowler による AI 開発ツールの実践的考察。
