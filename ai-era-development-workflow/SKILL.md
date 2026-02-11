# AI 時代の開発ワークフロー

> AI はソフトウェア開発のあり方を根本的に変える。AI コーディング支援、AI ペアプログラミング、AI 駆動テスト、AI レビュー、プロンプト駆動開発まで、AI 時代の開発手法を解説する。

## このSkillの対象者

- AI ツールを開発ワークフローに統合したいエンジニア
- 開発生産性を最大化したい方
- AI 時代のエンジニアリングスキルを身につけたい方

## 前提知識

- ソフトウェア開発の実務経験
- Git/CI の基礎知識
- LLM の基礎概念

## 学習ガイド

### 00-fundamentals — AI 開発の基礎

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/00-fundamentals/00-ai-dev-landscape.md]] | AI 開発ツールの全体像、カテゴリ、トレンド |
| 01 | [[docs/00-fundamentals/01-ai-dev-mindset.md]] | AI 時代のエンジニアマインドセット、スキルシフト |
| 02 | [[docs/00-fundamentals/02-prompt-driven-development.md]] | プロンプト駆動開発、仕様記述、コード生成 |

### 01-ai-coding — AI コーディング

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/01-ai-coding/00-github-copilot.md]] | GitHub Copilot、補完、チャット、エディタ統合 |
| 01 | [[docs/01-ai-coding/01-claude-code.md]] | Claude Code、CLI、MCP、自律的タスク実行 |
| 02 | [[docs/01-ai-coding/02-cursor-and-windsurf.md]] | Cursor、Windsurf、AI IDE、コード理解 |
| 03 | [[docs/01-ai-coding/03-ai-coding-best-practices.md]] | AI コーディングのベストプラクティス、品質担保 |

### 02-workflow — AI ワークフロー

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/02-workflow/00-ai-testing.md]] | AI テスト生成、テストケース自動作成、カバレッジ改善 |
| 01 | [[docs/02-workflow/01-ai-code-review.md]] | AI コードレビュー、PR 自動レビュー、品質チェック |
| 02 | [[docs/02-workflow/02-ai-documentation.md]] | AI ドキュメント生成、API 仕様、README、変更履歴 |
| 03 | [[docs/02-workflow/03-ai-debugging.md]] | AI デバッグ、エラー分析、ログ解析、根本原因特定 |

### 03-team — AI チーム開発

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/03-team/00-ai-team-practices.md]] | AI 活用のチームプラクティス、ガイドライン策定 |
| 01 | [[docs/03-team/01-ai-onboarding.md]] | AI ツール導入、チームオンボーディング、測定 |
| 02 | [[docs/03-team/02-future-of-development.md]] | ソフトウェア開発の未来、AI ネイティブ開発 |

## クイックリファレンス

```
AI 開発ツール推奨スタック:
  コーディング: Claude Code + GitHub Copilot
  IDE:         Cursor or VS Code + 拡張
  テスト:      AI テスト生成 + Vitest
  レビュー:    AI レビュー + 人間レビュー
  ドキュメント: AI 生成 + 人間レビュー
  デバッグ:    AI 分析 + 従来デバッガー

生産性向上の原則:
  ✓ AI に丸投げせず、レビュー必須
  ✓ コンテキストを十分に与える
  ✓ 小さなタスクに分割して依頼
  ✓ テストで品質を担保
```

## 参考文献

1. GitHub. "Copilot Documentation." docs.github.com/copilot, 2024.
2. Anthropic. "Claude Code." claude.ai/claude-code, 2024.
3. Cursor. "Documentation." cursor.com/docs, 2024.
