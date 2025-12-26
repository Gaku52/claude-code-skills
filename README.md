# Claude Code Skills

<!-- PROGRESS_BADGES_START -->
![Progress](https://img.shields.io/badge/Progress-19%25-yellow)
![Skills](https://img.shields.io/badge/Skills-5%2F26-blue)
![Characters](https://img.shields.io/badge/Characters-717K-informational)
![Guides](https://img.shields.io/badge/Guides-15-success)
<!-- PROGRESS_BADGES_END -->

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Stars](https://img.shields.io/github/stars/Gaku52/claude-code-skills?style=social)](https://github.com/Gaku52/claude-code-skills/stargazers)
[![GitHub Issues](https://img.shields.io/github/issues/Gaku52/claude-code-skills)](https://github.com/Gaku52/claude-code-skills/issues)
[![GitHub Last Commit](https://img.shields.io/github/last-commit/Gaku52/claude-code-skills)](https://github.com/Gaku52/claude-code-skills/commits/main)

> **Status:** ✅ Complete (Phase 1 Final) - No further development planned

ソフトウェア開発における設計・実装・テスト・デプロイまでの全ライフサイクルをカバーする包括的なSkills体系。
iOS、Web、Backend、Script開発からDevOps・品質管理まで、プラットフォームを横断した開発知識を体系化。

**このリポジトリは Phase 1（Skills）のみで完結し、Agent開発（Phase 2-5）は中止しました。**

## 🎯 目的

- **知識の体系化**: 開発に必要な全ての知識を構造化して保存
- **失敗の防止**: 過去の失敗事例を記録し、同じ過ちを繰り返さない
- **Claude Code連携**: Claude Codeが開発時に自動参照
- **ポートフォリオ**: 知識の体系化能力を証明

## 🚀 Quick Start

### Claude Codeでの使用（推奨）

Claude Codeは `~/.claude/skills/` ディレクトリを自動的に参照します。

```bash
# 1. このリポジトリをClone
git clone https://github.com/Gaku52/claude-code-skills.git ~/.claude/skills

# 2. Claude Codeを起動
# Skillsが自動的に参照されます
```

**使用例:**
```
あなた: 「新しいブランチを作りたい」
Claude Code: git-workflow Skillを参照 → ブランチ命名規則を提案

あなた: 「Next.jsプロジェクトを作って」
Claude Code: nextjs-development Skillを参照 → ベストプラクティスに従って実装
```

### 手動での参照

```bash
# 特定のSkillを読む
cat ~/.claude/skills/git-workflow/SKILL.md

# コミットメッセージ規約を確認
cat ~/.claude/skills/git-workflow/guides/05-commit-messages.md

# チェックリストを表示
cat ~/.claude/skills/code-review/checklists/review-checklist.md
```

### ディレクトリ構造

各Skillは統一された構造を持ちます:

```
skill-name/
├── SKILL.md              # 目次・概要
├── README.md             # 使い方
├── guides/               # 詳細ガイド
├── checklists/           # チェックリスト
├── templates/            # テンプレート
├── references/           # リファレンス
├── incidents/            # 過去の問題事例（使いながら蓄積）
└── scripts/              # 自動化スクリプト
```

## 📊 進捗状況

**現在の進捗**: 15% (4/26 スキル完成)

詳細な進捗状況は [PROGRESS.md](./PROGRESS.md) をご覧ください。

### 完了済みスキル ✅

- ✅ **react-development** - React開発ベストプラクティス (107,513文字、3ガイド)
- ✅ **nextjs-development** - Next.js App Router開発 (96,166文字、3ガイド)
- ✅ **frontend-performance** - フロントエンド最適化 (82,662文字、3ガイド)
- ✅ **web-development** - モダンWeb開発の基礎 (95,074文字、3ガイド)

### 進行中の領域

- 🔄 **WEB開発** (80%完成) - あと1スキル（web-accessibility）で完了
- ⏳ **iOS開発** (0%完成) - 5スキル
- ⏳ **Backend開発** (0%完成) - 3スキル
- ⏳ **DevOps・品質** (0%完成) - 7スキル
- ⏳ **その他** (0%完成) - 6スキル

## 🔍 進捗トラッキング

このリポジトリは自動的に進捗を測定し、レポートを生成します。

### 自動測定機能

- **文字数カウント**: 各スキルの総文字数を自動計測
- **ガイド数カウント**: 詳細ガイド（20,000文字以上）の数を自動カウント
- **完成度判定**: ガイド数に基づいて自動判定
  - ✅ **Complete**: 3本以上のガイド
  - 🔄 **In Progress**: 1-2本のガイド
  - 📝 **Basic**: SKILL.mdのみ（5,000文字以上）
  - ⬜ **Not Started**: 未着手

### 手動更新

```bash
# 進捗レポートを手動生成
npm run track

# 生成されるファイル:
# - PROGRESS.md: 詳細な進捗レポート
# - README.md: バッジセクションが自動更新
```

### 自動更新（GitHub Actions）

main ブランチへのpush時に自動的に進捗が更新されます：

- `**/SKILL.md` または `**/guides/**/*.md` の変更を検知
- 進捗レポートを自動生成・コミット

---

## 🚀 使い方

### Claude Codeでの使用

Claude Codeは自動的にこれらのSkillsを参照します：

```
「新しいブランチを作る」
→ git-workflow Skillが自動参照される
→ ブランチ命名規則に従った名前を提案

「PRを作成して」
→ git-workflow Skillのテンプレートを使用
→ チェックリストで漏れを防止
```

### 手動参照

各SkillのSKILL.mdから詳細ドキュメントにアクセス：

```bash
# スキル一覧を表示
ls -la

# 特定のスキルを読む
cat react-development/SKILL.md

# 詳細ガイドを読む
cat react-development/guides/hooks/hooks-mastery.md
```

---

## 📚 スキル一覧

全26スキルの詳細は [PROGRESS.md](./PROGRESS.md) をご覧ください。

### 領域別スキル

**WEB開発** (5スキル)
- react-development, nextjs-development, frontend-performance, web-development, web-accessibility

**iOS開発** (5スキル)
- ios-development, swiftui-patterns, ios-security, ios-project-setup, networking-data

**Backend開発** (3スキル)
- backend-development, nodejs-development, database-design

**DevOps・品質** (7スキル)
- testing-strategy, ci-cd-automation, git-workflow, code-review, quality-assurance, incident-logger, lessons-learned

**その他** (6スキル)
- python-development, cli-development, script-development, mcp-development, documentation, dependency-management

```bash
# 例: Git Workflowの詳細を見る
cat ~/.claude/skills/git-workflow/SKILL.md

# コミットメッセージ規約を確認
cat ~/.claude/skills/git-workflow/guides/05-commit-messages.md
```

### Agentとの連携

Skillsは各種Agentと連携して並行実行・自動化を実現：

```
「リリース前チェックを実行」
→ 複数のAgentが並行起動
  - code-review-agent
  - test-runner-agent
  - security-scanner-agent
  - performance-tester-agent
→ 15分で完了（従来は6時間）
```

## 📁 構成

各Skillは統一された構造：

```
skill-name/
├── SKILL.md              # 目次・概要・トリガー
├── README.md             # 使い方
├── guides/               # 詳細ガイド
├── checklists/           # チェックリスト
├── templates/            # テンプレート
├── references/           # リファレンス・ベストプラクティス
├── incidents/            # 過去の問題事例
└── scripts/              # 自動化スクリプト
```

## 🎓 学習・成長システム

### 失敗から学ぶ仕組み

1. **問題発生時**: `incident-logger` Skillで即座に記録
2. **分析**: 原因・解決方法・予防策を文書化
3. **共有**: 各Skillの `incidents/` に事例追加
4. **予防**: チェックリストに反映、同じ失敗を防ぐ

### 継続的改善

```
実際の開発
  ↓
問題・気づき発生
  ↓
incidents/ に記録
  ↓
Skillsにフィードバック
  ↓
次回から自動的に考慮される
```

## 🔗 Skills間の連携

Skillsは独立していますが、開発フローに沿って連携します：

### Web開発フロー
```
web-development → react-development / nextjs-development
                                      ↓
                          frontend-performance
                                      ↓
                           web-accessibility
                                      ↓
                              testing-strategy
                                      ↓
                              code-review
                                      ↓
                              ci-cd-automation
```

### バックエンド開発フロー
```
backend-development → nodejs-development / python-development
                                      ↓
                              database-design
                                      ↓
                              testing-strategy
                                      ↓
                              code-review
                                      ↓
                              ci-cd-automation
```

### iOS開発フロー
```
ios-project-setup → ios-development
                                      ↓
                              testing-strategy
                                      ↓
                              code-review
                                      ↓
                              ci-cd-automation
```

### スクリプト・自動化フロー
```
script-development / cli-development / mcp-development
                                      ↓
                              testing-strategy
                                      ↓
                              code-review
```

### 全プロジェクト共通
```
git-workflow (常時)
     ↓
documentation (必要に応じて)
     ↓
incident-logger (問題発生時)
     ↓
lessons-learned (フィードバック)
```

## 📝 開発ロードマップ

### Phase 1: Web開発基盤（完了✅）
- [x] `web-development` - モダンWeb開発基礎
- [x] `react-development` - React開発
- [x] `nextjs-development` - Next.js開発
- [x] `frontend-performance` - フロントエンド最適化
- [x] `web-accessibility` - アクセシビリティ対応
- [x] `documentation` - ドキュメンテーション

### Phase 2: バックエンド基盤（完了✅）
- [x] `backend-development` - バックエンド開発基礎
- [x] `nodejs-development` - Node.js開発
- [x] `python-development` - Python開発
- [x] `database-design` - データベース設計

### Phase 3: スクリプト・自動化（完了✅）
- [x] `script-development` - スクリプト開発
- [x] `cli-development` - CLIツール開発
- [x] `mcp-development` - MCP Server開発

### Phase 4: DevOps・品質管理（完了✅）
- [x] `git-workflow` - Git運用・ブランチ戦略
- [x] `testing-strategy` - テスト戦略
- [x] `code-review` - コードレビュー
- [x] `ci-cd-automation` - CI/CD自動化
- [x] `incident-logger` - 問題記録・インシデント管理
- [x] `lessons-learned` - 教訓データベース

### Phase 5: iOS開発（完了✅）
- [x] `ios-development` - iOS開発ベストプラクティス
- [x] `ios-project-setup` - プロジェクト初期設定

### Phase 6: 残りのSkills（完了✅）
- [x] `swiftui-patterns` - SwiftUI開発パターン
- [x] `networking-data` - ネットワーク・データ永続化
- [x] `ios-security` - セキュリティ実装
- [x] `quality-assurance` - 品質保証・QA
- [x] `dependency-management` - 依存関係管理

---

## 🎊 全26 Skills完成！

フルスタック開発における全ライフサイクルをカバーする包括的なSkills体系が完成しました。
iOS、Web、Backend、Script開発、DevOps、品質管理まで、プラットフォームを横断した開発知識を体系化。

---

## 🚀 Phase 2: 製品開発ファースト（収益化最優先）

**Phase 1（Skills）** = 知識ベース（完成✅）
**Phase 2** = 製品開発に100%集中

### 💰 戦略: Agent開発はスキップ、収益化を最優先

**判断理由:**
```
Agent開発（90時間）の機会損失:
- 収益化が1ヶ月遅れる
- 市場タイミングを逃すリスク
- Year 1の製品数 -2%
- 失敗時の90時間が完全な無駄

vs

製品開発に集中:
✅ 今日から収益化に向けて開始
✅ 市場フィードバックを早期獲得
✅ リスク最小化
✅ Year 1で6.6個の製品
```

### 🎯 Phase 2の方針

**やること:**
- Claude Code ($20/月) をフル活用
- 製品開発に1000時間投資
- Year 1で6-7個の製品リリース
- 収益化を最優先

**やらないこと:**
- ❌ Agent開発（収益化の歯止めになる）
- ❌ ツール作り（既存ツールで十分）
- ❌ 過度な自動化（時期尚早）

### 📚 Phase 1（Skills）の活用方法

**26個のSkillsは既に完成:**
- ポートフォリオとして活用
- GitHub公開（MIT License）
- 採用面接でのアピール材料
- 自分用のチートシート

**Claude CodeがSkillsを自動参照:**
- 開発時のベストプラクティス提示
- コードレビュー時の指摘
- アーキテクチャ判断の支援

### 💡 将来的なAgent開発（条件付き）

**製品が軌道に乗ってから検討:**
```
条件:
- 月$1,000以上の安定収益
- 同じパターンの製品を3つ以上作った
- 自動化の必要性が明確

その時に:
- 必要最小限のツールだけ作る
- 製品開発の副産物として作る
- 過度な投資はしない
```

**今は: 製品開発に100%集中 🚀**

## 🤝 コントリビューション

詳細は [CONTRIBUTING.md](CONTRIBUTING.md) を参照

### 新しいSkillの追加

1. テンプレートからコピー
2. SKILL.mdを記述
3. 必要な詳細ドキュメントを追加
4. READMEの進捗表を更新
5. PRを作成

### 既存Skillの改善

1. 実際の開発で気づいた点をIssue化
2. incidents/ に事例を追加
3. ガイド・チェックリストを更新
4. PRを作成

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details

Feel free to use, modify, and distribute this knowledge base.

## 🙏 Acknowledgments

- Created with [Claude Code](https://claude.com/claude-code)
- Built over 300+ hours of software development research and practice

## 📧 Contact

For questions, suggestions, or collaboration:
- Open an [Issue](https://github.com/Gaku52/claude-code-skills/issues)
- Start a [Discussion](https://github.com/Gaku52/claude-code-skills/discussions)

---

**最終更新**: 2025-12-25
**バージョン**: 1.1.0 (Final) - Phase 1 Complete, Phase 2+ Cancelled
**ステータス**: ✅ Complete - No further development planned
