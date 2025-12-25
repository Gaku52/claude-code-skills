# Claude Code Skills - Full-Stack Development Framework

ソフトウェア開発における設計・実装・テスト・デプロイまでの全ライフサイクルをカバーする包括的なSkills体系。
iOS、Web、Backend、Script開発からDevOps・品質管理まで、プラットフォームを横断した開発知識を体系化。

## 🎯 目的

- **知識の体系化**: 開発に必要な全ての知識を構造化して保存
- **失敗の防止**: 過去の失敗事例を記録し、同じ過ちを繰り返さない
- **効率化**: Agent連携により作業を自動化・並列化
- **品質向上**: チェックリストとベストプラクティスで高品質を維持

## 📊 進捗状況

### 完成度：100% (26/26) 🎉

| # | Skill | 状態 | 説明 |
|---|-------|------|------|
| **Web開発** (6/6) ||||
| 1 | `documentation` | ✅ **完成** | 技術ドキュメント・README作成 |
| 2 | `web-development` | ✅ **完成** | モダンWeb開発の基礎 |
| 3 | `react-development` | ✅ **完成** | React開発ベストプラクティス |
| 4 | `nextjs-development` | ✅ **完成** | Next.js App Router開発 |
| 5 | `frontend-performance` | ✅ **完成** | フロントエンド最適化 |
| 6 | `web-accessibility` | ✅ **完成** | アクセシビリティ対応 |
| **バックエンド開発** (4/4) ||||
| 7 | `backend-development` | ✅ **完成** | バックエンド開発基礎 |
| 8 | `nodejs-development` | ✅ **完成** | Node.js開発ガイド |
| 9 | `python-development` | ✅ **完成** | Python開発ガイド |
| 10 | `database-design` | ✅ **完成** | データベース設計 |
| **スクリプト・自動化** (3/3) ||||
| 11 | `script-development` | ✅ **完成** | スクリプト開発ガイド |
| 12 | `cli-development` | ✅ **完成** | CLIツール開発 |
| 13 | `mcp-development` | ✅ **完成** | MCP Server開発 |
| **iOS開発** (5/5) ||||
| 14 | `ios-development` | ✅ **完成** | iOS開発ベストプラクティス |
| 15 | `ios-project-setup` | ✅ **完成** | プロジェクト初期設定 |
| 16 | `swiftui-patterns` | ✅ **完成** | SwiftUI開発パターン |
| 17 | `networking-data` | ✅ **完成** | ネットワーク・データ永続化 |
| 18 | `ios-security` | ✅ **完成** | セキュリティ実装 |
| **品質・テスト** (3/3) ||||
| 19 | `testing-strategy` | ✅ **完成** | テスト戦略 |
| 20 | `code-review` | ✅ **完成** | コードレビュー |
| 21 | `quality-assurance` | ✅ **完成** | 品質保証・QA |
| **DevOps・CI/CD** (3/3) ||||
| 22 | `git-workflow` | ✅ **完成** | Git運用・ブランチ戦略 |
| 23 | `ci-cd-automation` | ✅ **完成** | CI/CD自動化 |
| 24 | `dependency-management` | ✅ **完成** | 依存関係管理 |
| **ナレッジ管理** (2/2) ||||
| 25 | `incident-logger` | ✅ **完成** | 問題記録・インシデント管理 |
| 26 | `lessons-learned` | ✅ **完成** | 教訓データベース |

**凡例**:
- ✅ 完成 - SKILL.md、ガイド、チェックリスト、テンプレート、リファレンス全て完成
- 🔨 作業中 - SKILL.mdは完成、詳細ドキュメント作成中
- 📋 構造のみ - フォルダ構造のみ作成済み

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

## 📄 ライセンス

Private（個人使用）

将来的にMIT LicenseでOSS化も検討

## 📧 お問い合わせ

Issue または Discussion で

---

**最終更新**: 2025-12-25
**バージョン**: 1.0.1 (Stable) - Security & Environment Setup Enhanced 🔒
