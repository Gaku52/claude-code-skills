# Claude Code Skills - iOS Development Framework

iOS開発における設計、製造、テスト、リリースまでの全ライフサイクルをカバーする包括的なSkills体系。

## 🎯 目的

- **知識の体系化**: 開発に必要な全ての知識を構造化して保存
- **失敗の防止**: 過去の失敗事例を記録し、同じ過ちを繰り返さない
- **効率化**: Agent連携により作業を自動化・並列化
- **品質向上**: チェックリストとベストプラクティスで高品質を維持

## 📊 進捗状況

### 完成度：4% (1/25)

| # | Skill | 状態 | 説明 |
|---|-------|------|------|
| **企画・設計** ||||
| 1 | `product-planning` | 📋 構造のみ | プロダクト企画・要件定義 |
| 2 | `ux-design` | 📋 構造のみ | UX/UI設計 |
| 3 | `system-architecture` | 📋 構造のみ | システムアーキテクチャ設計 |
| 4 | `api-design` | 📋 構造のみ | API設計 |
| 5 | `database-design` | 📋 構造のみ | データモデリング |
| **iOS開発** ||||
| 6 | `ios-project-setup` | 📋 構造のみ | プロジェクト初期設定 |
| 7 | `ios-development` | 📋 構造のみ | iOS開発ベストプラクティス |
| 8 | `swiftui-patterns` | 📋 構造のみ | SwiftUI開発パターン |
| 9 | `networking-data` | 📋 構造のみ | ネットワーク・データ永続化 |
| 10 | `ios-performance` | 📋 構造のみ | パフォーマンス最適化 |
| 11 | `ios-security` | 📋 構造のみ | セキュリティ実装 |
| **品質・テスト** ||||
| 12 | `testing-strategy` | 📋 構造のみ | テスト戦略 |
| 13 | `code-review` | 📋 構造のみ | コードレビュー |
| 14 | `quality-assurance` | 📋 構造のみ | 品質保証・QA |
| 15 | `accessibility` | 📋 構造のみ | アクセシビリティ |
| **DevOps・CI/CD** ||||
| 16 | `git-workflow` | ✅ **完成** | Git運用・ブランチ戦略 |
| 17 | `ci-cd-automation` | 📋 構造のみ | CI/CD自動化 |
| 18 | `code-signing` | 📋 構造のみ | 証明書・プロビジョニング管理 |
| 19 | `dependency-management` | 📋 構造のみ | 依存関係管理 |
| **リリース・運用** ||||
| 20 | `release-process` | 📋 構造のみ | リリースプロセス |
| 21 | `app-store-submission` | 📋 構造のみ | App Store申請 |
| 22 | `monitoring-analytics` | 📋 構造のみ | 監視・分析・インシデント対応 |
| **ナレッジ管理** ||||
| 23 | `incident-logger` | 📋 構造のみ | 問題記録・インシデント管理 |
| 24 | `lessons-learned` | 📋 構造のみ | 教訓データベース |
| 25 | `documentation` | 📋 構造のみ | ドキュメンテーション |

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

Skills は独立していますが、相互に連携します：

```
product-planning → ux-design → system-architecture
                                      ↓
                              ios-project-setup
                                      ↓
                              ios-development
                                      ↓
                            testing-strategy
                                      ↓
                              code-review
                                      ↓
                              ci-cd-automation
                                      ↓
                              release-process
                                      ↓
                          app-store-submission
                                      ↓
                          monitoring-analytics
                                      ↓
                          incident-logger (常時)
                                      ↓
                          lessons-learned (フィードバック)
```

## 📝 開発ロードマップ

### Phase 1: 基盤構築（完了）
- [x] 25個のSkills骨格作成
- [x] `git-workflow` 完全版作成（テンプレート）
- [x] リポジトリ作成・初期コミット

### Phase 2: コアSkills（優先度：高）
- [ ] `ios-development` - 毎日使う
- [ ] `testing-strategy` - 品質の要
- [ ] `code-review` - レビュー効率化
- [ ] `ci-cd-automation` - 自動化の基盤
- [ ] `incident-logger` - 失敗記録の要
- [ ] `lessons-learned` - 学習システムの核

### Phase 3: 開発サイクルSkills（優先度：中）
- [ ] `ios-project-setup`
- [ ] `swiftui-patterns`
- [ ] `networking-data`
- [ ] `release-process`
- [ ] `app-store-submission`

### Phase 4: 設計・最適化Skills（優先度：中）
- [ ] `system-architecture`
- [ ] `api-design`
- [ ] `ios-performance`
- [ ] `ios-security`
- [ ] `accessibility`

### Phase 5: 企画・運用Skills（優先度：低）
- [ ] `product-planning`
- [ ] `ux-design`
- [ ] `monitoring-analytics`
- [ ] `documentation`

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

**最終更新**: 2024-12-24
**バージョン**: 0.1.0 (Alpha)
