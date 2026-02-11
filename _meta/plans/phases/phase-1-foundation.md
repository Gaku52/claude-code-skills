# 🏗️ Phase 1: 基盤整備 (Week 1-4)

**期間**: 2026-01-01 〜 2026-01-26
**総工数**: 80時間
**目標**: 開発の基盤となる4つのスキルを徹底改善

---

## 📋 Phase概要

### 目的
開発の基盤となるテスト戦略、Git運用、データベース設計、Python開発の4スキルを優先的に改善し、後続のPhaseの土台を構築する。

### 対象スキル

| # | Skill | 現状 | 目標 | 工数 | 優先度 |
|---|-------|------|------|------|--------|
| 1 | testing-strategy | 🔴 低 | 🟢 高 | 20h | 🔴 最高 |
| 2 | git-workflow | 🟡 中 | 🟢 高 | 15h | 🟡 高 |
| 3 | database-design | 🟡 中 | 🟢 高 | 15h | 🟡 高 |
| 4 | python-development | 🔴 低 | 🟢 高 | 20h | 🔴 高 |

### 成果指標
- [ ] 4スキル全てが🟢高解像度に到達
- [ ] 総文字数 +300,000文字以上
- [ ] 新規ガイド 12個以上
- [ ] 実用的なテンプレート/スクリプト 15個以上

---

## 📅 Week 1: testing-strategy

**期間**: 2026-01-01 〜 2026-01-05
**工数**: 20時間
**現状**: 🔴 低解像度 (68,222 chars, 2/3 guides)
**目標**: 🟢 高解像度 (100,000+ chars, 3/3 guides + 充実した参考資料)

### 📖 背景と重要性

testing-strategyは品質保証の中核スキル。このスキルが充実していることで:
- 他のスキルの品質も担保できる
- テストコードの実例を他スキルで参照できる
- CI/CD構築時の基礎知識となる

### 🎯 週の目標

#### 数値目標
- 総文字数: 68,222 → 100,000+ chars
- ガイド数: 2 → 3個 (全て20,000+ chars)
- ケーススタディ: 3つ以上
- チェックリスト: 3個以上
- テンプレート: 5個以上

#### 品質目標
- コピペで動くコード例を全ガイドに含める
- 実際のプロジェクトで使える実践的な内容
- 失敗事例から学べる構成

### 📝 詳細タスク

#### Day 1 (月曜, 4h): リサーチと既存分析

**タスク**
```bash
cd testing-strategy

# 既存ガイドの確認
cat SKILL.md
cat guides/unit-testing-best-practices.md
cat guides/integration-testing-patterns.md

# 不足している要素を洗い出し
```

**成果物**
- [ ] 既存ガイドのレビューメモ
- [ ] 追加するガイドの計画書
- [ ] ケーススタディの収集リスト

**時間配分**
- 既存ガイドレビュー: 1.5h
- 追加計画作成: 1.5h
- リソース収集: 1h

---

#### Day 2 (火曜, 4h): テストピラミッド実践ガイド作成 (Part 1)

**タスク**
新規ガイド `guides/test-pyramid-practice.md` の作成開始

**構成**
```markdown
# テストピラミッド実践ガイド

## 1. テストピラミッドとは
- 概念の説明
- なぜ重要か
- 各層の役割

## 2. ケーススタディ1: Reactアプリケーション
- プロジェクト概要
- テスト構成 (70% Unit, 20% Integration, 10% E2E)
- 実際のコード例
  - Unit: コンポーネントテスト
  - Integration: API連携テスト
  - E2E: ユーザーフローテスト

## 3. よくある失敗パターン
- ピラミッドが逆転している
- E2Eテストが遅すぎる
- Unitテストが実装に依存しすぎ
```

**成果物**
- [ ] ガイドの前半部分 (10,000+ chars)
- [ ] Reactコンポーネントのテスト例 3つ

**時間配分**
- 構成設計: 1h
- コンテンツ作成: 2.5h
- コード例作成: 0.5h

---

#### Day 3 (水曜, 4h): テストピラミッド実践ガイド作成 (Part 2)

**タスク**
`guides/test-pyramid-practice.md` の完成

**構成 (続き)**
```markdown
## 4. ケーススタディ2: API統合テスト
- プロジェクト概要
- テスト構成
- 実際のコード例
  - Supertest + Jest
  - データベースのセットアップ/クリーンアップ
  - 認証を含むテスト

## 5. ケーススタディ3: E2Eテストの最適化
- Playwrightの実例
- テストの並列化
- フレーク対策

## 6. チェックリスト
- テスト設計時の確認項目
- コードレビュー時の確認項目
```

**成果物**
- [ ] ガイドの完成 (25,000+ chars)
- [ ] API統合テストの完全な例
- [ ] E2Eテストの実例

**時間配分**
- ケーススタディ2作成: 1.5h
- ケーススタディ3作成: 1.5h
- チェックリスト作成: 1h

---

#### Day 4 (木曜, 4h): TDD/BDD実践ガイド作成

**タスク**
新規ガイド `guides/tdd-bdd-workflow.md` の作成

**構成**
```markdown
# TDD/BDD実践ガイド

## 1. TDDの基本ワークフロー
- Red-Green-Refactorサイクル
- 実践例: 関数開発をTDDで

## 2. BDDとの使い分け
- Given-When-Then
- BDDフレームワーク (Jest/Cucumber)

## 3. 実際のプロジェクト例
- フィーチャー開発をTDDで進める
- ステップバイステップの実例

## 4. よくある失敗
- テストを後から書く
- テストが実装の詳細に依存する
- テストが脆い
```

**成果物**
- [ ] TDD/BDDガイド完成 (20,000+ chars)
- [ ] Red-Green-Refactorの完全な実例
- [ ] BDDのコード例

**時間配分**
- TDD部分: 2h
- BDD部分: 1.5h
- 失敗事例: 0.5h

---

#### Day 5 (金曜, 4h): チェックリスト・テンプレート・トラブルシューティング

**タスク**
実用的な補助資料の作成

**成果物**

1. **チェックリスト** (`checklists/`)
   - [ ] `test-strategy-checklist.md` - テスト戦略チェックリスト
   - [ ] `pr-review-test-checklist.md` - PRレビュー時のテスト観点
   - [ ] `test-coverage-checklist.md` - カバレッジ確認項目

2. **テンプレート** (`templates/`)
   - [ ] `jest-setup-template/` - Jestセットアップ
     - `jest.config.js`
     - `setupTests.ts`
     - `testUtils.ts`
   - [ ] `testing-library-helpers/` - Testing Libraryヘルパー
   - [ ] `api-test-template/` - API テストテンプレート

3. **リファレンス** (`references/`)
   - [ ] `common-testing-failures.md` - よくある失敗10選
   - [ ] `troubleshooting-guide.md` - トラブルシューティング

**時間配分**
- チェックリスト作成: 1h
- テンプレート作成: 2h
- トラブルシューティング: 1h

---

### ✅ Week 1完了チェックリスト

#### ガイド構成
- [ ] `guides/test-pyramid-practice.md` 完成 (25,000+ chars)
- [ ] `guides/tdd-bdd-workflow.md` 完成 (20,000+ chars)
- [ ] 既存の2ガイドも強化済み

#### 内容の質
- [ ] ケーススタディ 3つ以上
- [ ] コピペで動くコード例 15+ 個
- [ ] 失敗事例 10+ 個
- [ ] トラブルシューティング 15+ 項目

#### 実用性
- [ ] チェックリスト 3個以上
- [ ] テンプレート 5個以上
- [ ] 全てのコード例が動作確認済み

#### 品質保証
- [ ] Markdownリント通過
- [ ] 内部リンクの検証
- [ ] `npm run track` で🟢高に到達
- [ ] 総文字数 100,000+ chars

#### Git管理
- [ ] 毎日コミット実施
- [ ] コミットメッセージが規約に準拠
- [ ] 最終日に統合コミット

---

### 🎯 Week 1の成功基準

**必須条件**
- testing-strategyが🟢高解像度に到達
- 総文字数が100,000文字以上
- 実用的なテンプレートが揃っている

**理想条件**
- 他の開発者がこのガイドだけでテスト戦略を構築できる
- コピペでプロジェクトに組み込める
- 失敗事例から学べる構成になっている

---

## 📅 Week 2: git-workflow

**期間**: 2026-01-06 〜 2026-01-12
**工数**: 15時間
**詳細**: [../skills/git-workflow.md](../skills/git-workflow.md)

### 主要タスク
- ブランチ戦略の比較ガイド
- Conventional Commits実践ガイド
- Gitトラブルシューティング20選
- Git hooksテンプレート集

---

## 📅 Week 3: database-design

**期間**: 2026-01-13 〜 2026-01-19
**工数**: 15時間
**詳細**: [../skills/database-design.md](../skills/database-design.md)

### 主要タスク
- 正規化の実践的ケーススタディ
- インデックス最適化ガイド
- Prisma/TypeORM実践ガイド
- スキーマ設計テンプレート

---

## 📅 Week 4: python-development

**期間**: 2026-01-20 〜 2026-01-26
**工数**: 20時間
**詳細**: [../skills/python-development.md](../skills/python-development.md)

### 主要タスク
- 型ヒント実践ガイド
- FastAPI開発パターン
- asyncio実践ガイド
- FastAPIボイラープレート

---

## 📊 Phase 1進捗管理

### 週次チェックポイント

#### Week 1終了時 (2026-01-05)
- [ ] testing-strategy: 🟢高到達
- [ ] Phase 1進捗: 25%
- [ ] 総文字数: +30,000+ chars
- [ ] 週次レポート作成

#### Week 2終了時 (2026-01-12)
- [ ] git-workflow: 🟢高到達
- [ ] Phase 1進捗: 50%
- [ ] 総文字数: +75,000+ chars
- [ ] 週次レポート作成

#### Week 3終了時 (2026-01-19)
- [ ] database-design: 🟢高到達
- [ ] Phase 1進捗: 75%
- [ ] 総文字数: +145,000+ chars
- [ ] 週次レポート作成

#### Week 4終了時 (2026-01-26)
- [ ] python-development: 🟢高到達
- [ ] Phase 1進捗: 100%
- [ ] 総文字数: +300,000+ chars
- [ ] Phase 1完了レポート作成

### Phase 1完了時の成果物

#### スキル改善
- [ ] 4スキル全てが🟢高解像度
- [ ] 新規ガイド 12個以上
- [ ] 総文字数 +300,000+ chars

#### 実用ツール
- [ ] チェックリスト 12個以上
- [ ] テンプレート 15個以上
- [ ] スクリプト 5個以上

#### ドキュメント
- [ ] 週次レポート 4個
- [ ] Phase 1完了レポート 1個
- [ ] 次Phaseへの引き継ぎ事項

---

## 🚀 Phase 1開始方法

### Day 0: 準備 (2026-01-01朝)

```bash
# リポジトリに移動
cd /Users/gaku/claude-code-skills

# mcp-developmentを削除
rm -rf mcp-development

# 進捗更新
npm run track

# コミット
./scripts/safe-commit-push.sh "chore: remove mcp-development skill based on selection sheet"

# Week 1の計画を確認
cat plans/phases/phase-1-foundation.md
cat plans/skills/testing-strategy.md

# 作業ディレクトリに移動
cd testing-strategy

# 準備完了
echo "Phase 1 Week 1 開始準備完了!"
```

---

## 🔄 調整ルール

### 進捗が遅れた場合
1. 毎日の終了時に進捗確認
2. 金曜日に週次レビュー
3. 遅延の原因を分析
4. 次週のタスクを調整

### 品質が基準に満たない場合
1. 完成度 < 80%の場合は次週に持ち越し
2. 他のスキルの工数を削減してでも品質優先
3. Phase全体のスケジュール調整

---

**最終更新**: 2026-01-01
**次回レビュー**: 2026-01-05 (Week 1完了時)
