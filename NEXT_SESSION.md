# 次回セッション開始ガイド

**このファイルは次回のClaude Codeセッション開始時に参照してください**

## 🎯 次回セッションの目的

**Phase 2: Sub Agents実装の開始**
- code-reviewer-agent の基盤構築
- TypeScript開発環境のセットアップ
- 共通ライブラリ（skill-loader等）の実装

## 📝 次回セッション開始時にClaude Codeに伝える内容

以下をコピペして、新しいClaude Codeセッションで伝えてください：

```
Phase 2のcode-reviewer-agent開発を開始します。

1. STATUS.md を読んで現在の状態を確認してください
2. docs/phase2/QUICKSTART.md に従って進めてください
3. 午前中に基盤（lib/skill-loader.ts等）を構築
4. 午後にcode-reviewer-agent本体を実装

質問があれば随時聞いてください。
```

## ✅ 事前準備チェックリスト

次回セッション開始「前」に確認：

- [ ] Node.js 18以上インストール済み（`node --version`）
- [ ] npm インストール済み（`npm --version`）
- [ ] GitHub Personal Access Token取得済み
  - https://github.com/settings/tokens
  - scope: `repo`（全てにチェック）
  - トークンをメモ帳などに保存

## 🚀 次回セッションの流れ（予定）

### ステップ1: 環境確認（5分）
```bash
# リポジトリ最新化
cd ~/.claude/skills
git pull

# Node.js確認
node --version  # 18以上であることを確認

# GitHub Token確認
echo $GITHUB_TOKEN
# 未設定なら: export GITHUB_TOKEN=ghp_xxxxx
```

### ステップ2: プロジェクト初期化（10分）
```bash
cd ~/.claude/skills
mkdir -p agents/lib agents/code-reviewer
cd agents

# package.json作成
# tsconfig.json作成
# 依存関係インストール
```
**→ QUICKSTART.mdの「Step 2」に詳細手順あり**

### ステップ3: 共通ライブラリ実装（30-60分）
```typescript
// lib/skill-loader.ts
// lib/types.ts
// lib/logger.ts
```
**→ PHASE2_DESIGN.mdにコード例あり**

### ステップ4: テスト（10分）
```bash
npm run test
tsx test.ts
```

### ステップ5: code-reviewer-agent実装（2-3時間）
```typescript
// code-reviewer/index.ts
// code-reviewer/reviewer.ts
```
**→ PHASE2_DESIGN.mdに詳細設計あり**

## 📚 参照ドキュメント（優先順）

1. **QUICKSTART.md** ← まずこれ！
   - `cat docs/phase2/QUICKSTART.md`
   - 全手順がステップバイステップで書いてある

2. **PHASE2_DESIGN.md** ← 実装時に参照
   - `cat docs/phase2/PHASE2_DESIGN.md`
   - コード例、技術スタック、詳細設計

3. **STATUS.md** ← 現在の状態確認
   - `cat STATUS.md`
   - Phase 1の成果、Phase 2の準備状況

4. **ROADMAP.md** ← 全体像の把握
   - `cat docs/phase2/ROADMAP.md`
   - Phase 1-5の全体計画

## 🎯 成功基準

次回セッション終了時に以下ができていればOK：

### 必須（午前中）
- [ ] agents/ ディレクトリ作成
- [ ] package.json, tsconfig.json 設定完了
- [ ] lib/skill-loader.ts 実装・テスト完了
- [ ] lib/types.ts, lib/logger.ts 実装完了
- [ ] skill-loader が実際にSKILL.mdを読み込める

### 推奨（午後）
- [ ] code-reviewer/index.ts 作成
- [ ] code-reviewer/reviewer.ts 基本実装
- [ ] GitHub API連携確認
- [ ] 簡単なテスト実行

### 理想（可能なら）
- [ ] 実際のPRでcode-reviewerをテスト
- [ ] 自動レビューコメント投稿成功

## 💡 つまづきそうなポイントと対策

### 問題1: GITHUB_TOKEN未設定
**症状:** GitHub APIアクセス時にエラー
**対策:**
```bash
export GITHUB_TOKEN=ghp_xxxxxxxxxxxxx
# または .env ファイルに記載
echo "GITHUB_TOKEN=ghp_xxxxx" > agents/.env
```

### 問題2: TypeScriptエラー
**症状:** 型エラーが出る
**対策:**
```bash
# 型チェックスキップで実行
tsx code-reviewer/index.ts

# または any 型で一旦回避
```

### 問題3: SKILL.md読み込めない
**症状:** loadSkill() でエラー
**対策:**
- パスが正しいか確認
- agents/ から ../skills/ への相対パス
- ファイルの存在確認: `ls ../skills/code-review/SKILL.md`

### 問題4: npm install失敗
**症状:** パッケージインストールエラー
**対策:**
```bash
# npmキャッシュクリア
npm cache clean --force
# 再インストール
npm install
```

## 🔄 進捗報告

次回セッション終了時に、このセクションを更新してコミット：

**実施日:** ____年__月__日
**所要時間:** ___時間
**完了項目:**
- [ ] 基盤構築
- [ ] skill-loader実装
- [ ] code-reviewer実装

**次回の課題:**
- 

**メモ:**
- 

---

**準備は完璧です！次回セッションを楽しんでください！** 🚀
