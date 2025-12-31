# ✅ Skill改善チェックリスト (共通テンプレート)

このチェックリストは、各Skillの改善時に使用する共通の品質基準です。
全ての項目を満たすことで🟢高解像度に到達します。

---

## 📊 改善前の準備

### 現状分析
- [ ] 現在の総文字数を確認 (`npm run track` または手動計算)
- [ ] 既存ガイドの数を確認
- [ ] 既存ガイドの内容をレビュー
- [ ] 不足している要素をリストアップ
- [ ] 改善計画書を作成 (`plans/skills/{skill-name}.md`)

### 環境準備
- [ ] 作業ブランチの作成 (必要に応じて)
- [ ] エディタの準備
- [ ] 参考資料の収集

---

## 📝 ガイド構成の基準

### 必須要件: ガイド数と文字数

#### 🔴 低解像度スキル (20h投資)
- [ ] 最低3つの詳細ガイド
- [ ] 各ガイド20,000文字以上
- [ ] 総文字数 80,000+ 文字
- [ ] SKILL.mdの更新

#### 🟡 中解像度スキル (15h投資)
- [ ] 既存ガイドの強化 + 1-2個の新規ガイド
- [ ] 各ガイド20,000文字以上
- [ ] 総文字数 80,000+ 文字
- [ ] SKILL.mdの更新

### 推奨構成
```
skill-name/
├── SKILL.md (Frontmatter + TOC + Overview)
├── README.md (使い方ガイド)
├── guides/ (3+ ガイド)
│   ├── guide-1.md (20,000+ chars)
│   ├── guide-2.md (20,000+ chars)
│   └── guide-3.md (20,000+ chars)
├── checklists/ (実行時の確認項目)
├── templates/ (コピペ可能なテンプレート)
├── references/ (ベストプラクティス・アンチパターン)
└── scripts/ (自動化スクリプト, 任意)
```

---

## 🎯 内容の質の基準

### ケーススタディ (必須)
- [ ] 最低3つの実際のケーススタディ
- [ ] 各ケーススタディは実際のプロジェクトベース
- [ ] 完全なコード例が含まれる
- [ ] Before/After比較がある (該当する場合)

### コード例 (必須)
- [ ] コピペで動くコード例が15個以上
- [ ] 各コード例にコメント付き
- [ ] 完全なファイル構成を示す
- [ ] TypeScript/型定義を含む (該当言語の場合)
- [ ] 実行方法を明記

#### コード例の品質基準
```typescript
// ✅ Good: 完全で実行可能
// src/utils/validator.ts
export interface ValidationRule {
  field: string;
  required?: boolean;
  minLength?: number;
}

export function validate(data: Record<string, any>, rules: ValidationRule[]): string[] {
  const errors: string[] = [];

  for (const rule of rules) {
    const value = data[rule.field];

    if (rule.required && !value) {
      errors.push(`${rule.field} is required`);
    }

    if (rule.minLength && value.length < rule.minLength) {
      errors.push(`${rule.field} must be at least ${rule.minLength} characters`);
    }
  }

  return errors;
}

// 使用例
const errors = validate(
  { email: 'test@example.com', password: '123' },
  [
    { field: 'email', required: true },
    { field: 'password', required: true, minLength: 8 },
  ]
);
console.log(errors); // ["password must be at least 8 characters"]
```

### 失敗事例 (必須)
- [ ] 最低5-10個の失敗事例
- [ ] 各失敗事例の構成:
  - [ ] **症状**: 何が起こるか
  - [ ] **原因**: なぜ起こるか
  - [ ] **解決策**: どう直すか
  - [ ] **予防策**: 今後どう防ぐか
  - [ ] **時間的影響**: 修正にかかった時間

#### 失敗事例のフォーマット
```markdown
### 失敗1: テストがランダムに失敗する

**症状**
- 同じコードで成功したり失敗したりする
- CIでは失敗するがローカルでは成功する
- 週に2-3回発生

**原因**
- 非同期処理の待機不足
- グローバルステートへの依存
- 時間依存のテスト

**解決策**
\`\`\`typescript
// ❌ Bad
it('loads data', () => {
  fetchData();
  expect(data).toBeDefined(); // タイミングで失敗
});

// ✅ Good
it('loads data', async () => {
  await waitFor(() => {
    expect(data).toBeDefined();
  });
});
\`\`\`

**予防策**
- 非同期処理は必ずawait/waitForを使う
- グローバルステートを避ける
- beforeEach/afterEachでクリーンアップ

**時間的影響**
- 調査: 2時間
- 修正: 1時間
- 同様の問題の修正: 4時間
- **合計**: 7時間のロス
```

### トラブルシューティング (必須)
- [ ] 最低10項目のトラブルシューティング
- [ ] 各項目に具体的な解決手順
- [ ] エラーメッセージの例を含む
- [ ] 関連リソースへのリンク

---

## 🛠️ 実用性の基準

### チェックリスト (推奨)
- [ ] 最低2-3個のチェックリスト
- [ ] 実際のタスク実行時に使える
- [ ] チェックボックス形式
- [ ] 優先順位付き (必要に応じて)

#### チェックリストの例
```markdown
# テスト実装チェックリスト

## 新機能開発時
- [ ] Unit Testsを先に作成したか
- [ ] テストが独立して実行できるか
- [ ] エッジケースをカバーしているか
- [ ] エラーハンドリングをテストしているか

## PRレビュー時
- [ ] 全てのテストが通るか
- [ ] テストカバレッジが低下していないか
- [ ] テストコードも読みやすいか
```

### テンプレート (推奨)
- [ ] 最低2-3個のテンプレート
- [ ] コピペですぐ使える
- [ ] README.mdで使い方を説明
- [ ] カスタマイズ方法を記載

#### テンプレートの構成例
```
templates/jest-setup-template/
├── README.md (使い方ガイド)
├── jest.config.js (設定ファイル)
├── setupTests.ts (セットアップ)
├── testUtils.ts (ヘルパー関数)
└── example.test.ts (使用例)
```

### スクリプト (任意)
- [ ] 自動化できる部分はスクリプト化
- [ ] 実行方法を明記
- [ ] エラーハンドリング実装済み
- [ ] ヘルプメッセージ付き

---

## 📚 品質保証の基準

### Markdown品質
- [ ] 見出しレベルが適切 (H1は1つ、H2-H6の階層)
- [ ] コードブロックに言語指定
- [ ] リンクが全て有効
- [ ] 画像がある場合は代替テキスト
- [ ] 箇条書きの記法が統一

### コード品質
- [ ] 全てのコード例が動作確認済み
- [ ] 依存関係が明記されている
- [ ] 型定義が正確
- [ ] ベストプラクティスに準拠

### 内部リンク
- [ ] SKILL.mdから各ガイドへのリンク
- [ ] ガイド間の相互リンク
- [ ] チェックリスト・テンプレートへのリンク
- [ ] 全てのリンクが有効

---

## 🎯 完成度チェック

### 文字数の確認
```bash
# 特定のスキルの総文字数
cd /Users/gaku/claude-code-skills/{skill-name}
find . -name "*.md" -exec wc -c {} + | tail -1

# ガイドごとの文字数
wc -c guides/*.md
```

### ガイド数の確認
```bash
# ガイド数
ls -1 guides/*.md | wc -l

# 各ガイドの行数
wc -l guides/*.md
```

### リンクの検証
```bash
# 内部リンクの抽出
grep -r "\[.*\](.*)" . | grep -v ".git"

# 壊れたリンクの検出 (手動確認推奨)
```

### 進捗の更新
```bash
# ルートディレクトリに戻る
cd /Users/gaku/claude-code-skills

# 進捗を更新
npm run track

# PROGRESS.mdを確認
cat PROGRESS.md | grep -A 5 "{skill-name}"
```

---

## ✅ 最終チェックリスト

### 必須基準 (Must Have)
- [ ] 総文字数 80,000+ chars
- [ ] ガイド数 3個以上 (各20,000+ chars)
- [ ] ケーススタディ 3つ以上
- [ ] コピペで動くコード例 15+ 個
- [ ] 失敗事例 5+ 個
- [ ] `npm run track` で🟢高に到達

### 推奨基準 (Should Have)
- [ ] トラブルシューティング 10+ 項目
- [ ] チェックリスト 2-3個
- [ ] テンプレート 2-3個
- [ ] 全てのコード例が動作確認済み

### 理想基準 (Nice to Have)
- [ ] 他の開発者のレビューで「即使える」評価
- [ ] スクリプトによる自動化
- [ ] ビジュアル要素 (図表、スクリーンショット等)
- [ ] 動画・デモへのリンク

---

## 🚀 コミット前の最終確認

### Git操作
```bash
# 変更内容の確認
git status
git diff

# ステージング
git add .

# コミット (safe-commit-push.shを使用)
./scripts/safe-commit-push.sh "feat({skill-name}): complete comprehensive guides"

# プッシュ確認
git log -1
```

### 完了後の確認
```bash
# PROGRESS.mdの確認
cat PROGRESS.md

# リポジトリ全体の進捗
npm run track
```

---

## 📝 改善ログ (記録用)

### 作業記録
```markdown
## {skill-name} 改善ログ

**期間**: YYYY-MM-DD 〜 YYYY-MM-DD
**総工数**: XXh

### Day 1
- 作業内容:
- 成果物:
- 時間: Xh

### Day 2
...

### 完成時の状態
- 総文字数: XX,XXX chars
- ガイド数: X個
- ケーススタディ: X個
- コード例: X個
- 解像度: 🟢高
```

---

## 🔄 改善が必要な場合

### 基準未達の場合
- [ ] 未達の項目を特定
- [ ] 追加で必要な作業を見積もり
- [ ] スケジュール調整
- [ ] 優先度の再評価

### 品質に問題がある場合
- [ ] 問題点の洗い出し
- [ ] 改善計画の作成
- [ ] レビュー依頼 (可能であれば)

---

**このチェックリストを使用して**: 全14スキルを🟢高解像度に引き上げましょう!

**最終更新**: 2026-01-01
