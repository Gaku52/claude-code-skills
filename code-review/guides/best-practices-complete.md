# コードレビュー ベストプラクティス 完全ガイド
**作成日**: 2025年1月
**対象**: 全エンジニア（初級〜上級）
**レベル**: 総合

---

## 目次

1. [レビューの原則](#1-レビューの原則)
2. [レビュワーの視点](#2-レビュワーの視点)
3. [作成者の視点](#3-作成者の視点)
4. [メンテナーの視点](#4-メンテナーの視点)
5. [建設的なフィードバック技術](#5-建設的なフィードバック技術)
6. [セルフレビュー戦略](#6-セルフレビュー戦略)
7. [レビューツール活用](#7-レビューツール活用)
8. [言語別ベストプラクティス](#8-言語別ベストプラクティス)
9. [チーム文化の構築](#9-チーム文化の構築)
10. [ケーススタディ](#10-ケーススタディ)

---

## 1. レビューの原則

### 1.1 コードレビューの目的

```typescript
// コードレビューの主要な目的
enum ReviewPurpose {
  FIND_BUGS = 'バグの早期発見',
  IMPROVE_QUALITY = 'コード品質の向上',
  SHARE_KNOWLEDGE = 'ナレッジ共有',
  ENSURE_CONSISTENCY = 'コーディング規約の統一',
  MENTOR_DEVELOPERS = '開発者の育成',
  PREVENT_SECURITY_ISSUES = 'セキュリティ問題の予防',
  MAINTAIN_ARCHITECTURE = 'アーキテクチャの維持',
}

interface ReviewBenefits {
  immediate: string[];
  shortTerm: string[];
  longTerm: string[];
}

const reviewBenefits: ReviewBenefits = {
  immediate: [
    'デプロイ前のバグ検出（修正コスト1/10）',
    'セキュリティ脆弱性の発見',
    'パフォーマンス問題の特定',
  ],
  shortTerm: [
    'コード品質の向上',
    'チーム内のコーディング規約統一',
    '技術的負債の削減',
  ],
  longTerm: [
    'チーム全体のスキル向上',
    'バス因子の低減（知識の分散）',
    'オンボーディング時間の短縮',
    'システムの保守性向上',
  ],
};
```

### 1.2 効果的なレビューの黄金律

```markdown
## 10の黄金律

### 1. コードを批判し、人を批判しない
❌ 悪い例: 「なんでこんな実装したの？」
✅ 良い例: 「この実装だとエッジケースで問題が起きる可能性があります」

### 2. 具体的で建設的なフィードバックを提供
❌ 悪い例: 「ここは良くない」
✅ 良い例: 「この関数は50行あり、単一責任原則に違反しています。バリデーション部分を別関数に分離することを提案します」

### 3. 提案と理由をセットで提示
❌ 悪い例: 「async/awaitを使ってください」
✅ 良い例: 「async/awaitを使うと、エラーハンドリングがtry-catchで統一でき、可読性が向上します」

### 4. 良いコードを称賛する
✅ 「このエラーハンドリングは完璧です！エッジケースまで考慮されていますね」
✅ 「テストケースが充実していて素晴らしいです」

### 5. 質問形式で指摘する
❌ 「これは間違っています」
✅ 「この場合、nullが返る可能性はありませんか？」

### 6. コンテキストを提供する
✅ 「このプロジェクトでは、エラーハンドリングにResult型パターンを使用しています（参考: src/utils/result.ts）」

### 7. 優先順位を明確にする
✅ [Critical] セキュリティ: SQLインジェクションの可能性
✅ [Nice-to-have] リファクタリング: この関数は分割できます

### 8. タイムリーにレビューする
- 小さなPR: 2時間以内
- 中規模PR: 4時間以内
- 大規模PR: 8時間以内（または分割を提案）

### 9. 完璧を求めすぎない
- 「今」必要な品質か「理想」の品質かを見極める
- 技術的負債として記録し、後で対応する選択肢もある

### 10. 学習機会として活用する
- 新しい技術・パターンを学ぶ
- なぜそのアプローチを選んだか質問する
- ベストプラクティスを共有する
```

### 1.3 レビューの範囲

```typescript
// レビューで確認すべき項目
interface ReviewScope {
  critical: ReviewItem[];
  high: ReviewItem[];
  medium: ReviewItem[];
  low: ReviewItem[];
}

interface ReviewItem {
  category: string;
  items: string[];
  estimatedTime: number; // minutes
}

const reviewScope: ReviewScope = {
  critical: [
    {
      category: 'セキュリティ',
      items: [
        'SQLインジェクション',
        'XSS脆弱性',
        '認証・認可の欠陥',
        '機密情報の露出',
      ],
      estimatedTime: 10,
    },
    {
      category: '機能性',
      items: [
        '要件の充足',
        'エッジケースの考慮',
        'エラーハンドリング',
      ],
      estimatedTime: 15,
    },
  ],
  high: [
    {
      category: 'パフォーマンス',
      items: [
        'N+1クエリ',
        'メモリリーク',
        '不要なループ',
      ],
      estimatedTime: 8,
    },
    {
      category: 'テスト',
      items: [
        'ユニットテストの網羅性',
        'エッジケースのテスト',
        'テストの独立性',
      ],
      estimatedTime: 10,
    },
  ],
  medium: [
    {
      category: 'コード品質',
      items: [
        '命名の適切性',
        '関数の長さ',
        '重複コード',
      ],
      estimatedTime: 8,
    },
    {
      category: '保守性',
      items: [
        'コメントの適切性',
        'マジックナンバー',
        '依存関係',
      ],
      estimatedTime: 5,
    },
  ],
  low: [
    {
      category: 'スタイル',
      items: [
        'フォーマット',
        'インデント',
        'タイポ',
      ],
      estimatedTime: 2,
    },
  ],
};
```

---

## 2. レビュワーの視点

### 2.1 効果的なレビューの進め方

```typescript
// レビューのステップバイステップガイド
class CodeReviewer {
  async reviewPullRequest(pr: PullRequest): Promise<Review> {
    console.log('🔍 Starting code review...\n');

    // Step 1: 理解する（5-10分）
    await this.understandContext(pr);

    // Step 2: 全体を把握する（5分）
    await this.getOverview(pr);

    // Step 3: 詳細レビュー（20-40分）
    const issues = await this.detailedReview(pr);

    // Step 4: フィードバック作成（10-15分）
    const feedback = await this.createFeedback(issues);

    // Step 5: 総合判断（5分）
    const decision = await this.makeDecision(feedback);

    return { feedback, decision };
  }

  private async understandContext(pr: PullRequest) {
    console.log('📖 Step 1: Understanding context');

    // PR説明を読む
    console.log(`  Title: ${pr.title}`);
    console.log(`  Description: ${pr.description}`);

    // 関連Issueを確認
    if (pr.relatedIssues.length > 0) {
      console.log(`  Related Issues: ${pr.relatedIssues.join(', ')}`);
    }

    // 自動チェックの結果を確認
    const checks = await pr.getChecks();
    console.log(`  CI/CD Status: ${checks.status}`);

    // 質問: なぜこの変更が必要か理解できたか？
    const understood = await this.confirm('Context understood?');
    if (!understood) {
      await this.askQuestion(pr, 'この変更の背景を教えてください');
    }
  }

  private async getOverview(pr: PullRequest) {
    console.log('\n📊 Step 2: Getting overview');

    const stats = {
      filesChanged: pr.files.length,
      additions: pr.additions,
      deletions: pr.deletions,
      total: pr.additions + pr.deletions,
    };

    console.log(`  Files changed: ${stats.filesChanged}`);
    console.log(`  Lines added: ${stats.additions}`);
    console.log(`  Lines deleted: ${stats.deletions}`);
    console.log(`  Total changes: ${stats.total}`);

    // PRサイズの評価
    if (stats.total > 400) {
      await this.suggestSplit(pr);
    }

    // 影響範囲の確認
    const impactedAreas = this.analyzeImpact(pr.files);
    console.log(`  Impacted areas: ${impactedAreas.join(', ')}`);
  }

  private async detailedReview(pr: PullRequest): Promise<Issue[]> {
    console.log('\n🔬 Step 3: Detailed review');

    const issues: Issue[] = [];

    for (const file of pr.files) {
      console.log(`\n  Reviewing: ${file.path}`);

      // ファイルタイプに応じたレビュー
      if (file.path.match(/\.(ts|tsx|js|jsx)$/)) {
        issues.push(...await this.reviewJavaScript(file));
      } else if (file.path.match(/\.(py)$/)) {
        issues.push(...await this.reviewPython(file));
      } else if (file.path.match(/\.(swift)$/)) {
        issues.push(...await this.reviewSwift(file));
      }

      // 共通のレビュー項目
      issues.push(...await this.reviewCommon(file));
    }

    return issues;
  }

  private async reviewJavaScript(file: File): Promise<Issue[]> {
    const issues: Issue[] = [];

    // TypeScript特有のチェック
    if (file.path.endsWith('.ts') || file.path.endsWith('.tsx')) {
      // 型安全性
      if (file.content.includes('any')) {
        issues.push({
          severity: 'medium',
          type: 'type-safety',
          message: '`any`型の使用を避けてください',
          line: this.findLine(file, 'any'),
          suggestion: '具体的な型を定義するか、`unknown`を使用してください',
        });
      }
    }

    // 非同期処理
    if (file.content.includes('Promise') && !file.content.includes('catch')) {
      issues.push({
        severity: 'high',
        type: 'error-handling',
        message: 'Promise のエラーハンドリングが不足しています',
        suggestion: '.catch()またはtry-catchを追加してください',
      });
    }

    // パフォーマンス
    const nestedLoops = this.findNestedLoops(file);
    if (nestedLoops.length > 0) {
      issues.push({
        severity: 'medium',
        type: 'performance',
        message: 'ネストしたループがあります',
        suggestion: 'アルゴリズムの見直しを検討してください',
      });
    }

    return issues;
  }

  private async createFeedback(issues: Issue[]): Promise<Feedback> {
    console.log('\n💬 Step 4: Creating feedback');

    // 重要度でグループ化
    const grouped = {
      critical: issues.filter(i => i.severity === 'critical'),
      high: issues.filter(i => i.severity === 'high'),
      medium: issues.filter(i => i.severity === 'medium'),
      low: issues.filter(i => i.severity === 'low'),
    };

    console.log(`  Critical: ${grouped.critical.length}`);
    console.log(`  High: ${grouped.high.length}`);
    console.log(`  Medium: ${grouped.medium.length}`);
    console.log(`  Low: ${grouped.low.length}`);

    return {
      summary: this.createSummary(grouped),
      comments: this.createComments(issues),
      positives: this.findPositives(),
    };
  }

  private async makeDecision(feedback: Feedback): Promise<Decision> {
    console.log('\n⚖️  Step 5: Making decision');

    const criticalCount = feedback.summary.critical;
    const highCount = feedback.summary.high;

    if (criticalCount > 0) {
      return {
        action: 'request-changes',
        reason: `${criticalCount}件のクリティカルな問題があります`,
      };
    }

    if (highCount > 3) {
      return {
        action: 'request-changes',
        reason: `${highCount}件の重要な問題があります`,
      };
    }

    if (highCount > 0) {
      return {
        action: 'comment',
        reason: 'いくつか改善提案がありますが、ブロッカーではありません',
      };
    }

    return {
      action: 'approve',
      reason: '問題なし。素晴らしいコードです！',
    };
  }
}
```

### 2.2 レビューのテンプレート

```markdown
# コードレビューコメントテンプレート

## 🐛 バグ指摘テンプレート

### テンプレート
```
**問題**: [具体的な問題の説明]

**影響**: [どのような問題が発生するか]

**再現手順**:
1. [ステップ1]
2. [ステップ2]
3. [結果]

**修正案**:
\```[language]
[修正後のコード]
\```

**参考**: [関連ドキュメント・Issue]
```

### 実例
```
**問題**: null チェックが不足しています

**影響**: ユーザーがログアウトしている状態でこのページにアクセスすると、アプリケーションがクラッシュします

**再現手順**:
1. ログアウト状態でダッシュボードにアクセス
2. `user.name`にアクセス
3. TypeError: Cannot read property 'name' of null

**修正案**:
\```typescript
// Before
const userName = user.name;

// After
const userName = user?.name ?? 'Guest';
\```

**参考**: 類似の問題 #123
```

## 💡 改善提案テンプレート

### テンプレート
```
**現状**: [現在の実装]

**提案**: [改善案]

**理由**: [なぜ改善すべきか]

**メリット**:
- [メリット1]
- [メリット2]

**デメリット**:
- [デメリット1（もしあれば）]

**参考**: [ベストプラクティス・ドキュメント]
```

### 実例
```
**現状**: 複数の場所で同じバリデーションロジックが重複しています

**提案**: バリデーション関数を共通化しましょう

**理由**: DRY原則に従い、保守性を向上させるため

**メリット**:
- バグ修正が一箇所で済む
- テストが容易
- 一貫性の保証

**実装例**:
\```typescript
// utils/validation.ts
export function validateEmail(email: string): boolean {
  const regex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return regex.test(email);
}

// 使用箇所
import { validateEmail } from '@/utils/validation';

if (!validateEmail(email)) {
  throw new Error('Invalid email');
}
\```

**参考**: https://refactoring.guru/dry-principle
```

## ❓ 質問テンプレート

### テンプレート
```
**質問**: [理解したい点]

**背景**: [なぜ質問するか]

**懸念点**: [もしあれば]
```

### 実例
```
**質問**: この setTimeout は何のために使用していますか？

**背景**: 非同期処理は通常 Promise/async-await で行うため、意図を理解したいです

**懸念点**: タイミング依存のバグが発生しないか心配です
```

## 👍 称賛テンプレート

### テンプレート
```
**Good**: [良かった点]

**理由**: [なぜ良いか]

**学び**: [自分が学んだこと（オプション）]
```

### 実例
```
**Good**: エッジケースを全てカバーするテストが追加されていますね！

**理由**:
- 空配列のケース
- 大量データのケース
- 境界値のケース
全てテストされており、品質が高いです

**学び**: この parametrized test のアプローチは参考になります
```

## ⚠️ セキュリティ指摘テンプレート

### テンプレート
```
**セキュリティリスク**: [OWASP分類]

**脆弱性**: [具体的な脆弱性]

**攻撃シナリオ**:
[攻撃者がどのように悪用できるか]

**修正**: [必須]
\```[language]
[安全なコード]
\```

**参考**: [OWASP/CWEリンク]
```

### 実例
```
**セキュリティリスク**: SQL Injection (OWASP A03:2021)

**脆弱性**: ユーザー入力が直接SQLクエリに埋め込まれています

**攻撃シナリオ**:
攻撃者が `userId` に `"1 OR 1=1"` を送信すると、全ユーザーのデータが取得できます

**修正**: [必須]
\```typescript
// Before ❌
const query = `SELECT * FROM users WHERE id = ${userId}`;

// After ✅
const query = 'SELECT * FROM users WHERE id = ?';
db.query(query, [userId]);
\```

**参考**: https://owasp.org/www-community/attacks/SQL_Injection
```

## 🎯 パフォーマンス指摘テンプレート

### テンプレート
```
**パフォーマンス問題**: [問題の種類]

**現在の計算量**: O([complexity])

**影響**: [どのくらい遅くなるか]

**最適化案**:
\```[language]
[最適化後のコード]
\```

**改善後の計算量**: O([complexity])
```

### 実例
```
**パフォーマンス問題**: N+1 クエリ

**現在の計算量**: O(n) - ユーザー数に比例

**影響**: 1000ユーザーいる場合、1000回のDBクエリが発生します

**最適化案**:
\```typescript
// Before ❌ O(n)
const users = await User.findAll();
for (const user of users) {
  user.posts = await Post.findAll({ where: { userId: user.id } });
}

// After ✅ O(1)
const users = await User.findAll({
  include: [{ model: Post }]
});
\```

**改善後の計算量**: O(1) - 1回のクエリで完了

**測定結果**:
- Before: 2.5秒
- After: 0.05秒
- **50倍の高速化**
```

## 🔍 Nitpick（細かい指摘）テンプレート

### テンプレート
```
**Nit**: [小さな改善提案]

**理由**: [なぜ改善すると良いか]

※ ブロッカーではありません
```

### 実例
```
**Nit**: 変数名を `data` から `userData` に変更してはどうでしょうか

**理由**: より具体的な名前の方が、コードの意図が明確になります

※ ブロッカーではありません。お好みで対応してください
```
```

---

## 3. 作成者の視点

### 3.1 セルフレビューの重要性

```typescript
// セルフレビューチェックリスト
interface SelfReviewChecklist {
  phase: string;
  items: ChecklistItem[];
}

interface ChecklistItem {
  category: string;
  question: string;
  action: string;
}

const selfReviewProcess: SelfReviewChecklist[] = [
  {
    phase: 'コード完成直後',
    items: [
      {
        category: '機能性',
        question: '全ての要件を満たしているか？',
        action: 'Issue/チケットと照らし合わせる',
      },
      {
        category: 'テスト',
        question: 'テストを追加したか？',
        action: 'カバレッジを確認する',
      },
      {
        category: 'エラーハンドリング',
        question: '全てのエラーケースを処理したか？',
        action: 'try-catchの追加、エラーメッセージの確認',
      },
    ],
  },
  {
    phase: 'PR作成前',
    items: [
      {
        category: 'コード品質',
        question: 'デバッグコードを削除したか？',
        action: 'console.log, debuggerの検索',
      },
      {
        category: 'コミット',
        question: 'コミットメッセージは適切か？',
        action: 'Conventional Commits準拠の確認',
      },
      {
        category: 'ドキュメント',
        question: 'README/ドキュメントを更新したか？',
        action: 'API変更時は必須',
      },
    ],
  },
  {
    phase: 'PR作成時',
    items: [
      {
        category: 'PR説明',
        question: 'レビュワーが理解できる説明か？',
        action: 'What, Why, Howを記載',
      },
      {
        category: 'スクリーンショット',
        question: 'UI変更の場合、画像を追加したか？',
        action: 'Before/Afterを添付',
      },
      {
        category: 'レビュー観点',
        question: '特に見てほしい点を明記したか？',
        action: '「レビュー観点」セクションを追加',
      },
    ],
  },
];

// セルフレビュー実行
async function performSelfReview(): Promise<void> {
  console.log('📋 Self Review Checklist\n');

  for (const phase of selfReviewProcess) {
    console.log(`\n## ${phase.phase}`);

    for (const item of phase.items) {
      console.log(`\n### ${item.category}`);
      console.log(`Q: ${item.question}`);
      console.log(`→ ${item.action}`);

      const completed = await prompt('完了しましたか？ (y/n): ');

      if (completed.toLowerCase() !== 'y') {
        console.log('⚠️  このアイテムを完了してからPRを作成してください');
        return;
      }
    }
  }

  console.log('\n✅ セルフレビュー完了！PRを作成できます。');
}
```

### 3.2 効果的なPR説明の書き方

```markdown
# 効果的なPR説明テンプレート

## 基本テンプレート

\```markdown
## 概要
<!-- このPRで何を達成するか（1-2文） -->

## 変更の背景
<!-- なぜこの変更が必要か -->
- 問題: [現在の問題]
- 解決: [このPRでどう解決するか]

## 変更内容
<!-- 主な変更点をリスト形式で -->
- ✨ [新機能]
- 🐛 [バグ修正]
- ♻️  [リファクタリング]
- 📝 [ドキュメント]

## テスト
<!-- どのようにテストしたか -->
### ユニットテスト
- [ ] 新規テスト追加
- [ ] 既存テスト更新
- [ ] カバレッジ維持/向上

### 手動テスト
- [ ] 正常系動作確認
- [ ] エラーケース確認
- [ ] 境界値テスト

### テスト環境
- OS: [macOS/Windows/Linux]
- ブラウザ: [Chrome/Firefox/Safari]
- デバイス: [Desktop/Mobile]

## スクリーンショット
<!-- UI変更がある場合 -->
### Before
[画像]

### After
[画像]

## パフォーマンス影響
<!-- パフォーマンスに影響がある場合 -->
- Before: [測定結果]
- After: [測定結果]
- 改善率: [%]

## Breaking Changes
<!-- 破壊的変更がある場合 -->
- [ ] なし
- [ ] あり: [詳細]

## マイグレーション
<!-- DB変更がある場合 -->
- [ ] マイグレーションファイル追加
- [ ] ロールバック手順確認

## デプロイメント
<!-- 特別なデプロイ手順がある場合 -->
- [ ] 通常デプロイ
- [ ] 環境変数追加: [詳細]
- [ ] 設定変更必要: [詳細]

## 関連
<!-- 関連Issue、PR -->
- Closes #123
- Related to #456
- Depends on #789

## レビュー観点
<!-- レビュワーに特に見てほしい点 -->
- [ ] [観点1]
- [ ] [観点2]

## チェックリスト
<!-- 作成者のセルフチェック -->
- [ ] コードを自分でレビューした
- [ ] テストを追加した
- [ ] ドキュメントを更新した
- [ ] Lintエラーがない
- [ ] ビルドが通る
\```

## 実例

\```markdown
## 概要
ユーザー検索機能のパフォーマンスを改善し、レスポンスタイムを80%削減しました。

## 変更の背景
- 問題: ユーザー検索が1000件を超えると5秒以上かかる
- 原因: N+1クエリとフルテーブルスキャン
- 解決: クエリ最適化とインデックス追加

## 変更内容
- ♻️  ユーザー検索クエリの最適化（N+1解消）
- 🗄️ `users.email`にインデックス追加
- ⚡ Redis キャッシュ導入
- ✅ パフォーマンステスト追加

## テスト
### ユニットテスト
- ✅ 検索機能のテスト追加（10ケース）
- ✅ キャッシュ動作のテスト追加
- ✅ カバレッジ: 85% → 92%

### パフォーマンステスト
\```typescript
// 1000件のユーザーでテスト
describe('Performance', () => {
  it('should search within 500ms', async () => {
    const start = Date.now();
    await searchUsers({ query: 'test' });
    const duration = Date.now() - start;
    expect(duration).toBeLessThan(500);
  });
});
\```

### 手動テスト
- ✅ 空文字列検索
- ✅ 特殊文字検索
- ✅ 1万件データでの動作確認

## パフォーマンス影響
| ユーザー数 | Before | After | 改善率 |
|----------|--------|-------|------|
| 100件    | 500ms  | 50ms  | 90%  |
| 1,000件  | 5s     | 200ms | 96%  |
| 10,000件 | 50s    | 800ms | 98%  |

## Breaking Changes
- なし

## デプロイメント
### マイグレーション
\```sql
-- マイグレーションファイル: 20250102_add_email_index.sql
CREATE INDEX idx_users_email ON users(email);
\```

### 環境変数
\```bash
# .env に追加
REDIS_URL=redis://localhost:6379
SEARCH_CACHE_TTL=3600
\```

## 関連
- Closes #234 (検索が遅い)
- Related to #145 (パフォーマンス改善タスク)

## レビュー観点
- ✅ クエリの最適化が適切か
- ✅ キャッシュ戦略が妥当か
- ✅ パフォーマンステストが十分か

## チェックリスト
- ✅ コードを自分でレビューした
- ✅ テストを追加した（カバレッジ+7%）
- ✅ READMEにキャッシュ設定を追加
- ✅ Lintエラーなし
- ✅ ビルド成功
\```
```

---

## 4. メンテナーの視点

### 4.1 メンテナーの責務

```typescript
// メンテナーのチェックリスト
interface MaintainerChecklist {
  category: string;
  responsibilities: string[];
}

const maintainerDuties: MaintainerChecklist[] = [
  {
    category: 'コード品質の維持',
    responsibilities: [
      'アーキテクチャ原則の遵守',
      'コーディング規約の統一',
      '技術的負債の管理',
      'リファクタリング機会の特定',
    ],
  },
  {
    category: 'プロジェクト全体の整合性',
    responsibilities: [
      '既存コードとの一貫性確認',
      '設計パターンの統一',
      '命名規則の統一',
      'ディレクトリ構造の維持',
    ],
  },
  {
    category: 'レビュープロセスの管理',
    responsibilities: [
      'レビュー優先順位の決定',
      'レビュワーの割り当て',
      'レビュー遅延の解消',
      'コンフリクト解決',
    ],
  },
  {
    category: 'チーム育成',
    responsibilities: [
      'ベストプラクティスの共有',
      '建設的なフィードバック',
      'メンタリング',
      'レビュー文化の醸成',
    ],
  },
];
```

### 4.2 最終承認の判断基準

```typescript
// 最終承認の判断フロー
class Maintainer {
  async decideFinalApproval(pr: PullRequest): Promise<Decision> {
    console.log('⚖️  Final approval decision process\n');

    // 1. 必須条件のチェック
    const mandatory = await this.checkMandatory(pr);
    if (!mandatory.passed) {
      return {
        action: 'block',
        reason: `必須条件を満たしていません: ${mandatory.failures.join(', ')}`,
      };
    }

    // 2. 品質基準のチェック
    const quality = await this.checkQuality(pr);
    if (quality.score < 70) {
      return {
        action: 'request-changes',
        reason: `品質スコア${quality.score}点が基準（70点）を下回っています`,
      };
    }

    // 3. アーキテクチャ整合性のチェック
    const architecture = await this.checkArchitecture(pr);
    if (!architecture.consistent) {
      return {
        action: 'request-discussion',
        reason: 'アーキテクチャへの影響について議論が必要です',
      };
    }

    // 4. テストカバレッジのチェック
    const coverage = await this.checkCoverage(pr);
    if (coverage.current < coverage.threshold) {
      return {
        action: 'request-changes',
        reason: `カバレッジ${coverage.current}%が基準（${coverage.threshold}%）を下回っています`,
      };
    }

    // 5. セキュリティチェック
    const security = await this.checkSecurity(pr);
    if (security.vulnerabilities.length > 0) {
      return {
        action: 'block',
        reason: `セキュリティ脆弱性が検出されました: ${security.vulnerabilities.join(', ')}`,
      };
    }

    // 6. パフォーマンスへの影響
    const performance = await this.checkPerformance(pr);
    if (performance.degradation > 10) {
      return {
        action: 'request-discussion',
        reason: `パフォーマンスが${performance.degradation}%低下しています`,
      };
    }

    // 7. ドキュメント確認
    const docs = await this.checkDocumentation(pr);
    if (!docs.updated && docs.required) {
      return {
        action: 'request-changes',
        reason: 'ドキュメントの更新が必要です',
      };
    }

    // 全ての条件をクリア
    return {
      action: 'approve',
      reason: '全ての基準を満たしています。素晴らしいPRです！',
      comments: this.generateApprovalComments(pr),
    };
  }

  private async checkMandatory(pr: PullRequest): Promise<CheckResult> {
    const failures: string[] = [];

    // CI/CDの成功
    if (!pr.ciPassed) {
      failures.push('CI/CDが失敗しています');
    }

    // コンフリクト解消
    if (pr.hasConflicts) {
      failures.push('マージコンフリクトがあります');
    }

    // 最小レビュー数
    const requiredReviews = 2;
    const approvals = pr.reviews.filter(r => r.state === 'approved').length;
    if (approvals < requiredReviews) {
      failures.push(`承認が${approvals}件（必要: ${requiredReviews}件）`);
    }

    // 未解決のコメント
    const unresolvedComments = pr.comments.filter(c => !c.resolved).length;
    if (unresolvedComments > 0) {
      failures.push(`未解決のコメントが${unresolvedComments}件あります`);
    }

    return {
      passed: failures.length === 0,
      failures,
    };
  }

  private generateApprovalComments(pr: PullRequest): string {
    return `
## ✅ 承認

このPRは以下の基準を全て満たしており、マージを承認します：

### 品質基準
- ✅ テストカバレッジ: ${pr.coverage}%
- ✅ コード品質スコア: ${pr.qualityScore}/100
- ✅ セキュリティスキャン: 問題なし
- ✅ パフォーマンス影響: 許容範囲内

### レビュー状況
- ✅ ${pr.approvals}名のレビュワーが承認
- ✅ 全てのコメントが解決済み
- ✅ CI/CD成功

### 次のステップ
1. マージ後、stagingで動作確認
2. 本番デプロイ前にQAチームに通知
3. リリースノートに追加

お疲れ様でした！ 🎉
    `;
  }
}
```

---

## 5. 建設的なフィードバック技術

### 5.1 フィードバックの原則

```typescript
// 効果的なフィードバックの要素
interface EffectiveFeedback {
  principle: string;
  description: string;
  bad: string;
  good: string;
}

const feedbackPrinciples: EffectiveFeedback[] = [
  {
    principle: '1. Specific（具体的に）',
    description: '曖昧な表現を避け、具体的に指摘する',
    bad: 'このコードは良くないです',
    good: 'この関数は50行あり、バリデーション・変換・保存の3つの責務を持っています。単一責任原則に従い、3つの関数に分割することを提案します',
  },
  {
    principle: '2. Actionable（実行可能）',
    description: '改善方法を明示する',
    bad: 'パフォーマンスが悪いです',
    good: `このループはO(n²)です。Setを使うとO(n)に改善できます：
\`\`\`typescript
const seen = new Set();
for (const item of items) {
  if (!seen.has(item)) {
    seen.add(item);
    // 処理
  }
}
\`\`\``,
  },
  {
    principle: '3. Kind（親切に）',
    description: '敬意を持って伝える',
    bad: 'なんでこんなコード書いたの？',
    good: 'この実装の意図を教えていただけますか？別のアプローチも検討できるかもしれません',
  },
  {
    principle: '4. Balanced（バランス良く）',
    description: '改善点だけでなく、良い点も伝える',
    bad: '（改善点のみ指摘）',
    good: `👍 エラーハンドリングが完璧です！全てのエッジケースをカバーしていますね。

💡 一点、ログレベルを見直すと良いかもしれません：
- ユーザーエラー → warning
- システムエラー → error`,
  },
];
```

### 5.2 難しい状況でのフィードバック

```markdown
## 難しい状況でのコミュニケーション

### ケース1: 大幅な書き直しが必要な場合

❌ **悪い例**
「このコード全部書き直してください」

✅ **良い例**
\```
この実装について相談させてください。

現在の実装は動作しますが、将来的な保守性を考えると、
いくつか改善できる点があります：

1. **アーキテクチャ**: MVVMパターンに従うと、テストが容易になります
2. **依存関係**: Dependency Injectionを使うと、モック化が簡単になります
3. **エラーハンドリング**: Result型を使うと、エラーフローが明確になります

大きな変更になりますが、段階的にリファクタリングするのはどうでしょうか？
最初のステップとして、[具体的な提案] から始めることを提案します。

ペアプログラミングでサポートすることもできます。いかがでしょうか？
\```

### ケース2: 同じ指摘を繰り返す場合

❌ **悪い例**
「また同じミスをしていますね」

✅ **良い例**
\```
この問題は以前のPR #123 でも発生していました。

パターンとして、[共通の問題点] があるようです。
これを防ぐために、以下の方法を検討しませんか？

1. **Lintルール追加**: ESLintでこのパターンを検出
2. **コードスニペット**: VSCodeスニペットで正しい実装をテンプレート化
3. **ドキュメント**: チーム内ガイドラインに追加

ペアプログラミングでベストプラクティスを一緒に確認しましょうか？
\```

### ケース3: デザインに根本的な問題がある場合

❌ **悪い例**
「この設計は間違っています」

✅ **良い例**
\```
この設計について、いくつか懸念点があります。
一度、設計を見直すミーティングを持ちませんか？

**懸念点**:
1. **スケーラビリティ**: ユーザーが10万人を超えると、このアプローチは厳しいかもしれません
2. **依存関係**: 循環依存が発生する可能性があります
3. **テスト容易性**: 現在の設計だと、ユニットテストが困難です

**代替案**:
- イベント駆動アーキテクチャ
- CQRS パターン

設計意図を理解したいので、30分ほど時間をいただけませんか？
画面共有しながら議論しましょう。
\```

### ケース4: 時間がない中で急いで作られたコード

❌ **悪い例**
「雑なコードですね」

✅ **良い例**
\```
緊急対応お疲れ様でした！ 🙏
ホットフィックスとして機能していますね。

今後のために、いくつか改善点を技術的負債として記録しましょう：

**今すぐ対応が必要**:
- [ ] #456 エラーログに機密情報が含まれている（セキュリティ）

**次のスプリントで対応**:
- [ ] #457 テストの追加
- [ ] #458 エラーハンドリングの改善

**将来的に対応**:
- [ ] #459 リファクタリング（関数の分割）

緊急時は品質と速度のトレードオフが必要ですが、
技術的負債として記録することで、後で返済できます。
\```
```

---

## 6. セルフレビュー戦略

### 6.1 段階的セルフレビュー

```typescript
// 3段階のセルフレビュー
class SelfReviewer {
  async performThreeStageReview(code: Code): Promise<ReviewResult> {
    console.log('📝 Three-Stage Self Review\n');

    // Stage 1: 即座のレビュー（完成直後）
    const immediate = await this.immediateReview(code);

    // Stage 2: 休憩後のレビュー（30分後）
    await this.takeBreak(30);
    const afterBreak = await this.freshEyesReview(code);

    // Stage 3: ペアレビュー（同僚に依頼）
    const peerReview = await this.peerReview(code);

    return this.combineResults([immediate, afterBreak, peerReview]);
  }

  private async immediateReview(code: Code): Promise<ReviewResult> {
    console.log('Stage 1: Immediate Review\n');

    const checks = [
      // 機能性
      {
        question: '全ての要件を満たしているか？',
        check: () => this.checkRequirements(code),
      },
      // テスト
      {
        question: 'テストを追加したか？',
        check: () => this.checkTests(code),
      },
      // デバッグコード
      {
        question: 'console.log / debugger を削除したか？',
        check: () => !code.includes('console.log') && !code.includes('debugger'),
      },
      // エラーハンドリング
      {
        question: 'エラーハンドリングは適切か？',
        check: () => this.checkErrorHandling(code),
      },
    ];

    return this.runChecks(checks);
  }

  private async freshEyesReview(code: Code): Promise<ReviewResult> {
    console.log('Stage 2: Fresh Eyes Review (after break)\n');

    const checks = [
      // 可読性
      {
        question: '初めて読む人が理解できるか？',
        check: () => this.checkReadability(code),
      },
      // 命名
      {
        question: '変数名・関数名は明確か？',
        check: () => this.checkNaming(code),
      },
      // コメント
      {
        question: 'コメントは適切か（Why重視）？',
        check: () => this.checkComments(code),
      },
      // 重複
      {
        question: '重複コードはないか？',
        check: () => this.checkDuplication(code),
      },
    ];

    return this.runChecks(checks);
  }
}
```

### 6.2 セルフレビューツール

```bash
#!/bin/bash
# scripts/self-review.sh

echo "🔍 Self Review Tool"
echo "==================="

# 1. デバッグコードチェック
echo -e "\n📋 Checking for debug code..."
debug_count=$(git diff --cached | grep -E "console\.(log|debug|info)" | wc -l)
if [ $debug_count -gt 0 ]; then
  echo "⚠️  Found $debug_count console statements"
  git diff --cached | grep -n -E "console\.(log|debug|info)"
  read -p "Remove them? (y/n): " remove
  if [ "$remove" = "y" ]; then
    # 削除処理
    echo "Please remove manually"
  fi
fi

# 2. TODOコメントチェック
echo -e "\n📋 Checking for TODO comments..."
todo_count=$(git diff --cached | grep -E "TODO|FIXME" | wc -l)
if [ $todo_count -gt 0 ]; then
  echo "⚠️  Found $todo_count TODO/FIXME comments"
  git diff --cached | grep -n -E "TODO|FIXME"
  echo "Consider creating issues for these"
fi

# 3. テストファイルチェック
echo -e "\n📋 Checking for test files..."
src_files=$(git diff --cached --name-only | grep -E "src/.*\.(ts|tsx|js|jsx)$" | grep -v ".test." | wc -l)
test_files=$(git diff --cached --name-only | grep -E "\.(test|spec)\.(ts|tsx|js|jsx)$" | wc -l)

if [ $src_files -gt 0 ] && [ $test_files -eq 0 ]; then
  echo "⚠️  Source files changed but no test files"
  read -p "Did you add tests? (y/n): " has_tests
  if [ "$has_tests" != "y" ]; then
    echo "❌ Please add tests before committing"
    exit 1
  fi
fi

# 4. カバレッジチェック
echo -e "\n📋 Checking test coverage..."
npm test -- --coverage --silent
coverage=$(cat coverage/coverage-summary.json | jq '.total.lines.pct')
threshold=80

if (( $(echo "$coverage < $threshold" | bc -l) )); then
  echo "⚠️  Coverage $coverage% is below threshold $threshold%"
  read -p "Continue anyway? (y/n): " continue
  if [ "$continue" != "y" ]; then
    exit 1
  fi
fi

# 5. Lintチェック
echo -e "\n📋 Running linter..."
npm run lint

# 6. ビルドチェック
echo -e "\n📋 Building..."
npm run build

echo -e "\n✅ Self review completed!"
echo "Ready to create PR? (y/n): "
read ready

if [ "$ready" = "y" ]; then
  echo "Great! Don't forget to:"
  echo "  1. Write a clear PR description"
  echo "  2. Add screenshots if UI changed"
  echo "  3. Link related issues"
  echo "  4. Request reviewers"
fi
```

---

## 7. レビューツール活用

### 7.1 GitHub の効果的な使い方

```markdown
## GitHub レビュー機能の活用

### 1. Suggested Changes（提案変更）

コードの修正案を直接提示できます：

\`\`\`suggestion
const userName = user?.name ?? 'Guest';
\`\`\`

レビュイーは「Commit suggestion」ボタンで即座に適用可能

### 2. Review Comments のスレッド

関連するコメントをスレッド化：
- 議論を整理
- 解決済みマーク
- 追跡が容易

### 3. Code Owners

自動的に適切なレビュワーを割り当て：

\`\`\`
# .github/CODEOWNERS
/src/api/** @backend-team
/src/components/** @frontend-team
\`\`\`

### 4. Review Templates

.github/PULL_REQUEST_TEMPLATE.md で標準化：
- 必要な情報を漏らさない
- レビュワーの負担軽減

### 5. Draft PR

開発中のフィードバックを得る：
- 早期のアーキテクチャレビュー
- 方向性の確認
- 完成前の相談

### 6. Multiple Reviewers

複数の観点でレビュー：
- Backend エンジニア: ロジック・パフォーマンス
- Frontend エンジニア: UI/UX
- Security: セキュリティ
- DevOps: インフラへの影響
```

### 7.2 自動レビューツールの組み合わせ

```yaml
# .github/workflows/automated-review.yml
name: Automated Code Review

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  # 1. Static Analysis
  static-analysis:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run ESLint
        run: npm run lint
      - name: Run TypeScript Check
        run: npm run type-check

  # 2. Security Scan
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run Snyk
        uses: snyk/actions/node@master
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
      - name: Run CodeQL
        uses: github/codeql-action/analyze@v3

  # 3. Test & Coverage
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm test -- --coverage
      - name: Coverage Check
        run: |
          coverage=$(cat coverage/coverage-summary.json | jq '.total.lines.pct')
          if (( $(echo "$coverage < 80" | bc -l) )); then
            echo "Coverage $coverage% below 80%"
            exit 1
          fi

  # 4. Complexity Check
  complexity:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Check Complexity
        run: npx complexity-report src/

  # 5. Dependency Audit
  dependencies:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm audit --audit-level=high

  # 6. Bundle Size
  bundle-size:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm run build
      - name: Check bundle size
        run: |
          size=$(du -sh dist | cut -f1)
          echo "Bundle size: $size"
```

---

## 8. 言語別ベストプラクティス

### 8.1 TypeScript/JavaScript

```typescript
// TypeScript レビューチェックリスト
const typeScriptChecklist = {
  '型安全性': {
    items: [
      '`any` を避ける',
      '`unknown` を適切に使う',
      'Type guards を実装',
      'Generics を活用',
    ],
    examples: {
      bad: `
// ❌ any を使用
function processData(data: any) {
  return data.value;
}
      `,
      good: `
// ✅ 具体的な型
interface Data {
  value: string;
}

function processData(data: Data): string {
  return data.value;
}
      `,
    },
  },

  '非同期処理': {
    items: [
      'async/await を使う',
      'Promise をチェーン化しない',
      'エラーハンドリングを忘れない',
      'Promise.all で並行処理',
    ],
    examples: {
      bad: `
// ❌ Promiseチェーン
getData()
  .then(data => processData(data))
  .then(result => saveResult(result))
  .catch(err => console.error(err));
      `,
      good: `
// ✅ async/await
try {
  const data = await getData();
  const result = await processData(data);
  await saveResult(result);
} catch (err) {
  logger.error('Processing failed', err);
  throw new ProcessingError('Failed to process data', { cause: err });
}
      `,
    },
  },

  'React Hooks': {
    items: [
      'useMemo で重い計算をメモ化',
      'useCallback でコールバックをメモ化',
      '依存配列を正しく指定',
      'カスタムフックで再利用',
    ],
    examples: {
      bad: `
// ❌ 毎回再計算
function UserList({ users }) {
  const sortedUsers = users.sort((a, b) =>
    a.name.localeCompare(b.name)
  );
  return <div>{sortedUsers.map(u => <User key={u.id} user={u} />)}</div>;
}
      `,
      good: `
// ✅ useMemo でメモ化
function UserList({ users }) {
  const sortedUsers = useMemo(
    () => users.sort((a, b) => a.name.localeCompare(b.name)),
    [users]
  );
  return <div>{sortedUsers.map(u => <User key={u.id} user={u} />)}</div>;
}
      `,
    },
  },
};
```

### 8.2 Python

```python
# Python レビューチェックリスト
python_checklist = {
    '型ヒント': {
        'items': [
            '関数シグネチャに型ヒント',
            'mypy でチェック',
            'Optional を明示',
        ],
        'bad': """
def get_user(user_id):
    return database.get(user_id)
        """,
        'good': """
from typing import Optional

def get_user(user_id: int) -> Optional[User]:
    return database.get(user_id)
        """,
    },

    'エラーハンドリング': {
        'items': [
            '具体的な例外をキャッチ',
            '例外を握りつぶさない',
            'カスタム例外を定義',
        ],
        'bad': """
try:
    result = risky_operation()
except:
    pass  # ❌ 例外を握りつぶす
        """,
        'good': """
try:
    result = risky_operation()
except SpecificException as e:
    logger.error(f"Operation failed: {e}")
    raise ProcessingError("Failed to process") from e
        """,
    },

    'リスト内包表記': {
        'items': [
            '複雑すぎる内包表記は避ける',
            'ジェネレータ式を活用',
            '可読性重視',
        ],
        'bad': """
# ❌ 複雑すぎる
result = [item.upper() for sublist in data
          for item in sublist
          if item and len(item) > 3
          if not item.startswith('_')]
        """,
        'good': """
# ✅ 通常のループで明確に
result = []
for sublist in data:
    for item in sublist:
        if item and len(item) > 3 and not item.startswith('_'):
            result.append(item.upper())
        """,
    },
}
```

### 8.3 Swift

```swift
// Swift レビューチェックリスト
let swiftChecklist: [String: Any] = [
    "Optional": [
        "items": [
            "強制アンラップを避ける",
            "Optional Binding を使う",
            "nil coalescing を活用",
        ],
        "bad": """
// ❌ 強制アンラップ
let name = user!.name
        """,
        "good": """
// ✅ Optional Binding
guard let user = user else {
    return
}
let name = user.name

// または
let name = user?.name ?? "Unknown"
        """,
    ],

    "メモリ管理": [
        "items": [
            "循環参照を避ける",
            "[weak self] を適切に使う",
            "Combine の保持を確認",
        ],
        "bad": """
// ❌ 循環参照
class ViewController: UIViewController {
    var closure: (() -> Void)?

    func setup() {
        closure = {
            self.doSomething()  // 循環参照
        }
    }
}
        """,
        "good": """
// ✅ weak self
class ViewController: UIViewController {
    var closure: (() -> Void)?

    func setup() {
        closure = { [weak self] in
            self?.doSomething()
        }
    }
}
        """,
    ],

    "エラーハンドリング": [
        "items": [
            "throws 関数を適切に扱う",
            "Result型を活用",
            "カスタムエラーを定義",
        ],
        "good": """
// ✅ Result型
enum NetworkError: Error {
    case invalidURL
    case noData
    case decodingFailed
}

func fetchData(from url: String) -> Result<Data, NetworkError> {
    guard let url = URL(string: url) else {
        return .failure(.invalidURL)
    }

    // 処理
    guard let data = data else {
        return .failure(.noData)
    }

    return .success(data)
}
        """,
    ],
]
```

---

## 9. チーム文化の構築

### 9.1 心理的安全性の確保

```markdown
## 心理的安全性を高めるレビュー文化

### 1. 失敗を学習機会とする

❌ 悪い文化:
「またバグ入れたの？」
「前も同じミスしてたよね」

✅ 良い文化:
「このバグから学べることをチームで共有しましょう」
「同じ問題を防ぐために、Lint ルールを追加しませんか？」

### 2. 質問しやすい環境

❌ 悪い文化:
「こんなことも知らないの？」
「ドキュメント読んだ？」

✅ 良い文化:
「良い質問ですね！一緒に調べましょう」
「ドキュメントが分かりにくいですね。改善しましょう」

### 3. 異なる意見を歓迎

❌ 悪い文化:
「私のやり方に従ってください」
「議論の余地はありません」

✅ 良い文化:
「興味深い視点ですね。メリット・デメリットを議論しましょう」
「両方のアプローチを試して、データで判断しましょう」

### 4. 成長をサポート

❌ 悪い文化:
「これくらい自分で調べて」
「レベル低すぎる」

✅ 良い文化:
「参考になる記事を共有します」
「ペアプログラミングで一緒にやりましょう」
```

### 9.2 レビュー文化の指標

```typescript
// レビュー文化の健全性を測る指標
interface ReviewCultureMetrics {
  participation: number; // 参加率
  responseTime: number; // 応答時間
  constructiveness: number; // 建設性スコア
  learningOpportunities: number; // 学習機会の数
}

async function measureReviewCulture(): Promise<ReviewCultureMetrics> {
  const metrics = {
    // 参加率: 全メンバーがレビューに参加しているか
    participation: await calculateParticipation(),

    // 応答時間: レビューがタイムリーか
    responseTime: await calculateAvgResponseTime(),

    // 建設性: ポジティブなコメントの割合
    constructiveness: await calculateConstructiveness(),

    // 学習機会: TIL、参考リンクの共有数
    learningOpportunities: await countLearningMoments(),
  };

  console.log('📊 Review Culture Health Check\n');
  console.log(`Participation: ${metrics.participation}%`);
  console.log(`Avg Response Time: ${metrics.responseTime}h`);
  console.log(`Constructiveness Score: ${metrics.constructiveness}/100`);
  console.log(`Learning Moments: ${metrics.learningOpportunities}/month`);

  if (metrics.participation < 80) {
    console.warn('⚠️  Low participation. Everyone should be involved in reviews.');
  }

  if (metrics.responseTime > 4) {
    console.warn('⚠️  Slow response time. Set SLA for reviews.');
  }

  if (metrics.constructiveness < 70) {
    console.warn('⚠️  Review tone needs improvement. Focus on constructive feedback.');
  }

  return metrics;
}
```

---

## 10. ケーススタディ

### ケース1: 大規模リファクタリングのレビュー

```markdown
## 状況
- 500行のレガシーコードを完全リライト
- 3週間の作業
- 15ファイル変更

## 課題
- レビューに時間がかかりすぎる
- レビュワーが全体像を把握できない
- フィードバックが遅れてブロック

## 解決策

### 1. PR を段階的に分割
\`\`\`
PR #1: モデル層のリファクタリング（100行）
PR #2: サービス層のリファクタリング（150行）
PR #3: コントローラー層のリファクタリング（120行）
PR #4: ビュー層のリファクタリング（130行）
\`\`\`

### 2. Design Doc を先に共有
リファクタリング計画を事前にレビュー：
- アーキテクチャ図
- 移行戦略
- テスト戦略

### 3. Draft PR で早期フィードバック
完成前に方向性を確認

## 結果
- レビュー時間: 8時間 → 各PR 1時間（計4時間）
- フィードバック品質: 向上
- マージまでの期間: 3週間 → 1週間
```

### ケース2: セキュリティ脆弱性の指摘

```markdown
## 状況
認証機能のPRでセキュリティ脆弱性を発見

## 発見した脆弱性
- パスワードが平文でログに出力
- SQLインジェクションの可能性
- セッショントークンが予測可能

## 対応

### 1. 即座にPRをブロック
\`\`\`markdown
🚨 **Critical Security Issue**

このPRには重大なセキュリティ脆弱性があり、マージをブロックします。

### 問題1: パスワードの平文ログ出力
**重要度**: Critical
**OWASP**: A09:2021 - Security Logging and Monitoring Failures

\`\`\`typescript
// ❌ 現在のコード
logger.info('Login attempt', { email, password });
\`\`\`

**修正**:
\`\`\`typescript
// ✅ 修正後
logger.info('Login attempt', { email });
\`\`\`

### 問題2: SQLインジェクション
[詳細な説明と修正方法]

### 問題3: セッショントークンの脆弱性
[詳細な説明と修正方法]
\`\`\`

### 2. セキュリティチームに escalate
即座にセキュリティチームに通知

### 3. 同様の問題がないか確認
\`\`\`bash
# 全コードベースをスキャン
grep -r "logger.*password" src/
\`\`\`

### 4. 再発防止策
- Snyk/CodeQL の導入
- セキュリティガイドラインの整備
- セキュリティトレーニングの実施

## 学び
- セキュリティレビューは最優先
- 自動スキャンツールの重要性
- チーム全体のセキュリティ意識向上が必要
```

### ケース3: パフォーマンス問題の発見

```markdown
## 状況
ユーザー一覧表示のPRでパフォーマンス問題を発見

## 問題のコード
\`\`\`typescript
async function getUsersWithPosts() {
  const users = await User.findAll();

  for (const user of users) {
    user.posts = await Post.findAll({
      where: { userId: user.id }
    });
  }

  return users;
}
\`\`\`

## 問題点
- N+1 クエリ問題
- 1000ユーザーで1001回のDBクエリ

## レビューコメント
\`\`\`markdown
## ⚡ パフォーマンス問題: N+1クエリ

**現状**:
ユーザー数に比例してクエリ数が増加します。

**パフォーマンス測定**:
| ユーザー数 | クエリ数 | 処理時間 |
|----------|--------|--------|
| 100      | 101    | 0.5s   |
| 1,000    | 1,001  | 5s     |
| 10,000   | 10,001 | 50s    |

**修正案**:
\`\`\`typescript
async function getUsersWithPosts() {
  const users = await User.findAll({
    include: [{
      model: Post,
      separate: false  // JOIN を使用
    }]
  });

  return users;
}
\`\`\`

**改善後**:
| ユーザー数 | クエリ数 | 処理時間 | 改善率 |
|----------|--------|--------|------|
| 100      | 1      | 0.05s  | 90%  |
| 1,000    | 1      | 0.2s   | 96%  |
| 10,000   | 1      | 0.8s   | 98%  |

**テスト方法**:
\`\`\`typescript
// パフォーマンステストを追加
it('should load users efficiently', async () => {
  await createManyUsers(1000);

  const start = Date.now();
  await getUsersWithPosts();
  const duration = Date.now() - start;

  expect(duration).toBeLessThan(500); // 500ms以内
});
\`\`\`
\`\`\`

## 結果
- 作成者が修正を実施
- パフォーマンステストを追加
- ドキュメントにベストプラクティスとして追加

## 学び
- パフォーマンスレビューの重要性
- データで示すことの効果
- テストで継続的に監視
```

---

## まとめ

効果的なコードレビューは、単なるバグ発見ツールではなく、チーム全体の成長を促進する文化です。

### レビューの価値

```typescript
const reviewValue = {
  immediate: 'バグの早期発見',
  shortTerm: 'コード品質の向上',
  longTerm: 'チーム全体のスキル向上',
  culture: '心理的安全性の高いチーム',
};
```

### 継続的改善

```markdown
## レビュープロセスの改善サイクル

1. **測定**: メトリクスを収集
   - レビュー時間
   - バグ検出率
   - フィードバック品質

2. **分析**: ボトルネックを特定
   - レビューが遅い原因
   - 見逃されるバグのパターン
   - コミュニケーション課題

3. **改善**: プロセスを最適化
   - 自動化の導入
   - ガイドライン更新
   - トレーニング実施

4. **検証**: 改善効果を測定
   - Before/After比較
   - チームフィードバック収集
```

### 次のステップ

1. セルフレビューを習慣化
2. 建設的なフィードバックを実践
3. 自動化ツールを導入
4. チーム文化を醸成
5. 継続的に改善

---

**更新日**: 2025年1月2日
**次回更新予定**: 四半期ごと
**フィードバック**: skill-feedback@example.com
