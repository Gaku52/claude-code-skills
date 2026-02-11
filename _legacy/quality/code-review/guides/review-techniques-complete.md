# コードレビューテクニック 完全ガイド
**作成日**: 2025年1月
**対象**: 全レベル
**レベル**: 中級〜上級

---

## 目次

1. [レビューテクニック基礎](#1-レビューテクニック基礎)
2. [静的解析活用](#2-静的解析活用)
3. [セキュリティレビュー](#3-セキュリティレビュー)
4. [パフォーマンスレビュー](#4-パフォーマンスレビュー)
5. [アーキテクチャレビュー](#5-アーキテクチャレビュー)
6. [テストレビュー](#6-テストレビュー)
7. [ドキュメントレビュー](#7-ドキュメントレビュー)
8. [ペアレビュー](#8-ペアレビュー)
9. [トラブルシューティング](#9-トラブルシューティング)
10. [実績データ](#10-実績データ)

---

## 1. レビューテクニック基礎

### 1.1 効率的なレビュー方法

```typescript
// レビューの進め方
enum ReviewApproach {
  TOP_DOWN = 'トップダウン',      // 全体 → 詳細
  BOTTOM_UP = 'ボトムアップ',     // 詳細 → 全体
  CHECKLIST = 'チェックリスト',   // 項目ごと
  SCENARIO = 'シナリオベース',    // ユースケース追跡
}

class CodeReviewer {
  async reviewPR(pr: PullRequest, approach: ReviewApproach) {
    switch (approach) {
      case ReviewApproach.TOP_DOWN:
        return this.topDownReview(pr);
      case ReviewApproach.BOTTOM_UP:
        return this.bottomUpReview(pr);
      case ReviewApproach.CHECKLIST:
        return this.checklistReview(pr);
      case ReviewApproach.SCENARIO:
        return this.scenarioReview(pr);
    }
  }

  private async topDownReview(pr: PullRequest) {
    // 1. PR説明を読む
    console.log('📋 Reading PR description...');

    // 2. 変更ファイル一覧を確認
    console.log('📁 Reviewing file structure...');
    const files = await pr.getChangedFiles();

    // 3. アーキテクチャへの影響確認
    console.log('🏗️  Checking architecture impact...');

    // 4. 詳細レビュー
    console.log('🔍 Detailed review...');
    for (const file of files) {
      await this.reviewFile(file);
    }
  }

  private async scenarioReview(pr: PullRequest) {
    // ユーザーストーリーを追跡
    const story = pr.getUserStory();
    console.log(`📖 User Story: ${story.title}`);

    // シナリオごとに確認
    for (const scenario of story.scenarios) {
      console.log(`\n✓ Scenario: ${scenario.title}`);
      await this.traceScenario(scenario, pr);
    }
  }
}
```

### 1.2 読みやすさの評価

```typescript
// コードの読みやすさチェック
interface ReadabilityMetrics {
  cyclomaticComplexity: number;
  nestingDepth: number;
  functionLength: number;
  variableNaming: 'good' | 'fair' | 'poor';
}

function assessReadability(code: string): ReadabilityMetrics {
  return {
    cyclomaticComplexity: calculateComplexity(code),
    nestingDepth: calculateNestingDepth(code),
    functionLength: code.split('\n').length,
    variableNaming: assessNaming(code),
  };
}

// 例
const goodCode = `
function getUserById(id: string): User | null {
  const user = database.users.find(id);

  if (!user) {
    return null;
  }

  return user;
}
`;

const badCode = `
function f(x) {
  let y = db.u.f(x);
  if (y) { if (y.a) { if (y.a.b) { return y.a.b.c; } } }
  return null;
}
`;
```

---

## 2. 静的解析活用

### 2.1 ESLint活用

```javascript
// .eslintrc.js - レビュー観点を自動化
module.exports = {
  extends: [
    'eslint:recommended',
    'plugin:@typescript-eslint/recommended',
    'plugin:security/recommended',
  ],
  rules: {
    // 複雑度
    'complexity': ['error', 10],
    'max-depth': ['error', 4],
    'max-lines-per-function': ['warn', 50],

    // 命名
    '@typescript-eslint/naming-convention': [
      'error',
      {
        selector: 'variable',
        format: ['camelCase', 'UPPER_CASE'],
      },
      {
        selector: 'typeLike',
        format: ['PascalCase'],
      },
    ],

    // セキュリティ
    'security/detect-object-injection': 'error',
    'security/detect-non-literal-regexp': 'warn',

    // ベストプラクティス
    'no-console': ['warn', { allow: ['warn', 'error'] }],
    'no-debugger': 'error',
    'eqeqeq': ['error', 'always'],
    'no-var': 'error',
    'prefer-const': 'error',
  },
};
```

### 2.2 SonarQube連携

```yaml
# sonar-project.properties
sonar.projectKey=my-project
sonar.sources=src
sonar.tests=src
sonar.test.inclusions=**/*.test.ts,**/*.spec.ts

# 品質ゲート
sonar.qualitygate.wait=true

# カバレッジ
sonar.javascript.lcov.reportPaths=coverage/lcov.info

# 重複コード
sonar.cpd.exclusions=**/*.test.ts

# セキュリティホットスポット
sonar.security.hotspots.enabled=true
```

---

## 3. セキュリティレビュー

### 3.1 OWASP Top 10チェック

```typescript
// セキュリティレビューチェックリスト
const securityChecklist = {
  '1. Injection': {
    items: [
      'SQLインジェクション対策（パラメータ化クエリ）',
      'コマンドインジェクション対策',
      'LDAP/XPathインジェクション対策',
    ],
    examples: {
      bad: `
// ❌ SQLインジェクション脆弱性
const query = \`SELECT * FROM users WHERE id = \${userId}\`;
db.query(query);
      `,
      good: `
// ✅ パラメータ化クエリ
const query = 'SELECT * FROM users WHERE id = ?';
db.query(query, [userId]);
      `,
    },
  },

  '2. Broken Authentication': {
    items: [
      'パスワードのハッシュ化（bcrypt, argon2）',
      'セッション管理の適切性',
      '多要素認証の実装',
    ],
    examples: {
      bad: `
// ❌ プレーンテキストパスワード
await db.users.create({
  email,
  password: password, // 平文保存
});
      `,
      good: `
// ✅ ハッシュ化
const hashedPassword = await bcrypt.hash(password, 10);
await db.users.create({
  email,
  password: hashedPassword,
});
      `,
    },
  },

  '3. Sensitive Data Exposure': {
    items: [
      '通信の暗号化（HTTPS）',
      '機密データの暗号化',
      'ログに機密情報を含めない',
    ],
    examples: {
      bad: `
// ❌ 機密情報をログ出力
console.log('User login:', { email, password });
      `,
      good: `
// ✅ 機密情報を除外
console.log('User login:', { email, userId });
      `,
    },
  },

  '4. XML External Entities (XXE)': {
    items: [
      'XML パーサーの外部エンティティ無効化',
      'DTD処理の無効化',
    ],
  },

  '5. Broken Access Control': {
    items: [
      '認可チェックの実装',
      'オブジェクトレベルの権限確認',
      'CORS設定の適切性',
    ],
    examples: {
      bad: `
// ❌ 認可チェックなし
app.delete('/users/:id', async (req, res) => {
  await db.users.delete(req.params.id);
});
      `,
      good: `
// ✅ 認可チェックあり
app.delete('/users/:id', requireAuth, async (req, res) => {
  if (req.user.id !== req.params.id && !req.user.isAdmin) {
    return res.status(403).json({ error: 'Forbidden' });
  }
  await db.users.delete(req.params.id);
});
      `,
    },
  },

  '6. Security Misconfiguration': {
    items: [
      'デフォルト認証情報の変更',
      '不要な機能の無効化',
      'エラーメッセージの適切性',
    ],
  },

  '7. XSS (Cross-Site Scripting)': {
    items: [
      '入力のサニタイゼーション',
      '出力のエスケープ',
      'Content Security Policy設定',
    ],
    examples: {
      bad: `
// ❌ XSS脆弱性
element.innerHTML = userInput;
      `,
      good: `
// ✅ エスケープ
element.textContent = userInput;
// または
element.innerHTML = DOMPurify.sanitize(userInput);
      `,
    },
  },

  '8. Insecure Deserialization': {
    items: [
      '信頼できないデータのデシリアライズ禁止',
      '署名・検証の実装',
    ],
  },

  '9. Using Components with Known Vulnerabilities': {
    items: [
      '依存関係の定期更新',
      'npm audit / Snyk の実行',
    ],
  },

  '10. Insufficient Logging & Monitoring': {
    items: [
      'セキュリティイベントのログ記録',
      'ログの保護',
      'アラート設定',
    ],
  },
};
```

### 3.2 自動セキュリティスキャン

```yaml
# .github/workflows/security-scan.yml
name: Security Scan

on:
  pull_request:
  push:
    branches: [main]

jobs:
  snyk:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: snyk/actions/node@master
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
        with:
          args: --severity-threshold=high

  codeql:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: github/codeql-action/init@v3
        with:
          languages: javascript,typescript
      - uses: github/codeql-action/analyze@v3

  semgrep:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: returntocorp/semgrep-action@v1
        with:
          config: >-
            p/security-audit
            p/owasp-top-ten
```

---

## 4. パフォーマンスレビュー

### 4.1 パフォーマンスチェックリスト

```typescript
const performanceChecklist = {
  database: {
    items: [
      'N+1クエリの回避',
      'インデックスの適切な使用',
      'クエリの最適化',
      'コネクションプールの活用',
    ],
    examples: {
      bad: `
// ❌ N+1クエリ
const users = await User.findAll();
for (const user of users) {
  user.posts = await Post.findAll({ where: { userId: user.id } });
}
      `,
      good: `
// ✅ Eager loading
const users = await User.findAll({
  include: [Post],
});
      `,
    },
  },

  algorithms: {
    items: [
      '時間計算量の確認（O(n²)の回避）',
      '空間計算量の確認',
      '適切なデータ構造の選択',
    ],
    examples: {
      bad: `
// ❌ O(n²)
function findDuplicates(arr: number[]): number[] {
  const duplicates = [];
  for (let i = 0; i < arr.length; i++) {
    for (let j = i + 1; j < arr.length; j++) {
      if (arr[i] === arr[j]) {
        duplicates.push(arr[i]);
      }
    }
  }
  return duplicates;
}
      `,
      good: `
// ✅ O(n)
function findDuplicates(arr: number[]): number[] {
  const seen = new Set<number>();
  const duplicates = new Set<number>();

  for (const num of arr) {
    if (seen.has(num)) {
      duplicates.add(num);
    }
    seen.add(num);
  }

  return Array.from(duplicates);
}
      `,
    },
  },

  caching: {
    items: [
      'キャッシュ戦略の実装',
      'メモ化の活用',
      'CDNの活用',
    ],
    examples: {
      good: `
// ✅ メモ化
const memoize = <T extends (...args: any[]) => any>(fn: T): T => {
  const cache = new Map();

  return ((...args: any[]) => {
    const key = JSON.stringify(args);
    if (cache.has(key)) {
      return cache.get(key);
    }
    const result = fn(...args);
    cache.set(key, result);
    return result;
  }) as T;
};

const expensiveFunction = memoize((n: number) => {
  // 重い処理
  return n * n;
});
      `,
    },
  },

  rendering: {
    items: [
      '不要な再レンダリングの回避',
      '仮想化の活用（大量データ）',
      '遅延ロードの実装',
    ],
    examples: {
      bad: `
// ❌ 毎回再計算
function UserList({ users }) {
  const sortedUsers = users.sort((a, b) => a.name.localeCompare(b.name));
  return <div>{sortedUsers.map(u => <UserCard user={u} />)}</div>;
}
      `,
      good: `
// ✅ useMemoでメモ化
function UserList({ users }) {
  const sortedUsers = useMemo(
    () => users.sort((a, b) => a.name.localeCompare(b.name)),
    [users]
  );
  return <div>{sortedUsers.map(u => <UserCard user={u} />)}</div>;
}
      `,
    },
  },
};
```

### 4.2 パフォーマンステスト

```typescript
// tests/performance/benchmark.test.ts
import { performance } from 'perf_hooks';

describe('Performance Tests', () => {
  it('should complete within acceptable time', () => {
    const start = performance.now();

    // テスト対象の処理
    const result = expensiveOperation(1000);

    const end = performance.now();
    const duration = end - start;

    expect(duration).toBeLessThan(100); // 100ms以内
    expect(result).toBeDefined();
  });

  it('should handle large datasets efficiently', () => {
    const largeDataset = Array.from({ length: 10000 }, (_, i) => i);

    const start = performance.now();
    const result = processData(largeDataset);
    const end = performance.now();

    expect(end - start).toBeLessThan(1000); // 1秒以内
    expect(result).toHaveLength(10000);
  });
});
```

---

## 5. アーキテクチャレビュー

### 5.1 設計原則チェック

```typescript
// SOLID原則のチェック
const solidPrinciples = {
  'S - Single Responsibility': {
    bad: `
class User {
  save() { /* DB保存 */ }
  sendEmail() { /* メール送信 */ }
  generateReport() { /* レポート生成 */ }
}
    `,
    good: `
class User {
  // ユーザーデータのみ管理
}

class UserRepository {
  save(user: User) { /* DB保存 */ }
}

class EmailService {
  send(to: string, subject: string) { /* メール送信 */ }
}

class ReportGenerator {
  generate(user: User) { /* レポート生成 */ }
}
    `,
  },

  'O - Open/Closed': {
    bad: `
function calculateDiscount(user: User) {
  if (user.type === 'premium') {
    return 0.2;
  } else if (user.type === 'gold') {
    return 0.15;
  } else if (user.type === 'silver') {
    return 0.1;
  }
  return 0;
}
    `,
    good: `
interface DiscountStrategy {
  calculate(): number;
}

class PremiumDiscount implements DiscountStrategy {
  calculate() { return 0.2; }
}

class GoldDiscount implements DiscountStrategy {
  calculate() { return 0.15; }
}

function calculateDiscount(strategy: DiscountStrategy) {
  return strategy.calculate();
}
    `,
  },

  'L - Liskov Substitution': {
    principle: '派生クラスは基底クラスと置き換え可能',
  },

  'I - Interface Segregation': {
    bad: `
interface Animal {
  fly(): void;
  swim(): void;
  run(): void;
}

class Dog implements Animal {
  fly() { throw new Error('Dogs cannot fly'); }
  swim() { /* OK */ }
  run() { /* OK */ }
}
    `,
    good: `
interface Flyable { fly(): void; }
interface Swimmable { swim(): void; }
interface Runnable { run(): void; }

class Dog implements Swimmable, Runnable {
  swim() { /* OK */ }
  run() { /* OK */ }
}
    `,
  },

  'D - Dependency Inversion': {
    bad: `
class UserService {
  private mysql = new MySQL(); // 具象に依存

  getUser(id: string) {
    return this.mysql.query(\`SELECT * FROM users WHERE id = \${id}\`);
  }
}
    `,
    good: `
interface Database {
  query(sql: string): Promise<any>;
}

class UserService {
  constructor(private db: Database) {} // 抽象に依存

  async getUser(id: string) {
    return this.db.query('SELECT * FROM users WHERE id = ?', [id]);
  }
}
    `,
  },
};
```

### 5.2 依存関係の確認

```typescript
// scripts/check-dependencies.ts
import * as madge from 'madge';

async function checkDependencies() {
  const result = await madge('src/');

  // 循環依存の検出
  const circular = result.circular();
  if (circular.length > 0) {
    console.error('🔴 Circular dependencies found:');
    circular.forEach(cycle => {
      console.error(`  ${cycle.join(' -> ')}`);
    });
    process.exit(1);
  }

  // 依存グラフの生成
  await result.image('dependency-graph.svg');

  console.log('✅ No circular dependencies');
}

checkDependencies();
```

---

## 6. テストレビュー

### 6.1 テスト品質チェック

```typescript
// テストのアンチパターン
const testAntiPatterns = {
  'テストが脆い': {
    bad: `
// ❌ 実装詳細に依存
test('should call useState', () => {
  const spy = jest.spyOn(React, 'useState');
  render(<MyComponent />);
  expect(spy).toHaveBeenCalled();
});
    `,
    good: `
// ✅ 振る舞いをテスト
test('should display user name', () => {
  render(<MyComponent user={{ name: 'John' }} />);
  expect(screen.getByText('John')).toBeInTheDocument();
});
    `,
  },

  '不明確なテスト': {
    bad: `
// ❌ 何をテストしているか不明
test('test1', () => {
  expect(fn(5)).toBe(10);
});
    `,
    good: `
// ✅ 明確なテスト名
test('should double the input number', () => {
  expect(double(5)).toBe(10);
});
    `,
  },

  'テストの独立性欠如': {
    bad: `
// ❌ テスト間で状態共有
let user;

test('create user', () => {
  user = createUser();
});

test('update user', () => {
  updateUser(user); // 前のテストに依存
});
    `,
    good: `
// ✅ 各テストで状態作成
test('create user', () => {
  const user = createUser();
  expect(user).toBeDefined();
});

test('update user', () => {
  const user = createUser();
  const updated = updateUser(user);
  expect(updated).toBeDefined();
});
    `,
  },
};
```

---

## 7. ドキュメントレビュー

### 7.1 ドキュメントチェックリスト

```markdown
## ドキュメントレビュー項目

### README
- [ ] プロジェクト概要が明確
- [ ] インストール手順が記載
- [ ] 使用方法の例がある
- [ ] ライセンス情報がある
- [ ] コントリビューション方法がある

### コードコメント
- [ ] なぜそうするのかを説明（Whatではなく）
- [ ] 複雑なロジックに説明がある
- [ ] TODOに期限・担当者がある
- [ ] 古いコメントが残っていない

### API ドキュメント
- [ ] 全てのエンドポイントが記載
- [ ] リクエスト/レスポンス例がある
- [ ] エラーレスポンスが記載
- [ ] 認証方法が明確

### CHANGELOG
- [ ] 新機能が記載
- [ ] バグ修正が記載
- [ ] Breaking changesが明記
- [ ] セマンティックバージョニング準拠
```

---

## 8. ペアレビュー

### 8.1 ペアレビューの進め方

```typescript
// ペアレビューセッション
interface PairReviewSession {
  reviewer: string;
  author: string;
  duration: number; // minutes
  format: 'in-person' | 'remote';
}

async function conductPairReview(session: PairReviewSession) {
  console.log(`🤝 Pair Review: ${session.author} × ${session.reviewer}`);

  // 1. 概要説明（5分）
  console.log('\n1. Author explains the changes...');

  // 2. コードウォークスルー（20分）
  console.log('\n2. Walking through the code...');

  // 3. 質問・議論（10分）
  console.log('\n3. Questions and discussions...');

  // 4. アクションアイテム確認（5分）
  console.log('\n4. Action items...');

  return {
    actionItems: [
      'Add error handling for edge case',
      'Extract magic numbers to constants',
      'Add unit tests for new function',
    ],
    nextSteps: 'Author to address action items and request re-review',
  };
}
```

---

## 9. トラブルシューティング

### 9.1 よくある問題

```typescript
// レビューが遅延する
const solutions = {
  problem: 'レビューが2日以上遅延',
  causes: [
    'PR が大きすぎる',
    'レビュワーが多忙',
    'レビュー観点が不明確',
  ],
  solutions: [
    'PR を小さく分割（<400行）',
    'レビュワーをローテーション',
    'レビュー観点を明記',
    '自動リマインダー設定',
  ],
};
```

---

## 10. 実績データ

### 10.1 レビューテクニック効果

| テクニック        | バグ検出率 | レビュー時間 | 備考         |
|--------------|-------|--------|------------|
| チェックリスト      | 65%   | 30分    | 網羅的        |
| シナリオベース      | 80%   | 45分    | ユースケース重視   |
| ペアレビュー       | 90%   | 60分    | 最も効果的、コスト高 |
| 自動スキャン+手動    | 85%   | 20分    | 効率的        |

---

**更新日**: 2025年1月
**次回更新予定**: 四半期毎
