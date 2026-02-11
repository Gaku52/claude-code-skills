# 🧪 testing-strategy 改善計画

**Skill名**: testing-strategy
**現状**: 🔴 低解像度
**目標**: 🟢 高解像度
**期間**: Week 1 (2026-01-01 〜 2026-01-05)
**工数**: 20時間

---

## 📊 現状分析

### 現在の状態
- **総文字数**: 68,222 chars
- **ガイド数**: 2/3 guides (In Progress)
- **解像度**: 🔴 低
- **不足要素**:
  - ケーススタディが少ない
  - 実行可能なコード例が不足
  - 失敗事例が少ない
  - チェックリストが未整備
  - テンプレートが未整備

### 既存ガイド
1. `guides/unit-testing-best-practices.md` (存在確認必要)
2. `guides/integration-testing-patterns.md` (存在確認必要)

---

## 🎯 改善目標

### 数値目標
- **総文字数**: 68,222 → 100,000+ chars (+31,778+ chars)
- **ガイド数**: 2 → 3個 (全て20,000+ chars)
- **ケーススタディ**: 0 → 3つ以上
- **コード例**: 数個 → 15+ 個
- **失敗事例**: 不明 → 10+ 個
- **チェックリスト**: 0 → 3個
- **テンプレート**: 0 → 5個

### 品質目標
- [ ] 実際のプロジェクトで即使える内容
- [ ] コピペで動くコード例
- [ ] 失敗から学べる構成
- [ ] チェックリストで実行をサポート
- [ ] テンプレートで時間短縮

---

## 📅 5日間の詳細計画

### Day 1 (月曜, 4h): リサーチと分析

#### 午前 (2h): 既存ガイドの分析

**タスク**
```bash
cd /Users/gaku/claude-code-skills/testing-strategy

# ディレクトリ構造確認
tree -L 2

# SKILL.mdの確認
cat SKILL.md | wc -l
cat SKILL.md

# 既存ガイドの確認
ls -la guides/
cat guides/*.md | wc -w

# 不足要素の洗い出し
echo "## 不足要素" > /tmp/testing-strategy-gaps.md
```

**成果物**
- [ ] 既存ガイドレビューメモ (`/tmp/testing-strategy-review.md`)
- [ ] 不足要素リスト (`/tmp/testing-strategy-gaps.md`)
- [ ] 改善優先度リスト

**時間配分**
- ディレクトリ構造確認: 15分
- SKILL.md確認: 30分
- 既存ガイド精読: 1時間15分

---

#### 午後 (2h): 追加コンテンツの計画

**タスク**
新規作成するガイドの詳細設計

**計画内容**

1. **guides/test-pyramid-practice.md** (25,000 chars)
   ```markdown
   # 目次
   1. テストピラミッドとは
   2. ケーススタディ1: Reactアプリケーション
   3. ケーススタディ2: API統合テスト
   4. ケーススタディ3: E2Eテストの最適化
   5. よくある失敗パターン 10選
   6. チェックリスト
   ```

2. **guides/tdd-bdd-workflow.md** (20,000 chars)
   ```markdown
   # 目次
   1. TDDの基本ワークフロー
   2. BDDとの使い分け
   3. 実際のプロジェクト例
   4. よくある失敗 7選
   5. チェックリスト
   ```

3. **既存ガイドの強化**
   - `unit-testing-best-practices.md`: +5,000 chars
   - `integration-testing-patterns.md`: +5,000 chars

**成果物**
- [ ] ガイド設計書 (`/tmp/testing-strategy-guide-design.md`)
- [ ] 各セクションの文字数配分計画
- [ ] 必要なコード例のリスト

**時間配分**
- 新規ガイド設計: 1時間
- 既存ガイド強化計画: 30分
- 全体レビュー: 30分

---

### Day 2 (火曜, 4h): テストピラミッド実践ガイド (Part 1)

#### 午前 (2h): ガイドの前半部分作成

**作成内容**

```markdown
# テストピラミッド実践ガイド

## 1. テストピラミッドとは (2,000 chars)
- 概念の説明
- 理論的背景 (Martin Fowler等の引用)
- なぜ重要か
- 各層の役割と責務

## 2. テストピラミッドの構成 (3,000 chars)
- Unit Tests (70%): 役割、範囲、技術スタック
- Integration Tests (20%): 役割、範囲、技術スタック
- E2E Tests (10%): 役割、範囲、技術スタック
- 比率の根拠

## 3. ケーススタディ1: Reactアプリケーション (10,000 chars)

### プロジェクト概要
- ECサイトのフロントエンド
- React + TypeScript + Jest + Testing Library
- 全体のテスト構成

### Unit Tests (70%)の実例

#### 例1: Buttonコンポーネント
\`\`\`typescript
// src/components/Button/Button.tsx
import React from 'react';

interface ButtonProps {
  label: string;
  onClick: () => void;
  disabled?: boolean;
}

export const Button: React.FC<ButtonProps> = ({
  label,
  onClick,
  disabled = false,
}) => {
  return (
    <button onClick={onClick} disabled={disabled}>
      {label}
    </button>
  );
};
\`\`\`

\`\`\`typescript
// src/components/Button/Button.test.tsx
import { render, screen, fireEvent } from '@testing-library/react';
import { Button } from './Button';

describe('Button', () => {
  it('renders with label', () => {
    render(<Button label="Click me" onClick={() => {}} />);
    expect(screen.getByText('Click me')).toBeInTheDocument();
  });

  it('calls onClick when clicked', () => {
    const handleClick = jest.fn();
    render(<Button label="Click me" onClick={handleClick} />);

    fireEvent.click(screen.getByText('Click me'));
    expect(handleClick).toHaveBeenCalledTimes(1);
  });

  it('does not call onClick when disabled', () => {
    const handleClick = jest.fn();
    render(<Button label="Click me" onClick={handleClick} disabled />);

    fireEvent.click(screen.getByText('Click me'));
    expect(handleClick).not.toHaveBeenCalled();
  });
});
\`\`\`

#### 例2: カスタムHook (useCart)
(詳細なコード例...)

#### 例3: ユーティリティ関数
(詳細なコード例...)
```

**成果物**
- [ ] ガイド前半 (10,000+ chars)
- [ ] Reactコンポーネントのテスト例 3つ

**時間配分**
- セクション1-2作成: 45分
- ケーススタディ1設計: 30分
- コード例作成: 45分

---

#### 午後 (2h): ケーススタディ1の完成

**作成内容**

```markdown
### Integration Tests (20%)の実例

#### 例1: ショッピングカート機能
\`\`\`typescript
// src/features/cart/Cart.integration.test.tsx
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Cart } from './Cart';
import { server } from '../../mocks/server';
import { rest } from 'msw';

describe('Cart Integration', () => {
  it('adds item to cart and shows total', async () => {
    render(<Cart />);

    // 商品を追加
    const addButton = screen.getByRole('button', { name: /add to cart/i });
    await userEvent.click(addButton);

    // APIレスポンスを待つ
    await waitFor(() => {
      expect(screen.getByText(/total: \$99/i)).toBeInTheDocument();
    });
  });

  it('handles API error gracefully', async () => {
    // APIエラーのモック
    server.use(
      rest.post('/api/cart', (req, res, ctx) => {
        return res(ctx.status(500));
      })
    );

    render(<Cart />);
    // エラーハンドリングのテスト...
  });
});
\`\`\`

### E2E Tests (10%)の実例

#### 例1: チェックアウトフロー
\`\`\`typescript
// e2e/checkout.spec.ts
import { test, expect } from '@playwright/test';

test('complete checkout flow', async ({ page }) => {
  await page.goto('http://localhost:3000');

  // 商品をカートに追加
  await page.click('[data-testid="add-to-cart"]');
  await expect(page.locator('[data-testid="cart-count"]')).toHaveText('1');

  // チェックアウトページに移動
  await page.click('[data-testid="checkout-button"]');

  // フォーム入力
  await page.fill('[name="email"]', 'test@example.com');
  await page.fill('[name="cardNumber"]', '4242424242424242');

  // 購入完了
  await page.click('[data-testid="submit-order"]');
  await expect(page.locator('text=Order confirmed')).toBeVisible();
});
\`\`\`

### テスト構成のまとめ
- Unit Tests: 45個 (70%)
- Integration Tests: 13個 (20%)
- E2E Tests: 6個 (10%)
- 総実行時間: 2分以内
- カバレッジ: 85%以上
```

**成果物**
- [ ] Integration Testsの実例 2つ
- [ ] E2E Testsの実例 1つ
- [ ] テスト構成のまとめ

**時間配分**
- Integration Tests作成: 1時間
- E2E Tests作成: 45分
- まとめ作成: 15分

---

### Day 3 (水曜, 4h): テストピラミッド実践ガイド (Part 2)

#### 午前 (2h): ケーススタディ2作成

**作成内容**

```markdown
## 4. ケーススタディ2: API統合テスト (7,000 chars)

### プロジェクト概要
- Node.js + Express + TypeScript
- Prisma + PostgreSQL
- Supertest + Jest

### テスト環境セットアップ

#### データベースのセットアップ
\`\`\`typescript
// tests/setup.ts
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

beforeAll(async () => {
  // テストデータベースのマイグレーション
  await prisma.$executeRawUnsafe('DROP SCHEMA IF EXISTS test CASCADE');
  await prisma.$executeRawUnsafe('CREATE SCHEMA test');
  // マイグレーション実行...
});

afterEach(async () => {
  // 各テスト後にクリーンアップ
  const tables = ['users', 'posts', 'comments'];
  for (const table of tables) {
    await prisma.$executeRawUnsafe(\`TRUNCATE TABLE \${table} CASCADE\`);
  }
});

afterAll(async () => {
  await prisma.$disconnect();
});
\`\`\`

### API統合テストの実例

#### 例1: 認証付きAPIテスト
\`\`\`typescript
// tests/api/auth.test.ts
import request from 'supertest';
import { app } from '../../src/app';

describe('POST /api/auth/login', () => {
  it('returns token for valid credentials', async () => {
    // ユーザー作成
    await request(app)
      .post('/api/users')
      .send({
        email: 'test@example.com',
        password: 'SecurePassword123',
      });

    // ログインテスト
    const response = await request(app)
      .post('/api/auth/login')
      .send({
        email: 'test@example.com',
        password: 'SecurePassword123',
      });

    expect(response.status).toBe(200);
    expect(response.body).toHaveProperty('token');
    expect(response.body.token).toMatch(/^[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+$/);
  });

  it('returns 401 for invalid credentials', async () => {
    const response = await request(app)
      .post('/api/auth/login')
      .send({
        email: 'test@example.com',
        password: 'WrongPassword',
      });

    expect(response.status).toBe(401);
    expect(response.body.error).toBe('Invalid credentials');
  });
});
\`\`\`

#### 例2: データベース連携テスト
(詳細なコード例...)

#### 例3: エラーハンドリングテスト
(詳細なコード例...)
```

**成果物**
- [ ] ケーススタディ2完成 (7,000+ chars)
- [ ] API統合テストの完全な実例 3つ

**時間配分**
- セットアップ部分: 45分
- テスト例1作成: 45分
- テスト例2-3作成: 30分

---

#### 午後 (2h): ケーススタディ3とよくある失敗

**作成内容**

```markdown
## 5. ケーススタディ3: E2Eテストの最適化 (5,000 chars)

### Playwrightによる高速E2Eテスト

#### 並列実行の設定
\`\`\`typescript
// playwright.config.ts
import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  workers: 4, // 並列数
  retries: 2,
  use: {
    baseURL: 'http://localhost:3000',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },
});
\`\`\`

#### フレーク対策
(詳細な実例...)

## 6. よくある失敗パターン 10選 (3,000 chars)

### 失敗1: ピラミッドが逆転している
**症状**
- E2Eテストが全体の50%以上
- テスト実行に30分以上かかる
- CIが頻繁にタイムアウト

**原因**
- 「E2Eが最も信頼できる」という誤解
- Unit/Integration層の設計不足

**解決策**
- テスト構成の見直し (70/20/10)
- Unit層の充実化

**予防策**
- テスト追加時に構成比を確認
- CI実行時間を監視

### 失敗2: テストが脆い (Flaky Tests)
(詳細...)

### 失敗3-10
(各失敗パターンの詳細...)

## 7. チェックリスト

### テスト戦略設計時
- [ ] テストピラミッドの比率を確認 (70/20/10)
- [ ] 各層の責務が明確か
- [ ] テスト実行時間は適切か (CI: 5分以内推奨)
...
```

**成果物**
- [ ] ケーススタディ3完成 (5,000+ chars)
- [ ] よくある失敗10選 (3,000+ chars)
- [ ] チェックリスト

**時間配分**
- ケーススタディ3: 1時間
- 失敗パターン: 45分
- チェックリスト: 15分

---

### Day 4 (木曜, 4h): TDD/BDD実践ガイド作成

#### 全日 (4h): 新規ガイド作成

**作成内容**

```markdown
# TDD/BDD実践ガイド

## 1. TDDの基本ワークフロー (5,000 chars)

### Red-Green-Refactorサイクル

#### Step 1: Red (失敗するテストを書く)
\`\`\`typescript
// sum.test.ts
describe('sum', () => {
  it('adds two numbers', () => {
    expect(sum(1, 2)).toBe(3);
  });
});

// 実行結果: FAIL - sum is not defined
\`\`\`

#### Step 2: Green (最小限の実装)
\`\`\`typescript
// sum.ts
export function sum(a: number, b: number): number {
  return a + b;
}

// 実行結果: PASS
\`\`\`

#### Step 3: Refactor (リファクタリング)
(改善例...)

### 実践例: 複雑な機能をTDDで開発
(ステップバイステップの実例...)

## 2. BDDとの使い分け (5,000 chars)

### Given-When-Then
\`\`\`typescript
describe('User Login', () => {
  it('should show dashboard after successful login', () => {
    // Given: ユーザーが存在する
    const user = createUser({ email: 'test@example.com' });

    // When: 正しい認証情報でログインする
    const result = login(user.email, 'password');

    // Then: ダッシュボードが表示される
    expect(result.redirectTo).toBe('/dashboard');
  });
});
\`\`\`

### BDDフレームワーク (Cucumber)の実例
(詳細...)

## 3. 実際のプロジェクト例 (7,000 chars)
(フィーチャー開発をTDDで進める完全な実例...)

## 4. よくある失敗 7選 (3,000 chars)
(失敗事例と対策...)
```

**成果物**
- [ ] TDD/BDDガイド完成 (20,000+ chars)
- [ ] Red-Green-Refactorの完全な実例
- [ ] BDDのコード例

**時間配分**
- TDD部分: 2時間
- BDD部分: 1.5時間
- 失敗事例: 30分

---

### Day 5 (金曜, 4h): チェックリスト・テンプレート・仕上げ

#### 午前 (2h): チェックリストとテンプレート作成

**成果物**

1. **checklists/test-strategy-checklist.md**
```markdown
# テスト戦略チェックリスト

## 新機能開発時
- [ ] Unit Testsを先に作成したか
- [ ] テストピラミッドの比率を守っているか (70/20/10)
- [ ] 各テストが独立して実行できるか
- [ ] テストが速い (Unit: <100ms, Integration: <1s)
- [ ] 失敗時のエラーメッセージがわかりやすいか
...

## PRレビュー時
...

## リファクタリング前
...
```

2. **checklists/pr-review-test-checklist.md**
3. **checklists/test-coverage-checklist.md**

4. **templates/jest-setup-template/**
```
jest-setup-template/
├── jest.config.js
├── setupTests.ts
├── testUtils.ts
└── README.md
```

5. **templates/testing-library-helpers/**
6. **templates/api-test-template/**

**時間配分**
- チェックリスト3個作成: 1時間
- テンプレート作成: 1時間

---

#### 午後 (2h): トラブルシューティングと最終レビュー

**成果物**

1. **references/common-testing-failures.md**
```markdown
# よくあるテスト失敗 10選

## 1. テストがランダムに失敗する (Flaky)
**症状**: 同じコードで成功したり失敗したりする
**原因**:
- タイミング依存
- 外部リソース依存
- グローバルステート

**解決策**:
- waitForを使う
- モックを使う
- テストの独立性を確保

**コード例**:
\`\`\`typescript
// ❌ Bad
it('loads data', () => {
  fetchData();
  expect(data).toBeDefined(); // タイミングによって失敗
});

// ✅ Good
it('loads data', async () => {
  await waitFor(() => {
    expect(data).toBeDefined();
  });
});
\`\`\`

## 2-10
(各失敗パターン...)
```

2. **references/troubleshooting-guide.md**

**時間配分**
- よくある失敗10選: 1時間
- トラブルシューティング: 30分
- 最終レビュー: 30分

---

#### 最終レビュー (30分)

**チェック項目**
```bash
# 文字数確認
cd /Users/gaku/claude-code-skills/testing-strategy
find . -name "*.md" -exec wc -c {} + | tail -1

# ガイド数確認
ls -1 guides/*.md | wc -l

# リンク検証
grep -r "\[.*\](.*)" . | grep -v ".git"

# 進捗更新
cd ..
npm run track

# コミット
./scripts/safe-commit-push.sh "feat(testing-strategy): complete comprehensive testing guides"
```

**完了確認**
- [ ] 総文字数 100,000+ chars
- [ ] ガイド数 3個 (各20,000+ chars)
- [ ] ケーススタディ 3つ
- [ ] コード例 15+ 個
- [ ] 失敗事例 10+ 個
- [ ] チェックリスト 3個
- [ ] テンプレート 5個
- [ ] 全てのリンクが有効
- [ ] npm run track で🟢高に到達

---

## 📁 最終的なディレクトリ構造

```
testing-strategy/
├── SKILL.md (更新)
├── README.md (更新)
├── guides/
│   ├── test-pyramid-practice.md (新規, 25,000+ chars) ✨
│   ├── tdd-bdd-workflow.md (新規, 20,000+ chars) ✨
│   ├── unit-testing-best-practices.md (強化, +5,000 chars)
│   └── integration-testing-patterns.md (強化, +5,000 chars)
├── checklists/ (新規フォルダ) ✨
│   ├── test-strategy-checklist.md
│   ├── pr-review-test-checklist.md
│   └── test-coverage-checklist.md
├── templates/ (新規フォルダ) ✨
│   ├── jest-setup-template/
│   │   ├── jest.config.js
│   │   ├── setupTests.ts
│   │   ├── testUtils.ts
│   │   └── README.md
│   ├── testing-library-helpers/
│   │   └── ...
│   └── api-test-template/
│       └── ...
└── references/ (新規フォルダ) ✨
    ├── common-testing-failures.md
    └── troubleshooting-guide.md
```

---

## ✅ 完了基準

### 必須基準 (Must Have)
- [ ] 総文字数 100,000+ chars
- [ ] 新規ガイド 2個完成 (各20,000+ chars)
- [ ] ケーススタディ 3つ以上
- [ ] コピペで動くコード例 15+ 個
- [ ] npm run track で🟢高に到達

### 推奨基準 (Should Have)
- [ ] 失敗事例 10+ 個
- [ ] チェックリスト 3個
- [ ] テンプレート 5個
- [ ] トラブルシューティング 15+ 項目

### 理想基準 (Nice to Have)
- [ ] 他の開発者がレビューして「すぐ使える」と評価
- [ ] 全てのコード例が動作確認済み
- [ ] 内部リンクが全て有効

---

## 🚀 実行コマンド

### Day 1
```bash
cd /Users/gaku/claude-code-skills/testing-strategy
cat SKILL.md
ls -la guides/
cat guides/*.md | wc -w
```

### Day 2-4
```bash
# ガイド作成 (エディタで)
code guides/test-pyramid-practice.md
code guides/tdd-bdd-workflow.md
```

### Day 5
```bash
# テンプレート作成
mkdir -p templates/jest-setup-template
mkdir -p checklists references

# 最終チェック
find . -name "*.md" -exec wc -c {} + | tail -1
npm run track
./scripts/safe-commit-push.sh "feat(testing-strategy): complete comprehensive testing guides"
```

---

**最終更新**: 2026-01-01
**実行予定**: 2026-01-01 〜 2026-01-05
