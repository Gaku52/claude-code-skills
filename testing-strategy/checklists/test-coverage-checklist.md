# テストカバレッジチェックリスト

**目的**: テストカバレッジを適切に管理し、品質を担保するための観点リスト

---

## 1. カバレッジ目標設定

### プロジェクト全体
- [ ] 全体のカバレッジ目標を設定した（推奨: 80%以上）
- [ ] カバレッジの測定方法を決定した
- [ ] カバレッジレポートの生成を自動化した
- [ ] カバレッジの定期的なレビュー体制を確立した

### レイヤー別目標
- [ ] Unit Testsのカバレッジ目標（推奨: 90%以上）
- [ ] Integration Testsのカバレッジ目標（推奨: 70%以上）
- [ ] E2E Testsのカバレッジ目標（推奨: クリティカルフロー100%）

---

## 2. カバレッジの種類

### Statement Coverage（行カバレッジ）
- [ ] 全てのコード行が実行されているか確認
- [ ] 未実行の行を特定
- [ ] 未実行の理由を文書化（Dead code, 意図的な除外）

**例**:
```typescript
function divide(a: number, b: number): number {
  if (b === 0) {
    throw new Error('Division by zero'); // ← この行もテスト必要
  }
  return a / b;
}

// ✅ 両方の行をカバー
it('should divide numbers', () => {
  expect(divide(10, 2)).toBe(5);
});

it('should throw error for division by zero', () => {
  expect(() => divide(10, 0)).toThrow('Division by zero');
});
```

### Branch Coverage（分岐カバレッジ）
- [ ] 全てのif/elseブランチがテストされているか確認
- [ ] switchの全ケースがテストされているか確認
- [ ] 三項演算子の両側がテストされているか確認

**例**:
```typescript
function getDiscount(price: number): number {
  if (price > 1000) {
    return price * 0.2; // ✅ テスト必要
  } else if (price > 500) {
    return price * 0.1; // ✅ テスト必要
  } else {
    return 0; // ✅ テスト必要
  }
}

// 3つの分岐を全てテスト
it.each([
  [1500, 300],    // > 1000
  [700, 70],      // > 500
  [300, 0],       // その他
])('should calculate discount for %i', (price, expected) => {
  expect(getDiscount(price)).toBe(expected);
});
```

### Function Coverage（関数カバレッジ）
- [ ] 全ての関数が最低1回は呼ばれているか確認
- [ ] 未使用の関数を特定
- [ ] Dead codeを削除

### Line Coverage（行カバレッジ）
- [ ] 全ての論理行が実行されているか確認
- [ ] マルチラインステートメントが完全にテストされているか確認

---

## 3. カバレッジツール設定

### Jest設定
- [ ] jest.config.jsにcoverageを設定
- [ ] collectCoverageFromで対象ファイルを指定
- [ ] coverageThresholdで最低基準を設定
- [ ] coveragePathIgnorePatternsで除外ファイルを指定

**設定例**:
```javascript
// jest.config.js
module.exports = {
  collectCoverage: true,
  collectCoverageFrom: [
    'src/**/*.{ts,tsx}',
    '!src/**/*.d.ts',
    '!src/**/*.stories.tsx',
    '!src/**/__tests__/**',
  ],
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80,
    },
    './src/core/**': {
      branches: 90,
      functions: 90,
      lines: 90,
      statements: 90,
    },
  },
  coverageReporters: ['text', 'lcov', 'html'],
};
```

### Vitest設定
- [ ] vitest.config.tsにcoverageを設定
- [ ] coverage providerを選択（c8 or istanbul）
- [ ] 除外パターンを設定

**設定例**:
```typescript
// vitest.config.ts
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    coverage: {
      provider: 'c8',
      reporter: ['text', 'json', 'html'],
      exclude: [
        'node_modules/',
        'src/**/*.test.ts',
        'src/**/*.spec.ts',
      ],
      lines: 80,
      functions: 80,
      branches: 80,
      statements: 80,
    },
  },
});
```

---

## 4. カバレッジレポート確認

### HTMLレポート
- [ ] ブラウザでHTMLレポートを開いた
- [ ] 未カバーの行をハイライトで確認
- [ ] ファイル別のカバレッジを確認
- [ ] 関数別のカバレッジを確認

### ターミナルレポート
- [ ] カバレッジサマリーを確認
- [ ] 目標未達のファイルを特定
- [ ] カバレッジの推移を追跡

**レポート例**:
```bash
--------------------------|---------|----------|---------|---------|
File                      | % Stmts | % Branch | % Funcs | % Lines |
--------------------------|---------|----------|---------|---------|
All files                 |   85.42 |    78.95 |   82.14 |   86.11 |
 src/utils                |   95.45 |    90.00 |   94.12 |   96.00 |
  validators.ts           |   98.00 |    95.00 |  100.00 |   98.50 |
  formatters.ts           |   92.00 |    85.00 |   88.24 |   93.50 |
 src/services             |   75.30 |    67.90 |   70.25 |   76.00 | ⚠️ 低い
  api.ts                  |   65.00 |    55.00 |   60.00 |   66.00 | ⚠️ 改善必要
--------------------------|---------|----------|---------|---------|
```

---

## 5. 未カバーコードの分析

### 意図的な除外
- [ ] テスト不要なコードを特定した
  - エントリーポイント（main.ts, index.ts）
  - 型定義ファイル（.d.ts）
  - ストーリーファイル（.stories.tsx）
  - モックデータ
- [ ] 除外理由を文書化した
- [ ] `/* istanbul ignore next */` コメントを追加（必要に応じて）

**例**:
```typescript
// ✅ 意図的な除外
/* istanbul ignore next */
if (process.env.NODE_ENV === 'development') {
  console.log('Debug mode'); // デバッグコードはカバー不要
}

// ❌ 悪い例: 本来テストすべきコードを除外
/* istanbul ignore next */
function validateUser(user: User) {
  // これはテストすべき
  if (!user.email) throw new Error('Email required');
}
```

### カバーすべき未カバーコード
- [ ] エラーハンドリングブランチ
- [ ] エッジケース
- [ ] 条件分岐の全パターン
- [ ] デフォルト値の設定ロジック

---

## 6. クリティカルパスの100%カバレッジ

### クリティカル機能の特定
- [ ] 認証・認可ロジック
- [ ] 決済処理
- [ ] データ保存・削除
- [ ] セキュリティ関連機能
- [ ] ビジネスロジックの中核

### 確認項目
- [ ] クリティカル機能が100%カバーされている
- [ ] 全ての異常系がテストされている
- [ ] 境界値テストが存在する
- [ ] エッジケースがカバーされている

**例**:
```typescript
// 決済処理 = クリティカル → 100%カバレッジ必須
describe('PaymentService', () => {
  it('should process valid payment', () => { /* ... */ });
  it('should reject invalid card', () => { /* ... */ });
  it('should handle network error', () => { /* ... */ });
  it('should handle timeout', () => { /* ... */ });
  it('should rollback on failure', () => { /* ... */ });
  it('should validate amount limits', () => { /* ... */ });
  // 全てのケースをカバー
});
```

---

## 7. カバレッジの罠を避ける

### カバレッジ ≠ 品質
- [ ] カバレッジだけでなく、テストの質も確認
- [ ] 意味のないテストでカバレッジを稼いでいないか確認
- [ ] アサーションが適切か確認

**例**:
```typescript
// ❌ 悪い例: カバレッジは100%だが品質が低い
it('should work', () => {
  const result = calculate(10, 5); // アサーションなし
  result; // 実行だけでカバレッジは上がる
});

// ✅ 良い例: カバレッジ + 適切なアサーション
it('should calculate sum correctly', () => {
  const result = calculate(10, 5);
  expect(result).toBe(15);
  expect(typeof result).toBe('number');
});
```

### 過度なカバレッジ追求の弊害
- [ ] カバレッジ100%を目指さない（80-90%が現実的）
- [ ] テスト困難なコードを無理にテストしない
- [ ] 実装の詳細をテストしない

---

## 8. CI/CDでのカバレッジ管理

### 自動チェック
- [ ] PRでカバレッジレポートが自動生成される
- [ ] カバレッジが低下するPRをブロック
- [ ] カバレッジのトレンドを可視化
- [ ] カバレッジバッジをREADMEに追加

**GitHub Actions例**:
```yaml
# .github/workflows/test.yml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run tests with coverage
        run: npm run test:coverage
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage/lcov.info
          fail_ci_if_error: true
      - name: Check coverage threshold
        run: |
          npm run test:coverage:check || exit 1
```

### カバレッジ可視化
- [ ] Codecov / Coveralls を統合
- [ ] カバレッジバッジを表示
- [ ] 履歴グラフを確認
- [ ] ファイル別の詳細を確認

---

## 9. カバレッジ改善計画

### 低カバレッジファイルの対応
- [ ] カバレッジ70%未満のファイルをリストアップ
- [ ] 優先度を付ける（クリティカル度 × 変更頻度）
- [ ] 改善計画を立てる
- [ ] 定期的に進捗を確認

**改善プラン例**:
```markdown
## カバレッジ改善プラン

### 高優先度（即対応）
- [ ] src/services/payment.ts (45% → 90%) - クリティカル
- [ ] src/utils/validators.ts (60% → 85%) - 頻繁に変更

### 中優先度（1ヶ月以内）
- [ ] src/components/Form.tsx (55% → 75%)
- [ ] src/hooks/useAuth.ts (65% → 80%)

### 低優先度（リファクタリング時に対応）
- [ ] src/legacy/oldService.ts (30% → 50%) - 非推奨
```

### 定期レビュー
- [ ] 月次でカバレッジをレビュー
- [ ] トレンドを分析
- [ ] 目標未達の原因を特定
- [ ] 改善案を実施

---

## 10. チーム運用

### ルール設定
- [ ] PRマージ時の最低カバレッジを設定（推奨: 80%）
- [ ] カバレッジ低下を許さないルールを設定
- [ ] 例外ルールを明確化
- [ ] ルールをドキュメント化

**例外ルール例**:
```markdown
## カバレッジ例外ルール

以下の場合、カバレッジ低下を許容:
1. レガシーコードのリファクタリング（段階的に改善）
2. 実験的な機能の追加（フィーチャーフラグで保護）
3. 外部ライブラリのラッパー（統合テストでカバー）

承認プロセス:
- テックリードの承認が必要
- 改善計画をissueで作成
```

### 教育・共有
- [ ] カバレッジの重要性をチームに共有
- [ ] ベストプラクティスを文書化
- [ ] 定期的な勉強会を開催
- [ ] カバレッジ改善の成功事例を共有

---

## カバレッジ目標一覧表

| レイヤー | 目標 | 最低ライン | 備考 |
|---------|------|-----------|------|
| **Unit Tests** | 90% | 80% | ビジネスロジック中心 |
| **Integration Tests** | 70% | 60% | API, DB統合 |
| **E2E Tests** | クリティカルフロー100% | 主要フロー80% | ユーザーシナリオ |
| **全体** | 85% | 75% | プロジェクト全体 |

---

## 最終チェック

### 承認前確認
- [ ] 全体カバレッジが目標以上
- [ ] クリティカル機能が100%カバー
- [ ] カバレッジが低下していない
- [ ] 未カバーコードの理由が明確
- [ ] テストの品質が確保されている

---

**最終更新**: 2026-01-02
**関連ドキュメント**:
- [Test Strategy Checklist](./test-strategy-checklist.md)
- [PR Review Test Checklist](./pr-review-test-checklist.md)
