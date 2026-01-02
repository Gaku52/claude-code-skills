# PRレビュー時のテスト観点チェックリスト

**目的**: Pull Request レビュー時にテストの品質を確保するための観点リスト

---

## 1. テストの存在確認

### 基本確認
- [ ] 新規コードに対応するテストが追加されている
- [ ] 修正されたコードのテストが更新されている
- [ ] 削除されたコードの不要なテストも削除されている
- [ ] テストファイルの命名が規約に従っている

### カバレッジ
- [ ] カバレッジが低下していない
- [ ] 新規コードのカバレッジが80%以上
- [ ] クリティカルなパスが全てテストされている
- [ ] エッジケースがカバーされている

---

## 2. テストの品質

### テスト設計
- [ ] 1テスト = 1つの振る舞いになっている
- [ ] テスト名が明確で理解しやすい
- [ ] AAA（Arrange-Act-Assert）パターンに従っている
- [ ] Given-When-Thenで記述されている（BDD）

### テストの独立性
- [ ] テスト間の依存がない
- [ ] テストの実行順序に依存しない
- [ ] 各テストで必要なセットアップが完結している
- [ ] グローバル状態に依存していない

### アサーション
- [ ] アサーションが適切（多すぎず、少なすぎず）
- [ ] エラーメッセージが明確
- [ ] 期待値が具体的（toBeではなくtoBe(5)など）
- [ ] 型安全なアサーションを使用

---

## 3. Unit Tests

### カバレッジ
- [ ] 正常系がテストされている
- [ ] 異常系がテストされている
- [ ] 境界値がテストされている
- [ ] エッジケースがテストされている

### モック
- [ ] 外部依存が適切にモック化されている
- [ ] モックの使用が最小限
- [ ] モックの振る舞いが明確
- [ ] スパイの検証が適切

**チェック例**:
```typescript
// ✅ Good
it('should call API with correct params', () => {
  const mockApi = jest.fn().mockResolvedValue({ data: 'test' });
  await service.fetchData(mockApi, 'param');
  expect(mockApi).toHaveBeenCalledWith('param');
});

// ❌ Bad
it('should work', () => {
  // モックなしで外部API呼び出し
  const result = await service.fetchData();
  expect(result).toBeTruthy(); // 曖昧なアサーション
});
```

---

## 4. Integration Tests

### スコープ
- [ ] 複数モジュールの統合がテストされている
- [ ] API統合テストが存在する（該当する場合）
- [ ] データベース統合がテストされている（該当する場合）
- [ ] 外部サービスが適切にモック化されている

### データ管理
- [ ] テストデータの作成・削除が適切
- [ ] トランザクションロールバックを活用
- [ ] データの独立性が確保されている
- [ ] センシティブデータが含まれていない

**チェック例**:
```typescript
// ✅ Good
describe('User API Integration', () => {
  beforeEach(async () => {
    await db.transaction(async (trx) => {
      await trx('users').insert(testUser);
    });
  });

  afterEach(async () => {
    await db('users').where('email', testUser.email).del();
  });
});

// ❌ Bad
describe('User API Integration', () => {
  it('test', async () => {
    // データクリーンアップなし
    await db('users').insert(testUser);
    // ...
  });
});
```

---

## 5. E2E Tests

### シナリオ
- [ ] クリティカルなユーザーフローがテストされている
- [ ] シナリオが現実的
- [ ] 複数のユーザーロールでテストされている（該当する場合）
- [ ] エラーハンドリングがテストされている

### 安定性
- [ ] テストが安定している（フレイキーではない）
- [ ] 適切な待機処理が実装されている
- [ ] タイムアウト設定が適切
- [ ] リトライロジックが適切（必要な場合）

**チェック例**:
```typescript
// ✅ Good
test('checkout flow', async ({ page }) => {
  await page.goto('/products');
  await page.click('[data-testid="add-to-cart"]');
  await page.waitForSelector('[data-testid="cart-count"]');
  await expect(page.locator('[data-testid="cart-count"]')).toHaveText('1');
});

// ❌ Bad
test('checkout flow', async ({ page }) => {
  await page.goto('/products');
  await page.click('button'); // セレクタが脆弱
  await new Promise(r => setTimeout(r, 3000)); // 固定待機
});
```

---

## 6. パフォーマンス

### 実行速度
- [ ] テストの実行時間が適切（Unit: <1s, Integration: <5s, E2E: <30s）
- [ ] 不要な待機がない
- [ ] 並列実行可能なテストが並列化されている
- [ ] 遅いテストに`@slow`タグが付与されている

### リソース使用
- [ ] メモリリークがない
- [ ] 不要なファイル生成がない
- [ ] データベース接続が適切にクローズされている
- [ ] 一時ファイルが削除されている

---

## 7. コード品質

### 可読性
- [ ] テストコードが読みやすい
- [ ] コメントが適切（複雑な部分のみ）
- [ ] マジックナンバーが定数化されている
- [ ] ヘルパー関数が適切に使用されている

### DRY原則
- [ ] 重複コードがない
- [ ] beforeEach/afterEachが適切に使用されている
- [ ] 共通ヘルパー関数が活用されている
- [ ] テストデータがファクトリー/Fixtureで管理されている

**チェック例**:
```typescript
// ✅ Good
describe('Calculator', () => {
  let calculator: Calculator;

  beforeEach(() => {
    calculator = new Calculator();
  });

  it('should add numbers', () => {
    expect(calculator.add(2, 3)).toBe(5);
  });

  it('should subtract numbers', () => {
    expect(calculator.subtract(5, 3)).toBe(2);
  });
});

// ❌ Bad
it('test 1', () => {
  const calculator = new Calculator(); // 重複
  expect(calculator.add(2, 3)).toBe(5);
});

it('test 2', () => {
  const calculator = new Calculator(); // 重複
  expect(calculator.subtract(5, 3)).toBe(2);
});
```

---

## 8. エラーハンドリング

### 異常系テスト
- [ ] エラーケースがテストされている
- [ ] 例外処理が適切にテストされている
- [ ] エラーメッセージが検証されている
- [ ] エラー型が検証されている

**チェック例**:
```typescript
// ✅ Good
it('should throw error for invalid input', () => {
  expect(() => validateEmail('invalid')).toThrow(ValidationError);
  expect(() => validateEmail('invalid')).toThrow('Invalid email format');
});

// ❌ Bad
it('should handle error', () => {
  try {
    validateEmail('invalid');
  } catch (e) {
    // エラーの検証なし
  }
});
```

---

## 9. テストのメンテナンス性

### 将来の変更への対応
- [ ] テストが実装の詳細に依存していない
- [ ] 公開APIのみをテストしている
- [ ] データテストIDが使用されている（E2E）
- [ ] ページオブジェクトパターンが使用されている（E2E）

### ドキュメント
- [ ] 複雑なテストにコメントがある
- [ ] テストの意図が明確
- [ ] 特殊なセットアップが文書化されている

---

## 10. CI/CD統合

### 自動化
- [ ] CIでテストが自動実行される
- [ ] カバレッジレポートが生成される
- [ ] テスト失敗時にPRがブロックされる
- [ ] フレイキーテストが検出される

### パフォーマンス
- [ ] CI実行時間が適切（<10分推奨）
- [ ] 並列実行が設定されている
- [ ] キャッシュが活用されている
- [ ] 不要なテストが除外されている

---

## 承認前の最終確認

### 必須項目（全てチェック必須）
- [ ] 全てのテストが成功している
- [ ] カバレッジが低下していない
- [ ] テストが独立している
- [ ] テストが高速である（Unit: <1s）
- [ ] 実装の詳細に依存していない

### 推奨項目（80%以上チェック推奨）
- [ ] エッジケースがカバーされている
- [ ] エラーハンドリングがテストされている
- [ ] テストコードが読みやすい
- [ ] DRY原則に従っている

---

## レビューコメント例

### 良いコメント例

```markdown
✅ テストカバレッジが向上していて良いですね！

🔍 エッジケース追加提案:
`calculateDiscount(0)` のケースも追加してはどうでしょうか？

💡 リファクタリング提案:
beforeEachでcalculatorの初期化を共通化すると、重複が減りそうです。
```

### 避けるべきコメント例

```markdown
❌ テストが足りない（具体性がない）
❌ これは良くない（理由不明）
❌ 全部書き直して（建設的でない）
```

---

**最終更新**: 2026-01-02
**関連ドキュメント**:
- [Test Strategy Checklist](./test-strategy-checklist.md)
- [Test Coverage Checklist](./test-coverage-checklist.md)
