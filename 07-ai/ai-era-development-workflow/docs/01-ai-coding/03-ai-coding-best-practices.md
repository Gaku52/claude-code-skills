# AIコーディングのベストプラクティス ── レビュー、検証、品質保証

> AIが生成したコードの品質を保証するための体系的なレビュー手法と検証プロセスを学び、AI支援コーディングにおける信頼性と保守性を確保する。

---

## この章で学ぶこと

1. **AIコード品質の評価フレームワーク** ── AI出力を体系的にレビューする基準と手順を確立する
2. **検証プロセスの設計** ── AI生成コードを安全にプロダクションに投入するためのゲートを構築する
3. **継続的な品質改善** ── AI活用の品質を組織的に底上げするフィードバックループを構築する

---

## 1. AI生成コードの品質評価

### 1.1 レビューの5層モデル

```
┌──────────────────────────────────────────────────┐
│            AI生成コード レビュー5層モデル           │
│                                                  │
│  Layer 5: ビジネスロジック検証                     │
│  ┌────────────────────────────────────────────┐  │
│  │ 要件との整合性、エッジケース、ドメインルール  │  │
│  └────────────────────────────────────────────┘  │
│                                                  │
│  Layer 4: セキュリティ検証                        │
│  ┌────────────────────────────────────────────┐  │
│  │ 入力検証、認証・認可、暗号化、脆弱性チェック  │  │
│  └────────────────────────────────────────────┘  │
│                                                  │
│  Layer 3: パフォーマンス検証                      │
│  ┌────────────────────────────────────────────┐  │
│  │ 計算量、メモリ使用量、N+1問題、キャッシュ     │  │
│  └────────────────────────────────────────────┘  │
│                                                  │
│  Layer 2: 設計品質検証                            │
│  ┌────────────────────────────────────────────┐  │
│  │ SOLID原則、命名、凝集度、結合度、テスト容易性 │  │
│  └────────────────────────────────────────────┘  │
│                                                  │
│  Layer 1: 構文・スタイル検証                       │
│  ┌────────────────────────────────────────────┐  │
│  │ リンター、フォーマッター、型チェック（自動化） │  │
│  └────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
  ※ Layer 1-2は自動化可能。Layer 3-5は人間の判断が必要
```

### 1.2 品質ゲートのフロー

```
AI生成コード
    │
    ▼
┌──────────────┐  失敗  ┌──────────┐
│ Gate 1: Lint │──────►│ AIに修正  │
│ + Format     │       │ を依頼    │──┐
└──────┬───────┘       └──────────┘  │
       │ 通過                         │
       ▼                             │
┌──────────────┐  失敗               │
│ Gate 2: 型   │──────────────────────┤
│ チェック     │                      │
└──────┬───────┘                      │
       │ 通過                         │
       ▼                             │
┌──────────────┐  失敗               │
│ Gate 3: テスト│──────────────────────┤
│ (自動)       │                      │
└──────┬───────┘                      │
       │ 通過                         │
       ▼                             │
┌──────────────┐  問題あり            │
│ Gate 4: 人間 │──────────────────────┘
│ レビュー     │
└──────┬───────┘
       │ 承認
       ▼
  プロダクション
```

---

## 2. 具体的なレビュー手法

### コード例1: セキュリティレビューチェックリスト

```python
# AI生成コードに対するセキュリティレビューの実施例

# === AIが生成した認証コード ===
from fastapi import Depends, HTTPException
from jose import jwt

SECRET_KEY = "my-secret-key-12345"  # ⚠️ 問題1: ハードコード
ALGORITHM = "HS256"

async def get_current_user(token: str):
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    user_id = payload.get("sub")  # ⚠️ 問題2: 型チェックなし
    if user_id is None:
        raise HTTPException(status_code=401)
    return user_id  # ⚠️ 問題3: ユーザー存在チェックなし

# === レビュー後の修正版 ===
import os
from fastapi import Depends, HTTPException, status
from jose import jwt, JWTError
from datetime import datetime, timezone

SECRET_KEY = os.environ["JWT_SECRET_KEY"]  # 修正1: 環境変数から取得
ALGORITHM = "HS256"

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str | None = payload.get("sub")  # 修正2: 型注釈
        if user_id is None:
            raise credentials_exception
        # 修正3: 期限チェック
        exp = payload.get("exp")
        if exp and datetime.fromtimestamp(exp, tz=timezone.utc) < datetime.now(timezone.utc):
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    # 修正4: ユーザー存在チェック
    user = await db.get(User, int(user_id))
    if user is None or not user.is_active:
        raise credentials_exception
    return user
```

### コード例2: パフォーマンスレビュー

```python
# AI生成コードのパフォーマンス問題を特定する

# === AIが生成したコード（N+1問題あり） ===
async def get_orders_with_items(db: AsyncSession) -> list[dict]:
    orders = await db.execute(select(Order))
    result = []
    for order in orders.scalars():
        # ⚠️ N+1問題: 注文ごとにクエリが発行される
        items = await db.execute(
            select(OrderItem).where(OrderItem.order_id == order.id)
        )
        result.append({
            "order": order,
            "items": items.scalars().all()
        })
    return result

# === レビュー後の修正版（Eager Loading） ===
async def get_orders_with_items(db: AsyncSession) -> list[dict]:
    # JOINで1クエリに統合
    query = (
        select(Order)
        .options(selectinload(Order.items))  # Eager Loading
        .order_by(Order.created_at.desc())
        .limit(100)  # ページネーション
    )
    result = await db.execute(query)
    orders = result.scalars().unique().all()
    return [
        {"order": order, "items": order.items}
        for order in orders
    ]
```

### コード例3: テスト生成と検証

```python
# AIにテストを生成させ、その品質を検証する

# Step 1: AIにテスト生成を依頼
# プロンプト: "calculate_discount関数のテストを生成して。
#             正常系3件、異常系3件、境界値2件"

# Step 2: AIが生成したテスト
import pytest
from decimal import Decimal
from app.pricing import calculate_discount

class TestCalculateDiscount:
    """割引計算のテスト"""

    # 正常系
    def test_percentage_discount(self):
        assert calculate_discount(Decimal("1000"), "SAVE10") == Decimal("900")

    def test_fixed_amount_discount(self):
        assert calculate_discount(Decimal("5000"), "FLAT500") == Decimal("4500")

    def test_no_discount(self):
        assert calculate_discount(Decimal("1000"), None) == Decimal("1000")

    # 異常系
    def test_expired_coupon(self):
        with pytest.raises(CouponExpiredError):
            calculate_discount(Decimal("1000"), "EXPIRED01")

    def test_invalid_coupon(self):
        with pytest.raises(InvalidCouponError):
            calculate_discount(Decimal("1000"), "INVALID")

    def test_negative_amount(self):
        with pytest.raises(ValueError):
            calculate_discount(Decimal("-100"), "SAVE10")

    # 境界値
    def test_zero_amount(self):
        assert calculate_discount(Decimal("0"), "SAVE10") == Decimal("0")

    def test_discount_exceeds_amount(self):
        # 割引額が商品額を超える場合 → 0になるべき
        assert calculate_discount(Decimal("100"), "FLAT500") == Decimal("0")

# Step 3: 人間がレビューすべきポイント
# ✓ テストが実際のビジネスルールを反映しているか
# ✓ エッジケースが網羅されているか
# ✓ テストの独立性が保たれているか
# ✗ 不足: 並行実行テストが含まれていない → 追加指示
```

### コード例4: AI生成コードのリファクタリング判断

```typescript
// AIが一度に生成した大きな関数を分割する判断基準

// === AIが生成した200行の関数（要リファクタリング） ===
async function processOrder(orderData: OrderInput): Promise<OrderResult> {
  // バリデーション (30行) → 分離対象
  // 在庫チェック (20行)  → 分離対象
  // 価格計算 (40行)     → 分離対象
  // 決済処理 (30行)     → 分離対象
  // 在庫更新 (20行)     → 分離対象
  // 通知送信 (20行)     → 分離対象
  // ログ記録 (20行)     → 分離対象
  // ... 全て1関数内
}

// === リファクタリング後 ===
async function processOrder(orderData: OrderInput): Promise<OrderResult> {
  const validatedOrder = validateOrder(orderData);
  await checkInventory(validatedOrder.items);
  const pricing = calculatePricing(validatedOrder);
  const payment = await processPayment(pricing);
  await updateInventory(validatedOrder.items);
  await sendNotifications(validatedOrder, payment);
  await recordAuditLog('order_processed', { orderId: payment.orderId });

  return { orderId: payment.orderId, total: pricing.total };
}

// 各関数は単一責任で20-30行以内に収まる
```

### コード例5: 品質メトリクス自動計測

```yaml
# .github/workflows/ai-code-quality.yml
# AI生成コードの品質を自動計測するCI

name: AI Code Quality Check
on: [pull_request]

jobs:
  quality-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Type Check
        run: npx tsc --noEmit

      - name: Lint
        run: npx eslint . --max-warnings 0

      - name: Test Coverage
        run: |
          npx vitest run --coverage
          # カバレッジが80%未満なら失敗
          npx istanbul check-coverage --lines 80

      - name: Security Audit
        run: npm audit --audit-level=high

      - name: Complexity Check
        run: |
          # 循環的複雑度が10を超える関数がないかチェック
          npx eslint . --rule '{"complexity": ["error", 10]}'

      - name: Bundle Size Check
        run: |
          npm run build
          npx bundlesize
```

---

## 3. AI活用の品質パターン集

### 効果的なパターン比較

| パターン | 説明 | 品質への影響 |
|---------|------|-------------|
| テストファースト | テストを先に書いてからAIに実装させる | 非常に高い |
| 段階的生成 | 小さな単位で生成→検証を繰り返す | 高い |
| コンテキスト注入 | 既存コードの規約をAIに提供 | 高い |
| ワンショット生成 | 1回で全て生成させる | 低い |
| 無検証マージ | AI出力をそのままマージ | 危険 |

### AI活用成熟度と品質指標

| 成熟度レベル | テストカバレッジ | バグ発生率 | レビュー時間 |
|------------|---------------|-----------|-------------|
| Level 1: 補完のみ | 40-50% | 従来と同等 | 従来と同等 |
| Level 2: 生成+手動レビュー | 60-70% | 20%減少 | 30%短縮 |
| Level 3: 生成+自動テスト | 80-90% | 50%減少 | 50%短縮 |
| Level 4: エージェント+CI | 90%以上 | 70%減少 | 70%短縮 |

---

## 4. 検証の自動化

```
┌────────────────────────────────────────────────────┐
│           AI生成コード検証パイプライン               │
│                                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ 静的解析  │  │ テスト   │  │ セキュリ │        │
│  │ (自動)   │─►│ (自動)   │─►│ ティ検査 │        │
│  │          │  │          │  │ (自動)   │        │
│  │ ・ESLint │  │ ・Unit   │  │ ・SAST   │        │
│  │ ・tsc    │  │ ・Integra│  │ ・Dep    │        │
│  │ ・mypy   │  │ ・E2E    │  │  Audit   │        │
│  └──────────┘  └──────────┘  └────┬─────┘        │
│                                    │              │
│                                    ▼              │
│                 ┌──────────────────────────┐      │
│                 │ 人間レビュー（残り10-20%） │      │
│                 │ ・ビジネスロジック検証      │      │
│                 │ ・アーキテクチャ適合性      │      │
│                 │ ・ドメイン知識との整合       │      │
│                 └──────────────────────────┘      │
└────────────────────────────────────────────────────┘
```

---

## アンチパターン

### アンチパターン 1: テストなしマージ

```python
# BAD: AIが生成したコードをテストなしでマージ
# "AIが書いたんだから正しいだろう" → 危険な思い込み

# 実際にあった事故例:
def send_notification(user_id: int, message: str) -> bool:
    """AIが生成した通知送信関数"""
    users = get_all_users()  # ← 全ユーザーを取得してしまう
    for user in users:
        if user.id == user_id:
            email_service.send(user.email, message)
            return True
    return False
    # 問題: 10万ユーザーいる場合、毎回全件取得 → 性能破壊

# GOOD: テストで性能問題を検出
def test_send_notification_performance():
    """通知送信が1秒以内に完了すること"""
    with time_limit(seconds=1):
        send_notification(user_id=42, message="test")
```

### アンチパターン 2: AI生成コードの過剰信頼スコアリング

```
❌ BAD: "AIの出力確率が高いから品質も高い"
   - LLMの出力確率と正確性は別物
   - 「自信満々に間違える」のがLLMの特性
   - 統計的に正しそうに見えるが論理的に誤っている場合がある

✅ GOOD: 実行ベースの品質検証
   - テストを実行して結果で判断
   - 型チェッカーで整合性を確認
   - 既存テストが全てパスすることを確認
   - 人間がドメイン知識で最終判断
```

---

## FAQ

### Q1: AI生成コードのレビューにどれくらい時間をかけるべきか？

目安は「AI生成にかかった時間の30-50%」。10分でAIが生成したコードなら3-5分のレビュー。ただし、セキュリティクリティカルな部分（認証、決済、個人情報処理）は通常の2-3倍の時間をかける。レビュー効率を上げるには、Layer 1-3を自動化し、人間はLayer 4-5に集中する。

### Q2: AIが生成したテストコードの品質はどう保証するか？

AIが生成したテスト自体を検証する必要がある。具体的には(1) ミューテーションテスト（Stryker等）でテストの有効性を検証、(2) テストが実際に失敗すべきケースで失敗するか確認（assertを削除して確認）、(3) カバレッジだけでなく、ビジネスルールの網羅性を人間が確認する。

### Q3: チームでAIコード品質基準をどう統一するか？

3つのレベルで統一する。(1) 自動化レベル: CI/CDに品質ゲートを組み込む（lint、型チェック、テスト、カバレッジ）、(2) ガイドラインレベル: 「AIコードレビューチェックリスト」をチームWikiに作成、(3) 文化レベル: 定期的なAIコードレビュー会で知見を共有し、ベストプラクティスを更新する。

---

## まとめ

| 項目 | 要点 |
|------|------|
| レビュー5層 | 構文→設計→性能→セキュリティ→ビジネスロジック |
| 品質ゲート | Lint→型→テスト→人間レビューの4段階 |
| 自動化範囲 | Layer 1-3（80%）は自動化、Layer 4-5は人間 |
| テスト戦略 | テストファーストでAIに実装させるのが最高品質 |
| メトリクス | カバレッジ、複雑度、セキュリティ監査を自動計測 |
| チーム運用 | CI/CD + ガイドライン + 定期レビュー会の3層 |

---

## 次に読むべきガイド

- [../02-workflow/00-ai-testing.md](../02-workflow/00-ai-testing.md) ── AIテストの詳細手法
- [../02-workflow/01-ai-code-review.md](../02-workflow/01-ai-code-review.md) ── AIコードレビューの実践
- [../03-team/00-ai-team-practices.md](../03-team/00-ai-team-practices.md) ── チーム品質基準の策定

---

## 参考文献

1. Google Engineering Practices, "Code Review Developer Guide," 2024. https://google.github.io/eng-practices/review/
2. OWASP, "OWASP Code Review Guide," 2024. https://owasp.org/www-project-code-review-guide/
3. Martin Fowler, "Refactoring: Improving the Design of Existing Code," Addison-Wesley, 2018.
