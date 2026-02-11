# AIテスト ── テスト生成、カバレッジ向上

> AIを活用してテストコードを効率的に生成し、テストカバレッジを大幅に向上させるための戦略と具体的な手法を体系的に学ぶ。

---

## この章で学ぶこと

1. **AIテスト生成の手法** ── 単体テスト、結合テスト、E2Eテストの自動生成パターンを習得する
2. **カバレッジ向上戦略** ── AIを使ってテストの網羅性を効率的に高めるアプローチを学ぶ
3. **テスト品質の検証** ── AI生成テストが本当に有効かを検証する方法を身につける

---

## 1. AIテスト生成の全体像

### 1.1 テスト生成のアプローチ分類

```
┌──────────────────────────────────────────────────────┐
│            AIテスト生成アプローチ                       │
│                                                      │
│  アプローチ1: コードからテスト生成                     │
│  ┌──────────┐    AI分析    ┌──────────────┐          │
│  │ 実装コード │────────────►│ テストコード  │          │
│  └──────────┘             └──────────────┘          │
│  ・既存コードを読んでテストを逆生成                    │
│  ・カバレッジの穴を自動検出                           │
│                                                      │
│  アプローチ2: 仕様からテスト生成（TDD）               │
│  ┌──────────┐    AI生成    ┌──────────────┐         │
│  │ 仕様書   │────────────►│ テストコード  │          │
│  └──────────┘             └──────┬───────┘          │
│                                  │ AI実装            │
│                            ┌─────▼──────┐           │
│                            │ 実装コード  │           │
│                            └────────────┘           │
│                                                      │
│  アプローチ3: 変更差分からテスト生成                   │
│  ┌──────────┐    AI分析    ┌──────────────┐         │
│  │ git diff │────────────►│ 差分テスト    │          │
│  └──────────┘             └──────────────┘          │
│  ・変更された部分に絞ってテストを追加                  │
└──────────────────────────────────────────────────────┘
```

### 1.2 テストピラミッドとAI活用

```
                    ┌─────┐
                   /  E2E  \          AI活用度: 中
                  /  テスト  \         ・シナリオ生成
                 /───────────\        ・セレクタ生成
                / 結合テスト   \       AI活用度: 高
               / (Integration) \      ・API呼び出しパターン
              /─────────────────\     ・モック生成
             /   単体テスト      \    AI活用度: 非常に高い
            /    (Unit Test)     \   ・関数ごとのテスト自動生成
           /─────────────────────\  ・エッジケース網羅
          /                       \
         /─────────────────────────\
        テスト数:  多い ← ──────── → 少ない
        実行速度:  速い ← ──────── → 遅い
```

---

## 2. 単体テストの自動生成

### コード例1: 関数からテストを生成

```python
# 対象関数
def calculate_bmi(weight_kg: float, height_m: float) -> dict:
    """BMIを計算し、判定結果を返す"""
    if weight_kg <= 0 or height_m <= 0:
        raise ValueError("体重と身長は正の数でなければなりません")
    bmi = weight_kg / (height_m ** 2)
    if bmi < 18.5:
        category = "低体重"
    elif bmi < 25:
        category = "普通体重"
    elif bmi < 30:
        category = "肥満(1度)"
    else:
        category = "肥満(2度以上)"
    return {"bmi": round(bmi, 1), "category": category}

# AIプロンプト: "calculate_bmi関数のテストを網羅的に生成して。
#               正常系、異常系、境界値を含めて"

# AI生成テスト
import pytest
from app.health import calculate_bmi

class TestCalculateBMI:
    """BMI計算のテスト"""

    # 正常系: 各カテゴリ
    def test_underweight(self):
        result = calculate_bmi(50, 1.75)
        assert result["category"] == "低体重"
        assert result["bmi"] == 16.3

    def test_normal_weight(self):
        result = calculate_bmi(65, 1.70)
        assert result["category"] == "普通体重"
        assert result["bmi"] == 22.5

    def test_overweight(self):
        result = calculate_bmi(85, 1.70)
        assert result["category"] == "肥満(1度)"
        assert result["bmi"] == 29.4

    def test_obese(self):
        result = calculate_bmi(100, 1.70)
        assert result["category"] == "肥満(2度以上)"
        assert result["bmi"] == 34.6

    # 境界値
    @pytest.mark.parametrize("weight,height,expected_category", [
        (56.6, 1.75, "低体重"),    # BMI ≈ 18.49
        (56.7, 1.75, "普通体重"),   # BMI ≈ 18.51
        (76.5, 1.75, "普通体重"),   # BMI ≈ 24.98
        (76.6, 1.75, "肥満(1度)"),  # BMI ≈ 25.01
    ])
    def test_boundary_values(self, weight, height, expected_category):
        result = calculate_bmi(weight, height)
        assert result["category"] == expected_category

    # 異常系
    def test_negative_weight(self):
        with pytest.raises(ValueError):
            calculate_bmi(-70, 1.70)

    def test_zero_height(self):
        with pytest.raises(ValueError):
            calculate_bmi(70, 0)

    def test_zero_weight(self):
        with pytest.raises(ValueError):
            calculate_bmi(0, 1.70)
```

### コード例2: APIテストの自動生成

```python
# FastAPI エンドポイントからテストを生成

# プロンプト: "以下のFastAPIエンドポイントの結合テストを生成して。
#             httpxのAsyncClientを使い、DBはモック"

import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch
from app.main import app

@pytest.fixture
def mock_db():
    """DBモックの設定"""
    mock = AsyncMock()
    mock.get.return_value = {"id": 1, "name": "テスト太郎", "email": "test@example.com"}
    return mock

@pytest.mark.asyncio
class TestUserAPI:
    """ユーザーAPI結合テスト"""

    async def test_get_user_success(self, mock_db):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            with patch("app.deps.get_db", return_value=mock_db):
                response = await client.get("/api/v1/users/1")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "テスト太郎"

    async def test_get_user_not_found(self, mock_db):
        mock_db.get.return_value = None
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            with patch("app.deps.get_db", return_value=mock_db):
                response = await client.get("/api/v1/users/999")
        assert response.status_code == 404

    async def test_create_user_validation_error(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/v1/users", json={"name": ""})
        assert response.status_code == 422
```

### コード例3: Property-Based Testの生成

```python
# AIに Property-Based Test を生成させる

# プロンプト: "calculate_bmi関数のproperty-based testを
#             Hypothesisライブラリで生成して"

from hypothesis import given, assume, settings
from hypothesis.strategies import floats

class TestCalculateBMIProperty:
    """BMI計算のProperty-Based Test"""

    @given(
        weight=floats(min_value=0.1, max_value=500),
        height=floats(min_value=0.3, max_value=3.0),
    )
    @settings(max_examples=1000)
    def test_bmi_always_positive(self, weight, height):
        """BMI値は常に正の数"""
        result = calculate_bmi(weight, height)
        assert result["bmi"] > 0

    @given(
        weight=floats(min_value=0.1, max_value=500),
        height=floats(min_value=0.3, max_value=3.0),
    )
    def test_category_always_valid(self, weight, height):
        """カテゴリは4種類のいずれか"""
        result = calculate_bmi(weight, height)
        valid_categories = {"低体重", "普通体重", "肥満(1度)", "肥満(2度以上)"}
        assert result["category"] in valid_categories

    @given(
        weight=floats(min_value=-1000, max_value=0),
        height=floats(min_value=0.3, max_value=3.0),
    )
    def test_negative_weight_raises(self, weight, height):
        """負の体重はValueError"""
        with pytest.raises(ValueError):
            calculate_bmi(weight, height)
```

### コード例4: E2Eテストの生成

```typescript
// Playwrightを使ったE2Eテストの自動生成

// プロンプト: "ログイン→商品検索→カート追加→購入のE2Eテストを生成"

import { test, expect } from '@playwright/test';

test.describe('購入フロー', () => {
  test('商品を検索してカートに入れ購入を完了する', async ({ page }) => {
    // 1. ログイン
    await page.goto('/login');
    await page.fill('[data-testid="email"]', 'test@example.com');
    await page.fill('[data-testid="password"]', 'password123');
    await page.click('[data-testid="login-button"]');
    await expect(page).toHaveURL('/dashboard');

    // 2. 商品検索
    await page.fill('[data-testid="search-input"]', 'TypeScript入門');
    await page.press('[data-testid="search-input"]', 'Enter');
    await expect(page.locator('[data-testid="search-results"]')).toBeVisible();

    // 3. カートに追加
    await page.click('[data-testid="product-card"]:first-child');
    await page.click('[data-testid="add-to-cart"]');
    await expect(page.locator('[data-testid="cart-count"]')).toHaveText('1');

    // 4. 購入手続き
    await page.click('[data-testid="cart-icon"]');
    await page.click('[data-testid="checkout-button"]');
    await page.fill('[data-testid="card-number"]', '4242424242424242');
    await page.fill('[data-testid="card-expiry"]', '12/28');
    await page.fill('[data-testid="card-cvc"]', '123');
    await page.click('[data-testid="confirm-purchase"]');

    // 5. 完了確認
    await expect(page).toHaveURL(/\/orders\/\d+/);
    await expect(page.locator('[data-testid="order-success"]')).toBeVisible();
  });
});
```

### コード例5: テストカバレッジギャップの特定

```bash
# AIにカバレッジレポートを分析させる

# Step 1: カバレッジレポート生成
pytest --cov=src --cov-report=json:coverage.json

# Step 2: Claude Codeに分析を依頼
claude "coverage.jsonを読んで、カバレッジが低いファイルのうち
       ビジネスロジック的に重要なものをリストアップして。
       各ファイルに対して追加すべきテストケースを提案して"

# AIの出力例:
# 1. src/services/payment.py (カバレッジ: 45%)
#    - 決済失敗時のリトライロジックがテストされていない
#    - 部分返金のテストが不足
#    - タイムアウト時の挙動テストなし
#
# 2. src/domain/order.py (カバレッジ: 62%)
#    - 注文ステータス遷移の全パターンが網羅されていない
#    - 同時注文の競合テストなし
```

---

## 3. テスト生成ツール比較

### 3.1 AIテスト生成ツール

| ツール | 対応言語 | 特徴 | 精度 |
|--------|---------|------|------|
| Copilot (/tests) | 全般 | エディタ内で即座に生成 | 中 |
| Claude Code | 全般 | コンテキスト理解が深い | 高 |
| Codium AI (Qodo) | Python/JS/TS | テスト専門、提案型 | 高 |
| Diffblue Cover | Java | JUnit自動生成 | 非常に高い |
| EvoSuite | Java | 遺伝的アルゴリズム | 高 |

### 3.2 テスト種類別の最適アプローチ

| テスト種類 | 推奨手法 | AIの寄与度 | 人間の役割 |
|-----------|---------|-----------|-----------|
| 単体テスト | AIで自動生成 | 90% | 境界値の確認 |
| 結合テスト | AIで骨格生成+人間調整 | 70% | モック設計 |
| E2Eテスト | AIでシナリオ生成 | 60% | ユーザーフロー検証 |
| Property Test | AIで生成 | 80% | 不変条件の定義 |
| パフォーマンステスト | AIで雛形生成 | 50% | 閾値の決定 |
| セキュリティテスト | 人間主導+AI補助 | 30% | 脅威モデルの設計 |

---

## 4. テスト品質の検証

```
┌──────────────────────────────────────────────────┐
│       AI生成テストの品質検証プロセス                │
│                                                  │
│  Step 1: ミューテーションテスト                    │
│  ┌──────────────────────────────────────────┐    │
│  │ コードに意図的なバグを注入                 │    │
│  │ → テストが検出できるかチェック              │    │
│  │ → 検出率(Mutation Score)が80%以上なら合格  │    │
│  └──────────────────────────────────────────┘    │
│                                                  │
│  Step 2: テストの独立性チェック                    │
│  ┌──────────────────────────────────────────┐    │
│  │ テストの実行順序をランダム化               │    │
│  │ → 順序依存の失敗がないか確認              │    │
│  └──────────────────────────────────────────┘    │
│                                                  │
│  Step 3: 意味のあるアサーションか確認              │
│  ┌──────────────────────────────────────────┐    │
│  │ assert True のような無意味なテストを検出    │    │
│  │ → アサーション削除でテストが失敗するか     │    │
│  └──────────────────────────────────────────┘    │
└──────────────────────────────────────────────────┘
```

---

## アンチパターン

### アンチパターン 1: カバレッジ数字だけを追いかける

```python
# BAD: カバレッジ100%だが意味のないテスト
def test_create_order():
    order = create_order(user_id=1, items=[{"id": 1, "qty": 1}])
    assert order is not None  # ← 何も検証していない！
    # カバレッジは上がるが、バグは見つけられない

# GOOD: ビジネスルールを検証するテスト
def test_create_order_calculates_total():
    order = create_order(
        user_id=1,
        items=[
            {"id": 1, "qty": 2, "price": 1000},
            {"id": 2, "qty": 1, "price": 500},
        ]
    )
    assert order.total == 2500  # 実際の計算結果を検証
    assert order.item_count == 3
    assert order.status == OrderStatus.PENDING
```

### アンチパターン 2: AI生成テストの無批判受け入れ

```python
# BAD: AIが生成したテストをそのまま使う
# AIはしばしば「実装の動作」をテストしてしまう（実装テスト）

# AIが生成しがちな実装テスト（BAD）
def test_calculate_discount():
    """内部実装に依存したテスト"""
    result = calculate_discount(1000, "SAVE10")
    assert result == 1000 * 0.9  # ← 実装の計算式をそのまま書いている

# GOOD: 仕様をテストする
def test_calculate_discount():
    """10%割引クーポンで100円割引される"""
    result = calculate_discount(1000, "SAVE10")
    assert result == 900  # ← 期待する「結果」を明示
```

---

## FAQ

### Q1: AI生成テストはどの程度信頼できるか？

AI生成テストの信頼度は「正常系で80-90%、異常系で60-70%」程度。正常系のテストは高品質だが、ドメイン固有のエッジケースやビジネスルールの境界条件はAIが見逃しやすい。ミューテーションテストで有効性を検証し、人間がドメイン知識に基づくテストケースを補完するのが理想的。

### Q2: テスト生成のプロンプトで最も重要なことは？

「テストの目的を明確にすること」が最重要。「テストを生成して」だけでは汎用的なテストしか得られない。代わりに「注文キャンセル時に在庫が正しく復元されることを検証するテスト」のように、検証すべきビジネスルールを明示する。入出力の具体例を示すとさらに精度が上がる。

### Q3: 既存のテストがないレガシーコードにAIテストを追加する方法は？

段階的アプローチが有効。(1) まずAIにコードを読ませて「テストの書きやすさ」を評価させる。(2) リファクタリングなしでテスト可能な部分（純粋関数等）から開始。(3) テスト困難な部分はSeams（テスト可能な接合点）を見つけてからテストを追加。(4) カバレッジを少しずつ上げながら、リファクタリングとテスト追加を並行する。

---

## まとめ

| 項目 | 要点 |
|------|------|
| 生成アプローチ | コードから逆生成、仕様から生成、差分から生成の3種 |
| テストピラミッド | 単体テストでAI活用度が最も高い（90%） |
| 品質検証 | ミューテーションテストで生成テストの有効性を確認 |
| ツール選定 | Copilot(速度) vs Claude Code(品質) vs Codium(専門性) |
| カバレッジ戦略 | 数字よりビジネスルールの網羅性を重視 |
| 注意点 | 実装テストの回避、アサーションの意味確認 |

---

## 次に読むべきガイド

- [01-ai-code-review.md](./01-ai-code-review.md) ── AIコードレビューとの統合
- [03-ai-debugging.md](./03-ai-debugging.md) ── テスト失敗時のAIデバッグ
- [../01-ai-coding/03-ai-coding-best-practices.md](../01-ai-coding/03-ai-coding-best-practices.md) ── テスト戦略の全体像

---

## 参考文献

1. Martin Fowler, "TestPyramid," martinfowler.com, 2012. https://martinfowler.com/bliki/TestPyramid.html
2. David R. MacIver, "Hypothesis: Property-based testing for Python," 2024. https://hypothesis.readthedocs.io/
3. Pitest, "Mutation Testing," 2024. https://pitest.org/
