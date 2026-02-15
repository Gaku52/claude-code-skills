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

## 5. テスト生成の実践パターン

### 5.1 TDD + AI の統合ワークフロー

```python
# TDD（テスト駆動開発）とAIテスト生成を組み合わせるワークフロー

class TDDAIWorkflow:
    """
    TDD + AI のハイブリッドワークフロー

    手順:
    1. 人間が仕様をプロンプトで記述
    2. AIがテストコードを生成（Red）
    3. 人間がテストの妥当性を検証
    4. AIが実装コードを生成（Green）
    5. AIがリファクタリングを提案（Refactor）
    6. 人間が最終確認
    """

    # Step 1: 仕様をプロンプトで記述
    SPEC_PROMPT = """
    以下の仕様に基づいてテストを生成してください。

    ## 機能: 割引計算サービス

    ### ビジネスルール
    1. 通常割引: 購入金額10,000円以上で5%割引
    2. 会員割引: ゴールド会員は追加10%割引
    3. クーポン: 有効なクーポンコードで固定額割引
    4. 割引上限: 合計割引率は30%を超えない
    5. 最低支払額: 割引後の金額は1円以上

    ### テストケース要件
    - 各ルールの境界値テスト
    - ルールの組み合わせテスト
    - 異常系テスト（無効なクーポン、負の金額等）
    - 並行アクセステスト（同一クーポンの二重使用）

    ## 技術仕様
    - pytest + pytest-asyncio
    - Given-When-Then形式のテスト名
    - テストデータはファクトリパターンで生成
    """

    # Step 2: AIが生成したテスト例
    GENERATED_TEST = """
import pytest
from decimal import Decimal
from discount_service import DiscountService, DiscountRequest
from factories import UserFactory, CouponFactory

class TestDiscountService:
    \"\"\"割引計算サービスのテスト\"\"\"

    @pytest.fixture
    def service(self):
        return DiscountService()

    @pytest.fixture
    def gold_user(self):
        return UserFactory.create(membership="gold")

    @pytest.fixture
    def regular_user(self):
        return UserFactory.create(membership="regular")

    # --- 通常割引テスト ---

    def test_given_amount_below_10000_when_calculate_then_no_discount(
        self, service, regular_user
    ):
        request = DiscountRequest(user=regular_user, amount=Decimal("9999"))
        result = service.calculate(request)
        assert result.discount_amount == Decimal("0")
        assert result.final_amount == Decimal("9999")

    def test_given_amount_exactly_10000_when_calculate_then_5_percent_discount(
        self, service, regular_user
    ):
        request = DiscountRequest(user=regular_user, amount=Decimal("10000"))
        result = service.calculate(request)
        assert result.discount_amount == Decimal("500")
        assert result.final_amount == Decimal("9500")

    def test_given_amount_above_10000_when_calculate_then_5_percent_discount(
        self, service, regular_user
    ):
        request = DiscountRequest(user=regular_user, amount=Decimal("20000"))
        result = service.calculate(request)
        assert result.discount_amount == Decimal("1000")
        assert result.final_amount == Decimal("19000")

    # --- 会員割引テスト ---

    def test_given_gold_member_when_calculate_then_additional_10_percent(
        self, service, gold_user
    ):
        request = DiscountRequest(user=gold_user, amount=Decimal("10000"))
        result = service.calculate(request)
        # 5% + 10% = 15%
        assert result.discount_amount == Decimal("1500")
        assert result.final_amount == Decimal("8500")

    # --- 割引上限テスト ---

    def test_given_max_discounts_combined_when_calculate_then_cap_at_30_percent(
        self, service, gold_user
    ):
        coupon = CouponFactory.create(discount_percent=Decimal("20"))
        request = DiscountRequest(
            user=gold_user, amount=Decimal("10000"), coupon=coupon
        )
        result = service.calculate(request)
        # 5% + 10% + 20% = 35% → 上限30%に制限
        assert result.discount_rate <= Decimal("0.30")
        assert result.final_amount >= Decimal("7000")

    # --- 最低支払額テスト ---

    def test_given_huge_discount_when_calculate_then_minimum_1_yen(
        self, service, regular_user
    ):
        coupon = CouponFactory.create(discount_amount=Decimal("99999"))
        request = DiscountRequest(
            user=regular_user, amount=Decimal("100"), coupon=coupon
        )
        result = service.calculate(request)
        assert result.final_amount >= Decimal("1")

    # --- 異常系テスト ---

    def test_given_negative_amount_when_calculate_then_raise_error(
        self, service, regular_user
    ):
        with pytest.raises(ValueError, match="金額は正の値"):
            DiscountRequest(user=regular_user, amount=Decimal("-100"))

    def test_given_expired_coupon_when_calculate_then_raise_error(
        self, service, regular_user
    ):
        expired_coupon = CouponFactory.create(expired=True)
        request = DiscountRequest(
            user=regular_user, amount=Decimal("10000"),
            coupon=expired_coupon
        )
        with pytest.raises(ValueError, match="クーポンは有効期限切れ"):
            service.calculate(request)
    """
```

### 5.2 テストファクトリパターン

```python
# AIテスト生成と相性の良いテストデータファクトリ

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional
import uuid

@dataclass
class User:
    id: str
    name: str
    email: str
    membership: str
    created_at: datetime

@dataclass
class Coupon:
    code: str
    discount_type: str  # "percent" or "fixed"
    discount_value: Decimal
    expires_at: datetime
    max_uses: int
    used_count: int

class UserFactory:
    """テスト用ユーザーデータのファクトリ"""

    _counter = 0

    @classmethod
    def create(cls, **overrides) -> User:
        cls._counter += 1
        defaults = {
            "id": str(uuid.uuid4()),
            "name": f"テストユーザー{cls._counter}",
            "email": f"test{cls._counter}@example.com",
            "membership": "regular",
            "created_at": datetime.now(),
        }
        defaults.update(overrides)
        return User(**defaults)

    @classmethod
    def create_gold_member(cls, **overrides) -> User:
        return cls.create(membership="gold", **overrides)

    @classmethod
    def create_batch(cls, count: int, **overrides) -> list[User]:
        return [cls.create(**overrides) for _ in range(count)]

class CouponFactory:
    """テスト用クーポンデータのファクトリ"""

    @classmethod
    def create(cls, **overrides) -> Coupon:
        defaults = {
            "code": f"TEST-{uuid.uuid4().hex[:8].upper()}",
            "discount_type": "percent",
            "discount_value": Decimal("10"),
            "expires_at": datetime.now() + timedelta(days=30),
            "max_uses": 100,
            "used_count": 0,
        }

        # expired=True のショートカット
        if overrides.pop("expired", False):
            defaults["expires_at"] = datetime.now() - timedelta(days=1)

        # discount_percent のショートカット
        if "discount_percent" in overrides:
            defaults["discount_type"] = "percent"
            defaults["discount_value"] = overrides.pop("discount_percent")

        # discount_amount のショートカット
        if "discount_amount" in overrides:
            defaults["discount_type"] = "fixed"
            defaults["discount_value"] = overrides.pop("discount_amount")

        defaults.update(overrides)
        return Coupon(**defaults)
```

### 5.3 テスト生成プロンプトのベストプラクティス

```python
# 高品質なテストを生成するためのプロンプトテンプレート集

TEST_GENERATION_PROMPTS = {
    "unit_test": """
    以下のコードに対する単体テストを生成してください。

    ## 対象コード
    {source_code}

    ## テスト要件
    1. Given-When-Then形式のテスト名
    2. 各公開メソッドに対して最低3つのテストケース
       - 正常系（典型的な入力）
       - 境界値（最小/最大/空）
       - 異常系（不正入力、例外）
    3. テストの独立性を保証（共有状態なし）
    4. モックは最小限（外部依存のみ）
    5. アサーションは具体的な値を検証（assert is not None 禁止）

    ## 技術スタック
    - pytest
    - unittest.mock
    - freezegun（時刻固定）
    """,

    "integration_test": """
    以下のサービス間の結合テストを生成してください。

    ## 対象サービス
    {service_code}

    ## 依存関係
    {dependencies}

    ## テスト要件
    1. 実際のDB（testcontainers使用）でテスト
    2. 外部APIはモック（responses / httpx-mock）
    3. トランザクションのロールバック確認
    4. 並行アクセスのテスト（asyncio.gather）
    5. エラー伝播の検証（サービスAのエラー→サービスBの挙動）

    ## 技術スタック
    - pytest-asyncio
    - testcontainers-python
    - httpx-mock
    """,

    "snapshot_test": """
    以下のReactコンポーネントに対するスナップショットテストを生成してください。

    ## 対象コンポーネント
    {component_code}

    ## テスト要件
    1. 各propsパターンでのレンダリング結果をスナップショット
    2. インタラクション（クリック、入力）後の状態変化
    3. ローディング、エラー、空状態の各UIステートをテスト
    4. レスポンシブ対応（モバイル/デスクトップ）のスナップショット

    ## 技術スタック
    - Vitest
    - @testing-library/react
    - @testing-library/user-event
    """,
}
```

### 5.4 テストカバレッジ分析と改善提案

```python
# カバレッジレポートをAIで分析して改善提案を生成

import json
from pathlib import Path

class CoverageAnalyzer:
    """テストカバレッジをAIで分析するツール"""

    def __init__(self, coverage_json_path: str):
        self.coverage_data = json.loads(Path(coverage_json_path).read_text())

    def analyze(self) -> dict:
        """カバレッジデータを分析"""
        files = self.coverage_data.get("files", {})

        analysis = {
            "total_coverage": self.coverage_data.get("totals", {}).get(
                "percent_covered", 0
            ),
            "low_coverage_files": [],
            "uncovered_critical_paths": [],
            "improvement_suggestions": [],
        }

        for file_path, file_data in files.items():
            coverage = file_data.get("summary", {}).get("percent_covered", 0)

            if coverage < 70:
                missing_lines = file_data.get("missing_lines", [])
                analysis["low_coverage_files"].append({
                    "path": file_path,
                    "coverage": coverage,
                    "missing_lines": missing_lines[:20],
                    "total_missing": len(missing_lines),
                })

        # 優先順位をつけてソート（カバレッジが低く、ファイルが大きいものを優先）
        analysis["low_coverage_files"].sort(
            key=lambda x: (x["coverage"], -x["total_missing"])
        )

        return analysis

    def generate_improvement_prompt(self, analysis: dict) -> str:
        """カバレッジ改善のためのAIプロンプトを生成"""
        low_files = analysis["low_coverage_files"][:5]

        prompt = f"""
以下のテストカバレッジ分析結果に基づいて、テスト追加の優先順位と
具体的なテストケースを提案してください。

## 全体カバレッジ: {analysis['total_coverage']:.1f}%

## カバレッジが低いファイル（上位5件）
"""
        for f in low_files:
            prompt += f"""
### {f['path']} (カバレッジ: {f['coverage']:.1f}%)
未カバー行数: {f['total_missing']}行
未カバー行の例: {f['missing_lines'][:10]}
"""

        prompt += """
## 回答形式
1. 各ファイルの優先度（High/Medium/Low）と理由
2. 各ファイルに追加すべきテストケースの一覧
3. テストコードのスケルトン
"""
        return prompt
```

---

## 6. CI/CDパイプラインとの統合

### 6.1 テスト自動生成パイプライン

```yaml
# .github/workflows/ai-test-generation.yml
name: AI Test Generation

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  analyze-coverage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Run tests with coverage
        run: pytest --cov=src --cov-report=json:coverage.json

      - name: Analyze coverage gaps
        run: |
          python scripts/analyze_coverage.py \
            --coverage coverage.json \
            --changed-files "$(git diff --name-only origin/main...HEAD)" \
            --output coverage-analysis.json

      - name: Comment coverage analysis on PR
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const analysis = JSON.parse(
              fs.readFileSync('coverage-analysis.json')
            );

            let body = '## テストカバレッジ分析\n\n';
            body += `全体カバレッジ: **${analysis.total_coverage}%**\n\n`;

            if (analysis.uncovered_changes.length > 0) {
              body += '### テストが不足している変更\n\n';
              for (const file of analysis.uncovered_changes) {
                body += `- \`${file.path}\` (${file.coverage}%)\n`;
              }
              body += '\n以下のテストを追加することを推奨します。\n';
            }

            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: body,
            });

  mutation-testing:
    runs-on: ubuntu-latest
    needs: analyze-coverage
    steps:
      - uses: actions/checkout@v4

      - name: Run mutation testing
        run: |
          pip install mutmut
          mutmut run --paths-to-mutate=src/ \
            --tests-dir=tests/ \
            --runner="pytest -x -q" \
            || true  # mutmutは検出されたミュータントがあるとexit 1

      - name: Generate mutation report
        run: mutmut results > mutation-report.txt

      - name: Check mutation score
        run: |
          python scripts/check_mutation_score.py \
            --report mutation-report.txt \
            --min-score 80
```

### 6.2 テスト品質ゲート

```python
# テスト品質の自動チェックゲート

class TestQualityGate:
    """テスト品質の自動チェック"""

    def __init__(self, config: dict = None):
        self.config = config or {
            "min_coverage": 80,
            "min_mutation_score": 75,
            "max_test_duration_sec": 300,
            "min_assertion_density": 1.5,  # テストあたりの最小アサーション数
            "max_test_complexity": 10,     # テスト関数の最大サイクロマティック複雑度
        }

    def check_all(self, metrics: dict) -> dict:
        """全てのゲートをチェック"""
        results = {
            "passed": True,
            "checks": [],
        }

        checks = [
            self._check_coverage(metrics),
            self._check_mutation_score(metrics),
            self._check_test_duration(metrics),
            self._check_assertion_density(metrics),
            self._check_test_independence(metrics),
        ]

        for check in checks:
            results["checks"].append(check)
            if not check["passed"]:
                results["passed"] = False

        return results

    def _check_coverage(self, metrics: dict) -> dict:
        coverage = metrics.get("coverage", 0)
        min_cov = self.config["min_coverage"]
        return {
            "name": "カバレッジ",
            "passed": coverage >= min_cov,
            "value": f"{coverage:.1f}%",
            "threshold": f"{min_cov}%",
            "message": f"カバレッジが基準値（{min_cov}%）を{'達成' if coverage >= min_cov else '未達成'}",
        }

    def _check_mutation_score(self, metrics: dict) -> dict:
        score = metrics.get("mutation_score", 0)
        min_score = self.config["min_mutation_score"]
        return {
            "name": "ミューテーションスコア",
            "passed": score >= min_score,
            "value": f"{score:.1f}%",
            "threshold": f"{min_score}%",
            "message": f"テストの有効性が基準値（{min_score}%）を{'達成' if score >= min_score else '未達成'}",
        }

    def _check_test_duration(self, metrics: dict) -> dict:
        duration = metrics.get("total_duration_sec", 0)
        max_dur = self.config["max_test_duration_sec"]
        return {
            "name": "テスト実行時間",
            "passed": duration <= max_dur,
            "value": f"{duration:.0f}秒",
            "threshold": f"{max_dur}秒以下",
            "message": f"テスト実行時間が基準値（{max_dur}秒）を{'クリア' if duration <= max_dur else '超過'}",
        }

    def _check_assertion_density(self, metrics: dict) -> dict:
        density = metrics.get("assertion_density", 0)
        min_density = self.config["min_assertion_density"]
        return {
            "name": "アサーション密度",
            "passed": density >= min_density,
            "value": f"{density:.1f}",
            "threshold": f"{min_density}以上",
            "message": f"テストあたりのアサーション数が{'十分' if density >= min_density else '不足'}",
        }

    def _check_test_independence(self, metrics: dict) -> dict:
        order_dependent = metrics.get("order_dependent_tests", 0)
        return {
            "name": "テスト独立性",
            "passed": order_dependent == 0,
            "value": f"{order_dependent}件",
            "threshold": "0件",
            "message": f"実行順序に依存するテストが{order_dependent}件{'あります' if order_dependent > 0 else 'ありません'}",
        }
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

### Q4: ミューテーションテストをCI/CDに導入する際のコツは？

実行時間が長くなりがちなため、(1) 変更されたファイルのみを対象にする（全体実行は夜間バッチで）。(2) 高速なミュータントから順に実行し、タイムアウトを設定する。(3) 最初は閾値を低め（60%）に設定し、段階的に引き上げる。(4) 等価ミュータント（テストで検出不可能なミュータント）はホワイトリストに登録して除外する。Python なら mutmut、Java なら Pitest、JavaScript なら Stryker が代表的なツール。

### Q5: テストの保守コストを削減する方法は？

AIテストで特に重要な観点。(1) テストヘルパーとファクトリパターンを活用し、テストデータ作成の重複を排除する。(2) Page Object パターン（E2E）やBuilder パターンで変更に強いテストを設計する。(3) AIにテストのリファクタリングも依頼し、DRY原則を適用する。(4) フレーキー（不安定）テストを定期的に特定・修正する仕組みを構築する。(5) テストコードもプロダクションコードと同じ品質基準でレビューする。

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
