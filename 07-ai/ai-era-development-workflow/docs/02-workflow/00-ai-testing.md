# AIテスト ── テスト生成、カバレッジ向上

> AIを活用してテストコードを効率的に生成し、テストカバレッジを大幅に向上させるための戦略と具体的な手法を体系的に学ぶ。

---

## この章で学ぶこと

1. **AIテスト生成の手法** ── 単体テスト、結合テスト、E2Eテストの自動生成パターンを習得する
2. **カバレッジ向上戦略** ── AIを使ってテストの網羅性を効率的に高めるアプローチを学ぶ
3. **テスト品質の検証** ── AI生成テストが本当に有効かを検証する方法を身につける


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解

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

## 実践演習

### 演習1: 基本的な実装

以下の要件を満たすコードを実装してください。

**要件:**
- 入力データの検証を行うこと
- エラーハンドリングを適切に実装すること
- テストコードも作成すること

```python
# 演習1: 基本実装のテンプレート
class Exercise1:
    """基本的な実装パターンの演習"""

    def __init__(self):
        self.data = []

    def validate_input(self, value):
        """入力値の検証"""
        if value is None:
            raise ValueError("入力値がNoneです")
        return True

    def process(self, value):
        """データ処理のメインロジック"""
        self.validate_input(value)
        self.data.append(value)
        return self.data

    def get_results(self):
        """処理結果の取得"""
        return {
            'count': len(self.data),
            'data': self.data
        }

# テスト
def test_exercise1():
    ex = Exercise1()
    assert ex.process(1) == [1]
    assert ex.process(2) == [1, 2]
    assert ex.get_results()['count'] == 2

    try:
        ex.process(None)
        assert False, "例外が発生するべき"
    except ValueError:
        pass

    print("全テスト合格!")

test_exercise1()
```

### 演習2: 応用パターン

基本実装を拡張して、以下の機能を追加してください。

```python
# 演習2: 応用パターン
from typing import List, Dict, Optional
from datetime import datetime

class AdvancedExercise:
    """応用パターンの演習"""

    def __init__(self, max_size: int = 100):
        self._items: List[Dict] = []
        self._max_size = max_size
        self._created_at = datetime.now()

    def add(self, key: str, value: any) -> bool:
        """アイテムの追加（サイズ制限付き）"""
        if len(self._items) >= self._max_size:
            return False
        self._items.append({
            'key': key,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
        return True

    def find(self, key: str) -> Optional[Dict]:
        """キーによる検索"""
        for item in reversed(self._items):
            if item['key'] == key:
                return item
        return None

    def remove(self, key: str) -> bool:
        """キーによる削除"""
        for i, item in enumerate(self._items):
            if item['key'] == key:
                self._items.pop(i)
                return True
        return False

    def stats(self) -> Dict:
        """統計情報"""
        return {
            'total_items': len(self._items),
            'max_size': self._max_size,
            'usage_percent': len(self._items) / self._max_size * 100,
            'uptime': str(datetime.now() - self._created_at)
        }

# テスト
def test_advanced():
    ex = AdvancedExercise(max_size=3)
    assert ex.add("a", 1) == True
    assert ex.add("b", 2) == True
    assert ex.add("c", 3) == True
    assert ex.add("d", 4) == False  # サイズ制限
    assert ex.find("b")['value'] == 2
    assert ex.remove("b") == True
    assert ex.find("b") is None
    stats = ex.stats()
    assert stats['total_items'] == 2
    print("応用テスト全合格!")

test_advanced()
```

### 演習3: パフォーマンス最適化

以下のコードのパフォーマンスを改善してください。

```python
# 演習3: パフォーマンス最適化
import time
from functools import lru_cache

# 最適化前（O(n^2)）
def slow_search(data: list, target: int) -> int:
    """非効率な検索"""
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i] + data[j] == target:
                return (i, j)
    return (-1, -1)

# 最適化後（O(n)）
def fast_search(data: list, target: int) -> tuple:
    """ハッシュマップを使った効率的な検索"""
    seen = {}
    for i, num in enumerate(data):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
    return (-1, -1)

# ベンチマーク
def benchmark():
    import random
    data = list(range(5000))
    random.shuffle(data)
    target = data[100] + data[4000]

    start = time.time()
    result1 = slow_search(data, target)
    slow_time = time.time() - start

    start = time.time()
    result2 = fast_search(data, target)
    fast_time = time.time() - start

    print(f"非効率版: {slow_time:.4f}秒")
    print(f"効率版:   {fast_time:.6f}秒")
    print(f"高速化率: {slow_time/fast_time:.0f}倍")

benchmark()
```

**ポイント:**
- アルゴリズムの計算量を意識する
- 適切なデータ構造を選択する
- ベンチマークで効果を測定する

---

## トラブルシューティング

### よくあるエラーと解決策

| エラー | 原因 | 解決策 |
|--------|------|--------|
| 初期化エラー | 設定ファイルの不備 | 設定ファイルのパスと形式を確認 |
| タイムアウト | ネットワーク遅延/リソース不足 | タイムアウト値の調整、リトライ処理の追加 |
| メモリ不足 | データ量の増大 | バッチ処理の導入、ページネーションの実装 |
| 権限エラー | アクセス権限の不足 | 実行ユーザーの権限確認、設定の見直し |
| データ不整合 | 並行処理の競合 | ロック機構の導入、トランザクション管理 |

### デバッグの手順

1. **エラーメッセージの確認**: スタックトレースを読み、発生箇所を特定する
2. **再現手順の確立**: 最小限のコードでエラーを再現する
3. **仮説の立案**: 考えられる原因をリストアップする
4. **段階的な検証**: ログ出力やデバッガを使って仮説を検証する
5. **修正と回帰テスト**: 修正後、関連する箇所のテストも実行する

```python
# デバッグ用ユーティリティ
import logging
import traceback
from functools import wraps

# ロガーの設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def debug_decorator(func):
    """関数の入出力をログ出力するデコレータ"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"呼び出し: {func.__name__}(args={args}, kwargs={kwargs})")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"戻り値: {func.__name__} -> {result}")
            return result
        except Exception as e:
            logger.error(f"例外発生: {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

@debug_decorator
def process_data(items):
    """データ処理（デバッグ対象）"""
    if not items:
        raise ValueError("空のデータ")
    return [item * 2 for item in items]
```

### パフォーマンス問題の診断

パフォーマンス問題が発生した場合の診断手順:

1. **ボトルネックの特定**: プロファイリングツールで計測
2. **メモリ使用量の確認**: メモリリークの有無をチェック
3. **I/O待ちの確認**: ディスクやネットワークI/Oの状況を確認
4. **同時接続数の確認**: コネクションプールの状態を確認

| 問題の種類 | 診断ツール | 対策 |
|-----------|-----------|------|
| CPU負荷 | cProfile, py-spy | アルゴリズム改善、並列化 |
| メモリリーク | tracemalloc, objgraph | 参照の適切な解放 |
| I/Oボトルネック | strace, iostat | 非同期I/O、キャッシュ |
| DB遅延 | EXPLAIN, slow query log | インデックス、クエリ最適化 |

---

## 設計判断ガイド

### 選択基準マトリクス

技術選択を行う際の判断基準を以下にまとめます。

| 判断基準 | 重視する場合 | 妥協できる場合 |
|---------|------------|-------------|
| パフォーマンス | リアルタイム処理、大規模データ | 管理画面、バッチ処理 |
| 保守性 | 長期運用、チーム開発 | プロトタイプ、短期プロジェクト |
| スケーラビリティ | 成長が見込まれるサービス | 社内ツール、固定ユーザー |
| セキュリティ | 個人情報、金融データ | 公開データ、社内利用 |
| 開発速度 | MVP、市場投入スピード | 品質重視、ミッションクリティカル |

### アーキテクチャパターンの選択

```
┌─────────────────────────────────────────────────┐
│              アーキテクチャ選択フロー              │
├─────────────────────────────────────────────────┤
│                                                 │
│  ① チーム規模は？                                │
│    ├─ 小規模（1-5人）→ モノリス                   │
│    └─ 大規模（10人+）→ ②へ                       │
│                                                 │
│  ② デプロイ頻度は？                               │
│    ├─ 週1回以下 → モノリス + モジュール分割         │
│    └─ 毎日/複数回 → ③へ                          │
│                                                 │
│  ③ チーム間の独立性は？                            │
│    ├─ 高い → マイクロサービス                      │
│    └─ 中程度 → モジュラーモノリス                   │
│                                                 │
└─────────────────────────────────────────────────┘
```

### トレードオフの分析

技術的な判断には必ずトレードオフが伴います。以下の観点で分析を行いましょう:

**1. 短期 vs 長期のコスト**
- 短期的に速い方法が長期的には技術的負債になることがある
- 逆に、過剰な設計は短期的なコストが高く、プロジェクトの遅延を招く

**2. 一貫性 vs 柔軟性**
- 統一された技術スタックは学習コストが低い
- 多様な技術の採用は適材適所が可能だが、運用コストが増加

**3. 抽象化のレベル**
- 高い抽象化は再利用性が高いが、デバッグが困難になる場合がある
- 低い抽象化は直感的だが、コードの重複が発生しやすい

```python
# 設計判断の記録テンプレート
class ArchitectureDecisionRecord:
    """ADR (Architecture Decision Record) の作成"""

    def __init__(self, title: str):
        self.title = title
        self.context = ""
        self.decision = ""
        self.consequences = []
        self.alternatives = []

    def set_context(self, context: str):
        """背景と課題の記述"""
        self.context = context
        return self

    def set_decision(self, decision: str):
        """決定内容の記述"""
        self.decision = decision
        return self

    def add_consequence(self, consequence: str, positive: bool = True):
        """結果の追加"""
        self.consequences.append({
            'description': consequence,
            'type': 'positive' if positive else 'negative'
        })
        return self

    def add_alternative(self, name: str, reason_rejected: str):
        """却下した代替案の追加"""
        self.alternatives.append({
            'name': name,
            'reason_rejected': reason_rejected
        })
        return self

    def to_markdown(self) -> str:
        """Markdown形式で出力"""
        md = f"# ADR: {self.title}\n\n"
        md += f"## 背景\n{self.context}\n\n"
        md += f"## 決定\n{self.decision}\n\n"
        md += "## 結果\n"
        for c in self.consequences:
            icon = "✅" if c['type'] == 'positive' else "⚠️"
            md += f"- {icon} {c['description']}\n"
        md += "\n## 却下した代替案\n"
        for a in self.alternatives:
            md += f"- **{a['name']}**: {a['reason_rejected']}\n"
        return md
```

---

## 実務での適用シナリオ

### シナリオ1: スタートアップでのMVP開発

**状況:** 限られたリソースで素早くプロダクトをリリースする必要がある

**アプローチ:**
- シンプルなアーキテクチャを選択
- 必要最小限の機能に集中
- 自動テストはクリティカルパスのみ
- モニタリングは早期から導入

**学んだ教訓:**
- 完璧を求めすぎない（YAGNI原則）
- ユーザーフィードバックを早期に取得
- 技術的負債は意識的に管理する

### シナリオ2: レガシーシステムのモダナイゼーション

**状況:** 10年以上運用されているシステムを段階的に刷新する

**アプローチ:**
- Strangler Fig パターンで段階的に移行
- 既存のテストがない場合はCharacterization Testを先に作成
- APIゲートウェイで新旧システムを共存
- データ移行は段階的に実施

| フェーズ | 作業内容 | 期間目安 | リスク |
|---------|---------|---------|--------|
| 1. 調査 | 現状分析、依存関係の把握 | 2-4週間 | 低 |
| 2. 基盤 | CI/CD構築、テスト環境 | 4-6週間 | 低 |
| 3. 移行開始 | 周辺機能から順次移行 | 3-6ヶ月 | 中 |
| 4. コア移行 | 中核機能の移行 | 6-12ヶ月 | 高 |
| 5. 完了 | 旧システム廃止 | 2-4週間 | 中 |

### シナリオ3: 大規模チームでの開発

**状況:** 50人以上のエンジニアが同一プロダクトを開発する

**アプローチ:**
- ドメイン駆動設計で境界を明確化
- チームごとにオーナーシップを設定
- 共通ライブラリはInner Source方式で管理
- APIファーストで設計し、チーム間の依存を最小化

```python
# チーム間のAPI契約定義
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class APIContract:
    """チーム間のAPI契約"""
    endpoint: str
    method: str
    owner_team: str
    consumers: List[str]
    sla_ms: int  # レスポンスタイムSLA
    priority: Priority

    def validate_sla(self, actual_ms: int) -> bool:
        """SLA準拠の確認"""
        return actual_ms <= self.sla_ms

    def to_openapi(self) -> dict:
        """OpenAPI形式で出力"""
        return {
            'path': self.endpoint,
            'method': self.method,
            'x-owner': self.owner_team,
            'x-consumers': self.consumers,
            'x-sla-ms': self.sla_ms
        }

# 使用例
contracts = [
    APIContract(
        endpoint="/api/v1/users",
        method="GET",
        owner_team="user-team",
        consumers=["order-team", "notification-team"],
        sla_ms=200,
        priority=Priority.HIGH
    ),
    APIContract(
        endpoint="/api/v1/orders",
        method="POST",
        owner_team="order-team",
        consumers=["payment-team", "inventory-team"],
        sla_ms=500,
        priority=Priority.CRITICAL
    )
]
```

### シナリオ4: パフォーマンスクリティカルなシステム

**状況:** ミリ秒単位のレスポンスが求められるシステム

**最適化ポイント:**
1. キャッシュ戦略（L1: インメモリ、L2: Redis、L3: CDN）
2. 非同期処理の活用
3. コネクションプーリング
4. クエリ最適化とインデックス設計

| 最適化手法 | 効果 | 実装コスト | 適用場面 |
|-----------|------|-----------|---------|
| インメモリキャッシュ | 高 | 低 | 頻繁にアクセスされるデータ |
| CDN | 高 | 低 | 静的コンテンツ |
| 非同期処理 | 中 | 中 | I/O待ちが多い処理 |
| DB最適化 | 高 | 高 | クエリが遅い場合 |
| コード最適化 | 低-中 | 高 | CPU律速の場合 |
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
