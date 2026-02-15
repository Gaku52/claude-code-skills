# AIコーディングのベストプラクティス ── レビュー、検証、品質保証

> AIが生成したコードの品質を保証するための体系的なレビュー手法と検証プロセスを学び、AI支援コーディングにおける信頼性と保守性を確保する。

---

## この章で学ぶこと

1. **AIコード品質の評価フレームワーク** ── AI出力を体系的にレビューする基準と手順を確立する
2. **検証プロセスの設計** ── AI生成コードを安全にプロダクションに投入するためのゲートを構築する
3. **継続的な品質改善** ── AI活用の品質を組織的に底上げするフィードバックループを構築する
4. **言語別ベストプラクティス** ── Python、TypeScript、Goなど言語ごとのAIコード品質パターンを習得する
5. **プロンプト設計と品質の関係** ── 高品質コードを生成するためのプロンプトエンジニアリングを理解する

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

### 1.3 各層のレビュー観点詳細

各層で確認すべき具体的な項目を詳細に整理する。

```python
# 5層モデルの各層を定義するレビューチェックシステム
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

class ReviewLayer(Enum):
    SYNTAX_STYLE = 1
    DESIGN_QUALITY = 2
    PERFORMANCE = 3
    SECURITY = 4
    BUSINESS_LOGIC = 5

class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ReviewItem:
    """レビュー項目"""
    layer: ReviewLayer
    category: str
    description: str
    severity: Severity
    automated: bool  # 自動化可能か
    checker: Callable | None = None  # 自動チェック関数

@dataclass
class ReviewChecklist:
    """AI生成コード レビューチェックリスト"""
    items: list[ReviewItem] = field(default_factory=list)

    def add_layer1_items(self) -> None:
        """Layer 1: 構文・スタイル検証"""
        checks = [
            ("フォーマット", "コードフォーマッタ（black/prettier）に準拠しているか", True),
            ("リンティング", "リンターの警告・エラーがないか", True),
            ("型注釈", "型注釈が適切に付与されているか", True),
            ("import整理", "未使用importがなく、順序が正しいか", True),
            ("命名規則", "プロジェクトの命名規則に従っているか", True),
            ("コメント", "不要なコメントや説明不足がないか", False),
        ]
        for category, desc, automated in checks:
            self.items.append(ReviewItem(
                layer=ReviewLayer.SYNTAX_STYLE,
                category=category,
                description=desc,
                severity=Severity.WARNING,
                automated=automated,
            ))

    def add_layer2_items(self) -> None:
        """Layer 2: 設計品質検証"""
        checks = [
            ("単一責任", "各関数・クラスが1つの責任のみを持っているか", Severity.ERROR),
            ("DRY原則", "コードの重複がないか", Severity.WARNING),
            ("凝集度", "関連する機能が適切にグループ化されているか", Severity.WARNING),
            ("結合度", "モジュール間の依存が最小限か", Severity.WARNING),
            ("テスト容易性", "モック不要で単体テスト可能な設計か", Severity.ERROR),
            ("拡張性", "将来の変更に対して開放的な設計か", Severity.INFO),
            ("エラーハンドリング", "例外処理が適切で一貫しているか", Severity.ERROR),
            ("抽象化レベル", "関数内の抽象化レベルが統一されているか", Severity.WARNING),
        ]
        for category, desc, severity in checks:
            self.items.append(ReviewItem(
                layer=ReviewLayer.DESIGN_QUALITY,
                category=category,
                description=desc,
                severity=severity,
                automated=False,
            ))

    def add_layer3_items(self) -> None:
        """Layer 3: パフォーマンス検証"""
        checks = [
            ("計算量", "O(n²)以上のアルゴリズムが不必要に使われていないか", Severity.ERROR),
            ("N+1問題", "ループ内でDB/APIクエリが発行されていないか", Severity.CRITICAL),
            ("メモリ使用", "大量データをメモリに展開していないか", Severity.ERROR),
            ("キャッシュ", "キャッシュが活用されているか", Severity.WARNING),
            ("並行処理", "非同期処理が適切に使われているか", Severity.WARNING),
            ("インデックス", "DBクエリに適切なインデックスが考慮されているか", Severity.ERROR),
        ]
        for category, desc, severity in checks:
            self.items.append(ReviewItem(
                layer=ReviewLayer.PERFORMANCE,
                category=category,
                description=desc,
                severity=severity,
                automated=False,
            ))

    def add_layer4_items(self) -> None:
        """Layer 4: セキュリティ検証"""
        checks = [
            ("入力検証", "全ての外部入力がバリデーションされているか", Severity.CRITICAL),
            ("SQLインジェクション", "パラメータ化クエリが使われているか", Severity.CRITICAL),
            ("XSS", "出力がエスケープされているか", Severity.CRITICAL),
            ("認証・認可", "適切なアクセス制御が実装されているか", Severity.CRITICAL),
            ("秘密情報", "ハードコードされた秘密情報がないか", Severity.CRITICAL),
            ("CSRF対策", "状態変更リクエストにCSRFトークンがあるか", Severity.ERROR),
            ("レート制限", "API エンドポイントにレート制限があるか", Severity.WARNING),
        ]
        for category, desc, severity in checks:
            self.items.append(ReviewItem(
                layer=ReviewLayer.SECURITY,
                category=category,
                description=desc,
                severity=severity,
                automated=category in ("SQLインジェクション", "秘密情報"),
            ))

    def add_layer5_items(self) -> None:
        """Layer 5: ビジネスロジック検証"""
        checks = [
            ("要件整合性", "仕様書・ユーザーストーリーの要件を満たしているか", Severity.CRITICAL),
            ("エッジケース", "境界値やnull/空のケースが処理されているか", Severity.ERROR),
            ("ドメインルール", "業務ルール（消費税計算、在庫管理等）が正確か", Severity.CRITICAL),
            ("データ整合性", "トランザクション境界が適切か", Severity.CRITICAL),
            ("冪等性", "リトライ時に副作用が重複しないか", Severity.ERROR),
            ("監査証跡", "重要な操作のログが記録されているか", Severity.WARNING),
        ]
        for category, desc, severity in checks:
            self.items.append(ReviewItem(
                layer=ReviewLayer.BUSINESS_LOGIC,
                category=category,
                description=desc,
                severity=severity,
                automated=False,
            ))

    def generate_report(self) -> dict:
        """レビューレポートを生成"""
        report = {
            "total_items": len(self.items),
            "automated_count": sum(1 for item in self.items if item.automated),
            "manual_count": sum(1 for item in self.items if not item.automated),
            "by_layer": {},
            "by_severity": {},
        }
        for layer in ReviewLayer:
            layer_items = [i for i in self.items if i.layer == layer]
            report["by_layer"][layer.name] = {
                "count": len(layer_items),
                "automated": sum(1 for i in layer_items if i.automated),
            }
        for severity in Severity:
            report["by_severity"][severity.value] = sum(
                1 for i in self.items if i.severity == severity
            )
        return report

# 使用例
checklist = ReviewChecklist()
checklist.add_layer1_items()
checklist.add_layer2_items()
checklist.add_layer3_items()
checklist.add_layer4_items()
checklist.add_layer5_items()
report = checklist.generate_report()
# → automated_count: ~10, manual_count: ~25
# → Layer 1-2はほぼ自動化可能、Layer 3-5は人間の判断が中心
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
| Level 3: 生成+自動テスト | 80-90% | 50%減少 | 70%短縮 |
| Level 4: エージェント+CI | 90%以上 | 70%減少 | 70%短縮 |

### 3.1 パターン別の適用ガイド

```python
# 各パターンの選択基準を判定するヘルパー

from dataclasses import dataclass
from enum import Enum

class GenerationPattern(Enum):
    TEST_FIRST = "test_first"
    INCREMENTAL = "incremental"
    CONTEXT_INJECTION = "context_injection"
    ONE_SHOT = "one_shot"

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class TaskCharacteristics:
    """タスクの特性を表現"""
    complexity: int  # 1-10
    security_sensitivity: bool  # セキュリティ関連か
    has_existing_tests: bool  # 既存テストがあるか
    has_existing_code: bool  # 既存コードがあるか
    domain_complexity: int  # ドメイン複雑度 1-10
    team_familiarity: int  # チームの技術習熟度 1-10

def recommend_pattern(task: TaskCharacteristics) -> GenerationPattern:
    """タスク特性に基づいてAI生成パターンを推薦する"""

    # セキュリティ関連 or ドメイン複雑度が高い → テストファースト必須
    if task.security_sensitivity or task.domain_complexity >= 7:
        return GenerationPattern.TEST_FIRST

    # 複雑度が高い → 段階的生成
    if task.complexity >= 7:
        return GenerationPattern.INCREMENTAL

    # 既存コードがあり、規約に沿わせたい → コンテキスト注入
    if task.has_existing_code and task.team_familiarity >= 5:
        return GenerationPattern.CONTEXT_INJECTION

    # 単純なタスク → ワンショットでも可
    if task.complexity <= 3 and not task.security_sensitivity:
        return GenerationPattern.ONE_SHOT

    # デフォルト: 段階的生成
    return GenerationPattern.INCREMENTAL

def estimate_risk(task: TaskCharacteristics) -> RiskLevel:
    """AI生成コードのリスクレベルを推定"""
    score = 0
    score += task.complexity * 2
    score += task.domain_complexity * 2
    if task.security_sensitivity:
        score += 20
    if not task.has_existing_tests:
        score += 10
    if task.team_familiarity < 5:
        score += 5

    if score >= 40:
        return RiskLevel.CRITICAL
    elif score >= 25:
        return RiskLevel.HIGH
    elif score >= 15:
        return RiskLevel.MEDIUM
    else:
        return RiskLevel.LOW

# 使用例
task = TaskCharacteristics(
    complexity=6,
    security_sensitivity=True,
    has_existing_tests=True,
    has_existing_code=True,
    domain_complexity=8,
    team_familiarity=7,
)
pattern = recommend_pattern(task)
risk = estimate_risk(task)
# pattern = TEST_FIRST, risk = CRITICAL
# → テストを先に書いてからAIに実装させ、全層レビューを実施
```

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

### 4.1 自動検証パイプラインの実装

```python
# AI生成コード検証パイプラインの実装例

import subprocess
import json
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

class GateResult(Enum):
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"

@dataclass
class GateOutput:
    """ゲートの実行結果"""
    gate_name: str
    result: GateResult
    details: str
    duration_ms: float
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

class AICodeValidator:
    """AI生成コードを段階的に検証するバリデーター"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.gates: list[GateOutput] = []

    def run_gate1_lint(self, files: list[Path]) -> GateOutput:
        """Gate 1: リント + フォーマットチェック"""
        import time
        start = time.monotonic()
        errors = []
        warnings = []

        for file_path in files:
            if file_path.suffix == ".py":
                # Ruff（高速Pythonリンター）でチェック
                result = subprocess.run(
                    ["ruff", "check", str(file_path), "--output-format=json"],
                    capture_output=True, text=True
                )
                if result.returncode != 0:
                    lint_errors = json.loads(result.stdout)
                    for err in lint_errors:
                        msg = f"{err['filename']}:{err['location']['row']}: {err['code']} {err['message']}"
                        if err["code"].startswith("E"):
                            errors.append(msg)
                        else:
                            warnings.append(msg)

                # フォーマットチェック
                fmt_result = subprocess.run(
                    ["ruff", "format", "--check", str(file_path)],
                    capture_output=True, text=True
                )
                if fmt_result.returncode != 0:
                    errors.append(f"{file_path}: フォーマットが不正")

            elif file_path.suffix in (".ts", ".tsx"):
                # ESLint でチェック
                result = subprocess.run(
                    ["npx", "eslint", str(file_path), "--format=json"],
                    capture_output=True, text=True,
                    cwd=str(self.project_root)
                )
                if result.returncode != 0:
                    lint_data = json.loads(result.stdout)
                    for file_result in lint_data:
                        for msg in file_result.get("messages", []):
                            line = f"{file_result['filePath']}:{msg['line']}: {msg['message']}"
                            if msg["severity"] == 2:
                                errors.append(line)
                            else:
                                warnings.append(line)

        elapsed = (time.monotonic() - start) * 1000
        result_status = GateResult.FAIL if errors else (
            GateResult.WARN if warnings else GateResult.PASS
        )

        output = GateOutput(
            gate_name="Gate 1: Lint + Format",
            result=result_status,
            details=f"files={len(files)}, errors={len(errors)}, warnings={len(warnings)}",
            duration_ms=elapsed,
            errors=errors,
            warnings=warnings,
        )
        self.gates.append(output)
        return output

    def run_gate2_typecheck(self) -> GateOutput:
        """Gate 2: 型チェック"""
        import time
        start = time.monotonic()
        errors = []

        # Python: mypy
        mypy_result = subprocess.run(
            ["mypy", ".", "--strict", "--no-error-summary"],
            capture_output=True, text=True,
            cwd=str(self.project_root)
        )
        if mypy_result.returncode != 0:
            for line in mypy_result.stdout.strip().split("\n"):
                if line.strip():
                    errors.append(f"[mypy] {line}")

        # TypeScript: tsc
        tsc_result = subprocess.run(
            ["npx", "tsc", "--noEmit"],
            capture_output=True, text=True,
            cwd=str(self.project_root)
        )
        if tsc_result.returncode != 0:
            for line in tsc_result.stdout.strip().split("\n"):
                if line.strip():
                    errors.append(f"[tsc] {line}")

        elapsed = (time.monotonic() - start) * 1000
        output = GateOutput(
            gate_name="Gate 2: Type Check",
            result=GateResult.FAIL if errors else GateResult.PASS,
            details=f"errors={len(errors)}",
            duration_ms=elapsed,
            errors=errors,
        )
        self.gates.append(output)
        return output

    def run_gate3_tests(self, coverage_threshold: float = 80.0) -> GateOutput:
        """Gate 3: テスト実行 + カバレッジ"""
        import time
        start = time.monotonic()
        errors = []
        warnings = []

        # pytest実行
        test_result = subprocess.run(
            ["pytest", "--tb=short", "--cov=app", "--cov-report=json", "-q"],
            capture_output=True, text=True,
            cwd=str(self.project_root)
        )

        if test_result.returncode != 0:
            errors.append(f"テスト失敗:\n{test_result.stdout}")

        # カバレッジ確認
        coverage_file = self.project_root / "coverage.json"
        if coverage_file.exists():
            with open(coverage_file) as f:
                cov_data = json.load(f)
            total_coverage = cov_data.get("totals", {}).get("percent_covered", 0)
            if total_coverage < coverage_threshold:
                warnings.append(
                    f"カバレッジ {total_coverage:.1f}% < 閾値 {coverage_threshold}%"
                )

        elapsed = (time.monotonic() - start) * 1000
        output = GateOutput(
            gate_name="Gate 3: Tests + Coverage",
            result=GateResult.FAIL if errors else (
                GateResult.WARN if warnings else GateResult.PASS
            ),
            details=f"errors={len(errors)}, warnings={len(warnings)}",
            duration_ms=elapsed,
            errors=errors,
            warnings=warnings,
        )
        self.gates.append(output)
        return output

    def run_gate4_security(self) -> GateOutput:
        """Gate 4: セキュリティ検査"""
        import time
        start = time.monotonic()
        errors = []
        warnings = []

        # Bandit (Python SAST)
        bandit_result = subprocess.run(
            ["bandit", "-r", "app", "-f", "json", "-ll"],
            capture_output=True, text=True,
            cwd=str(self.project_root)
        )
        if bandit_result.stdout:
            bandit_data = json.loads(bandit_result.stdout)
            for issue in bandit_data.get("results", []):
                severity = issue["issue_severity"]
                msg = f"[Bandit {severity}] {issue['filename']}:{issue['line_number']}: {issue['issue_text']}"
                if severity in ("HIGH", "MEDIUM"):
                    errors.append(msg)
                else:
                    warnings.append(msg)

        # 秘密情報スキャン（gitleaks）
        gitleaks_result = subprocess.run(
            ["gitleaks", "detect", "--no-git", "-s", ".", "-r", "/tmp/gitleaks.json"],
            capture_output=True, text=True,
            cwd=str(self.project_root)
        )
        if gitleaks_result.returncode != 0:
            errors.append("秘密情報がコード内に検出されました")

        elapsed = (time.monotonic() - start) * 1000
        output = GateOutput(
            gate_name="Gate 4: Security Scan",
            result=GateResult.FAIL if errors else (
                GateResult.WARN if warnings else GateResult.PASS
            ),
            details=f"errors={len(errors)}, warnings={len(warnings)}",
            duration_ms=elapsed,
            errors=errors,
            warnings=warnings,
        )
        self.gates.append(output)
        return output

    def generate_summary(self) -> str:
        """全ゲートのサマリーを生成"""
        lines = ["=== AI生成コード検証サマリー ===\n"]
        all_passed = True

        for gate in self.gates:
            icon = {"pass": "✅", "fail": "❌", "warn": "⚠️"}[gate.result.value]
            lines.append(f"{icon} {gate.gate_name}: {gate.result.value} ({gate.duration_ms:.0f}ms)")
            if gate.errors:
                all_passed = False
                for err in gate.errors[:3]:  # 最初の3つだけ表示
                    lines.append(f"   - {err}")
                if len(gate.errors) > 3:
                    lines.append(f"   ... 他 {len(gate.errors) - 3} 件")

        lines.append("")
        if all_passed:
            lines.append("結果: 全ゲート通過 → 人間レビューに進めます")
        else:
            lines.append("結果: ゲート未通過 → AIに修正を依頼してください")

        return "\n".join(lines)
```

---

## 5. 言語別AIコーディングベストプラクティス

### 5.1 Python固有の品質チェック

```python
# Python AI生成コードで頻出する問題と修正パターン

# === 問題パターン1: ミュータブルデフォルト引数 ===
# AI生成コードでよく発生する

# BAD: AIが生成しがちなパターン
def add_item(item: str, items: list[str] = []) -> list[str]:
    items.append(item)  # ⚠️ デフォルト引数はミュータブル → 呼び出し間で共有される
    return items

# GOOD: 修正版
def add_item(item: str, items: list[str] | None = None) -> list[str]:
    if items is None:
        items = []
    items.append(item)
    return items


# === 問題パターン2: 例外のベアキャッチ ===
# BAD: AIが安全のために広い例外をキャッチしがち
def parse_config(config_str: str) -> dict:
    try:
        return json.loads(config_str)
    except Exception:  # ⚠️ 全例外キャッチ → バグを隠す
        return {}

# GOOD: 具体的な例外をキャッチ
def parse_config(config_str: str) -> dict:
    try:
        return json.loads(config_str)
    except json.JSONDecodeError as e:
        logger.warning("Invalid JSON config", error=str(e), input_length=len(config_str))
        raise ConfigParseError(f"設定の解析に失敗: {e}") from e


# === 問題パターン3: 同期/非同期の混在 ===
# BAD: AIが非同期コンテキストで同期的にI/Oを呼ぶ
async def get_user_data(user_id: int) -> dict:
    # ⚠️ requestsは同期 → イベントループをブロック
    response = requests.get(f"https://api.example.com/users/{user_id}")
    return response.json()

# GOOD: 非同期HTTPクライアントを使用
async def get_user_data(user_id: int) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.example.com/users/{user_id}")
        response.raise_for_status()
        return response.json()


# === 問題パターン4: コンテキストマネージャの不使用 ===
# BAD: AIがリソース解放を忘れる
def read_file(path: str) -> str:
    f = open(path, "r")  # ⚠️ 例外時にファイルが閉じられない
    content = f.read()
    f.close()
    return content

# GOOD: コンテキストマネージャを使用
def read_file(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# === 問題パターン5: 型ヒントの不足・不正確 ===
# BAD: AIが型ヒントを省略したり不正確にする
def process_data(data):  # ⚠️ 型ヒントなし
    result = {}
    for item in data:
        result[item['id']] = item['value'] * 2
    return result

# GOOD: 正確な型ヒントと入力検証
from typing import TypedDict

class DataItem(TypedDict):
    id: str
    value: float

def process_data(data: list[DataItem]) -> dict[str, float]:
    """データを処理してIDをキーとする辞書を返す"""
    return {item["id"]: item["value"] * 2 for item in data}
```

### 5.2 TypeScript/React固有の品質チェック

```typescript
// TypeScript AI生成コードで頻出する問題と修正パターン

// === 問題パターン1: any型の乱用 ===
// BAD: AIが型定義を面倒がってanyを使う
function processResponse(data: any): any {
  return data.results.map((item: any) => ({
    id: item.id,
    name: item.name,
  }));
}

// GOOD: 適切な型定義
interface ApiResponse {
  results: ApiItem[];
  total: number;
  page: number;
}

interface ApiItem {
  id: string;
  name: string;
  createdAt: string;
}

interface ProcessedItem {
  id: string;
  name: string;
}

function processResponse(data: ApiResponse): ProcessedItem[] {
  return data.results.map((item) => ({
    id: item.id,
    name: item.name,
  }));
}


// === 問題パターン2: useEffectの依存配列ミス ===
// BAD: AIが依存配列を不完全にする
function UserProfile({ userId }: { userId: string }) {
  const [user, setUser] = useState<User | null>(null);

  useEffect(() => {
    fetchUser(userId).then(setUser);
  }, []); // ⚠️ userIdが依存配列に含まれていない

  return <div>{user?.name}</div>;
}

// GOOD: 依存配列を正確に指定
function UserProfile({ userId }: { userId: string }) {
  const [user, setUser] = useState<User | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    fetchUser(userId)
      .then((data) => {
        if (!cancelled) setUser(data);
      })
      .catch((err) => {
        if (!cancelled) setError(err.message);
      });

    return () => { cancelled = true; };  // クリーンアップ
  }, [userId]);

  if (error) return <ErrorMessage message={error} />;
  if (!user) return <Skeleton />;
  return <div>{user.name}</div>;
}


// === 問題パターン3: メモ化の不適切な使用 ===
// BAD: AIが全てをmemoしがち
const UserList = React.memo(({ users }: { users: User[] }) => {
  // ⚠️ useMemoを使っているがdepsが毎回新しい配列を参照
  const sortedUsers = useMemo(
    () => users.sort((a, b) => a.name.localeCompare(b.name)),
    [users]  // usersの参照が変わるたびに再計算
  );

  // ⚠️ useCallbackの依存配列が空 → staleクロージャ
  const handleClick = useCallback((id: string) => {
    const user = users.find(u => u.id === id);  // 古いusersを参照
    console.log(user);
  }, []);

  return (
    <ul>
      {sortedUsers.map(user => (
        <li key={user.id} onClick={() => handleClick(user.id)}>
          {user.name}
        </li>
      ))}
    </ul>
  );
});

// GOOD: 適切なメモ化
function UserList({ users }: { users: User[] }) {
  // sort()は破壊的 → toSorted()を使用
  const sortedUsers = useMemo(
    () => users.toSorted((a, b) => a.name.localeCompare(b.name)),
    [users]
  );

  const handleClick = useCallback((id: string) => {
    const user = users.find(u => u.id === id);
    if (user) {
      console.log(user);
    }
  }, [users]);  // usersを依存配列に含める

  return (
    <ul>
      {sortedUsers.map(user => (
        <li key={user.id} onClick={() => handleClick(user.id)}>
          {user.name}
        </li>
      ))}
    </ul>
  );
}


// === 問題パターン4: エラーハンドリングの不足 ===
// BAD: AIがハッピーパスだけ実装
async function fetchData<T>(url: string): Promise<T> {
  const response = await fetch(url);
  return response.json();  // ⚠️ レスポンスステータスチェックなし
}

// GOOD: 堅牢なエラーハンドリング
class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
    public readonly body: unknown,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function fetchData<T>(
  url: string,
  options?: RequestInit,
): Promise<T> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 10_000);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });

    if (!response.ok) {
      const body = await response.text().catch(() => "");
      throw new ApiError(
        `API request failed: ${response.status} ${response.statusText}`,
        response.status,
        body,
      );
    }

    return (await response.json()) as T;
  } finally {
    clearTimeout(timeout);
  }
}
```

### 5.3 Go固有の品質チェック

```go
// Go AI生成コードで頻出する問題と修正パターン

// === 問題パターン1: エラーの握り潰し ===
// BAD: AIがエラー処理を省略
func GetUser(id int) *User {
    user, _ := db.FindUser(id)  // ⚠️ エラー無視
    return user
}

// GOOD: エラーを適切に伝播
func GetUser(ctx context.Context, id int) (*User, error) {
    user, err := db.FindUser(ctx, id)
    if err != nil {
        return nil, fmt.Errorf("finding user %d: %w", id, err)
    }
    if user == nil {
        return nil, ErrUserNotFound
    }
    return user, nil
}


// === 問題パターン2: goroutineリーク ===
// BAD: AIがgoroutineの終了条件を考慮しない
func ProcessItems(items []Item) {
    for _, item := range items {
        go func(i Item) {
            result := heavyProcess(i)  // ⚠️ タイムアウトなし
            saveToDB(result)            // ⚠️ エラーハンドリングなし
        }(item)
    }
    // ⚠️ goroutineの完了を待たない
}

// GOOD: errgroup + contextで制御
func ProcessItems(ctx context.Context, items []Item) error {
    g, ctx := errgroup.WithContext(ctx)
    g.SetLimit(10)  // 同時実行数を制限

    for _, item := range items {
        item := item
        g.Go(func() error {
            ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
            defer cancel()

            result, err := heavyProcess(ctx, item)
            if err != nil {
                return fmt.Errorf("processing item %s: %w", item.ID, err)
            }

            if err := saveToDB(ctx, result); err != nil {
                return fmt.Errorf("saving item %s: %w", item.ID, err)
            }
            return nil
        })
    }

    return g.Wait()
}


// === 問題パターン3: deferの誤用 ===
// BAD: ループ内でdeferを使う
func ProcessFiles(paths []string) error {
    for _, path := range paths {
        f, err := os.Open(path)
        if err != nil {
            return err
        }
        defer f.Close()  // ⚠️ 関数終了まで閉じられない → ファイルディスクリプタ枯渇

        // ファイル処理...
    }
    return nil
}

// GOOD: 個別の関数に分離
func ProcessFiles(paths []string) error {
    for _, path := range paths {
        if err := processFile(path); err != nil {
            return fmt.Errorf("processing %s: %w", path, err)
        }
    }
    return nil
}

func processFile(path string) error {
    f, err := os.Open(path)
    if err != nil {
        return err
    }
    defer f.Close()  // この関数のスコープで確実に閉じられる

    // ファイル処理...
    return nil
}
```

---

## 6. プロンプト設計と品質の関係

### 6.1 高品質コードを生成するプロンプト設計

```python
# 品質指向のプロンプトテンプレート

QUALITY_PROMPT_TEMPLATE = """
## 実装タスク
{task_description}

## 技術要件
- 言語: {language}
- フレームワーク: {framework}
- 対象Python/Nodeバージョン: {runtime_version}

## 品質要件（必須）
1. **型安全性**: 全ての関数に型ヒント/型注釈を付与
2. **エラーハンドリング**: 全ての外部呼び出しにtry-except/try-catchを設置
3. **入力検証**: 公開APIの引数をバリデーション
4. **ログ**: 重要な処理ポイントに構造化ログを追加
5. **テスト**: 実装と同時にユニットテストを生成

## コーディング規約
{coding_standards}

## 既存コードのパターン例
```{language}
{existing_code_example}
```

## 禁止事項
- any型の使用禁止（TypeScript）
- ベアexceptの使用禁止（Python）
- ハードコードされた秘密情報の禁止
- グローバル変数の使用禁止
- sleep/time.sleepによる待機の禁止

## 出力形式
1. 実装コード
2. ユニットテスト
3. 変更点の説明
"""


# プロンプト品質スコアリング
from dataclasses import dataclass

@dataclass
class PromptQualityScore:
    """プロンプトの品質を評価するスコア"""
    has_task_description: bool = False
    has_technical_context: bool = False
    has_quality_requirements: bool = False
    has_coding_standards: bool = False
    has_examples: bool = False
    has_constraints: bool = False
    has_output_format: bool = False

    @property
    def score(self) -> float:
        """0-100のスコアを返す"""
        weights = {
            "has_task_description": 20,
            "has_technical_context": 15,
            "has_quality_requirements": 20,
            "has_coding_standards": 15,
            "has_examples": 15,
            "has_constraints": 10,
            "has_output_format": 5,
        }
        total = sum(
            weight for attr, weight in weights.items()
            if getattr(self, attr)
        )
        return total

    @property
    def grade(self) -> str:
        """品質グレードを返す"""
        s = self.score
        if s >= 90:
            return "A: 高品質プロンプト → 高品質コード期待"
        elif s >= 70:
            return "B: 良好プロンプト → 中程度の品質修正が必要"
        elif s >= 50:
            return "C: 基本プロンプト → 大幅な品質修正が必要"
        else:
            return "D: 不十分 → コード生成前にプロンプト改善必須"

    def improvement_suggestions(self) -> list[str]:
        """改善提案を返す"""
        suggestions = []
        if not self.has_quality_requirements:
            suggestions.append("品質要件（型安全性、エラーハンドリング等）を追加")
        if not self.has_examples:
            suggestions.append("既存コードのパターン例を追加")
        if not self.has_coding_standards:
            suggestions.append("チームのコーディング規約を追加")
        if not self.has_constraints:
            suggestions.append("禁止事項・制約条件を明示")
        if not self.has_technical_context:
            suggestions.append("使用技術スタック・バージョンを明記")
        return suggestions

# 使用例
score = PromptQualityScore(
    has_task_description=True,
    has_technical_context=True,
    has_quality_requirements=True,
    has_coding_standards=False,
    has_examples=False,
    has_constraints=True,
    has_output_format=True,
)
print(f"Score: {score.score}/100")  # 70/100
print(f"Grade: {score.grade}")     # B
print(f"Suggestions: {score.improvement_suggestions()}")
```

### 6.2 段階的生成の具体的手順

```python
# 段階的にAI生成コードの品質を高める実践例

# ステップ1: インターフェース定義を先に生成させる
STEP1_PROMPT = """
以下のユーザーストーリーに基づいて、
Pythonのインターフェース（プロトコル/ABC）を定義してください。
実装は不要です。

ユーザーストーリー:
「管理者として、ユーザーのアクティビティレポートを
 CSV/PDF形式で生成・ダウンロードしたい」

要件:
- レポート生成は非同期で行う
- CSV と PDF の2形式に対応
- 日付範囲でフィルタリング可能
"""

# AIの出力例
from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path
from enum import Enum

class ReportFormat(Enum):
    CSV = "csv"
    PDF = "pdf"

class ReportFilter:
    """レポートのフィルタ条件"""
    def __init__(
        self,
        start_date: date,
        end_date: date,
        user_ids: list[int] | None = None,
    ):
        if start_date > end_date:
            raise ValueError("start_date must be before end_date")
        self.start_date = start_date
        self.end_date = end_date
        self.user_ids = user_ids

class ReportGenerator(ABC):
    """レポート生成のインターフェース"""

    @abstractmethod
    async def generate(
        self,
        report_filter: ReportFilter,
        format: ReportFormat,
    ) -> Path:
        """レポートを生成してファイルパスを返す"""
        ...

    @abstractmethod
    async def get_status(self, report_id: str) -> str:
        """レポート生成の進捗状況を返す"""
        ...


# ステップ2: インターフェースに基づいてテストを生成させる
STEP2_PROMPT = """
上記のReportGeneratorインターフェースに対するテストを生成してください。
正常系3件、異常系3件、境界値2件。
"""


# ステップ3: テストを満たす実装を生成させる
STEP3_PROMPT = """
上記のテストが全て通る ReportGenerator の実装を作成してください。
以下の制約に従ってください:
- SQLAlchemy AsyncSessionでDBアクセス
- レポート生成は10分のタイムアウト付き
- ファイルは一時ディレクトリに保存
"""


# ステップ4: レビュー + 改善指示
STEP4_PROMPT = """
生成された実装について以下の観点でレビューして改善してください:
1. エラーハンドリングの網羅性
2. パフォーマンス（大量データ対応）
3. セキュリティ（パストラバーサル対策）
4. ログ追加
"""
```

---

## 7. コードレビュー自動化ツールの構築

### 7.1 AI生成コード専用レビューボット

```python
# GitHub PRに対してAI生成コードの品質チェックを行うレビューボット

import re
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class ReviewComment:
    """レビューコメント"""
    file: str
    line: int
    severity: str  # "error", "warning", "info"
    category: str  # "security", "performance", "design", "style"
    message: str
    suggestion: str | None = None

class AICodeReviewBot:
    """AI生成コードに特化したレビューボット"""

    # AI生成コードで頻出する問題パターン
    PATTERNS: dict[str, list[dict]] = {
        "python": [
            {
                "name": "hardcoded_secret",
                "pattern": r'(?:password|secret|api_key|token)\s*=\s*["\'][^"\']+["\']',
                "severity": "error",
                "category": "security",
                "message": "ハードコードされた秘密情報が検出されました",
                "suggestion": "環境変数またはシークレットマネージャーを使用してください",
            },
            {
                "name": "bare_except",
                "pattern": r'except\s*:',
                "severity": "error",
                "category": "design",
                "message": "ベアexceptが検出されました",
                "suggestion": "具体的な例外クラスをキャッチしてください（例: except ValueError:）",
            },
            {
                "name": "mutable_default",
                "pattern": r'def\s+\w+\([^)]*(?::\s*list|:\s*dict|:\s*set)\s*=\s*(?:\[\]|\{\}|set\(\))',
                "severity": "warning",
                "category": "design",
                "message": "ミュータブルなデフォルト引数が検出されました",
                "suggestion": "None をデフォルトにして関数内で初期化してください",
            },
            {
                "name": "sync_in_async",
                "pattern": r'async\s+def.*\n(?:.*\n)*?.*requests\.(get|post|put|delete)',
                "severity": "error",
                "category": "performance",
                "message": "非同期関数内で同期HTTPクライアントが使用されています",
                "suggestion": "httpx.AsyncClient または aiohttp を使用してください",
            },
            {
                "name": "no_type_hints",
                "pattern": r'def\s+\w+\([^:)]*\)\s*:',
                "severity": "warning",
                "category": "style",
                "message": "型ヒントが不足しています",
                "suggestion": "全ての引数と戻り値に型ヒントを追加してください",
            },
            {
                "name": "string_format_sql",
                "pattern": r'(?:execute|query)\s*\(\s*f["\']|(?:execute|query)\s*\([^)]*%\s',
                "severity": "error",
                "category": "security",
                "message": "SQLインジェクションのリスクが検出されました",
                "suggestion": "パラメータ化クエリを使用してください",
            },
        ],
        "typescript": [
            {
                "name": "any_type",
                "pattern": r':\s*any\b',
                "severity": "warning",
                "category": "style",
                "message": "any型が使用されています",
                "suggestion": "具体的な型またはunknownを使用してください",
            },
            {
                "name": "empty_catch",
                "pattern": r'catch\s*\([^)]*\)\s*\{\s*\}',
                "severity": "error",
                "category": "design",
                "message": "空のcatchブロックが検出されました",
                "suggestion": "エラーをログに記録するか再throwしてください",
            },
            {
                "name": "no_error_handling_fetch",
                "pattern": r'await\s+fetch\([^)]+\)(?!\s*\.then|\s*;?\s*\n\s*if\s*\(!)',
                "severity": "warning",
                "category": "design",
                "message": "fetchの結果に対するエラーチェックが不足している可能性があります",
                "suggestion": "response.ok をチェックしてエラーハンドリングを追加してください",
            },
        ],
    }

    def __init__(self):
        self.comments: list[ReviewComment] = []

    def review_file(self, file_path: str, content: str) -> list[ReviewComment]:
        """ファイルをレビューしてコメントを返す"""
        comments = []

        # 言語判定
        ext = Path(file_path).suffix
        lang_map = {".py": "python", ".ts": "typescript", ".tsx": "typescript"}
        lang = lang_map.get(ext)

        if not lang or lang not in self.PATTERNS:
            return comments

        lines = content.split("\n")

        for pattern_def in self.PATTERNS[lang]:
            for match in re.finditer(pattern_def["pattern"], content, re.MULTILINE):
                # マッチ位置から行番号を特定
                line_num = content[:match.start()].count("\n") + 1

                comment = ReviewComment(
                    file=file_path,
                    line=line_num,
                    severity=pattern_def["severity"],
                    category=pattern_def["category"],
                    message=pattern_def["message"],
                    suggestion=pattern_def.get("suggestion"),
                )
                comments.append(comment)

        self.comments.extend(comments)
        return comments

    def generate_pr_review(self) -> str:
        """PR用のレビューサマリーを生成"""
        if not self.comments:
            return "AI生成コードレビュー: 問題は検出されませんでした ✅"

        errors = [c for c in self.comments if c.severity == "error"]
        warnings = [c for c in self.comments if c.severity == "warning"]

        lines = [
            "## AI生成コード レビュー結果\n",
            f"- エラー: {len(errors)}件",
            f"- 警告: {len(warnings)}件\n",
        ]

        if errors:
            lines.append("### エラー（修正必須）\n")
            for c in errors:
                lines.append(f"- **{c.file}:{c.line}** [{c.category}] {c.message}")
                if c.suggestion:
                    lines.append(f"  - 提案: {c.suggestion}")

        if warnings:
            lines.append("\n### 警告（確認推奨）\n")
            for c in warnings:
                lines.append(f"- **{c.file}:{c.line}** [{c.category}] {c.message}")
                if c.suggestion:
                    lines.append(f"  - 提案: {c.suggestion}")

        return "\n".join(lines)
```

### 7.2 pre-commitフックによるAI生成コード品質保証

```yaml
# .pre-commit-config.yaml
# AI生成コードの品質を自動チェックするpre-commitフック

repos:
  # Python: フォーマット + リント
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  # Python: 型チェック
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.13.0
    hooks:
      - id: mypy
        additional_dependencies: [types-requests, pydantic]
        args: [--strict]

  # セキュリティ: 秘密情報スキャン
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.21.0
    hooks:
      - id: gitleaks

  # セキュリティ: Python SAST
  - repo: https://github.com/PyCQA/bandit
    rev: 1.8.0
    hooks:
      - id: bandit
        args: [-r, -ll]
        exclude: tests/

  # TypeScript: ESLint
  - repo: local
    hooks:
      - id: eslint
        name: eslint
        entry: npx eslint --max-warnings 0
        language: system
        types: [typescript]

  # カスタム: AI生成コード品質チェック
  - repo: local
    hooks:
      - id: ai-code-quality
        name: AI Code Quality Check
        entry: python scripts/ai_code_quality_check.py
        language: python
        types: [python, typescript]
        additional_dependencies: [pyyaml]
```

```python
# scripts/ai_code_quality_check.py
# pre-commitフック用のAI生成コード品質チェッカー

import ast
import sys
from pathlib import Path

class PythonQualityChecker(ast.NodeVisitor):
    """Python ASTを解析して品質問題を検出"""

    def __init__(self, filename: str):
        self.filename = filename
        self.issues: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """関数定義の品質チェック"""
        # 型ヒントチェック
        if not node.returns:
            self.issues.append(
                f"{self.filename}:{node.lineno}: "
                f"関数 '{node.name}' に戻り値の型ヒントがありません"
            )

        for arg in node.args.args:
            if arg.arg != "self" and not arg.annotation:
                self.issues.append(
                    f"{self.filename}:{node.lineno}: "
                    f"引数 '{arg.arg}' に型ヒントがありません"
                )

        # 関数の行数チェック（50行以上は警告）
        end_line = node.end_lineno or node.lineno
        func_lines = end_line - node.lineno
        if func_lines > 50:
            self.issues.append(
                f"{self.filename}:{node.lineno}: "
                f"関数 '{node.name}' が {func_lines} 行あります（推奨: 50行以下）"
            )

        # docstringチェック
        if not (node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)):
            if not node.name.startswith("_"):
                self.issues.append(
                    f"{self.filename}:{node.lineno}: "
                    f"公開関数 '{node.name}' にdocstringがありません"
                )

        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        """例外ハンドラのチェック"""
        if node.type is None:
            self.issues.append(
                f"{self.filename}:{node.lineno}: "
                f"ベアexceptが検出されました。具体的な例外クラスを指定してください"
            )

        # 空のexceptブロック
        if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
            self.issues.append(
                f"{self.filename}:{node.lineno}: "
                f"空のexceptブロックが検出されました。エラーをログに記録してください"
            )

        self.generic_visit(node)

def check_file(filepath: str) -> list[str]:
    """ファイルを検査して問題リストを返す"""
    path = Path(filepath)
    if path.suffix != ".py":
        return []

    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=filepath)
    except SyntaxError as e:
        return [f"{filepath}:{e.lineno}: 構文エラー: {e.msg}"]

    checker = PythonQualityChecker(filepath)
    checker.visit(tree)
    return checker.issues

def main() -> int:
    """メインエントリポイント"""
    files = sys.argv[1:]
    all_issues: list[str] = []

    for filepath in files:
        issues = check_file(filepath)
        all_issues.extend(issues)

    if all_issues:
        print("AI生成コード品質チェック: 問題が検出されました\n")
        for issue in all_issues:
            print(f"  {issue}")
        print(f"\n合計: {len(all_issues)} 件の問題")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
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

### アンチパターン 3: コンテキスト不足のプロンプト

```python
# BAD: コンテキストなしでAIにコード生成を依頼
# プロンプト: "ユーザー登録APIを作って"

# → AIはフレームワーク、DB、認証方式、バリデーション要件が
#   わからないため、汎用的で低品質なコードを生成する

# GOOD: 十分なコンテキストを提供
# プロンプト:
# """
# FastAPI + SQLAlchemy(async) + PostgreSQL環境で
# ユーザー登録APIを実装してください。
#
# 既存パターン: app/api/v1/auth.py の login エンドポイントを参照
# バリデーション: Pydantic v2のモデルで入力検証
# パスワード: bcryptでハッシュ化
# レスポンス: app/schemas/user.py のUserResponseスキーマを使用
# テスト: tests/api/test_auth.py のパターンに従う
# """
```

### アンチパターン 4: 生成コードの丸コピ

```python
# BAD: 別プロジェクト用にAIが生成したコードをそのまま流用

# プロジェクトA用に生成されたコード（Django + MySQL）
class UserView(APIView):
    def post(self, request):
        serializer = UserSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(serializer.data, status=201)

# プロジェクトBにそのまま持ってくる
# → フレームワーク（FastAPI）、ORM（SQLAlchemy）、DB（PostgreSQL）が
#   全て異なるため動作しない or 設計が不適合

# GOOD: プロジェクト固有のコンテキストで再生成
# プロジェクトBの既存コードパターンをAIに提供し、
# そのプロジェクトの慣習に沿ったコードを新たに生成させる
```

---

## 8. チームでのAI品質運用

### 8.1 品質メトリクスダッシュボード

```python
# チーム全体のAI生成コード品質を可視化するダッシュボード

from dataclasses import dataclass, field
from datetime import datetime, date
from collections import defaultdict

@dataclass
class PRMetrics:
    """PR単位の品質メトリクス"""
    pr_number: int
    author: str
    ai_generated_lines: int
    total_lines: int
    test_coverage: float  # 0-100
    lint_errors: int
    type_errors: int
    security_issues: int
    review_comments: int
    review_time_minutes: int
    merged_at: datetime | None = None
    bugs_found_post_merge: int = 0

@dataclass
class TeamQualityDashboard:
    """チーム品質ダッシュボード"""

    metrics: list[PRMetrics] = field(default_factory=list)

    def ai_code_ratio(self) -> float:
        """AI生成コードの比率"""
        total = sum(m.total_lines for m in self.metrics)
        ai = sum(m.ai_generated_lines for m in self.metrics)
        return (ai / total * 100) if total > 0 else 0

    def average_coverage(self) -> float:
        """平均テストカバレッジ"""
        if not self.metrics:
            return 0
        return sum(m.test_coverage for m in self.metrics) / len(self.metrics)

    def defect_density(self) -> float:
        """欠陥密度（バグ数 / AI生成1000行あたり）"""
        total_ai_lines = sum(m.ai_generated_lines for m in self.metrics)
        total_bugs = sum(m.bugs_found_post_merge for m in self.metrics)
        if total_ai_lines == 0:
            return 0
        return (total_bugs / total_ai_lines) * 1000

    def review_efficiency(self) -> dict[str, float]:
        """レビュー効率の分析"""
        if not self.metrics:
            return {}
        return {
            "avg_review_time_min": sum(m.review_time_minutes for m in self.metrics) / len(self.metrics),
            "avg_comments_per_pr": sum(m.review_comments for m in self.metrics) / len(self.metrics),
            "avg_lines_per_minute": sum(m.total_lines for m in self.metrics) / max(1, sum(m.review_time_minutes for m in self.metrics)),
        }

    def quality_trend(self, period_days: int = 30) -> dict[str, list]:
        """品質トレンド（週次推移）"""
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(days=period_days)
        recent = [m for m in self.metrics if m.merged_at and m.merged_at > cutoff]

        weekly: dict[str, list[PRMetrics]] = defaultdict(list)
        for m in recent:
            if m.merged_at:
                week_key = m.merged_at.strftime("%Y-W%W")
                weekly[week_key].append(m)

        trend = {
            "weeks": [],
            "coverage": [],
            "defects": [],
            "review_time": [],
        }

        for week in sorted(weekly.keys()):
            prs = weekly[week]
            trend["weeks"].append(week)
            trend["coverage"].append(
                sum(p.test_coverage for p in prs) / len(prs)
            )
            trend["defects"].append(
                sum(p.bugs_found_post_merge for p in prs)
            )
            trend["review_time"].append(
                sum(p.review_time_minutes for p in prs) / len(prs)
            )

        return trend

    def generate_report(self) -> str:
        """月次品質レポートを生成"""
        lines = [
            "# AI生成コード品質レポート\n",
            f"- 対象PR数: {len(self.metrics)}",
            f"- AI生成コード比率: {self.ai_code_ratio():.1f}%",
            f"- 平均テストカバレッジ: {self.average_coverage():.1f}%",
            f"- 欠陥密度: {self.defect_density():.2f} bugs/1000行",
            "",
        ]

        efficiency = self.review_efficiency()
        if efficiency:
            lines.extend([
                "## レビュー効率",
                f"- 平均レビュー時間: {efficiency['avg_review_time_min']:.0f}分",
                f"- 平均コメント数: {efficiency['avg_comments_per_pr']:.1f}件/PR",
                f"- レビュー速度: {efficiency['avg_lines_per_minute']:.0f}行/分",
            ])

        return "\n".join(lines)
```

### 8.2 品質改善のフィードバックループ

```
AI生成コードの品質改善サイクル:

┌─────────────────────────────────────────────────┐
│                                                 │
│  ① 生成: AIがコードを生成                       │
│     │                                           │
│     ▼                                           │
│  ② 検証: 自動ゲート + 人間レビュー              │
│     │                                           │
│     ▼                                           │
│  ③ 計測: 品質メトリクスを記録                    │
│     │                                           │
│     ▼                                           │
│  ④ 分析: パターン別の品質傾向を分析              │
│     │                                           │
│     ▼                                           │
│  ⑤ 改善: プロンプト・ルール・CIを更新            │
│     │                                           │
│     └──────────► ① に戻る                       │
│                                                 │
│  各サイクル: 2週間スプリントで回す               │
│  KPI: カバレッジ、欠陥密度、レビュー時間         │
└─────────────────────────────────────────────────┘

改善アクション例:
- セキュリティ問題が多い → プロンプトにセキュリティ要件を追加
- 型エラーが多い → CIに strictモードの型チェックを追加
- N+1問題が多い → レビューチェックリストにDB関連項目を追加
- テストカバレッジが低い → テストファーストパターンを標準化
```

---

## FAQ

### Q1: AI生成コードのレビューにどれくらい時間をかけるべきか？

目安は「AI生成にかかった時間の30-50%」。10分でAIが生成したコードなら3-5分のレビュー。ただし、セキュリティクリティカルな部分（認証、決済、個人情報処理）は通常の2-3倍の時間をかける。レビュー効率を上げるには、Layer 1-3を自動化し、人間はLayer 4-5に集中する。

### Q2: AIが生成したテストコードの品質はどう保証するか？

AIが生成したテスト自体を検証する必要がある。具体的には(1) ミューテーションテスト（Stryker等）でテストの有効性を検証、(2) テストが実際に失敗すべきケースで失敗するか確認（assertを削除して確認）、(3) カバレッジだけでなく、ビジネスルールの網羅性を人間が確認する。

### Q3: チームでAIコード品質基準をどう統一するか？

3つのレベルで統一する。(1) 自動化レベル: CI/CDに品質ゲートを組み込む（lint、型チェック、テスト、カバレッジ）、(2) ガイドラインレベル: 「AIコードレビューチェックリスト」をチームWikiに作成、(3) 文化レベル: 定期的なAIコードレビュー会で知見を共有し、ベストプラクティスを更新する。

### Q4: AI生成コードに著作権やライセンスの問題はあるか？

AI生成コードの著作権は法的にグレーゾーンだが、実務上以下の対策が重要。(1) AIが生成したコードが既存のOSSコードと高い類似性を持っていないか確認する（特にコピーレフトライセンスのコード）、(2) 社内ポリシーでAI生成コードの利用範囲を明確にする、(3) AIツールの利用規約を確認し、生成コードの商用利用が許可されていることを確認する、(4) PRの説明にAI支援の範囲を記録し、監査証跡を残す。

### Q5: AI生成コードのパフォーマンスが悪い場合の対処法は？

AI生成コードのパフォーマンス問題は以下のステップで対処する。(1) ベンチマークテストで問題箇所を特定する（pytest-benchmark、k6等）、(2) プロファイラでボトルネックを可視化する（cProfile、py-spy、Chrome DevTools）、(3) AIに改善プロンプトを投げる際に、具体的なパフォーマンス要件（「1000リクエスト/秒」「レスポンス100ms以下」等）を明示する、(4) 改善前後でベンチマーク結果を比較する。パフォーマンス要件をプロンプトに含めることで、初回生成時から最適化されたコードを得やすくなる。

### Q6: 大規模プロジェクトでAIコード品質を管理するにはどうすればよいか？

大規模プロジェクトでは以下の戦略が有効。(1) モノレポ全体に統一された品質ゲートをCIで強制する（lint、型チェック、テスト、セキュリティスキャン）、(2) CODEOWNERS で各ディレクトリのレビュー担当を明確にし、AI生成コードも必ず人間レビューを通す、(3) 品質メトリクスダッシュボード（本章の8.1参照）でチーム横断的にトレンドを監視する、(4) 月次の品質振り返り会で、AI生成コードの欠陥パターンを共有し、プロンプトテンプレートやCIルールを更新する、(5) 新規メンバーのオンボーディングにAIコード品質ガイドを含める。

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
| 言語別対策 | Python/TypeScript/Goそれぞれの頻出問題を把握 |
| プロンプト品質 | コンテキスト充実が生成コード品質に直結 |

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
4. Microsoft, "Best Practices for AI-Assisted Development," 2025. https://learn.microsoft.com/en-us/ai/
5. Dan Abramov, "A Complete Guide to useEffect," 2024. https://overreacted.io/a-complete-guide-to-useeffect/
6. Go Wiki, "Code Review Comments," 2024. https://go.dev/wiki/CodeReviewComments
